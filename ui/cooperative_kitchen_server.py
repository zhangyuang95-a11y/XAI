"""Independent Kitchen research HTTP application. Run with python -m ui.cooperative_kitchen_server."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.cooperative_kitchen.study import KitchenStudy, ProgramBaseline
from ui.cooperative_kitchen_store import StudyError

ROOT=Path(__file__).resolve().parents[1]
STATIC=ROOT/"ui"/"cooperative_kitchen_web"
COOKIE="policylens_kitchen_research_session"


def configured_study(output=None):
    output=Path(output or os.environ.get("KITCHEN_OUTPUT", ROOT/"output"/"cooperative_kitchen"/"v3-id-pilot"))
    database_url=os.environ.get("DATABASE_URL")
    namespace=os.environ.get("KITCHEN_NAMESPACE","development")
    allow_sqlite=os.environ.get("KITCHEN_ALLOW_SQLITE")=="1"
    allow_freeplay_qa=os.environ.get("KITCHEN_FREEPLAY_QA","0")=="1"
    if not database_url:
        if not allow_sqlite:
            raise RuntimeError("DATABASE_URL is required. For local development only, set KITCHEN_ALLOW_SQLITE=1.")
        output.mkdir(parents=True,exist_ok=True)
        database_url=f"sqlite:///{output/'development.sqlite3'}"
    from backend.cooperative_kitchen.artifacts import load_release
    release=load_release(output)
    policy=ProgramBaseline()
    explainer=None
    if release.get("actor_path"):
        from backend.cooperative_kitchen.policy import NumpyKitchenPolicy
        policy=NumpyKitchenPolicy(release["actor_path"])
        from backend.cooperative_kitchen.explanations import ExplanationEngine
        explainer=ExplanationEngine(policy,program_path=release.get("program_path"),
            extraction_report=release.get("reports",{}).get("extraction",{}))
    return KitchenStudy(output,database_url,namespace=namespace,allow_sqlite=allow_sqlite,
                        policy=policy,explainer=explainer,release=release,
                        enrollment_mode=os.environ.get("KITCHEN_ENROLLMENT_MODE", "closed"),
                        allow_freeplay_qa=allow_freeplay_qa,
                        workers=int(os.environ.get("KITCHEN_QA_WORKERS","8")))


def create_app(study=None, *, start_workers=True, admin_key=None, secure_cookie=None, freeplay_study=None):
    study=study or configured_study()
    if freeplay_study is None:
        if study.store.namespace in {"pilot","confirmatory"}:
            freeplay_study=KitchenStudy(study.output,
                study.store.engine.url.render_as_string(hide_password=False),namespace="development",
                policy=study.policy,explainer=study.explainer,release=study.release,workers=study.workers,
                enrollment_mode="closed",allow_freeplay_qa=study.allow_freeplay_qa)
        else: freeplay_study=study
    applications=[study] if freeplay_study is study else [study,freeplay_study]
    admin_key=admin_key if admin_key is not None else os.environ.get("KITCHEN_ADMIN_KEY","")
    secure_cookie=secure_cookie if secure_cookie is not None else os.environ.get("KITCHEN_SECURE_COOKIE", "1" if study.store.namespace in {"pilot","confirmatory"} else "0")=="1"

    @asynccontextmanager
    async def lifespan(app):
        if start_workers:
            for application in applications: application.start_workers()
        yield
        if start_workers:
            for application in applications: application.stop_workers()

    app=FastAPI(title="PolicyLens Cooperative Kitchen",docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    app.state.study=study
    app.state.freeplay_study=freeplay_study

    @app.middleware("http")
    async def guards(request,call_next):
        # Same-origin cookie mutations. No permissive CORS and no URL credentials.
        if request.url.path.startswith("/api/"):
            if request.method not in {"GET","HEAD","OPTIONS"}:
                origin=request.headers.get("origin")
                if origin and urlparse(origin).netloc != request.headers.get("host"):
                    return JSONResponse({"error":"Cross-origin requests are forbidden","code":"origin_forbidden"},status_code=403)
                if int(request.headers.get("content-length","0") or 0)>65536:
                    return JSONResponse({"error":"Request is too large","code":"request_too_large"},status_code=413)
            response=await call_next(request)
            response.headers["Cache-Control"]="no-store, private"
            response.headers["Pragma"]="no-cache"
        else:
            response=await call_next(request)
            response.headers["Cache-Control"]="no-store" if response.headers.get("content-type","").startswith("text/html") else "no-cache"
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Referrer-Policy"]="same-origin"
        response.headers["X-Frame-Options"]="DENY"
        response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        return response

    @app.exception_handler(StudyError)
    async def known_error(request,exc):
        return JSONResponse({"error":str(exc),"code":exc.code},status_code=exc.status)

    def session(request):
        value=request.cookies.get(COOKIE)
        if not value: raise StudyError("No session",401,"session_not_found")
        # Namespace tags are routing hints only; every service independently
        # verifies the opaque token hash and its namespace in PostgreSQL.
        if value.startswith("d."): return freeplay_study,value[2:]
        if value.startswith("r."): return study,value[2:]
        return study,value

    def admin(request):
        supplied=request.headers.get("x-kitchen-admin-key","")
        if not admin_key or not hmac.compare_digest(supplied,admin_key):
            raise StudyError("Researcher authentication required",401,"admin_required")

    @app.get("/api/status")
    def status(request:Request):
        return JSONResponse(study.status())

    @app.post("/api/session")
    def join(request:Request,payload:dict):
        application=freeplay_study if payload.get("mode","freeplay")=="freeplay" else study
        existing_token=None
        if request.cookies.get(COOKIE):
            previous_application, previous_token=session(request)
            if previous_application is application:
                existing_token=previous_token
            elif previous_application is study and application is freeplay_study:
                # A free-play tab must not overwrite the only recovery token for
                # an enrolled participant in another tab. Ignore a forged or
                # expired cookie, but preserve every valid research run.
                try:
                    with study.store.transaction() as db:
                        previous_run=study.store.run(db,previous_token)
                except StudyError:
                    pass
                else:
                    if previous_run.get("mode")=="pilot":
                        raise StudyError("This browser already has a participant session; refresh to resume it",409,"participant_session_conflict")
        value,view=application.join(payload,existing_token=existing_token)
        response=JSONResponse(view)
        value=("d." if application is freeplay_study and freeplay_study is not study else "r.")+value
        response.set_cookie(COOKIE,value,max_age=30*86400,httponly=True,secure=secure_cookie,samesite="strict",path="/")
        return response

    @app.get("/api/view")
    def view(request:Request):
        application,value=session(request);return application.view(value)

    @app.post("/api/command")
    def command(request:Request,payload:dict):
        application,value=session(request);return application.command(value,payload)

    @app.get("/api/history")
    def history(request:Request,episode_id:str):
        application,value=session(request);return application.history(value,episode_id)

    @app.post("/api/question")
    def ask(request:Request,payload:dict):
        application,value=session(request);return application.ask(value,payload)

    @app.get("/api/question/{question_id}")
    def question(request:Request,question_id:str):
        application,value=session(request);return application.question(value,question_id)

    @app.post("/api/exposure")
    def exposure(request:Request,payload:dict):
        application,value=session(request);return application.exposure(value,payload)

    @app.post("/api/admin/invitations")
    def invitations(request:Request,payload:dict):
        admin(request);return study.create_invitations(payload)

    @app.post("/api/admin/retry")
    def retry(request:Request,payload:dict):
        admin(request);return study.technical_retry(payload)

    @app.get("/api/admin/status")
    def admin_status(request:Request):
        admin(request);return study.admin_status()

    @app.get("/api/admin/export")
    def export(request:Request,format:str="jsonl"):
        admin(request)
        content=study.store.export(format)
        media="text/csv" if format=="csv" else "application/x-ndjson"
        return Response(content,media_type=media,headers={"Content-Disposition":f'attachment; filename="kitchen-{study.store.namespace}.{format}"'})

    if STATIC.is_dir(): app.mount("/",StaticFiles(directory=STATIC,html=True),name="web")
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=int(os.environ.get("PORT","8003")))
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(configured_study(args.output)),host=args.host,port=args.port,access_log=False)


if __name__=="__main__": main()
