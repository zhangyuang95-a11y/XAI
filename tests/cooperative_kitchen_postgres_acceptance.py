"""Actual local HTTP/PostgreSQL acceptance. Results are NOT remote cloud QA capacity evidence."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.cookiejar import CookieJar
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class Client:
    def __init__(self,url):
        self.url=url;self.cookies=CookieJar();self.opener=build_opener(HTTPCookieProcessor(self.cookies))
    def request(self,path,data=None):
        payload=json.dumps(data).encode() if data is not None else None
        request=Request(self.url+path,data=payload,headers={"content-type":"application/json"} if payload else {})
        start=time.perf_counter()
        try:
            with self.opener.open(request,timeout=30) as response:return json.load(response),time.perf_counter()-start
        except HTTPError as exc:
            raise AssertionError(f"{path} HTTP {exc.code}: {exc.read().decode()}") from exc


def command(client,view,name,**kwargs):
    return client.request("/api/command",{"operation_id":uuid.uuid4().hex,"version":view["run"]["version"],"command":name,**kwargs})


def percentile(values,fraction):
    values=sorted(values)
    return values[min(len(values)-1,int((len(values)-1)*fraction))]


def load(url):
    status,_=Client(url).request("/api/status")
    if status.get("qa_configured"):
        raise AssertionError("This acceptance script is for local factual answers only; use a separate cloud-load test when a cloud QA provider is configured")
    barrier=threading.Barrier(20)
    def participant(number):
        client=Client(url)
        view,_=client.request("/api/session",{"operation_id":uuid.uuid4().hex,"mode":"freeplay","language":"en" if number%2 else "zh"})
        view,_=command(client,view,"next")
        run_id=view["run"]["id"];episode=view["run"]["episode_id"]
        barrier.wait(timeout=30)
        latencies=[];answer_times=[];answers=[]
        for step in range(10):
            payload={"operation_id":uuid.uuid4().hex,"version":view["run"]["version"],"command":"action","action":["UP","INTERACT","WAIT","RIGHT","DOWN"][step%5]}
            view,elapsed=client.request("/api/command",payload);latencies.append(elapsed)
            if step in (0,9):
                duplicate,_=client.request("/api/command",payload)
                assert duplicate["state"]["turn"]==view["state"]["turn"]
            assert view["run"]["id"]==run_id and view["run"]["episode_id"]==episode
            assert view["state"]["turn"]==step+1
            if step in (3,7):
                kind="why" if step==3 else "counterfactual"
                before=json.loads(json.dumps(view["state"]))
                begun=time.perf_counter()
                job,_=client.request("/api/question",{"operation_id":uuid.uuid4().hex,"version":view["run"]["version"],"episode_id":episode,"frame":0 if step==3 else step+1,"kind":kind,"question":""})
                while True:
                    answer,_=client.request("/api/question/"+job["id"])
                    if answer["status"] in {"complete","failed","cancelled"}:break
                    if time.perf_counter()-begun>30:raise AssertionError("Local explanation exceeded 30 seconds")
                    time.sleep(.025)
                answer_times.append(time.perf_counter()-begun)
                assert answer["status"]=="complete" and answer["answer"]["verified"]
                assert answer["answer"]["frame"]==(0 if step==3 else step+1)
                exposure={"operation_id":uuid.uuid4().hex,"question_id":job["id"],"event":"shown"}
                client.request("/api/exposure",exposure);client.request("/api/exposure",exposure)
                view,_=client.request("/api/view")
                assert view["state"]==before
                answers.append(job["id"])
        history,_=client.request("/api/history?episode_id="+episode)
        assert len(history["frames"])==11
        return {"run_id":run_id,"episode_id":episode,"actions":latencies,"answers":answer_times,"question_ids":answers,"turn":view["state"]["turn"]}
    start=time.time()
    with ThreadPoolExecutor(max_workers=20) as pool:results=list(pool.map(participant,range(20)))
    action_times=[v for r in results for v in r["actions"]]
    question_times=[v for r in results for v in r["answers"]]
    assert len({r["run_id"] for r in results})==20
    return {"passed":True,"scope":"local HTTP + PostgreSQL with the loaded policy and verified factual explanations; no cloud API calls; not a remote-load gate",
            "policy_kind":status["policy_kind"],"versions":status["versions"],"sessions":20,"actions":len(action_times),"questions":len(question_times),"duplicate_action_retries":40,"duplicate_exposure_retries":40,
            "action_p95_seconds":percentile(action_times,.95),"question_p95_seconds":percentile(question_times,.95),"duration_seconds":time.time()-start,
            "zero_duplicate_steps":True,"zero_session_mixups":True,"zero_confirmed_data_loss":True,"runs":results}


def serve_fixture(dsn,port,workers):
    from backend.cooperative_kitchen.study import KitchenStudy
    from ui.cooperative_kitchen_server import create_app
    import uvicorn
    release={"study_ready":False,"versions":{"fixture":"postgres-recovery-only-v1"}}
    study=KitchenStudy(ROOT/"output/cooperative_kitchen/v1",dsn,namespace="test",release=release,test_mode=True)
    uvicorn.run(create_app(study,start_workers=workers),host="127.0.0.1",port=port,access_log=False,log_level="error")


def recovery(dsn,output):
    import sqlalchemy
    from sqlalchemy.engine import make_url
    engine=sqlalchemy.create_engine(dsn)
    schema="kitchen_recovery_"+uuid.uuid4().hex
    with engine.begin() as db:db.execute(sqlalchemy.text(f'CREATE SCHEMA "{schema}"'))
    isolated=make_url(dsn).update_query_dict({"options":f"-csearch_path={schema}"}).render_as_string(hide_password=False)
    port=8005;url=f"http://127.0.0.1:{port}"
    env=os.environ.copy();env["KITCHEN_ACCEPTANCE_DSN"]=isolated
    process=None;log=(output/"postgres_recovery_server.log").open("ab")
    def launch(workers):
        args=[sys.executable,__file__,"--serve-fixture","--port",str(port)]
        if workers:args.append("--workers")
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        for _ in range(100):
            try:Client(url).request("/api/status");return child
            except Exception:
                if child.poll() is not None:raise AssertionError("Recovery fixture failed to start")
                time.sleep(.05)
        raise AssertionError("Recovery fixture did not become ready")
    try:
        process=launch(False)
        client=Client(url)
        view,_=client.request("/api/session",{"operation_id":"recovery-create","mode":"freeplay"})
        view,_=command(client,view,"next")
        payload={"operation_id":"confirmed-before-kill","version":view["run"]["version"],"command":"action","action":"UP"}
        view,_=client.request("/api/command",payload)
        before=view["state"]
        job,_=client.request("/api/question",{"operation_id":"durable-pending","version":view["run"]["version"],"episode_id":view["run"]["episode_id"],"frame":0,"kind":"counterfactual"})
        assert job["status"]=="pending"
        process.kill();process.wait(timeout=5)
        process=launch(True)
        restored,_=client.request("/api/view")
        assert restored["state"]==before
        repeated,_=client.request("/api/command",payload)
        assert repeated["state"]["turn"]==1
        deadline=time.time()+15
        while True:
            completed,_=client.request("/api/question/"+job["id"])
            if completed["status"]=="complete":break
            assert time.time()<deadline
            time.sleep(.05)
        final,_=client.request("/api/view")
        assert final["state"]==before and completed["answer"]["frame"]==0
        return {"passed":True,"scope":"Actual SIGKILL and fresh HTTP process against PostgreSQL; explicit program fixture, not trained-policy or remote-service acceptance",
                "confirmed_state_restored":True,"pending_job_recovered":True,"old_operation_id_did_not_advance":True,"explanation_did_not_change_state":True,
                "run_id":view["run"]["id"],"question_id":job["id"]}
    finally:
        if process and process.poll() is None:process.terminate();process.wait(timeout=15)
        log.close()
        with engine.begin() as db:db.execute(sqlalchemy.text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8003")
    parser.add_argument("--output",type=Path,default=ROOT/"output/cooperative_kitchen/v1")
    parser.add_argument("--serve-fixture",action="store_true")
    parser.add_argument("--workers",action="store_true")
    parser.add_argument("--port",type=int,default=8005)
    args=parser.parse_args()
    dsn=os.environ.get("KITCHEN_ACCEPTANCE_DSN")
    if args.serve_fixture:return serve_fixture(dsn,args.port,args.workers)
    if not dsn:raise SystemExit("Set KITCHEN_ACCEPTANCE_DSN to the local test PostgreSQL database")
    args.output.mkdir(parents=True,exist_ok=True)
    report={"local_load":load(args.url),"restart":recovery(dsn,args.output)}
    (args.output/"postgres_acceptance.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({"load":{k:v for k,v in report["local_load"].items() if k!="runs"},"restart":report["restart"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
