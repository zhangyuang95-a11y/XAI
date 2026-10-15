#!/usr/bin/env python3
"""Resume the real alignment HTTP QA after a lost response.

This is deliberately separate from the fresh-run checker: the original failed
run remains immutable, while this process recovers the exact pending operation
idempotently and finishes the four already allocated technical sessions.
"""
import argparse
from contextlib import closing
import http.cookiejar
import json
from pathlib import Path
import sqlite3

import check_warehouse_alignment_http as base


VERSION = "warehouse-alignment-real-http-resume.v1"


def jar_for(session_id):
    jar = http.cookiejar.CookieJar()
    jar.set_cookie(http.cookiejar.Cookie(
        0, base.COOKIE, session_id, None, False, "127.0.0.1", False, False,
        "/", True, False, None, True, None, None, {"HttpOnly": None}, False,
    ))
    return jar


def rows(database):
    with closing(base.database(database)) as db:
        values = [dict(row) for row in db.execute(
            "SELECT id,participant_id,condition,task_order,position,stage "
            "FROM sessions WHERE participant_id IS NOT NULL ORDER BY position"
        )]
        base.require(len(values) == 4, "Exactly four allocated technical sessions required")
        base.require({(r["condition"], r["task_order"]) for r in values} == {
            ("A", "XY"), ("A", "YX"), ("B", "XY"), ("B", "YX")
        }, "Existing four-person allocation block differs")
        return values


def confirmed_steps(database, session_ids):
    with closing(base.database(database)) as db:
        placeholders = ",".join("?" for _ in session_ids)
        return db.execute(
            "SELECT count(*) FROM operations WHERE kind='action' AND session_id IN (" +
            placeholders + ")", session_ids
        ).fetchone()[0]


def finish_questionnaire(client):
    items = client.view["questionnaire"]["items"]
    kinds = [item.get("prediction_kind") for item in items]
    base.require(len(items) == 11 and kinds.count("next_action") == 4
                 and kinds.count("wait_three") == 4
                 and sum(item["type"] == "scale" for item in items) == 3,
                 "Frozen questionnaire is not 4+4+3")
    answers = {item["id"]: (4 if item["type"] == "scale"
                            else item["options"][0]["value"]) for item in items}
    partial = dict(list(answers.items())[:4])
    client.command("questionnaire", answers=partial, submit=False)
    client.refresh()
    base.require(client.view["questionnaire"]["draft"] == partial,
                 "Questionnaire draft did not survive refresh")
    client.command("questionnaire", 400, "questionnaire_incomplete",
                   answers={}, submit=True)
    client.command("questionnaire", answers=answers, submit=True)
    client.refresh()


def execute(args):
    context = base.db_identity(args.database, args.expected_manifest_sha256)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((Path(args.failed_run).resolve() / "failure.json").read_text())
    base.require(prior.get("pending", {}).get("payload", {}).get("operation_id"),
                 "Prior failure has no exact pending operation")
    allocations = rows(args.database)
    plan = {
        "version": VERSION, "base": args.base.rstrip("/"),
        "database": str(Path(args.database).resolve()),
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "failed_run": str(Path(args.failed_run).resolve()),
        "failed_run_failure_sha256": base.sha(Path(args.failed_run) / "failure.json"),
        "technical_flow_only": True, "formal_sample": False,
        "recovery": "same operation_id and request digest, then current confirmed state",
        "service_context_sha256": base.digest(context),
    }
    base.save(output / "plan.json", plan)
    harness = base.Harness(plan["base"], output)
    clients = []
    try:
        for allocation in allocations:
            client = base.Client(harness, allocation["participant_id"], jar_for(allocation["id"]))
            client.get()
            base.require(client.view["session_id"] == allocation["id"]
                         and client.view["flow"]["participant_id"] == allocation["participant_id"],
                         "Recovered cookie belongs to another participant")
            clients.append(client)
        session_ids = [row["id"] for row in allocations]
        harness.gameplay_steps = confirmed_steps(args.database, session_ids)
        base.require(harness.gameplay_steps == prior["confirmed_gameplay_steps"] + 1,
                     "Lost response was not committed exactly once")

        pending = prior["pending"]
        pending_client = next(c for c in clients if c.name == pending["client"])
        recovered = harness.request(pending_client, pending["path"], pending["payload"])
        pending_client.view = recovered
        base.require(recovered["version"] == pending["payload"]["expected_version"] + 1,
                     "Idempotent pending action did not return its committed version")
        harness.checks.append("lost_response_recovered_without_duplicate_advance")

        for client, allocation in zip(clients, allocations):
            loops = 0
            while client.view["flow"]["stage"] != "completed":
                loops += 1
                base.require(loops <= 12, "Stage loop exceeded fixed six-round flow")
                stage = client.view["flow"]["stage"]
                if stage in ("task1", "task2"):
                    if stage == "task2" or allocation["condition"] == "B":
                        client.forbidden_question()
                    if stage == "task2":
                        with closing(base.database(args.database)) as db:
                            old = db.execute(
                                "SELECT id FROM questions WHERE session_id=? ORDER BY rowid LIMIT 1",
                                (client.view["session_id"],)
                            ).fetchone()
                        if old:
                            client.command("answer_seen", 403, "explanation_forbidden",
                                           answer_id=old[0])
                    while not client.view["ended"]:
                        base.require(client.view["state"]["frame"] < 120,
                                     "Unterminated round exceeded horizon")
                        client.action("WAIT")
                    base.history(client)
                    client.command("action", 409, "round_ended", action="WAIT")
                    client.command("next")
                elif stage == "questionnaire":
                    finish_questionnaire(client)
                else:
                    raise AssertionError("Unexpected resumable stage: " + str(stage))
            base.require(not client.view["answers"], "Completed view exposed old answers")

        harness.checks.append("remaining_rounds_permissions_history_questionnaire_completed")
        participants = base.final_database_check(args.database, clients, harness)
        checkpoint = [{
            "name": client.name,
            "cookie": next(cookie.value for cookie in client.jar if cookie.name == base.COOKIE),
            "view_sha256": base.digest(client.view),
            "session_id": client.view["session_id"],
        } for client in clients]
        base.save(output / "restart_checkpoint.json", {
            "plan": plan, "clients": checkpoint, "participants": participants,
            "confirmed_gameplay_steps": harness.gameplay_steps,
        })
        with closing(base.database(args.database)) as db:
            questions = [dict(row) for row in db.execute(
                "SELECT session_id,id,language,stage,status,shown FROM questions ORDER BY rowid"
            )]
        report = {
            "version": VERSION, "status": "passed_resumed_real_http_flow",
            "checks": harness.checks, "participants": participants,
            "prior_failure_preserved": str(Path(args.failed_run).resolve() / "failure.json"),
            "prior_http_requests": prior["http_requests"],
            "resume_http_requests": harness.requests,
            "combined_http_requests": prior["http_requests"] + harness.requests,
            "confirmed_gameplay_steps": harness.gameplay_steps,
            "questions": questions,
            "model_capability_claimed": False, "answer_quality_revalidated": False,
            "browser_verified": False, "process_restart_verified": False,
            "formal_sample": False,
        }
        base.save(output / "report.json", report)
        print(base.canonical(report))
        return report
    except BaseException as error:
        base.save(output / "failure.json", {
            "version": VERSION, "error": repr(error), "pending": harness.pending,
            "resume_http_requests": harness.requests,
            "confirmed_gameplay_steps": harness.gameplay_steps,
            "automatic_retry": False, "passed": False,
        })
        raise
    finally:
        harness.journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
