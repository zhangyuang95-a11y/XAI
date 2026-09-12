#!/usr/bin/env python3
"""Synthetic HTTPS service used only to exercise the v8 acceptance harnesses."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal

from scripts.serve_warehouse_r41_diagnostic_v8_https import https_server
from tests.test_warehouse_alignment_online_server import FakeRuntime
from tests.test_warehouse_r41_diagnostic_online_server import diagnostic_context
from ui import warehouse_alignment_online_server as online


class MovingFixtureRuntime(FakeRuntime):
    """Keep the fixture small while producing a real interpolated move."""

    def step(self, env, human_action):
        before = env.snapshot()
        submitted = {"robot_1": human_action, "robot_2": "UP"}
        _, rewards, terminated, truncated, info = env.step(submitted)
        decision = {
            "actor_sha256": self.actor_sha256,
            "frame": before["state"]["frame"],
            "policy_actions": {"robot_2": "UP"},
            "probabilities": {"robot_2": [1, 0, 0, 0, 0]},
            "post_policy_overrides": 0,
        }
        return {
            "before": before, "after": env.snapshot(),
            "decision": decision,
            "policy_actions": {"robot_1": "WAIT", "robot_2": "UP"},
            "submitted_actions": deepcopy(submitted),
            "executed_actions": deepcopy(info["executed_actions"]),
            "info": deepcopy(info), "rewards": rewards,
            "runtime_signature": self.signature,
            "done": bool(terminated or truncated),
        }


class LongFixtureExplainer:
    signature = "explainer-v1"

    def _assert_current(self, runtime):
        assert runtime.signature == "runtime-v1"

    def answer(self, request, record, runtime):
        assert request["frame"] == record["after"]["state"]["frame"]
        assert runtime.signature == record["runtime_signature"]
        if request["language"] == "zh":
            answer = ("机器人2在所选步骤向上移动，因为该动作在当时的公开状态下更接近一个可执行任务；"
                      "这句话只描述已核验的当前步骤，不承诺它之后会继续沿同一路线行动。")
            evidence = ("所选帧、实际提交动作和隔离重放已经核对。此测试文本特意较长，用来检查折叠证据在窄屏中仍可完整阅读。")
        else:
            answer = ("Robot 2 moved upward because that action approached an executable task in the selected public state. "
                      "This describes only the verified step and does not promise that it will keep the same route later.")
            evidence = ("The selected frame, submitted action, and isolated replay were checked. This deliberately long fixture detail tests complete rendering on the narrow viewport.")
        return {"answer": answer, "evidence_detail": evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    fixture_root = Path(args.database).expanduser().resolve().parent / "fixture_root"
    fixture_root.mkdir(parents=True, exist_ok=True)
    context = diagnostic_context(fixture_root)
    context.provenance.update(
        version=online.R41_DIAGNOSTIC_RELEASE_CONTEXT_VERSION_V8,
        actor_sha256="a" * 64,
    )
    context.runtime = MovingFixtureRuntime()
    context.explainer = LongFixtureExplainer()
    context.release["model_version"] = "synthetic-v8-harness-fixture"
    store = online.OnlineAlignmentStudyStore(
        context, database=args.database, storage_mode="ephemeral")
    origin = f"https://{args.host}:{args.port}"
    server = https_server(
        store, host=args.host, port=args.port, public_origin=origin,
        certificate=args.certificate, private_key=args.private_key)
    handlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: (_ for _ in ()).throw(
                KeyboardInterrupt()))
        print(json.dumps({"status": "ready", "origin": origin,
                          "synthetic_fixture": True}, sort_keys=True), flush=True)
        server.serve_forever(poll_interval=0.1)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        store.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
