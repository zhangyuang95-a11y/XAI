"""Small deterministic HTTP fixture for exercising the production Web assets."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.warehouse_view import warehouse_map_payload


WEB = ROOT / "ui" / "web"

ROBOT_PATHS = (
    (
        (7, 3), (6, 3), (6, 4), (5, 4),
        (5, 3), (5, 2), (5, 1), (5, 0),
    ),
    (
        (7, 5), (6, 5), (6, 6), (6, 7),
        (6, 8), (6, 7), (6, 6), (6, 5),
    ),
)


class FixtureState:
    def __init__(self) -> None:
        self.stage = "idle"
        self.version = 0
        self.frame = 0
        self.tutorial_index = 0
        self.task1 = None
        self.task2 = None
        self.condition = "explanation"
        self.explanation_target_agent = "robot_2"
        self.last_explanation = None
        self.locale = "en"
        self.operations = {}
        self.command_requests = []
        self.timeline_uploads = 0
        self.reference_requests = 0

    def commands(self):
        values = {
            "idle": (),
            "instructions": ("tutorial_advance", "tutorial_select", "begin_task1"),
            "task1": ("human_action",),
            "task1_complete": ("begin_task2",),
            "explanation": (
                "timeline_select",
                "timeline_back",
                "timeline_forward",
                "ask_explanation",
                "finish_explanation",
            ),
            "task2": ("human_action",),
            "survey": ("submit_survey",),
            "completed": (),
        }[self.stage]
        return ("set_language", "restart", *values) if self.stage != "idle" else values

    @staticmethod
    def path_position(display_frame: int, *, reverse: bool = False):
        path = ROBOT_PATHS[1 if reverse else 0]
        index = int(display_frame) % len(path)
        return list(path[index])

    def snapshot(self, display_frame=None):
        if display_frame is None:
            display_frame = self.tutorial_index if self.stage == "instructions" else self.frame
        score = -int(display_frame)
        return {
            "episode_id": 1,
            "frame": display_frame,
            "total_deliveries": 0,
            "active_count": 2,
            "collision_count": 0,
            "shutdown_count": 0,
            "terminated": False,
            "truncated": False,
            "terminal_reason": None,
            "selected_agent": "robot_1",
            "agents": [
                {
                    "id": "robot_1",
                    "position": self.path_position(display_frame),
                    "battery": max(10, 100 - 2 * (display_frame % 45)),
                    "carrying_task_id": None,
                    "deliveries_completed": 0,
                    "active": True,
                    "heading": "LEFT",
                    "last_action": "LEFT",
                    "last_executed_action": "LEFT",
                    "selected": True,
                    "carrying_label": None,
                },
                {
                    "id": "robot_2",
                    "position": self.path_position(display_frame, reverse=True),
                    "battery": max(10, 100 - 2 * (display_frame % 45)),
                    "carrying_task_id": "task_2",
                    "deliveries_completed": 0,
                    "active": True,
                    "heading": "UP",
                    "last_action": "UP",
                    "last_executed_action": "UP",
                    "selected": False,
                    "carrying_label": "A2",
                },
            ],
            "tasks": [
                {
                    "task_id": "task_1",
                    "pickup_position": [1, 1],
                    "delivery_position": [3, 1],
                    "status": "available",
                    "carrier_agent_id": None,
                    "created_frame": 0,
                    "claimed_frame": None,
                },
                {
                    "task_id": "task_2",
                    "pickup_position": [4, 8],
                    "delivery_position": [2, 7],
                    "status": "carried",
                    "carrier_agent_id": "robot_2",
                    "created_frame": 0,
                    "claimed_frame": 1,
                },
            ],
            "user_score": score,
            "score_breakdown": {
                "delivery": 0,
                "robot_collision": 0,
                "shutdown": 0,
                "time": -self.frame,
                "human_detour": 0,
            },
            "human_route_regret_units": 0,
            "robot_collision_events": 0,
            "invalid_move_count": 0,
            "events": None,
            "policy_hidden": True,
        }

    def transition(self, display_frame=None, *, loop=None):
        if display_frame is None:
            display_frame = self.tutorial_index if self.stage == "instructions" else self.frame
        if display_frame <= 0:
            return None
        reveal = self.stage in {"instructions", "explanation"}
        return {
            "from_frame": display_frame - 1,
            "to_frame": display_frame,
            "loop": self.stage == "explanation" if loop is None else bool(loop),
            "before_stage": self.stage,
            "before_state": self.snapshot(display_frame - 1),
            "agents": [
                {
                    "id": "robot_1",
                    "from_position": self.path_position(display_frame - 1),
                    "to_position": self.path_position(display_frame),
                    "proposed_action": "LEFT",
                    "executed_action": "LEFT",
                    "battery_before": 102 - 2 * display_frame,
                    "battery_after": 100 - 2 * display_frame,
                    "battery_delta": -2,
                    "blocked": False,
                    "invalid": False,
                    "collision": False,
                    "charging": False,
                },
                {
                    "id": "robot_2",
                    "from_position": self.path_position(display_frame - 1, reverse=True),
                    "to_position": self.path_position(display_frame, reverse=True),
                    "proposed_action": "UP" if reveal else None,
                    "executed_action": "UP",
                    "battery_before": 102 - 2 * display_frame,
                    "battery_after": 100 - 2 * display_frame,
                    "battery_delta": -2,
                    "blocked": False,
                    "invalid": False,
                    "collision": False,
                    "charging": False,
                },
            ],
        }

    @staticmethod
    def map_payload():
        return warehouse_map_payload()

    def reference_trajectory(self):
        self.reference_requests += 1
        event_frames = {
            8: ["pickup"], 18: ["conflict"], 22: ["yield"],
            48: ["charger_queue"], 50: ["charging"], 72: ["delivery"],
        }
        return {
            "schema_version": "warehouse-reference-timeline.v2",
            "trajectory_kind": "ai_ai_reference",
            "trajectory_seed": 42026,
            "trajectory_hash": "fixture-reference-hash",
            "agent_control": {"robot_1": "ai", "robot_2": "ai"},
            "map_layout_id": "warehouse_staggered_aisles_8x9_v1_three_cell_exit",
            "map": self.map_payload(),
            "frames": [
                {
                    "index": index,
                    "state": self.snapshot(index),
                    "transition": (
                        self.transition(index, loop=True) if index else None
                    ),
                    "event_tags": event_frames.get(index, []),
                }
                for index in range(121)
            ],
        }

    def view(self):
        tutorial_complete = self.tutorial_index >= 1
        summaries = {}
        if self.task1:
            summaries["task1"] = self.task1
        if self.task2:
            summaries["task2"] = self.task2
        return {
            "map": self.map_payload(),
            "state": self.snapshot(),
            "transition": self.transition(),
            "timeline": {
                "index": self.tutorial_index if self.stage == "instructions" else self.frame,
                "max_index": 2 if self.stage in {"task1_complete", "explanation"} else 1,
                "count": 3 if self.stage in {"task1_complete", "explanation"} else 2,
            },
            "study": {
                "run_id": "fixture-run" if self.stage != "idle" else None,
                "stage": self.stage,
                "state_version": self.version,
                "locale": self.locale,
                "participant_id": "browser-fixture" if self.stage != "idle" else "",
                "condition": self.condition if self.stage != "idle" else None,
                "group_code": (
                    "A" if self.condition == "explanation" else "B"
                ) if self.stage != "idle" else None,
                "group_explanation_available": (
                    self.condition == "explanation"
                ) if self.stage != "idle" else None,
                "test_condition_selector": True,
                "progress": self.frame if self.stage in {"task1", "task2"} else 0,
                "total": 120,
                "round_summaries": summaries,
                "score_delta": (
                    self.task2["score"] - self.task1["score"]
                    if self.task1 and self.task2 and self.stage == "completed"
                    else None
                ),
                "explanation_presented": False,
                "explanation_count": 0,
                "explanation_duration_seconds": 600,
                "explanation_seconds_remaining": 600 if self.stage == "explanation" else None,
                "controlled_agent": "robot_1",
                "explanation_target_agent": self.explanation_target_agent,
                "explanation_target_agents": ["robot_1", "robot_2"],
                "tutorial": {
                    "index": self.tutorial_index,
                    "max_played_index": self.tutorial_index,
                    "total_frames": 2,
                    "complete": tutorial_complete,
                },
                "survey_submitted": self.stage == "completed",
                "allowed_commands": list(self.commands()),
            },
            "trial": None,
            "last_explanation": self.last_explanation,
        }

    def command(self, envelope):
        operation = envelope["operation_id"]
        if operation in self.operations:
            return self.operations[operation]
        command = envelope["command"]
        self.command_requests.append(command)
        if command == "start":
            participant = str(envelope.get("payload", {}).get("participant_id", ""))
            self.explanation_target_agent = "robot_2"
            self.last_explanation = None
            override = str(envelope.get("payload", {}).get("condition_override", "auto"))
            self.condition = (
                override
                if override in {"control", "explanation"}
                else ("control" if "control" in participant.casefold() else "explanation")
            )
            self.locale = str(envelope.get("payload", {}).get("locale", "en"))
            self.stage = "instructions"
        elif command == "set_language":
            self.locale = str(envelope.get("payload", {}).get("locale", self.locale))
        elif command == "tutorial_advance":
            self.tutorial_index = min(1, self.tutorial_index + 1)
        elif command == "begin_task1":
            self.stage = "task1"
            self.frame = 0
        elif command == "human_action" and self.stage == "task1":
            self.frame += 1
            if self.frame >= 2:
                self.task1 = {
                    "round_name": "task1",
                    "seed": 1,
                    "score": -2,
                    "steps": 2,
                    "deliveries": 0,
                    "robot_collisions": 0,
                    "shutdowns": 0,
                    "human_route_regret_units": 0,
                    "mean_delivery_latency": None,
                    "terminal_reason": "horizon",
                }
                self.stage = "task1_complete" if self.condition == "control" else "explanation"
        elif command == "begin_task2" and self.stage == "task1_complete":
            self.stage = "task2"
            self.frame = 0
        elif command == "finish_explanation":
            self.stage = "task2"
            self.frame = 0
        elif command == "ask_explanation" and self.stage == "explanation":
            self.explanation_target_agent = str(
                envelope.get("payload", {}).get("target_agent", "robot_2")
            )
            question = str(envelope.get("payload", {}).get("question", ""))
            text = f"{self.explanation_target_agent}: {question}"
            self.last_explanation = {
                "explanation": text,
                "explanation_document": {"text": text},
                "target_agent": self.explanation_target_agent,
                "question_seed": 123456 + self.version,
            }
        elif command == "human_action" and self.stage == "task2":
            self.frame = 1
            self.task2 = {
                **self.task1,
                "round_name": "task2",
                "seed": 2,
                "score": -1,
                "steps": 1,
            }
            self.stage = "survey"
        elif command == "submit_survey":
            self.stage = "completed"
        elif command == "timeline_select":
            self.frame = max(1, int(envelope.get("payload", {}).get("index", 1)))
        elif command == "timeline_back":
            self.frame = max(1, self.frame - 1)
        elif command == "timeline_forward":
            self.frame = min(2, self.frame + 1)
        self.version += 1
        result = {"run_id": "fixture-run", "state_version": self.version, "view": self.view()}
        self.operations[operation] = result
        return result


STATE = FixtureState()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/view":
            return self.send_json(STATE.view())
        if self.path == "/api/study/reference-trajectory":
            return self.send_json(STATE.reference_trajectory())
        if self.path == "/api/fixture-metrics":
            return self.send_json({
                "command_requests": STATE.command_requests,
                "timeline_uploads": STATE.timeline_uploads,
                "reference_requests": STATE.reference_requests,
            })
        target = {
            "/": WEB / "index.html",
            "/index.html": WEB / "index.html",
            "/assets/styles.css": WEB / "styles.css",
            "/assets/app.js": WEB / "app.js",
            "/assets/favicon.svg": WEB / "favicon.svg",
        }.get(self.path)
        if target is None:
            self.send_error(404)
            return
        content = target.read_bytes()
        content_type = (
            "text/html; charset=utf-8"
            if target.suffix == ".html"
            else "text/css; charset=utf-8"
            if target.suffix == ".css"
            else "text/javascript; charset=utf-8"
            if target.suffix == ".js"
            else "image/svg+xml"
        )
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/study/command":
            return self.send_json(STATE.command(payload))
        if self.path == "/api/study/timeline-events":
            STATE.timeline_uploads += 1
            return self.send_json({"ok": True, "recorded": len(payload.get("events", []))})
        self.send_error(404)

    def send_json(self, payload):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
