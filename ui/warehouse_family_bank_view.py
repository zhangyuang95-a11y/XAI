"""Detached, read-only view of an externally anchored completed family bank.

Loading validates the saved independent replay through the actual finite driver.
It never constructs a runtime/Actor/FamilyQuestionBank or repeats that replay.
The original bank's pure public projection and grading methods are reused;
content availability is separate from model, explanation and study admission.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import re

from backend.training import warehouse_family_bank_run as driver
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from ui import warehouse_family_bank as bank_api
from ui.warehouse_public_history_bank import PublicHistoryQuestionBank as _PureMethods

VERSION = "warehouse-runtime-family-frozen-bank-view.v1"
REQUIRED_EXCLUSIONS = frozenset(("train", "calibration", "validation", "extraction",
                                "explanation_test", "final_test", "play"))
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_ITEM_FIELDS = frozenset(("id", "kind", "scenario_id", "frame", "snapshot", "snapshot_sha256",
                          "preview", "answer", "diversity_key", "evidence", "options", "prompt"))


def _hash(value, name):
    if type(value) is not str or not _HEX.fullmatch(value):
        raise ValueError(f"Explicit external {name} SHA256 is required")
    return value


def _pools(bank, excluded, fixture):
    if (type(excluded) is not dict or not REQUIRED_EXCLUSIONS <= set(excluded)
            or any(type(rows) is not list for rows in excluded.values())
            or (not fixture and any(not excluded[k] for k in REQUIRED_EXCLUSIONS))):
        raise ValueError("Seven complete exclusion pools are required; production pools cannot be empty")
    # Physical fingerprints were verified in the anchored generation/replay.
    # This is an additional saved-identity check, not another physics restore.
    def identity(scene):
        if type(scene) is not dict or type(scene.get("id")) is not str or not scene["id"]:
            raise ValueError("Invalid saved scenario identity")
        _hash(scene.get("fingerprint"), "scenario fingerprint")
        if type(scene.get("snapshot")) is not dict:
            raise ValueError("Missing saved exclusion snapshot")
        return scene["fingerprint"], digest(scene["snapshot"])
    excluded_ids = [identity(scene) for rows in excluded.values() for scene in rows]
    blocked_fingerprints = {x for x, _ in excluded_ids}
    blocked_snapshots = {x for _, x in excluded_ids}
    scenes = bank["pool_scenes"]
    if type(scenes) is not list or not 4 <= len(scenes) <= 100:
        raise ValueError("Invalid independent bank pool")
    found = [identity(scene) for scene in scenes]
    if (len({s["id"] for s in scenes}) != len(scenes) or len({x for x, _ in found}) != len(found)
            or any(x in blocked_fingerprints or y in blocked_snapshots for x, y in found)):
        raise ValueError("Independent bank pool overlaps an excluded pool or itself")
    return {scene["id"]: scene for scene in scenes}


def _items(bank, scenes):
    items = bank["items"]
    required_ids = {f"prediction_{kind}_{i}" for kind in bank_api.KINDS for i in range(1, 5)}
    if (type(items) is not list or len(items) != 8 or any(type(x) is not dict for x in items)
            or {x.get("id") for x in items} != required_ids):
        raise ValueError("Exactly four next-action and four wait-three questions are required")
    for item in items:
        kind, frame = item["kind"], item["frame"]
        if (set(item) != _ITEM_FIELDS or kind not in bank_api.KINDS
                or item["id"] not in {f"prediction_{kind}_{i}" for i in range(1, 5)}
                or item["scenario_id"] not in scenes or type(frame) is not int
                or not bank["minimum_frame"] <= frame < bank["trajectory_steps"]
                or digest(item["snapshot"]) != item["snapshot_sha256"]
                or item["snapshot"]["state"]["frame"] != frame):
            raise ValueError("Question schema, frame or saved snapshot differs")
        preview, snapshot = item["preview"], item["snapshot"]
        expected_keys = {"state", "map", "public_feedback"} | ({"question_markers"} if kind == "wait_three" else set())
        history = {k: v for k, v in snapshot["public_feedback_history"].items()
                   if k not in ("post_state_sha256", "unknown_reason", "previous_counts")}
        if (set(preview) != expected_keys or preview["state"]["frame"] != frame
                or snapshot["public_feedback_mode"] != "observed"
                or digest(preview["public_feedback"]) != digest(history)):
            raise ValueError("Public preview does not retain the same recorded history")
        if type(item["prompt"]) is not dict or set(item["prompt"]) != {"zh", "en"}:
            raise ValueError("Question needs the saved Chinese and English prompts")
        options = item["options"]
        if type(options) is not list or len(options) != (5 if kind == "next_action" else 4):
            raise ValueError("Question options differ")
        values = [x["value"] for x in options]
        if (any(type(x) is not str or not x for x in values) or len(set(values)) != len(values)
                or item["answer"] not in values or any(set(x["label"]) != {"zh", "en"} for x in options)):
            raise ValueError("Invalid saved answer or option labels")
        evidence = item["evidence"]
        if kind == "next_action":
            if (values != list(bank_api.original.ACTIONS)
                    or evidence["runtime_signature"] != bank["runtime_signature"]
                    or evidence["actor_sha256"] != bank["actor_sha256"]
                    or evidence["post_policy_overrides"] != 0 or evidence["masks"] is not False
                    or evidence["policy_actions"] != evidence["proposed_actions"]
                    or item["answer"] != evidence["policy_actions"]["robot_2"]):
                raise ValueError("Saved actual NN decision binding differs")
        else:
            rows = evidence["transitions"]
            if (len(rows) != 3 or evidence["assumed_player_actions"] != ["WAIT"] * 3
                    or digest(rows) != evidence["transitions_sha256"]
                    or digest(rows[0]["before"]) != item["snapshot_sha256"]):
                raise ValueError("Saved wait-three branch evidence differs")
            for row in rows:
                if (row["runtime_signature"] != bank["runtime_signature"] or row["done"] is not False
                        or row["submitted_actions"]["robot_1"] != "WAIT"
                        or row["submitted_actions"]["robot_2"] != row["policy_actions"]["robot_2"]):
                    raise ValueError("Saved counterfactual NN identity or submitted action differs")
    checks = bank_api._checks(items)
    if checks["passed"] is not True or digest(bank["checks"]) != digest(checks):
        raise ValueError("Question content checks did not pass")
    return checks


class FrozenFamilyBank:
    """A validated in-memory snapshot; no constructor or method can sample."""
    def __init__(self, workflow_root, *, expected_prepared_sha256, expected_replay_receipt_sha256,
                 expected_runtime_signature, expected_actor_sha256, expected_exclusions_sha256,
                 allow_test_fixture=False):
        anchors = {"prepared": expected_prepared_sha256, "replay_receipt": expected_replay_receipt_sha256,
                   "runtime_signature": expected_runtime_signature, "actor": expected_actor_sha256,
                   "exclusions": expected_exclusions_sha256}
        for name, value in anchors.items(): _hash(value, name)
        if type(allow_test_fixture) is not bool: raise ValueError("Fixture scope must be an explicit boolean")
        root = driver._absolute(workflow_root)
        # The caller's immutable receipt anchor is checked before the driver,
        # which in turn validates all original records before exposing bytes.
        receipt_raw = driver._read(root / "completion_receipt.json")
        if sha256(receipt_raw).hexdigest() != expected_replay_receipt_sha256:
            raise ValueError("External replay receipt byte anchor differs")
        result = driver.read_completed(root, expected_prepared_sha256=expected_prepared_sha256,
                                       allow_test_fixture=allow_test_fixture)
        if (result["replay_receipt_sha256"] != expected_replay_receipt_sha256
                or digest(result["replay_receipt"]) != digest(json.loads(receipt_raw))):
            raise ValueError("Completed replay receipt changed during loading")
        bank, receipt = result["bank_private_data"], result["replay_receipt"]
        if (set(bank) != bank_api._FIELDS or bank["version"] != bank_api.VERSION
                or bank["runtime_family"] not in ("observed197", "retained_beta197")
                or bank["test_fixture"] is not allow_test_fixture
                or bank["runtime_signature"] != expected_runtime_signature
                or bank["actor_sha256"] != expected_actor_sha256
                or bank["excluded_pools_sha256"] != expected_exclusions_sha256
                or receipt["runtime"]["runtime_signature"] != expected_runtime_signature
                or receipt["runtime"]["actor_sha256"] != expected_actor_sha256
                or result["report"]["independent_replay_completed"] is not True):
            raise ValueError("Externally selected runtime, Actor, exclusion or fixture identity differs")
        exclusion_raw = driver._read(root / "inputs/exclusion_pools.json")
        excluded = json.loads(exclusion_raw)
        if (digest(excluded) != expected_exclusions_sha256
                or driver._binding(exclusion_raw) != receipt["exclusion_pools_binding"]):
            raise ValueError("External complete exclusion pools differ")
        scenes = _pools(bank, excluded, allow_test_fixture)
        self._checks = _items(bank, scenes)
        self._bank, self._items, self._receipt = deepcopy(bank), deepcopy(bank["items"]), deepcopy(receipt)
        self.test_fixture = allow_test_fixture
        self.actor_sha256, self.runtime_signature = expected_actor_sha256, expected_runtime_signature
        self.excluded_pools_sha256 = expected_exclusions_sha256
        self.bank_sha256 = result["bank_binding"]["sha256"]
        self.replay_receipt_sha256 = expected_replay_receipt_sha256
        prepared_raw = driver._read(root / "prepared.json")
        if sha256(prepared_raw).hexdigest() != expected_prepared_sha256:
            raise ValueError("Prepared source binding changed during loading")
        self._sources = {**json.loads(prepared_raw)["identity"]["sources"],
                         str(Path(__file__).relative_to(ROOT)): file_hash(__file__)}
        self.signature = digest({"version": VERSION, "anchors": anchors, "bank_sha256": self.bank_sha256,
                                 "sources": self._sources})

    @property
    def content_eligible(self): return True
    @property
    def eligible(self): return False
    @property
    def participant_enabled(self): return False
    @property
    def formal_ready(self): return False
    @property
    def release_ready(self): return False
    @property
    def checks(self): return deepcopy(self._checks)
    @property
    def bank(self): return deepcopy(self._bank)
    @property
    def items(self): return deepcopy(self._items)
    @property
    def replay_receipt(self): return deepcopy(self._receipt)
    @property
    def sources(self): return deepcopy(self._sources)

    def public_items(self): return _PureMethods.public_items(self)
    def summary(self):
        return {**_PureMethods.summary(self), "version": VERSION, "eligible": False,
                "participant_enabled": False, "independent_replay_previously_verified": True,
                "physics_replay_on_load": False, "runtime_family": self._bank["runtime_family"],
                "bank_sha256": self.bank_sha256, "replay_receipt_sha256": self.replay_receipt_sha256}
    def grade(self, answers):
        if type(answers) is not dict: raise ValueError("Prediction answers must be an object")
        return _PureMethods.grade(self, answers)
