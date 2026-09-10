"""Build the hash-bound r4 internal-pilot archive from admitted components.

The existing online loader remains the only deployment reader.  This module
creates the same compact archive format from a new r4 Actor, high-conflict
play scenes, final RCPD program, and independently replayed questionnaire.
It refuses to build unless an external admission record binds every component
and marks each technical gate as passed.  Formal-study readiness remains
false because it requires human evidence.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from backend.training import warehouse_r4_production_admission as production_admission
from backend.training import warehouse_r4_question_bank
from backend.warehouse_alignment_online_explanation import OnlineAlignmentExplainer
from backend.warehouse_alignment_online_runtime import OnlineAlignmentRuntime
from env.warehouse_native.scenarios import scenario_fingerprint
from ui import warehouse_alignment_online_release as portable


VERSION = "warehouse-r4-online-release-builder.v1"
ADMISSION_VERSION = "warehouse-r4-local-pilot-admission.v1"
REQUIRED_GATES = (
    "active_policy", "action_authority", "high_conflict_scenes",
    "explanation_program", "questionnaire", "online_runtime",
)
_ADMISSION_FIELDS = frozenset((
    "version", "status", "formal_ready",
    "human_explanation_effect_validated", "gates", "bindings", "reports",
    "sources", "self_path",
))
_SELECTION_FIELDS = frozenset((
    "version", "actor_sha256", "source_scenario_manifest_sha256",
    "practice", "X", "Y", "pairs", "balance", "selection_score",
))
_SCENE_FIELDS = frozenset((
    "id", "seed", "fingerprint", "task_signature",
    "observed_state_signature", "snapshot",
))
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret(?:[_-]?key)?|password|passwd|"
    r"database[_-]?url|postgres(?:ql)?[_-]?url|authorization|bearer[_-]?token)(?:$|[_-])",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._-]{12,}|"
    r"\b(?:postgres(?:ql)?|mongodb(?:\+srv)?|mysql)://)",
    re.IGNORECASE,
)
ROOT = Path(__file__).resolve().parents[1]
SOURCE_CLOSURE_MANIFEST = (
    ROOT / "backend/training/warehouse_r4_release_source_closure.txt"
)
SOURCE_CLOSURE_SEEDS = (
    "backend/training/warehouse_r4_r3_preextract.py",
    "backend/training/warehouse_r4_active_run.py",
    "backend/training/warehouse_r4_active_evaluation.py",
    "backend/training/warehouse_r4_active_screen.py",
    "backend/training/warehouse_r4_final_rcpd.py",
    "backend/training/warehouse_r4_active_release.py",
    "backend/training/warehouse_r4_explanation_audit.py",
    "backend/training/warehouse_r4_conflict_scene_selection.py",
    "backend/training/warehouse_r4_question_bank.py",
    "backend/training/warehouse_r4_training_ledger.py",
    "backend/training/warehouse_r4_production_admission.py",
    "scripts/build_warehouse_r4_admission.py",
    "scripts/build_warehouse_r4_online_release.py",
    "ui/warehouse_alignment_r4_online_release.py",
)
SOURCE_CLOSURE_ASSETS = (
    "ui/warehouse_family_feedback_research/index.html",
    "ui/warehouse_family_feedback_research/styles.css",
    "ui/warehouse_family_feedback_research/app.js",
    "ui/warehouse_family_feedback_research/favicon.svg",
)
# These three modules are reachable from the r4 packaging entry point but are
# deliberately authenticated by the portable source group.  The two maps must
# remain disjoint; their union is the executable local import closure.
PORTABLE_DELEGATED_SOURCES = frozenset((
    "backend/warehouse_alignment_online_explanation.py",
    "backend/warehouse_alignment_online_runtime.py",
    "ui/warehouse_alignment_online_release.py",
))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _release_source_paths() -> tuple[Path, ...]:
    """Read the frozen, repository-relative r4 production import closure."""
    manifest = SOURCE_CLOSURE_MANIFEST
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("R4 release source-closure manifest is missing or unsafe")
    try:
        names = manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("R4 release source-closure manifest is not UTF-8") from error
    if not names or names != sorted(set(names)):
        raise ValueError("R4 release source-closure manifest must be sorted and unique")
    portable = PORTABLE_DELEGATED_SOURCES
    required = {
        "backend/training/warehouse_r4_active_trainer.py",
        "backend/training/warehouse_r4_final_rcpd.py",
        "backend/training/warehouse_r4_training_ledger.py",
        "backend/training/warehouse_family_stable_run.py",
        "scripts/build_warehouse_r4_admission.py",
        "scripts/build_warehouse_r4_online_release.py",
        "ui/warehouse_alignment_r4_online_release.py",
        "ui/warehouse_family_feedback_research/app.js",
    }
    expected = production_admission.local_source_hashes(
        tuple(ROOT / name for name in (*SOURCE_CLOSURE_SEEDS,
                                        *SOURCE_CLOSURE_ASSETS)))
    expected_names = sorted(set(expected) - portable)
    if (names != expected_names or not required.issubset(names)
            or portable.intersection(names)):
        raise ValueError("R4 release source-closure boundary differs")
    paths = []
    for name in names:
        part = PurePosixPath(name)
        if (part.is_absolute() or ".." in part.parts or "." in part.parts
                or part.as_posix() != name or not part.parts
                or part.parts[0] not in {"backend", "core", "env", "scripts", "ui"}):
            raise ValueError("Unsafe R4 release source path: " + name)
        path = ROOT.joinpath(*part.parts)
        if (not path.is_file() or path.is_symlink()
                or path.resolve() != path.absolute()):
            raise ValueError("R4 release source is missing or unsafe: " + name)
        paths.append(path)
    return (manifest, *paths)


def release_sources() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in _release_source_paths()
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return portable._parse_json(path.read_bytes(), str(path))


def _component_path(value: str | Path, label: str) -> Path:
    """Require an original regular repository file, never a followed link."""
    supplied = Path(value).expanduser()
    if not supplied.is_absolute():
        supplied = ROOT / supplied
    supplied = supplied.absolute()
    if (supplied.is_symlink() or not supplied.is_file()
            or supplied.resolve() != supplied or ROOT not in supplied.parents):
        raise ValueError(label + " must be a canonical regular repository file")
    return supplied


def _reject_secret_material(value: Any, label: str, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"Non-string field in {label}: {path}")
            if _SENSITIVE_KEY.search(key):
                raise ValueError(f"Credential-like field cannot enter {label}: {path}.{key}")
            _reject_secret_material(item, label, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secret_material(item, label, f"{path}[{index}]")
    elif isinstance(value, str) and _SENSITIVE_VALUE.search(value):
        raise ValueError(f"Credential-like value cannot enter {label}: {path}")


def _play_scenes(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    practice = selection.get("practice")
    x, y = selection.get("X"), selection.get("Y")
    if (set(selection) != _SELECTION_FIELDS
            or selection.get("version") != production_admission.SCENE_VERSION
            or not isinstance(practice, dict) or not isinstance(x, list)
            or not isinstance(y, list) or len(x) != 3 or len(y) != 3):
        raise ValueError("Exact r4 practice and X/Y scene selection required")
    play = [practice, *x, *y]
    if (any(not isinstance(scene, dict) or set(scene) != _SCENE_FIELDS
            for scene in play)
            or any(type(scene.get("id")) is not str or not scene["id"]
                   or type(scene.get("seed")) is not int
                   or any(type(scene.get(key)) is not str
                          or portable._HEX.fullmatch(scene[key]) is None
                          for key in ("fingerprint", "task_signature",
                                      "observed_state_signature"))
                   for scene in play)
            or len({scene.get("id") for scene in play}) != 7
            or len({scene.get("seed") for scene in play}) != 7
            or len({scene.get("fingerprint") for scene in play}) != 7
            or any(not isinstance(scene.get("snapshot"), dict) for scene in play)):
        raise ValueError("Seven distinct r4 physical play scenes are required")
    portable._sha(selection.get("source_scenario_manifest_sha256"),
                  "r4 source scenario manifest")
    expected_pairs = [[x[index]["id"], y[index]["id"]] for index in range(3)]
    if selection.get("pairs") != expected_pairs:
        raise ValueError("R4 X/Y scene pairing differs")
    return deepcopy(play)


def _validate_scene_runtime(runtime: OnlineAlignmentRuntime,
                            scene: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every public scene identity and one real submitted action."""
    env = runtime.environment(scene)
    before = digest(env.snapshot())
    actions, decision = runtime.decision(env)
    task_signature = digest(sorted(
        (task.pickup_position, task.delivery_position) for task in env.state.tasks
    ))
    observed_signature = digest({
        "tasks": task_signature,
        "agents": [{"position": agent.position, "heading": agent.heading,
                    "battery": agent.battery} for agent in env.state.agents],
    })
    branch = runtime.clone(env)
    transition = runtime.step(branch, "WAIT")
    valid = (
        int(env.state.frame) == 0
        and env.public_history().get("valid") is False
        and digest(env.snapshot()) == before
        and scenario_fingerprint(env) == scene.get("fingerprint")
        and task_signature == scene.get("task_signature")
        and observed_signature == scene.get("observed_state_signature")
        and decision.get("post_policy_overrides") == 0
        and actions.get("robot_2") == transition.get("policy_actions", {}).get("robot_2")
        and transition.get("policy_actions", {}).get("robot_2")
            == transition.get("submitted_actions", {}).get("robot_2")
        and transition.get("decision", {}).get("post_policy_overrides") == 0
    )
    if not valid:
        raise ValueError("R4 play scene failed identity or action-authority validation")
    return {"id": scene["id"], "policy_action": actions["robot_2"]}


def _inside_root(value: Any) -> Path:
    part = PurePosixPath(value) if isinstance(value, str) and value else None
    if (part is None or part.is_absolute() or ".." in part.parts
            or part.as_posix() != value):
        raise ValueError("R4 admission paths must be non-empty repo-relative paths")
    path = ROOT.joinpath(*part.parts)
    if (path == ROOT or ROOT not in path.parents or path.resolve() != path
            or path.is_symlink()):
        raise ValueError("R4 admission path is unsafe or escapes the repository")
    return path


def _validate_admission(admission: Mapping[str, Any], *, admission_path: Path,
                        admission_sha256: str,
                        runtime, explainer, selection, question,
                        selected_scenes_path: str | Path,
                        scenarios_path: str | Path):
    if (set(admission) != _ADMISSION_FIELDS
            or admission.get("version") != ADMISSION_VERSION
            or admission.get("status") != "local_pilot_technically_verified"
            or admission.get("formal_ready") is not False
            or admission.get("human_explanation_effect_validated") is not False):
        raise ValueError("Exact non-formal r4 admission record required")
    gates = admission.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATES) \
            or any(gates[name] is not True for name in REQUIRED_GATES):
        raise ValueError("Every r4 technical gate must pass before packaging")
    bindings = admission.get("bindings")
    expected = {
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256,
        "runtime_signature": runtime.signature,
        "program_sha256": explainer.program_sha256,
        "explainer_signature": explainer.signature,
        "selected_scenes_sha256": digest(selection),
        "question_bank_sha256": digest(question),
        "question_bank_signature": question.get("source_bank_signature"),
    }
    if not isinstance(bindings, dict) or bindings != expected:
        raise ValueError("R4 admission component binding differs")
    reports = admission.get("reports")
    if not isinstance(reports, dict) or set(reports) != set(production_admission.REPORT_NAMES):
        raise ValueError("R4 admission reports changed")
    try:
        report_paths, report_objects, report_hashes = {}, {}, {}
        for name, value in reports.items():
            if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
                raise ValueError
            path = _inside_root(value["path"])
            portable._sha(value["sha256"], "r4 admission report")
            if not path.is_file() or file_hash(path) != value["sha256"]:
                raise ValueError
            report_paths[name] = path
            report_hashes[name] = value["sha256"]
            report_objects[name] = _read_json(path)
    except (KeyError, TypeError, ValueError, OSError):
        raise ValueError("R4 admission reports changed") from None
    if len(report_paths) != len(set(report_paths.values())):
        raise ValueError("R4 admission report paths must be distinct")
    scenarios_path = Path(scenarios_path).resolve()
    if report_paths["validation_scenarios"] != scenarios_path:
        raise ValueError("R4 admission validation scenario path differs")
    context = production_admission.component_context(
        runtime=runtime, explainer=explainer, selection=selection,
        question=question, scenarios=report_objects["validation_scenarios"],
        selected_scenes_path=selected_scenes_path, scenarios_path=scenarios_path,
    )
    recomputed_gates = production_admission.validate_report_set(
        reports=report_objects, report_hashes=report_hashes,
        report_paths=report_paths, context=context, selection=selection,
        question=question, scenarios=report_objects["validation_scenarios"],
    )
    if gates != recomputed_gates:
        raise ValueError("R4 admission gates differ from real report evidence")
    sources = admission.get("sources")
    if sources != release_sources():
        raise ValueError("R4 admission source binding differs")
    self_path = admission_path.resolve()
    if (not isinstance(admission.get("self_path"), str)
            or _inside_root(admission["self_path"]) != self_path
            or file_hash(self_path) != admission_sha256):
        raise ValueError("R4 admission file changed")
    return deepcopy(sources)


def build(*, actor_path: str | Path, protocol_path: str | Path,
          program_path: str | Path, selected_scenes_path: str | Path,
          question_bank_path: str | Path, scenarios_path: str | Path,
          admission_path: str | Path,
          expected_admission_sha256: str, output_package: str | Path,
          output_base64: str | Path | None = None) -> dict[str, Any]:
    actor_path = _component_path(actor_path, "r4 Actor")
    protocol_path = _component_path(protocol_path, "r4 protocol")
    program_path = _component_path(program_path, "r4 program")
    selected_scenes_path = _component_path(selected_scenes_path, "r4 selected scenes")
    question_bank_path = _component_path(question_bank_path, "r4 question bank")
    scenarios_path = _component_path(scenarios_path, "r4 scenario manifest")
    selection = _read_json(selected_scenes_path)
    question = _read_json(question_bank_path)
    admission_path = _component_path(admission_path, "r4 admission")
    admission = _read_json(admission_path)
    if file_hash(admission_path) != expected_admission_sha256:
        raise ValueError("Admission bytes differ from the external SHA-256")
    protocol = _read_json(protocol_path)
    program_payload = _read_json(program_path)
    _reject_secret_material(protocol, "r4 protocol")
    _reject_secret_material(program_payload, "r4 program")
    _reject_secret_material(selection, "r4 scene selection")
    _reject_secret_material(question, "r4 questionnaire")
    runtime = OnlineAlignmentRuntime(
        actor_path, protocol=protocol,
        expected_actor_sha256=file_hash(actor_path),
        expected_protocol_sha256=digest(protocol), allow_test_fixture=False,
    )
    explainer = OnlineAlignmentExplainer(
        program_path, expected_program_sha256=file_hash(program_path),
        runtime=runtime, allow_test_fixture=False,
    )
    _reject_secret_material(runtime.actor.metadata, "r4 Actor metadata")
    warehouse_r4_question_bank.validate_payload(runtime, question)
    play = _play_scenes(selection)
    if selection.get("actor_sha256") != runtime.actor_sha256:
        raise ValueError("Selected scenes were audited with another Actor")
    for scene in play:
        _validate_scene_runtime(runtime, scene)
    source_scene_sha = digest(selection)
    scenarios = {
        "version": "warehouse-alignment-portable-play-scenes.v1",
        "test_fixture": False,
        "source_scenario_manifest_sha256": source_scene_sha,
        "splits": {"play": play},
    }
    identities = {
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256,
        "program_sha256": explainer.program_sha256,
        "parent_runtime_signature": runtime.signature,
        "parent_explainer_signature": explainer.signature,
        "parent_question_bank_signature": question.get("source_bank_signature"),
        "question_bank_private_items_sha256": question.get("private_items_sha256"),
        "question_bank_public_items_sha256": question.get("public_items_sha256"),
        "portable_scenarios_sha256": digest(scenarios),
        "play_scene_count": len(play),
        "play_scene_ids": [scene["id"] for scene in play],
    }
    portable._validate_question_projection(question, identities)
    sources = _validate_admission(
        admission, admission_path=admission_path,
        admission_sha256=expected_admission_sha256,
        runtime=runtime, explainer=explainer, selection=selection,
        question=question, selected_scenes_path=selected_scenes_path,
        scenarios_path=scenarios_path,
    )
    portable_sources = portable._portable_sources()
    if set(sources) & set(portable_sources):
        raise ValueError("R4 and portable runtime source groups overlap")
    actor_raw = actor_path.read_bytes()
    program_raw = program_path.read_bytes()
    if (sha256(actor_raw).hexdigest() != runtime.actor_sha256
            or sha256(program_raw).hexdigest() != explainer.program_sha256
            or runtime.verify_binding() != runtime.signature):
        raise ValueError("R4 Actor, program, or runtime changed during packaging")
    explainer._assert_current(runtime)
    artifacts = {
        "actor": actor_raw,
        "protocol": (canonical(protocol) + "\n").encode("utf-8"),
        "play_scenarios": (canonical(scenarios) + "\n").encode("utf-8"),
        "program": program_raw,
        "question_bank": (canonical(question) + "\n").encode("utf-8"),
    }
    records = portable._artifact_records(artifacts)
    message = {
        "zh": "r4 内部预实验：主动型神经队友、解释、冲突关卡与问卷已通过技术验收；正式研究结论仍待人类实验。",
        "en": "R4 internal pilot: the active neural teammate, explanations, conflict scenes, and questionnaire passed technical checks; formal conclusions still require a human study.",
    }
    manifest = {
        "version": portable.VERSION,
        "status": portable.STATUS,
        "test_fixture": False,
        "formal_ready": False,
        "parent": {
            "version": ADMISSION_VERSION,
            "status": "local_pilot_technically_verified",
            "manifest_sha256": expected_admission_sha256,
            "context_signature": digest({"admission": admission,
                                         "runtime": runtime.signature}),
            "source_binding_sha256": digest(sources),
            "scenario_manifest_sha256": source_scene_sha,
        },
        "artifacts": records,
        "identities": identities,
        "sources": {"parent": sources, "portable": portable_sources},
        "analysis": portable._analysis_protocol(),
        "release": portable._release_projection(message),
    }
    _reject_secret_material(manifest, "r4 archive manifest")
    portable._validate_manifest(manifest)
    raw = portable._archive_bytes(manifest, artifacts)
    package = portable._write_new(output_package, raw)
    encoded_path = None
    encoded = __import__("base64").b64encode(raw) + b"\n"
    if len(encoded) > portable.MAX_BASE64_BYTES:
        package.unlink(missing_ok=True)
        raise ValueError("R4 Base64 Secret File exceeds the deployment limit")
    if output_base64 is not None:
        try:
            encoded_path = portable._write_new(output_base64, encoded)
        except BaseException:
            package.unlink(missing_ok=True)
            raise
    manifest_sha = sha256((canonical(manifest) + "\n").encode("utf-8")).hexdigest()
    package_sha = sha256(raw).hexdigest()
    try:
        loaded = portable.load_online_release(
            package_path=package, expected_package_sha256=package_sha,
            expected_manifest_sha256=manifest_sha,
        )
        try:
            if (loaded.runtime.actor_sha256 != runtime.actor_sha256
                    or loaded.runtime.signature != runtime.signature
                    or loaded.explainer.program_sha256 != explainer.program_sha256
                    or loaded.explainer.signature != explainer.signature
                    or len(loaded.question_bank.public_items()) != 8
                    or len(loaded.scenarios["splits"]["play"]) != 7):
                raise ValueError("Independent r4 package reload differs")
            for scene in loaded.scenarios["splits"]["play"]:
                _validate_scene_runtime(loaded.runtime, scene)
        finally:
            loaded.close()
    except BaseException:
        # A package that failed its independent loader is not a deliverable.
        # Remove both copies so a later deployment command cannot pick up the
        # attractive but unverified archive left by a failed build.
        package.unlink(missing_ok=True)
        if encoded_path is not None:
            encoded_path.unlink(missing_ok=True)
        raise
    return {
        "version": VERSION, "status": "built_and_independently_reloaded",
        "package": str(package), "package_size": len(raw),
        "package_sha256": package_sha,
        "base64": str(encoded_path) if encoded_path else None,
        "base64_size": len(encoded), "manifest_sha256": manifest_sha,
        "admission_sha256": expected_admission_sha256,
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256,
        "program_sha256": explainer.program_sha256,
        "question_bank_signature": question["source_bank_signature"],
        "play_scene_ids": identities["play_scene_ids"],
        "formal_ready": False,
    }


__all__ = ["VERSION", "ADMISSION_VERSION", "REQUIRED_GATES", "build",
           "release_sources"]
