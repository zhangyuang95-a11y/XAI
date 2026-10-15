"""Immutable, fail-closed closeout for an exhausted warehouse r4.1 run.

This module is deliberately separate from the frozen trainer and runner.  It
can only close a run after all two million additional PPO joint steps have
been committed in contiguous 50k boundaries and none passed the frozen dual
suite gate.  It never selects an Actor, invokes a post-freeze producer, builds
an archive, or changes Render.

The closeout also retains the two defects discovered while the run was in
progress: the aliased ``fixed_yield`` validation partner and the incomplete
source list stored by the frozen protocol.  The latter is repaired only as
provenance: this record binds the omitted runner/evaluator bytes.  It does not
retroactively claim that the frozen protocol had a complete source closure.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping
from urllib.request import Request, urlopen
import zipfile

from backend.training import warehouse_r41_corrected_partner_audit as corrected_audit
from backend.training import warehouse_r41_production_admission as admission
from backend.training import warehouse_r41_training_ledger as training_ledger
from backend.training.warehouse_native_common import canonical, digest, file_hash


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-budget-failure-closeout.v1"
STATUS = "closed_budget_exhausted_no_eligible_actor"
BOUNDARY_INTERVAL = 50_000
MAXIMUM_ADDITIONAL_JOINT_STEPS = 2_000_000
EXPECTED_BOUNDARY_STEPS = list(range(
    BOUNDARY_INTERVAL, MAXIMUM_ADDITIONAL_JOINT_STEPS + 1, BOUNDARY_INTERVAL,
))
RELEASE_ARTIFACT_NAMES = (
    "production_admission.json",
    "warehouse_r41_online_release.zip",
    "warehouse_r41_online_release.b64",
    "release_receipt.json",
)
OMITTED_PROTOCOL_SOURCES = (
    "backend/training/warehouse_r41_active_run.py",
    "backend/training/warehouse_r41_active_evaluation.py",
)
REQUIRED_TREND_DEFECTS = {
    "validation_partner_alias",
    "incomplete_implementation_source_closure",
    "stale_running_status",
}
R3_RELEASE = {
    "package_path": (
        "output/warehouse_native/"
        "alignment_online_release_r3_animation_render_final2_395m_20260910/"
        "warehouse_alignment_online.zip"
    ),
    "package_sha256": "0d7ea02f618c252b864ead823f353e162e23494471d23095eb3a9c24164082df",
    "manifest_sha256": "39fa89d057112418a8cee9d4a6a5d9ff1b6432edaa668e1ce9acc307a4b849ce",
    "actor_sha256": "309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b",
    "play_scene_count": 12,
    "public_origin": "https://policylens-warehouse-study.onrender.com",
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _strict_json(path: str | Path, label: str) -> dict[str, Any]:
    supplied = Path(path).expanduser().absolute()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(label + " must be a canonical regular file")
    path = supplied.resolve()

    def pairs(items):
        value = {}
        for key, child in items:
            if key in value:
                raise ValueError("Duplicate JSON key in " + label)
            value[key] = child
        return value

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)),
    )
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object")
    return value


def _repo_file(path: str | Path, label: str) -> tuple[Path, str]:
    supplied = Path(path).expanduser().absolute()
    root = ROOT.resolve()
    if supplied.is_symlink() or not supplied.is_file():
        raise ValueError(label + " must be a canonical regular file")
    path = supplied.resolve()
    if path == root:
        raise ValueError(label + " must be a canonical regular file")
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        raise ValueError(label + " must stay inside the repository") from None
    return path, relative


def _external_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError("Exact external lowercase SHA-256 required for " + label)
    return value


def source_closure() -> dict[str, str]:
    """Bind the full closeout and release-gate implementation closure."""
    from backend.training.warehouse_r4_production_admission import local_source_hashes
    sources = local_source_hashes((
        Path(__file__), Path(training_ledger.__file__), Path(corrected_audit.__file__),
        Path(admission.__file__), ROOT / "scripts/run_warehouse_r41_postfreeze_release.py",
        ROOT / "scripts/preflight_warehouse_r41_render.py",
        ROOT / "scripts/build_warehouse_r41_failure_closeout.py",
    ))
    for relative in OMITTED_PROTOCOL_SOURCES:
        sources[relative] = file_hash(ROOT / relative)
    return dict(sorted(sources.items()))


def _reconstruct_ledger(run_root: Path, ledger_path: Path,
                        expected_ledger_sha256: str) -> dict[str, Any]:
    ledger = training_ledger.read_saved_ledger(
        ledger_path, expected_sha256=expected_ledger_sha256,
        require_selected=False,
    )
    with tempfile.TemporaryDirectory(prefix="warehouse-r41-failure-ledger-") as tmp:
        rebuilt = training_ledger.build(run_root, Path(tmp) / "ledger.json")
    if canonical(rebuilt) != canonical(ledger):
        raise ValueError("Failed r4.1 ledger differs from reconstructed boundaries")
    return ledger


def _assert_terminal_failure(ledger: Mapping[str, Any]) -> None:
    boundaries = ledger.get("boundaries")
    if (ledger.get("version") != training_ledger.VERSION
            or ledger.get("status") != "failed_no_eligible_actor"
            or ledger.get("admission_eligible") is not False
            or ledger.get("selected") is not None
            or ledger.get("runtime_action_override") is not False
            or ledger.get("maximum_additional_joint_steps")
                != MAXIMUM_ADDITIONAL_JOINT_STEPS
            or ledger.get("total_actual_additional_joint_steps")
                != MAXIMUM_ADDITIONAL_JOINT_STEPS
            or ledger.get("boundary_interval") != BOUNDARY_INTERVAL
            or not isinstance(boundaries, list)
            or [row.get("step") for row in boundaries] != EXPECTED_BOUNDARY_STEPS):
        raise ValueError("Closeout requires the exact exhausted, unselected r4.1 ledger")
    previous = 0
    for row in boundaries:
        segment = row.get("training_segment", {})
        authority = row.get("action_authority", {})
        required_hashes = (
            "actor_sha256", "actor_parameters_sha256", "checkpoint_sha256",
            "program_sha256", "tree_fit_sha256", "evaluation_sha256",
        )
        if (row.get("selected") is not False
                or segment.get("start_step") != previous
                or segment.get("end_step") != row["step"]
                or segment.get("sha256") is None
                or any(_HEX.fullmatch(str(row.get(name, ""))) is None
                       for name in required_hashes)
                or authority.get("overrides") != 0
                or authority.get("trainable") != authority.get("equal")):
            raise ValueError("A committed r4.1 failure boundary is incomplete or selected")
        previous = row["step"]


def _boundary_receipts(run_root: Path, ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    boundary_root = (run_root / "boundaries").resolve()
    for row in ledger["boundaries"]:
        directory = (boundary_root / f"step_{row['step']:07d}").resolve()
        if directory.is_symlink() or not directory.is_dir() or directory.parent != boundary_root:
            raise ValueError("R4.1 boundary directory differs from its step")
        files: dict[str, str] = {}
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise ValueError("R4.1 boundary evidence contains a link")
            if path.is_dir():
                continue
            if not path.is_file() or path.resolve() != path:
                raise ValueError("R4.1 boundary evidence contains a special file")
            files[path.relative_to(directory).as_posix()] = file_hash(path)
        required = {
            "actor.npz", "checkpoint.pt", "program.json", "summary.json",
            "training_segment.jsonl", "tree_fit.json",
            "evaluation/paired_dual_report.json",
        }
        if not required <= set(files):
            raise ValueError("R4.1 boundary evidence tree is incomplete")
        expected = {
            "actor.npz": row["actor_sha256"],
            "checkpoint.pt": row["checkpoint_sha256"],
            "program.json": row["program_sha256"],
            "tree_fit.json": row["tree_fit_sha256"],
            "training_segment.jsonl": row["training_segment"]["sha256"],
            "evaluation/paired_dual_report.json": row["evaluation_sha256"],
        }
        if any(files[name] != value for name, value in expected.items()):
            raise ValueError("R4.1 boundary files differ from the verified ledger")
        dual = _strict_json(directory / "evaluation/paired_dual_report.json",
                            "boundary dual evaluation")
        if (dual.get("selected") is not False or dual.get("status") != "failed"
                or not isinstance(dual.get("suite_decisions"), dict)):
            raise ValueError("Failure closeout found a passing dual evaluation")
        result.append({
            "step": row["step"],
            "actor_sha256": row["actor_sha256"],
            "selected_under_frozen_gate": False,
            "action_authority": deepcopy(row["action_authority"]),
            "suite_decisions": deepcopy(dual["suite_decisions"]),
            "files": files,
            "evidence_tree_sha256": digest(files),
        })
    return result


def _diagnostic(path: str | Path, expected_sha256: str,
                kind: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path, relative = _repo_file(path, kind + " diagnostic")
    expected = _external_sha(expected_sha256, kind + " diagnostic")
    if file_hash(path) != expected:
        raise ValueError(kind + " diagnostic SHA-256 differs")
    value = _strict_json(path, kind + " diagnostic")
    if value.get("read_only") is False:
        raise ValueError(kind + " diagnostic was not read-only")
    receipt = {
        "path": relative, "sha256": expected,
        "semantic_sha256": digest(value), "version": value.get("version"),
        "status": value.get("status"),
    }
    return value, receipt


def _diagnostic_evidence(paths: Mapping[str, str | Path],
                         shas: Mapping[str, str]) -> dict[str, Any]:
    if set(paths) != {"partner_alias", "energy_gate", "trend"} \
            or set(shas) != set(paths):
        raise ValueError("Exact r4.1 closeout diagnostic set required")
    alias, alias_receipt = _diagnostic(
        paths["partner_alias"], shas["partner_alias"], "partner alias")
    energy, energy_receipt = _diagnostic(
        paths["energy_gate"], shas["energy_gate"], "energy gate")
    trend, trend_receipt = _diagnostic(paths["trend"], shas["trend"], "trend")
    if (alias.get("version") != "warehouse-r41-fixed-yield-alias-diagnostic.v1"
            or alias.get("status") != "confirmed_validation_partner_alias"
            or not alias.get("root_cause", {}).get("alias_logic")
            or not alias.get("registered_eight_group_evidence")
            or energy.get("status") != "passed_consistency_no_attribution_bug"
            or energy.get("gate_subject") != "robot_2_neural_actor"
            or energy.get("registered_limit") != 0
            or trend.get("version") != "warehouse-r41-training-trend-audit.v1"
            or trend.get("read_only") is not True
            or trend.get("scope", {}).get(
                "no_frozen_source_or_training_artifact_modified") is not True):
        raise ValueError("Registered r4.1 diagnostic meaning differs")
    for label, receipt in (
        ("partner alias", alias.get("source_receipt")),
        ("energy gate", energy.get("sources")),
    ):
        if not isinstance(receipt, dict) or not receipt:
            raise ValueError(label + " diagnostic source receipt is missing")
        for relative, expected in receipt.items():
            path = ROOT / relative
            if (path.is_symlink() or not path.is_file()
                    or _HEX.fullmatch(str(expected)) is None
                    or file_hash(path) != expected):
                raise ValueError(label + " diagnostic source bytes differ: " + relative)
    defect_rows = trend.get("confirmed_implementation_defects")
    defect_ids = ({row.get("id") for row in defect_rows}
                  if isinstance(defect_rows, list) else set())
    if not REQUIRED_TREND_DEFECTS <= defect_ids:
        raise ValueError("Trend audit omits a registered release-blocking defect")
    trend_sources = trend.get("source_sha256")
    if not isinstance(trend_sources, dict):
        raise ValueError("Trend audit source receipt is missing")
    for relative, expected in trend_sources.items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Trend audit source bytes differ: " + relative)
    protocol_sources = trend.get("protocol_implementation_sources")
    if not isinstance(protocol_sources, dict) or any(
            relative in protocol_sources for relative in OMITTED_PROTOCOL_SOURCES):
        raise ValueError("Frozen protocol source-closure defect no longer matches evidence")
    return {
        "receipts": {
            "partner_alias": alias_receipt,
            "energy_gate": energy_receipt,
            "trend": trend_receipt,
        },
        "partner_alias_confirmed": True,
        "energy_gate_subject": "robot_2_neural_actor",
        "protocol_source_closure_complete": False,
        "protocol_source_closure_missing": list(OMITTED_PROTOCOL_SOURCES),
        "trend_defects": sorted(defect_ids),
        "corrected_six_partner_gate": {
            "version": corrected_audit.VERSION,
            "protocol_sha256": digest(corrected_audit.PROTOCOL),
            "producer_sources_sha256": digest(corrected_audit.producer_sources()),
            "all_committed_boundaries_replay_required_for_release": True,
            "status": "not_run_without_frozen_selected_actor",
            "admission_eligible": False,
            "reason": (
                "All boundaries failed the frozen dual-suite gate, so none can be "
                "rescued for release by a corrected-partner result."
            ),
        },
    }


def _r3_package_evidence() -> dict[str, Any]:
    path, relative = _repo_file(ROOT / R3_RELEASE["package_path"], "r3 package")
    if file_hash(path) != R3_RELEASE["package_sha256"]:
        raise ValueError("Registered r3 package bytes differ")
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if names.count("manifest.json") != 1:
            raise ValueError("Registered r3 package has no unique manifest")
        raw = archive.read("manifest.json")
    if sha256(raw).hexdigest() != R3_RELEASE["manifest_sha256"]:
        raise ValueError("Registered r3 manifest bytes differ")
    manifest = json.loads(raw)
    if (manifest.get("identities", {}).get("actor_sha256")
            != R3_RELEASE["actor_sha256"]
            or manifest.get("identities", {}).get("play_scene_count")
                != R3_RELEASE["play_scene_count"]
            or manifest.get("formal_ready") is not False):
        raise ValueError("Registered r3 package identity differs")
    return {
        **deepcopy(R3_RELEASE), "package_path": relative,
        "manifest_release_version": manifest.get("parent", {}).get("version"),
        "formal_ready": False,
    }


def _render_env(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        key = re.fullmatch(r"\s*-\s+key:\s*([^\s#]+)\s*", line)
        if key:
            current = key.group(1)
            continue
        value = re.fullmatch(r"\s+value:\s*([^\s#]+)\s*", line)
        if value and current:
            if current in result:
                raise ValueError("Origin render config repeats " + current)
            result[current] = value.group(1).strip("\"'")
            current = None
    return result


def capture_deployment_evidence(*, timeout: float = 30.) -> dict[str, Any]:
    """Perform read-only Git and HTTP checks of the still-active r3 release."""
    if timeout <= 0 or timeout > 60:
        raise ValueError("Deployment evidence timeout must be in (0, 60]")
    commit = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    render_text = subprocess.run(
        ["git", "show", "origin/main:render.yaml"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    responses = {}
    for endpoint in ("/health", "/api/view"):
        request = Request(
            R3_RELEASE["public_origin"] + endpoint,
            headers={"User-Agent": "warehouse-r41-closeout-readonly/1"},
            method="GET",
        )
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(512 * 1024)
            if response.status != 200:
                raise ValueError("Live r3 evidence endpoint did not return HTTP 200")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Live r3 evidence endpoint is not a JSON object")
        responses[endpoint] = {
            "value": value, "response_sha256": sha256(raw).hexdigest(),
            "response_bytes": len(raw),
        }
    return {
        "origin_commit": commit,
        "origin_render_text": render_text,
        "origin_render_sha256": sha256(render_text.encode("utf-8")).hexdigest(),
        "health": responses["/health"],
        "view": responses["/api/view"],
        "http_methods": ["GET"],
    }


def _validate_deployment_evidence(value: Mapping[str, Any],
                                  r3: Mapping[str, Any]) -> dict[str, Any]:
    commit = value.get("origin_commit")
    render_text = value.get("origin_render_text")
    if (not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or not isinstance(render_text, str)
            or value.get("origin_render_sha256")
                != sha256(render_text.encode("utf-8")).hexdigest()
            or value.get("http_methods") != ["GET"]):
        raise ValueError("Deployment retention evidence is malformed")
    env = _render_env(render_text)
    if (env.get("WAREHOUSE_RELEASE_PACKAGE_SHA256") != r3["package_sha256"]
            or env.get("WAREHOUSE_RELEASE_MANIFEST_SHA256")
                != r3["manifest_sha256"]
            or env.get("WAREHOUSE_PUBLIC_ORIGIN") != r3["public_origin"]
            or not re.search(r'^\s*autoDeployTrigger:\s*["\']?off["\']?\s*$',
                             render_text, re.MULTILINE)):
        raise ValueError("Origin deployment config no longer binds the r3 release")
    health_receipt = value.get("health", {})
    view_receipt = value.get("view", {})
    health = health_receipt.get("value", {})
    view = view_receipt.get("value", {})
    if (health.get("status") != "ok"
            or health.get("formal_ready") is not False
            or health.get("version") != "warehouse-alignment-online-study-server.v1"
            or view.get("play_scene_count") != r3["play_scene_count"]
            or view.get("provenance", {}).get("service_version")
                != "warehouse-alignment-online-study-server.v1"
            or view.get("enrollment", {}).get("formal_ready") is not False
            or "r4.1" in canonical({"health": health, "view": view}).lower()):
        raise ValueError("Live public service is not consistent with retained r3")
    return {
        "origin_commit": commit,
        "origin_render_sha256": value["origin_render_sha256"],
        "configured_package_sha256": env["WAREHOUSE_RELEASE_PACKAGE_SHA256"],
        "configured_manifest_sha256": env["WAREHOUSE_RELEASE_MANIFEST_SHA256"],
        "auto_deploy": False,
        "live_health_response_sha256": health_receipt.get("response_sha256"),
        "live_view_response_sha256": view_receipt.get("response_sha256"),
        "live_service_version": health["version"],
        "live_play_scene_count": view["play_scene_count"],
        "read_only_http_methods": ["GET"],
        "r3_retention_evidence_passed": True,
        "actor_identity_basis": (
            "The origin Render configuration still binds the locally reverified r3 "
            "package; the participant-safe live endpoint intentionally omits Actor hashes."
        ),
    }


def _new_output_root(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("Closeout output must stay inside the repository") from None
    if (path.exists() or path.is_symlink() or path.resolve() != path
            or not path.parent.is_dir() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent):
        raise ValueError("Closeout output must be a new canonical directory")
    return path


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def build_closeout(*, run_root: str | Path, ledger_path: str | Path,
                   expected_ledger_sha256: str,
                   diagnostic_paths: Mapping[str, str | Path],
                   diagnostic_sha256: Mapping[str, str],
                   output_root: str | Path,
                   deployment_evidence: Mapping[str, Any] | None = None,
                   allow_test_fixture: bool = False) -> dict[str, Any]:
    """Validate an exhausted run and write exactly one non-release record."""
    supplied_run_root = Path(run_root).expanduser().absolute()
    if supplied_run_root.is_symlink() or not supplied_run_root.is_dir():
        raise ValueError("R4.1 run root must be a canonical directory")
    run_root = supplied_run_root.resolve()
    ledger_path, _ = _repo_file(ledger_path, "r4.1 failure ledger")
    expected_ledger_sha256 = _external_sha(
        expected_ledger_sha256, "training ledger")
    try:
        run_root.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("R4.1 run root must stay inside the repository") from None
    ledger = _reconstruct_ledger(run_root, ledger_path, expected_ledger_sha256)
    _assert_terminal_failure(ledger)
    boundaries = _boundary_receipts(run_root, ledger)
    diagnostics = _diagnostic_evidence(diagnostic_paths, diagnostic_sha256)
    closure = source_closure()
    for relative in OMITTED_PROTOCOL_SOURCES:
        if closure.get(relative) != file_hash(ROOT / relative):
            raise ValueError("Additive closeout did not bind omitted source " + relative)
    r3 = _r3_package_evidence()
    if deployment_evidence is not None and not allow_test_fixture:
        raise ValueError("Supplied deployment evidence is allowed only in a test fixture")
    if deployment_evidence is None:
        deployment_evidence = capture_deployment_evidence()
    deployment = _validate_deployment_evidence(deployment_evidence, r3)
    if allow_test_fixture:
        # This bit is explicit in fixture records and can never qualify a release.
        test_fixture = True
    else:
        test_fixture = False
    report = {
        "version": VERSION,
        "status": STATUS,
        "test_fixture": test_fixture,
        "formal_ready": False,
        "internal_pilot_ready": False,
        "admission_eligible": False,
        "release_eligible": False,
        "deployment_allowed": False,
        "human_explanation_effect_validated": False,
        "diagnostic_models_only": True,
        "reason": (
            "The registered two-million-step r4.1 continuation exhausted its budget "
            "without an Actor passing both frozen validation suites."
        ),
        "training": {
            "ledger_path": ledger_path.relative_to(ROOT.resolve()).as_posix(),
            "ledger_sha256": expected_ledger_sha256,
            "ledger_semantic_sha256": digest(ledger),
            "run_root": run_root.relative_to(ROOT.resolve()).as_posix(),
            "source_r3_actor_sha256": ledger["source"]["actor_sha256"],
            "actual_additional_joint_steps": MAXIMUM_ADDITIONAL_JOINT_STEPS,
            "maximum_additional_joint_steps": MAXIMUM_ADDITIONAL_JOINT_STEPS,
            "boundary_interval": BOUNDARY_INTERVAL,
            "boundary_count": len(boundaries),
            "boundary_steps": EXPECTED_BOUNDARY_STEPS,
            "all_boundaries_failed_frozen_gate": True,
            "all_policy_actions_submitted_exactly": True,
            "action_override_count": 0,
            "boundaries": boundaries,
        },
        "diagnostics": diagnostics,
        "source_closure": {
            "protocol_source_closure_complete": False,
            "additive_closeout_source_closure_complete": True,
            "omitted_by_frozen_protocol": list(OMITTED_PROTOCOL_SOURCES),
            "files": closure,
            "sha256": digest(closure),
        },
        "retained_online_release": {
            **r3, "deployment_evidence": deployment,
            "r41_admission_created": False,
            "r41_package_created": False,
            "r41_base64_secret_created": False,
            "r41_release_receipt_created": False,
            "render_mutation_performed_by_closeout": False,
        },
        "prohibited_release_artifacts": {
            name: False for name in RELEASE_ARTIFACT_NAMES
        },
        "next_round": {
            "automatic_restart": False,
            "automatic_additional_training": False,
            "diagnostic_actor_may_be_packaged": False,
        },
    }
    output = _new_output_root(output_root)
    output.mkdir(mode=0o700)
    try:
        path = output / "failure_closeout.json"
        _write_new(path, report)
        if sorted(child.name for child in output.iterdir()) != ["failure_closeout.json"]:
            raise RuntimeError("Failure closeout created an unexpected release artifact")
    except BaseException:
        for child in output.iterdir():
            if child.is_file() and child.name == "failure_closeout.json":
                child.unlink()
        output.rmdir()
        raise
    return report


def read_saved_closeout(path: str | Path, *, expected_sha256: str,
                        require_current_sources: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    expected_sha256 = _external_sha(expected_sha256, "failure closeout")
    if file_hash(path) != expected_sha256:
        raise ValueError("Failure closeout bytes differ from external SHA-256")
    report = _strict_json(path, "r4.1 failure closeout")
    boundary_rows = report.get("training", {}).get("boundaries")
    retained = report.get("retained_online_release", {})
    diagnostics = report.get("diagnostics", {})
    if (report.get("version") != VERSION or report.get("status") != STATUS
            or report.get("formal_ready") is not False
            or report.get("internal_pilot_ready") is not False
            or report.get("admission_eligible") is not False
            or report.get("release_eligible") is not False
            or report.get("deployment_allowed") is not False
            or report.get("diagnostic_models_only") is not True
            or report.get("prohibited_release_artifacts")
                != {name: False for name in RELEASE_ARTIFACT_NAMES}
            or report.get("training", {}).get("boundary_steps")
                != EXPECTED_BOUNDARY_STEPS
            or report.get("training", {}).get("boundary_count")
                != len(EXPECTED_BOUNDARY_STEPS)
            or report.get("training", {}).get("all_boundaries_failed_frozen_gate")
                is not True
            or report.get("training", {}).get("action_override_count") != 0
            or not isinstance(boundary_rows, list)
            or [row.get("step") for row in boundary_rows] != EXPECTED_BOUNDARY_STEPS
            or any(row.get("selected_under_frozen_gate") is not False
                   or row.get("action_authority", {}).get("overrides") != 0
                   or row.get("action_authority", {}).get("trainable")
                        != row.get("action_authority", {}).get("equal")
                   for row in boundary_rows)
            or diagnostics.get("partner_alias_confirmed") is not True
            or diagnostics.get("protocol_source_closure_complete") is not False
            or diagnostics.get("corrected_six_partner_gate", {}).get(
                "admission_eligible") is not False
            or retained.get("package_sha256") != R3_RELEASE["package_sha256"]
            or retained.get("manifest_sha256") != R3_RELEASE["manifest_sha256"]
            or retained.get("actor_sha256") != R3_RELEASE["actor_sha256"]
            or retained.get("deployment_evidence", {}).get(
                "r3_retention_evidence_passed") is not True
            or any(retained.get(name) is not False for name in (
                "r41_admission_created", "r41_package_created",
                "r41_base64_secret_created", "r41_release_receipt_created",
                "render_mutation_performed_by_closeout",
            ))
            or report.get("next_round") != {
                "automatic_restart": False,
                "automatic_additional_training": False,
                "diagnostic_actor_may_be_packaged": False,
            }):
        raise ValueError("Saved r4.1 failure closeout is not fail-closed")
    closure = report.get("source_closure", {})
    if (closure.get("protocol_source_closure_complete") is not False
            or closure.get("additive_closeout_source_closure_complete") is not True
            or closure.get("omitted_by_frozen_protocol")
                != list(OMITTED_PROTOCOL_SOURCES)
            or closure.get("sha256") != digest(closure.get("files"))):
        raise ValueError("Saved r4.1 closeout source closure differs")
    if require_current_sources and closure.get("files") != source_closure():
        raise ValueError("Saved r4.1 closeout source bytes are no longer current")
    if sorted(child.name for child in path.parent.iterdir()) != [path.name]:
        raise ValueError("Failure closeout directory contains another artifact")
    return report


__all__ = [
    "VERSION", "STATUS", "EXPECTED_BOUNDARY_STEPS", "R3_RELEASE",
    "source_closure", "capture_deployment_evidence", "build_closeout",
    "read_saved_closeout",
]
