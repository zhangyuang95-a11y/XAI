#!/usr/bin/env python3
"""Report completed v6.21 BC measurements; never select, fit, reset or train."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v621_bc_protocol as protocol
from experiments.warehouse import v621_bc_selection as checked

RECIPES = ("additive_control", "shared_b035", "shared_b050")
LABELS = {"additive_control": "加性对照BC", "shared_b035": "共享结构主配方BC", "shared_b050": "共享结构备用BC"}
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
ROLES = ("robot_1", "robot_2")


def _number(value, *, optional=False, fraction=False):
    if value is None and optional:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or (fraction and not 0 <= value <= 1):
        raise RuntimeError("native metric must be a finite number in its declared range")
    return value


def _quality(outcomes):
    if not outcomes:
        raise RuntimeError("quality requires recorded natural episode outcomes")
    for row in outcomes:
        if any(type(row.get(k)) is not int or row[k] < 0 for k in ("deliveries", "robot_collisions")) or type(row.get("shutdown")) is not bool:
            raise RuntimeError("invalid physical outcome counters")
    return {"episodes": len(outcomes), "mean_deliveries": sum(r["deliveries"] for r in outcomes)/len(outcomes),
        "mean_collisions": sum(r["robot_collisions"] for r in outcomes)/len(outcomes),
        "shutdown_episode_rate": sum(r["shutdown"] for r in outcomes)/len(outcomes),
        "total_deliveries": sum(r["deliveries"] for r in outcomes),
        "total_collisions": sum(r["robot_collisions"] for r in outcomes), "shutdown_episodes": sum(r["shutdown"] for r in outcomes)}


def _metric(metric):
    if type(metric.get("sample_count")) is not int or metric["sample_count"] < 1:
        raise RuntimeError("native program metrics need their physical sample denominator")
    view = {"physical_frames": metric["sample_count"],
        "qualified_coverage": _number(metric.get("coverage"), fraction=True),
        "actual_actionable_coverage": _number(metric.get("actionable_coverage"), fraction=True),
        "agreement": _number(metric.get("agreement"), optional=True, fraction=True),
        "kl": _number(metric.get("kl"), optional=True)}
    if "expected_actionable_coverage" in metric:
        view["expected_actionable_coverage"] = _number(metric["expected_actionable_coverage"], fraction=True)
    for key in ("rows", "actionable_rows", "whole_tree_raw_illegal_count", "whole_tree_path_uncertified_count", "core_KL_bound_violations"):
        if key in metric:
            if type(metric[key]) is not int or metric[key] < 0:
                raise RuntimeError("invalid native program counter")
            view[key] = metric[key]
    if "rows" in view and not math.isclose(view["rows"]/view["physical_frames"], view["qualified_coverage"], abs_tol=1e-9):
        raise RuntimeError("qualified coverage differs from its bound frame counts")
    if "actionable_rows" in view and not math.isclose(view["actionable_rows"]/view["physical_frames"], view["actual_actionable_coverage"], abs_tol=1e-9):
        raise RuntimeError("actual actionable coverage differs from its bound frame counts")
    return view


def _strata(raw, denominator):
    """Use only recorded descriptive strata; never infer missing core fields."""
    if not isinstance(raw, dict):
        return {}
    result = {}
    for group in ("nn_top_action", "task"):
        if group not in raw:
            continue
        values = raw[group]
        if not isinstance(values, dict):
            raise RuntimeError("invalid recorded stratum map")
        total = 0
        for key, row in values.items():
            if type(row.get("physical_frames")) is not int or row["physical_frames"] < 0:
                raise RuntimeError("invalid recorded stratum denominator")
            total += row["physical_frames"]
            if "core_frames" in row and (type(row["core_frames"]) is not int or not 0 <= row["core_frames"] <= row["physical_frames"]):
                raise RuntimeError("invalid recorded core occupancy")
            for field in ("feedback_coverage", "explanation_coverage", "actual_feedback_coverage", "expected_feedback_coverage", "mean_nn_top_probability"):
                if field in row:
                    _number(row[field], optional=True, fraction=True)
        if total != denominator:
            raise RuntimeError("recorded strata do not partition the validation physical frames")
        result[group] = deepcopy(values)
    if "nn_top_action" in result and set(result["nn_top_action"]) != set(ACTIONS):
        raise RuntimeError("recorded action strata must include all five actions")
    for key in ("moving_nn_top", "wait_share_of_explanation_qualified", "purpose"):
        if key in raw:
            result[key] = deepcopy(raw[key])
    if "nn_top_action" in result and all("core_frames" in row for row in result["nn_top_action"].values()):
        core = sum(row["core_frames"] for row in result["nn_top_action"].values())
        result["overall_core_occupancy"] = {"core_frames": core, "physical_frames": denominator, "fraction": core/denominator}
    return result


def _probe(raw):
    if raw.get("applicable") is False:
        return {"applicable": False, "reason": raw.get("reason", "not_recorded_for_control")}
    if "variants" not in raw:
        raise RuntimeError("candidate is missing its native gradient probe")
    view = {"applicable": True, "passed": raw["passed"], "variants": deepcopy(raw["variants"]),
        "interpretation": "gradient_only_probe_on_virtual_post_75_percent_frozen_clone_not_actual_PPO_feedback"}
    for variant in ("raw", "regularized"):
        row = view["variants"][variant]
        for key in ("weighted_loss", "maximum_gradient_norm"):
            _number(row[key])
        if type(row["feedback_rows"]) is not int or row["feedback_rows"] < 0:
            raise RuntimeError("invalid gradient probe feedback count")
    for key in ("program_coefficient", "clone_virtual_frozen_at_step", "optimizer_steps", "actual_PPO_joint_steps", "actor_parameters_unchanged"):
        if key in raw:
            view[key] = raw[key]
    if raw.get("optimizer_steps") != 0 or raw.get("actual_PPO_joint_steps") != 0 or raw.get("actor_parameters_unchanged") is not True:
        raise RuntimeError("a BC gradient-only probe must not claim optimizer or PPO updates")
    r = view["variants"]["raw"]
    view["raw_external_feedback_loss_and_gradient_exactly_zero"] = (
        r["feedback_rows"] == 0 and r["weighted_loss"] == 0 and r["maximum_gradient_norm"] == 0)
    return view


def summarize_model(result, collection):
    checked.quality_score(result, collection, cases=protocol.diagnostics.source_cases())
    validation = [row for row in collection["outcomes"] if row["case"]["original_split"] == "validation"]
    row = {"initialization": result["initialization"], "recipe": result["recipe"], "label": LABELS[result["recipe"]],
        "candidate_admission_applicable": result["recipe"] != "additive_control",
        "quality": _quality(validation), "quality_by_task": {str(t): _quality([r for r in validation if r["case"]["task"] == t])
            for t in (1, 2, 3) if any(r["case"]["task"] == t for r in validation)},
        "native_gates": {k: result[k] for k in ("quality_passed", "program_passed", "gradient_probe_passed", "passed")},
        "roles": {}}
    for role in ROLES:
        native = result["roles"][role]
        validation_metrics = native["metrics"]["validation"]
        f, e = _metric(validation_metrics["feedback"]), _metric(validation_metrics["explanation"])
        if f["physical_frames"] != e["physical_frames"]:
            raise RuntimeError("role feedback/explanation frame denominators differ")
        evidence = {"feedback": f, "explanation": e, "native_validation_metrics": deepcopy(validation_metrics),
            "native_development_gate": deepcopy(native["development_gate"]), "gradient_probe": _probe(native["gradient_probe"])}
        if "validation_strata" in native:
            evidence["recorded_validation_strata"] = _strata(native["validation_strata"], f["physical_frames"])
        row["roles"][role] = evidence
    return row


def _delta(a, b, factor=1):
    return None if a is None or b is None else (a-b)*factor


def compare_models(a, b):
    """Descriptive differences on the same old cases, not a new causal test."""
    if a["initialization"] != b["initialization"]:
        raise RuntimeError("BC comparisons must be paired by initialization")
    quality = {k: a["quality"][k]-b["quality"][k] for k in ("mean_deliveries", "mean_collisions")}
    quality["shutdown_rate_percentage_points"] = 100*(a["quality"]["shutdown_episode_rate"]-b["quality"]["shutdown_episode_rate"])
    roles = {}
    for role in ROLES:
        roles[role] = {}
        for purpose in ("feedback", "explanation"):
            x, y = a["roles"][role][purpose], b["roles"][role][purpose]
            roles[role][purpose] = {k+"_percentage_points": _delta(x.get(k), y.get(k), 100)
                for k in ("qualified_coverage", "actual_actionable_coverage", "expected_actionable_coverage", "agreement")}
            roles[role][purpose]["kl_difference"] = _delta(x["kl"], y["kl"])
    return {"initialization": a["initialization"], "a_recipe": a["recipe"], "b_recipe": b["recipe"],
        "difference_direction": "a_minus_b", "quality": quality, "roles": roles}


def _summary(root, path, spec_ref, refs, results, *, backup, bound):
    ref = checked._pointer(root, path)
    value = checked._read_pointer(root, ref)
    candidate = "shared_b050" if backup else "shared_b035"
    passed = all(r["passed"] for r in results[candidate].values())
    quality_failed = any(not r["quality_passed"] for r in results["shared_b035"].values())
    expected_next = ("lock_common_bc_and_implement_fresh_precheck" if passed else
        "conditional_shared_b050_development" if not backup and quality_failed else "candidate_closed_development_gate_failed")
    if (value.get("schema") != "warehouse-v621-bc-development-summary.v1" or value.get("specification") != spec_ref
            or value.get("source_closure_sha256") != checked.SOURCE02_SHA256 or value.get("results") != refs
            or value.get("candidate_id") != protocol.diagnostics.CANDIDATE_ID or value.get("candidate_recipe") != candidate
            or value.get("all_three_candidate_initializations_passed") is not passed
            or value.get("main_quality_failure") is not quality_failed or value.get("next_stage") != expected_next
            or value.get("fresh_precheck_authorized") is not False or type(value.get("PPO_joint_steps")) is not int
            or value["PPO_joint_steps"] != 0 or value.get("publishable") is not False or value.get("pilot_opened") is not False
            or backup and not quality_failed):
        raise RuntimeError("native BC summary does not recompute from its complete bound cohorts")
    bound[ref["path"]] = ref["sha256"]
    return ref, value


def prepare_report(*, root, specification_path):
    root = Path(root).resolve(strict=True)
    implementation = {str(p): sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__).resolve(), Path(checked.__file__).resolve())}
    manifest = checked._verify_source_closure(root)
    manifest_ref = checked._pointer(root, root/"experiments/warehouse/evidence/warehouse-v621-bc-source-manifest-02.json")
    spec_ref = checked._pointer(root, specification_path)
    spec = checked._read_pointer(root, spec_ref)
    directory = root/"output/warehouse_v621"/protocol.BATCH
    if (spec.get("schema") != "warehouse-v621-bc-development-spec.v1"
            or spec.get("source_closure_sha256") != checked.SOURCE02_SHA256 or spec.get("source_files_sha256") != manifest
            or spec.get("candidate_id") != protocol.diagnostics.CANDIDATE_ID or spec.get("batch_id") != protocol.BATCH
            or spec.get("model_implementation_sha256") != protocol.digest({p: manifest[p] for p in checked.MODEL_CODE_FILES})
            or spec.get("output_directory") != str(directory) or spec.get("fresh_precheck_authorized") is not False
            or type(spec.get("PPO_joint_steps")) is not int or spec["PPO_joint_steps"] != 0
            or spec.get("publishable") is not False):
        raise RuntimeError("BC metrics require the current immutable source-bound development specification")
    config_refs = []
    for key, filename, expected in (("recipe", "warehouse_v621_bc_recipe.json", protocol.recipe()),
            ("seed_plan", "warehouse_v621_bc_seed_plan.json", protocol.seed_plan())):
        if checked._read_pointer(root, spec[key], expected=root/"configs"/filename) != expected:
            raise RuntimeError("BC reporting recipe or seed declaration changed")
        config_refs.append(spec[key])
    # Read completion metadata first. No Actor/model assets may be opened
    # until every required child and its native summary is complete.
    main_summary_ref = checked._pointer(root, directory/"main.summary.json")
    main_declared = checked._read_pointer(root, main_summary_ref)
    backup = main_declared.get("next_stage") == "conditional_shared_b050_development"
    required = RECIPES if backup else RECIPES[:2]
    if backup:
        checked._file(root, directory/"backup.summary.json")
        checked._file(root, directory/"backup.spec.json")
    elif (directory/"backup.summary.json").exists() or any((directory/f"init-{i}"/"shared_b050"/"capability.started.json").exists() for i in protocol.INITIALIZATIONS):
        raise RuntimeError("unexpected backup evidence without the registered main-quality branch")
    for recipe in required:
        for i in protocol.INITIALIZATIONS:
            folder = directory/f"init-{i}"/recipe
            if any((folder/name).exists() for name in ("capability.failed.json", "failure.json", "failed.json")):
                raise RuntimeError("BC report cannot consume an interrupted/failed execution")
            for name in ("completed.json", "capability.completed.json", "result.json"):
                checked._file(root, folder/name)
    bound = {spec_ref["path"]: spec_ref["sha256"], main_summary_ref["path"]: main_summary_ref["sha256"]}
    bound[manifest_ref["path"]] = manifest_ref["sha256"]
    bound.update({p["path"]: p["sha256"] for p in config_refs})
    # Verify every completion pointer chain as metadata, before the native
    # checker can open even the first model. A file's mere presence is not
    # evidence that a different child completed with an intact receipt.
    for recipe in required:
        for i in protocol.INITIALIZATIONS:
            folder = directory/f"init-{i}"/recipe
            complete_ref = checked._pointer(root, folder/"completed.json")
            complete = checked._read_pointer(root, complete_ref)
            result = checked._read_pointer(root, complete["result"], expected=folder/"result.json")
            cap = checked._read_pointer(root, complete["capability_completed"], expected=folder/"capability.completed.json")
            if (complete.get("source_closure_sha256") != checked.SOURCE02_SHA256
                    or cap.get("schema") != "warehouse-v621-bc-capability-completed.v1"
                    or cap.get("initialization") != i or cap.get("recipe") != recipe
                    or cap.get("preauthorized_resets") != 1005
                    or result.get("schema") != "warehouse-v621-bc-development-result.v1"
                    or result.get("status") != "completed" or result.get("initialization") != i
                    or result.get("recipe") != recipe or result.get("source_closure_sha256") != checked.SOURCE02_SHA256):
                raise RuntimeError("incomplete or mismatched native completion metadata")
            for ref in (complete_ref, complete["result"], complete["capability_completed"]):
                bound[ref["path"]] = ref["sha256"]
    results, collections, refs = {}, {}, {}
    for recipe in RECIPES[:2]:
        results[recipe], collections[recipe], refs[recipe] = checked._load_cohort(root, spec, spec_ref, recipe, bound)
        checked._cohort(results[recipe], recipe)
    main_ref, main_summary = _summary(root, directory/"main.summary.json", spec_ref, deepcopy(refs), results, backup=False, bound=bound)
    if main_ref != main_summary_ref:
        raise RuntimeError("native main summary changed after the completion preflight")
    summaries = {"main": {"pointer": main_ref, "value": main_summary}}
    if backup:
        backup_spec_ref = checked._pointer(root, directory/"backup.spec.json")
        backup_spec = checked._read_pointer(root, backup_spec_ref)
        if backup_spec != {**spec, "main_results": refs["shared_b035"]}:
            raise RuntimeError("backup metrics refer to a different parent cohort")
        bound[backup_spec_ref["path"]] = backup_spec_ref["sha256"]
        results["shared_b050"], collections["shared_b050"], refs["shared_b050"] = checked._load_cohort(root, backup_spec, backup_spec_ref, "shared_b050", bound)
        checked._cohort(results["shared_b050"], "shared_b050")
        ref, value = _summary(root, directory/"backup.summary.json", backup_spec_ref, refs, results, backup=True, bound=bound)
        summaries["backup"] = {"pointer": ref, "value": value}
    models = []
    for i in protocol.INITIALIZATIONS:
        for recipe in required:
            result, collection = results[recipe][i], collections[recipe][i]
            training_ref = result["training_report"]
            training_report = checked._read_pointer(root, training_ref,
                expected=directory/f"init-{i}"/recipe/"bc.training.json")
            bound[training_ref["path"]] = training_ref["sha256"]
            if training_report.get("external_program_feedback") != 0 or training_report.get("external_program_actor_updates") != 0:
                raise RuntimeError("BC metrics cannot be described as external-program PPO updates")
            measured = summarize_model(result, collection)
            measured["evidence"] = {"result": refs[recipe][str(i)], "collection": result["development_collection"],
                "training_report": training_ref, "actor": result["actor"],
                "programs": {r: result["roles"][r]["program"] for r in ROLES}}
            models.append(measured)
    lookup = {(r["initialization"], r["recipe"]): r for r in models}
    pairs = [("shared_b035", "additive_control")]
    if backup:
        pairs += [("shared_b050", "additive_control"), ("shared_b050", "shared_b035")]
    comparisons = [compare_models(lookup[i, a], lookup[i, b]) for i in protocol.INITIALIZATIONS for a, b in pairs]
    for path, expected in bound.items():
        if checked._pointer(root, path)["sha256"] != expected:
            raise RuntimeError("bound BC report evidence changed while reporting")
    if checked._verify_source_closure(root) != manifest or any(sha256(Path(p).read_bytes()).hexdigest() != h for p, h in implementation.items()):
        raise RuntimeError("BC reporting source changed during verification")
    return {"schema": "warehouse-v621-completed-bc-metrics.v1", "candidate_id": protocol.diagnostics.CANDIDATE_ID,
        "source_closure_sha256": checked.SOURCE02_SHA256, "reporting_sources_sha256": implementation,
        "source_manifest": manifest_ref,
        "native_gate_definitions": {"quality": deepcopy(protocol.recipe()["quality_gates"]),
            "program": deepcopy(protocol.diagnostics.GATES)},
        "specification": spec_ref, "native_summaries": summaries, "model_count": len(models), "models": models,
        "paired_descriptive_differences": comparisons, "bound_evidence_sha256": dict(sorted(bound.items())),
        "measurement_scope": {"development_scene_arguments_per_model": 275, "quality_natural_validation_episodes_per_model": 75,
            "scene_status": "previously_spent_v620_scenes_reexecuted_for_development_not_independent_validation",
            "same_scene_and_action_arguments_do_not_mean_same_trajectory_or_state_distribution": True,
            "gradient_probe_is_not_actual_PPO_feedback": True, "new_statistical_significance_claim": False},
        "model_selection_performed": False, "new_gate_introduced": False,
        "environment_resets": 0, "fit_calls": 0, "actor_updates": 0, "PPO_joint_steps": 0,
        "fresh_precheck_authorized": False, "pilot_opened": False, "publishable": False}


def _fmt(value, *, percent=False, signed=False):
    if value is None:
        return "—"
    return format(value*(100 if percent else 1), "+.2f" if signed else ".2f")+("%" if percent else "")


def _klfmt(value, *, signed=False):
    return "—" if value is None else format(value, "+.6g" if signed else ".6g")


def render_markdown(report):
    lines = ["# Warehouse v6.21 BC 实际指标对比", "",
        "数据范围：每模型重新执行已消耗的 275 个开发场景，其中质量来自 75 局自然采样验证轨迹。这些结果不是新的独立验证。",
        "同场景和动作参数不代表相同轨迹或状态分布；下列差值仅为配对开发描述，不作显著性或真人收益判断。",
        "raw/regularized 仅出现在无参数更新的梯度探针中；尚无实际 PPO 反馈。", "",
        "表格仅做显示舍入；全部未舍入数值和原生判定保留在JSON。", "",
        f"科学源码：`{report['source_closure_sha256']}`；完整模型数：{report['model_count']}。", "",
        "## 任务质量", "", "| 初始化 | BC配方 | 平均配送 | 平均碰撞 | 停机局/75 | 停机率 | 原生质量门槛 | 候选准入（原生） |",
        "|---|---|---:|---:|---:|---:|---|---|"]
    for row in report["models"]:
        q, gate = row["quality"], row["native_gates"]
        admission = ('通过' if gate['passed'] else '未通过') if row['candidate_admission_applicable'] else '不适用：诊断对照'
        lines.append(f"| {row['initialization']} | {row['label']} | {_fmt(q['mean_deliveries'])} | {_fmt(q['mean_collisions'])} | {q['shutdown_episodes']}/75 | {_fmt(q['shutdown_episode_rate'],percent=True)} | {'通过' if gate['quality_passed'] else '未通过'} | {admission} |")
    lines += ["", "## 程序覆盖与保真", "", "覆盖率分母均为该模型验证轨迹的全部物理帧；F=训练反馈资格，E=解释资格。实际可用还要求实际采样动作匹配；预期F按完整NN动作概率计算。", "",
        "| 初始化/配方 | 角色 | 用途 | 合格覆盖 | 实际动作可用 | 预期F | 动作一致率 | KL |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in report["models"]:
        for role, evidence in row["roles"].items():
            for purpose, label in (("feedback", "F"), ("explanation", "E")):
                m = evidence[purpose]
                lines.append(f"| {row['initialization']}/{row['label']} | {role} | {label} | {_fmt(m['qualified_coverage'],percent=True)} | {_fmt(m['actual_actionable_coverage'],percent=True)} | {_fmt(m.get('expected_actionable_coverage') if purpose=='feedback' else None,percent=True)} | {_fmt(m['agreement'],percent=True)} | {_klfmt(m['kl'])} |")
    lines += ["", "## 同初始化差值", "", "方向为前者减后者；停机差值单位为百分点。", "",
        "| 初始化 | 比较 | Δ平均配送 | Δ平均碰撞 | Δ停机百分点 |", "|---|---|---:|---:|---:|"]
    for row in report["paired_descriptive_differences"]:
        q = row["quality"]
        lines.append(f"| {row['initialization']} | {LABELS[row['a_recipe']]} − {LABELS[row['b_recipe']]} | {_fmt(q['mean_deliveries'],signed=True)} | {_fmt(q['mean_collisions'],signed=True)} | {_fmt(q['shutdown_rate_percentage_points'],signed=True)} |")
    lines += ["", "| 初始化/比较 | 角色 | ΔF合格百分点 | Δ实际F百分点 | Δ预期F百分点 | ΔF一致率百分点 | ΔF KL |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["paired_descriptive_differences"]:
        for role, evidence in row["roles"].items():
            m = evidence["feedback"]
            lines.append(f"| {row['initialization']}/{LABELS[row['a_recipe']]} − {LABELS[row['b_recipe']]} | {role} | {_fmt(m['qualified_coverage_percentage_points'],signed=True)} | {_fmt(m['actual_actionable_coverage_percentage_points'],signed=True)} | {_fmt(m['expected_actionable_coverage_percentage_points'],signed=True)} | {_fmt(m['agreement_percentage_points'],signed=True)} | {_klfmt(m['kl_difference'],signed=True)} |")
    lines += ["", "解释区域的对应差值保留在JSON。", "",
        "## 无更新梯度探针", "", "在虚拟75%冻结结构/core输出的副本上测量真实可微梯度，系数由原始probe收据给出；没有执行optimizer或PPO。对照未做该探针时不回填零。", "",
        "| 初始化/配方 | 角色 | raw损失 | raw最大梯度范数 | regularized最大梯度范数 | regularized反馈行 | 原生探针 |", "|---|---|---:|---:|---:|---:|---|"]
    for row in report["models"]:
        for role, evidence in row["roles"].items():
            p = evidence["gradient_probe"]
            if not p["applicable"]:
                continue
            raw, reg = p["variants"]["raw"], p["variants"]["regularized"]
            lines.append(f"| {row['initialization']}/{row['label']} | {role} | {raw['weighted_loss']:.6g} | {raw['maximum_gradient_norm']:.6g} | {reg['maximum_gradient_norm']:.6g} | {reg['feedback_rows']} | {'通过' if p['passed'] else '未通过'} |")
    lines += ["", "## 已记录的核心占用、五动作与task", "", "只展示原始收据确有的字段；未记录项用“—”，不会补成零。", "",
        "| 初始化/配方 | 角色 | 核心帧/物理帧 | 总体核心占用 |", "|---|---|---:|---:|"]
    for row in report["models"]:
        for role, evidence in row["roles"].items():
            core = evidence.get("recorded_validation_strata", {}).get("overall_core_occupancy")
            if core:
                lines.append(f"| {row['initialization']}/{row['label']} | {role} | {core['core_frames']}/{core['physical_frames']} | {_fmt(core['fraction'],percent=True)} |")
    lines += ["",
        "| 初始化/配方 | 角色 | NN首选动作 | 物理帧 | 核心帧 | F覆盖 | 实际F | 预期F |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in report["models"]:
        for role, evidence in row["roles"].items():
            for action, m in evidence.get("recorded_validation_strata", {}).get("nn_top_action", {}).items():
                lines.append(f"| {row['initialization']}/{row['label']} | {role} | {action} | {m['physical_frames']} | {m.get('core_frames','—')} | {_fmt(m.get('feedback_coverage'),percent=True)} | {_fmt(m.get('actual_feedback_coverage'),percent=True)} | {_fmt(m.get('expected_feedback_coverage'),percent=True)} |")
    lines += ["", "| 初始化/配方 | task | 局数 | 平均配送 | 平均碰撞 | 停机率 |", "|---|---|---:|---:|---:|---:|"]
    for row in report["models"]:
        for task, q in row["quality_by_task"].items():
            lines.append(f"| {row['initialization']}/{row['label']} | {task} | {q['episodes']} | {_fmt(q['mean_deliveries'])} | {_fmt(q['mean_collisions'])} | {_fmt(q['shutdown_episode_rate'],percent=True)} |")
    lines += ["", "| 初始化/配方 | 角色 | task | 物理帧 | 核心帧 | F覆盖 | 实际F | 预期F |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in report["models"]:
        for role, evidence in row["roles"].items():
            for task, m in evidence.get("recorded_validation_strata", {}).get("task", {}).items():
                lines.append(f"| {row['initialization']}/{row['label']} | {role} | {task} | {m['physical_frames']} | {m.get('core_frames','—')} | {_fmt(m.get('feedback_coverage'),percent=True)} | {_fmt(m.get('actual_feedback_coverage'),percent=True)} | {_fmt(m.get('expected_feedback_coverage'),percent=True)} |")
    lines += ["", "WAIT在合格解释中的比例等额外原生字段保留于JSON。", "", "## 可追溯来源", ""]
    for row in report["models"]:
        e = row["evidence"]
        lines.append(f"- {row['initialization']}/{row['label']}：[result]({e['result']['path']}) `{e['result']['sha256']}`；[collection]({e['collection']['path']}) `{e['collection']['sha256']}`。")
    lines += ["", "本报告不选择模型、不修改门槛、不授予fresh或pilot权限。完整source、Actor、program及summary指针见JSON。", ""]
    return "\n".join(lines)


def write_report(*, root, specification_path, output_directory):
    root = Path(root).resolve(strict=True)
    output = Path(output_directory)
    output = output if output.is_absolute() else root/output
    output = output.absolute()
    if output.parent != root/"output/warehouse_v621/bc-metrics-reports" or output.exists() or output.resolve() != output:
        raise RuntimeError("BC report requires a new isolated bc-metrics-reports directory without symlinks")
    report = prepare_report(root=root, specification_path=specification_path)
    text = render_markdown(report)
    output.mkdir(parents=True, exist_ok=False)
    if output.resolve(strict=True) != output:
        raise RuntimeError("BC report output directory became a symlink")
    json_path, markdown_path = output/"metrics.json", output/"metrics.zh.md"
    with json_path.open("x") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False); stream.write("\n")
    with markdown_path.open("x") as stream:
        stream.write(text)
    return {"json": checked._pointer(root, json_path), "markdown": checked._pointer(root, markdown_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(write_report(root=ROOT, specification_path=args.spec, output_directory=args.output_directory), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
