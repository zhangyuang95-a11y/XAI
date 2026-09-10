import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/warehouse_native/alignment_50k_pair_20260910/branches/feedback"
ACTOR = BASE / "actors/actor_0050000.npz"
PROTOCOL = BASE / "protocol.json"
PROGRAM = ROOT / "output/warehouse_native/alignment_explanation_system_acceptance_v3_395m_20260910/system_qualified_program.json"
SCENARIOS = ROOT / "output/warehouse_native/native_cycle_500k_candidate_20260909/scenarios.json"


def components():
    from backend.training.warehouse_native_common import digest, file_hash
    from backend.warehouse_alignment_online_runtime import OnlineAlignmentRuntime
    from backend.warehouse_alignment_online_explanation import OnlineAlignmentExplainer

    protocol = json.loads(PROTOCOL.read_text())
    runtime = OnlineAlignmentRuntime(
        ACTOR,
        protocol=protocol,
        expected_actor_sha256=file_hash(ACTOR),
        expected_protocol_sha256=digest(protocol),
    )
    explainer = OnlineAlignmentExplainer(
        PROGRAM,
        expected_program_sha256=file_hash(PROGRAM),
        runtime=runtime,
    )
    scene = json.loads(SCENARIOS.read_text())["splits"]["play"][0]
    return runtime, explainer, scene


def test_online_explainer_import_does_not_load_training_frameworks():
    code = "import sys; import backend.warehouse_alignment_online_explanation; " \
           "assert not [m for m in sys.modules if m.startswith(('torch','sklearn','backend.training'))]"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_participant_answer_is_plain_language_and_evidence_is_collapsed():
    runtime, explainer, scene = components()
    record = runtime.step(runtime.environment(scene), "WAIT")
    result = explainer.answer(
        {"question": "机器人2刚才为什么这样行动？", "language": "zh", "frame": 1},
        record,
        runtime,
    )
    assert set(result) == {"answer", "evidence_detail"}
    assert "机器人2刚才" in result["answer"]
    assert "距离" in result["answer"]
    for technical in ("NN", "argmax", "概率", "决策树", "阈值", "指示值", "SHA"):
        assert technical not in result["answer"]
    assert "动作概率" in result["evidence_detail"]
    assert "策略近似程序" in result["evidence_detail"]
    assert "抽象目标关系干预" in result["evidence_detail"]
    assert "隔离重放核验" in result["evidence_detail"]
    assert runtime.actor_sha256 in result["evidence_detail"]


def test_main_answer_boundary_rejects_audit_only_vocabulary():
    from backend.warehouse_alignment_online_explanation import _response

    for text in ("NN chose right.", "动作概率最高。", "这个决策树分支通过了阈值。",
                 "Actor SHA-256 is recorded.", "The policy probability was highest."):
        try:
            _response(text, "技术证据")
        except ValueError as error:
            assert "audit-only" in str(error)
        else:
            raise AssertionError(f"audit-only term reached participant answer: {text}")


def test_six_shortcuts_are_bilingual_and_fact_bound():
    runtime, explainer, scene = components()
    record = runtime.step(runtime.environment(scene), "WAIT")
    questions = [
        ("机器人2刚才为什么这样行动？", "executed", "zh"),
        ("机器人2刚才为什么等待？", "executed", "zh"),
        ("我们刚才为什么发生碰撞？", "executed", "zh"),
        ("我的上一步动作影响了机器人2吗？", "executed", "zh"),
        ("机器人2当前在朝哪个任务前进？", "next", "zh"),
        ("机器人2现在需要充电吗？", "next", "zh"),
        ("Why did Robot 2 choose that action?", "executed", "en"),
        ("Why did we just collide?", "executed", "en"),
        ("Did my last action affect Robot 2?", "executed", "en"),
        ("Which task is Robot 2 moving toward?", "next", "en"),
        ("Does Robot 2 need to charge now?", "next", "en"),
    ]
    for question, focus, language in questions:
        result = explainer.answer(
            {"question": question, "focus": focus, "language": language, "frame": 1},
            record,
            runtime,
        )
        assert result["answer"].strip()
        assert "Bound frame" in result["evidence_detail"] if language == "en" else "绑定帧" in result["evidence_detail"]


def test_counterfactual_does_not_mutate_live_state():
    from backend.warehouse_alignment_online_runtime import digest

    runtime, explainer, scene = components()
    env = runtime.environment(scene)
    record = runtime.step(env, "WAIT")
    before = digest(record)
    result = explainer.answer(
        {"question": "如果我等待三步会怎样？", "focus": "next", "language": "zh", "frame": 1},
        record,
        runtime,
    )
    assert digest(record) == before
    assert "真实回合没有改变" in result["answer"]
    assert "反事实" in result["evidence_detail"]
    assert "动作概率" not in result["answer"]


def test_tree_disagreement_never_becomes_participant_reason(monkeypatch):
    from core.program import ExecutableProgram

    runtime, explainer, scene = components()
    record = runtime.step(runtime.environment(scene), "WAIT")
    actual = record["submitted_actions"]["robot_2"]
    different = next(action for action in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT") if action != actual)
    monkeypatch.setattr(ExecutableProgram, "predict", lambda _self, _features: different)
    result = explainer.answer(
        {"question": "机器人2刚才为什么这样行动？", "focus": "executed", "language": "zh", "frame": 1},
        record,
        runtime,
    )
    assert "决策树" not in result["answer"]
    assert "近似程序" not in result["answer"]
    assert "刚才选择了" in result["answer"]
    assert "下一步会选择" not in result["answer"]
    assert "现有证据无法可靠说明更具体的原因" in result["answer"]
    assert "不一致" in result["evidence_detail"]
    assert "未用于主回答" in result["evidence_detail"]


def test_stationary_occupant_collision_identifies_whose_move_was_canceled(monkeypatch):
    import numpy as np

    runtime, explainer, scene = components()
    env = runtime.environment(scene)
    for agent, position in zip(env.state.agents, ((5, 2), (5, 3))):
        agent.position = position
        agent.battery = 100.0
        agent.active = True

    def wait_policy(observations, deterministic=True):
        assert deterministic is True
        return (
            {agent_id: "WAIT" for agent_id in observations},
            {agent_id: np.asarray((0, 0, 0, 0, 1), dtype=np.float32)
             for agent_id in observations},
        )

    monkeypatch.setattr(runtime.actor, "act", wait_policy)
    record = runtime.step(env, "RIGHT")
    assert record["after"]["public_feedback_history"]["collision_kind"] == "occupied_stationary"
    assert record["after"]["public_feedback_history"]["move_canceled"] == {
        "robot_1": True, "robot_2": False,
    }
    result = explainer.answer(
        {"question": "我们刚才为什么发生碰撞？", "focus": "executed",
         "language": "zh", "frame": 1},
        record,
        runtime,
    )
    assert "机器人2选择了等待并留在原格" in result["answer"]
    assert "取消了你的移动" in result["answer"]
    assert "取消了机器人2的移动" not in result["answer"]
