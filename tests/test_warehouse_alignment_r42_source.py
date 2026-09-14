from pathlib import Path

from ui.warehouse_alignment_r42_server import (
    R42_PUBLIC_RELEASE_VERSION,
    VERSION,
    _infer_quick_intent,
)


ROOT = Path(__file__).resolve().parents[1]


def test_r42_identity_and_fixed_intent_parser():
    assert VERSION == "warehouse-alignment-online-study-server.r4.2"
    assert R42_PUBLIC_RELEASE_VERSION == "r4.2-internal-pilot"
    cases = {
        "机器人2现在需要充电吗？": "charging_need",
        "Why did we collide?": "collision_reason",
        "我的上一步影响了机器人2吗？": "human_influence",
        "Which task is Robot 2 heading toward?": "task_direction",
        "机器人2刚才为什么等待？": "wait_reason",
        "Why did Robot 2 do that?": "action_reason",
    }
    for question, expected in cases.items():
        assert _infer_quick_intent(question) == expected


def test_render_blueprint_is_pinned_to_r42_package():
    blueprint = (ROOT / "render.yaml").read_text()
    assert "ui.warehouse_alignment_r42_server" in blueprint
    assert "ui.warehouse_alignment_r42_release" in blueprint
    assert "1c372b2294a3739aadee2d346a043c2cbdbe8d7c35c69d78a66ae8acd24837dd" in blueprint
    assert "56790275e22a017c015a4c47cc86109864e712d822ee521c05814960c6c715b5" in blueprint
    assert "warehouse_alignment_r41_diagnostic_release_v9" not in blueprint
