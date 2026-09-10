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


def test_online_explainer_import_does_not_load_training_frameworks():
    code = "import sys; import backend.warehouse_alignment_online_explanation; " \
           "assert not [m for m in sys.modules if m.startswith(('torch','sklearn','backend.training'))]"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_bilingual_answers_match_qualified_research_explainer():
    from backend.training.warehouse_native_common import digest, file_hash
    from backend.warehouse_alignment_runtime import AlignmentRuntime
    from backend.warehouse_alignment_diverse_explanation import DiverseAlignmentExplainer
    from backend.warehouse_alignment_online_runtime import OnlineAlignmentRuntime
    from backend.warehouse_alignment_online_explanation import OnlineAlignmentExplainer

    protocol = json.loads(PROTOCOL.read_text())
    kwargs = dict(protocol=protocol, expected_actor_sha256=file_hash(ACTOR),
                  expected_protocol_sha256=digest(protocol))
    reference, online = AlignmentRuntime(ACTOR, **kwargs), OnlineAlignmentRuntime(ACTOR, **kwargs)
    expected_program_sha256 = file_hash(PROGRAM)
    old_explainer = DiverseAlignmentExplainer(PROGRAM, expected_program_sha256=expected_program_sha256,
                                               runtime=reference)
    new_explainer = OnlineAlignmentExplainer(PROGRAM, expected_program_sha256=expected_program_sha256,
                                              runtime=online)
    scenario = json.loads(SCENARIOS.read_text())["splits"]["play"][0]
    old_record = reference.step(reference.environment(scenario), "WAIT")
    new_record = online.step(online.environment(scenario), "WAIT")
    questions = [
        {"question": "刚才为什么这样做？", "language": "zh"},
        {"question": "Why did you do that?", "language": "en"},
        {"question": "如果我等待三步会怎样？", "language": "zh", "focus": "next"},
        {"question": "What are the charging rules?", "language": "en", "focus": "next"},
    ]
    for request in questions:
        assert old_explainer.answer(request, old_record, reference) == \
               new_explainer.answer(request, new_record, online)

