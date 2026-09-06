"""Resume-local training counters must not masquerade as cumulative totals."""
import json
from backend.training.cooperative_kitchen import training_counter_summary


def test_legacy_resume_counts_are_summed_without_rewriting_log(tmp_path):
    path=tmp_path/'training.jsonl'
    rows=[{'joint_steps':100,'episodes':2,'successes':1},
          {'joint_steps':200,'episodes':5,'successes':2},
          {'joint_steps':300,'episodes':1,'successes':0},
          {'joint_steps':400,'episodes':4,'successes':3}]
    original='\n'.join(json.dumps(r) for r in rows)+'\n';path.write_text(original)
    result=training_counter_summary(path)
    assert result['completed_episodes_across_segments']==9
    assert result['successful_episodes_across_segments']==5
    assert [s['start_joint_steps'] for s in result['segments']]==[0,200]
    assert path.read_text()==original


def test_explicit_resume_markers_distinguish_zero_counter_segments(tmp_path):
    path=tmp_path/'training.jsonl'
    rows=[{'joint_steps':100,'episodes':0,'successes':0,'counter_scope':'resume_segment','segment_start_joint_steps':0},
          {'joint_steps':200,'episodes':0,'successes':0,'counter_scope':'resume_segment','segment_start_joint_steps':100},
          {'joint_steps':300,'episodes':3,'successes':1,'counter_scope':'resume_segment','segment_start_joint_steps':100}]
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    result=training_counter_summary(path)
    assert len(result['segments'])==2
    assert result['completed_episodes_across_segments']==3
    assert all(s['boundary_source']=='explicit_segment_marker' for s in result['segments'])
