"""Exact audit equivalence. No experiment resets or policy fitting."""
import json
from pathlib import Path
import pytest
from experiments.warehouse import seed_metadata_triage as slow,v628_seed_triage as fast
from experiments.warehouse.v628_data import exact_plan
from experiments.warehouse.exact_seed_audit import compile_families


@pytest.mark.parametrize('suffix',['.json','.jsonl'])
def test_superset_filter_preserves_exact_semantic_evidence(tmp_path,suffix):
    families=compile_families(exact_plan());file=tmp_path/('metadata'+suffix)
    rows=[dict(schema='test',candidate_id='fixture',base_seed=86280000,action_seed=5628000000,
        scene_seed=862800001,nonseed_integer=862800002,
        gap_numbers=[862800002,862800010,862800003,862816492],
        reserved_intervals=[dict(start=862800000,end_exclusive=862800032)],
        seed_range=[862800000,100],count=42,
        ordinary=[0,1,862800001,862800002,5628000001,5628009999,862816493]),
        dict(unrelated=[7,862812345,862800002],random_state=862800012)]
    file.write_text('\n'.join(json.dumps(r) for r in rows) if suffix=='.jsonl' else json.dumps(rows))
    old,om,oc=slow.stream_detail(file,families,tmp_path/'old.stderr')
    new,nm,nc=fast.stream_detail(file,families,tmp_path/'new.stderr')
    assert old==new and om==nm
    assert nc>=oc
    assert all(r['value']!=862800002 for r in new['candidate_matches'])


def test_envelopes_cover_all_exact_values_without_changing_families():
    families=compile_families(exact_plan());before=json.dumps(families,sort_keys=True)
    envelopes=[(min(s[0] for s in f['intervals']),max(s[1] for s in f['intervals'])) for f in families]
    for f,(lo,hi) in zip(families,envelopes,strict=True):
        assert all(lo<=v<hi for v in f['values'])
    assert json.dumps(families,sort_keys=True)==before and len(envelopes)==3
