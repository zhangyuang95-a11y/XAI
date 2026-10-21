"""Complete JSON event equivalence with exact seed and interval semantics."""
import json
from experiments.warehouse import v628_seed_triage_bounded as bounded,seed_metadata_audit as meta
from experiments.warehouse.v628_data import exact_plan
from experiments.warehouse.exact_seed_audit import compile_families


def test_complete_integer_prefilter_retains_exact_matches_and_all_reservations(tmp_path):
    f=compile_families(exact_plan())
    value=dict(schema='test',base_seed=86280000,env_seed=862800001,seed_ranges=dict(long=[0,900000000]),
        unrelated=[0,1,2,862800002,5628000000,5628009999,862816493,5628000000.0],
        reserved_intervals=[dict(start=0,end_exclusive=6000000000),dict(start=862800000,count=50)],
        arbitrary_seed_key=17,nested=dict(seed_range=[862800000,100]))
    path=tmp_path/'report.json';path.write_text(json.dumps(value))
    actual,sample,count=bounded.stream_detail(path,f,tmp_path/'unused.stderr')
    assert actual==meta.analyze_literal(value,f) and sample==value
    assert count>0


def test_top_level_list_classification_and_negative_or_boolean_nonseed_values(tmp_path):
    f=compile_families(exact_plan());value=[dict(schema='fixture',candidate_id='test',seed=-1),dict(x=True,y=False,number=2.0)]
    path=tmp_path/'report.json';path.write_text(json.dumps(value))
    actual,sample,count=bounded.stream_detail(path,f,tmp_path/'unused.stderr')
    assert actual==meta.analyze_literal(value,f) and sample==value[0]
