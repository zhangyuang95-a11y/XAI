import gzip
import json
from pathlib import Path
import pickle
import subprocess

import pytest

from experiments.warehouse import seed_metadata_audit as a
from experiments.warehouse import seed_metadata_triage as t


def proposal(tmp_path):
    path = tmp_path/"proposal.json"
    a.write_new(path, {"schema": "warehouse-seed-proposal.v1", "candidate_id": "warehouse-v620-01",
        "families": [{"name": "pilot", "kind": "init", "values": [6160]},
                     {"name": "fresh_env", "kind": "env", "start": 1002670000, "end_exclusive": 1002670010},
                     {"name": "actions", "kind": "action_rng", "start": 1710000000, "end_exclusive": 1710000216},
                     {"name": "critic", "kind": "critic", "values": [1810000000, 1810000001]},
                     {"name": "formal_query", "kind": "init", "values": [6101, 6102, 6103], "query_only": True}]})
    return path


def test_proposal_canonical_families_and_typed_intervals(tmp_path):
    p, families, pointer = a.read_proposal(proposal(tmp_path))
    assert pointer["sha256"] == a.file_sha(pointer["path"])
    assert families[0]["count"] == 1
    assert families[1]["count"] == 10
    assert a.matching_families(1002670005, families) == ["fresh_env"]
    assert a.matching_families(1002670010, families) == []


@pytest.mark.parametrize("mutation", ["boolean", "duplicate", "range", "kind", "overlap"])
def test_invalid_proposal_rejected(tmp_path, mutation):
    path = proposal(tmp_path)
    value = json.loads(path.read_bytes())
    if mutation == "boolean":
        value["families"][0]["values"] = [True]
    elif mutation == "duplicate":
        value["families"][0]["values"] = [6160, 6160]
    elif mutation == "range":
        value["families"][1]["end_exclusive"] = value["families"][1]["start"]
    elif mutation == "kind":
        value["families"][0]["kind"] = "unknown"
    else:
        value["families"].append({"name": "duplicate-purpose", "kind": "critic", "values": [1002670005]})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        a.read_proposal(path)


def test_initialization_and_environment_reservation_have_distinct_semantics(tmp_path):
    _, families, _ = a.read_proposal(proposal(tmp_path))
    value = {"reserved_intervals": [{"start": 41, "end_exclusive": 400297}],
             "pair_seed": 6160, "environment_seed": 6160}
    detail = a.analyze_literal(value, families)
    assert not detail["reservation_overlaps"]
    init = next(row for row in detail["candidate_matches"] if row["path"] == ["pair_seed"])
    env = next(row for row in detail["candidate_matches"] if row["path"] == ["environment_seed"])
    assert a.compatible("init", init["semantics"])
    assert not a.compatible("init", env["semantics"])


def test_explicit_model_initialization_interval_blocks_initializer(tmp_path):
    _, families, _ = a.read_proposal(proposal(tmp_path))
    detail = a.analyze_literal({"model_seed_reservation": {"start": 6150, "end_exclusive": 6170}}, families)
    assert [r["family"] for r in detail["reservation_overlaps"]] == ["pilot", "formal_query"][:1]
    assert detail["reservation_overlaps"][0]["semantics"] == "init"


def test_start_count_and_legacy_pair_ranges_conservatively_cover_seed_use(tmp_path):
    _, families, _ = a.read_proposal(proposal(tmp_path))
    value = {"seed_ranges": {"train": [1002670000, 10]},
             "reserved_seed_interval": {"start": 1710000000, "count": 216}}
    detail = a.analyze_literal(value, families)
    assert {r["family"] for r in detail["reservation_overlaps"]} == {"fresh_env", "actions"}
    assert any(r["range_interpretation"].startswith("conservative_union") for r in detail["reservation_overlaps"])


def test_state_key_heuristics_never_discard_possible_initialization(tmp_path):
    _, families, _ = a.read_proposal(proposal(tmp_path))
    detail = a.analyze_literal({"numpy_rng_state": ["MT19937", [1710000001, 6160]],
        "action_seed": 1710000001, "random_state": 6160}, families)
    assert not detail["ignored_internal_or_nonseed_matches"]
    assert len(detail["candidate_matches"]) == 4
    assert {r["semantics"] for r in detail["candidate_matches"]} == {
        "action_rng", "ambiguous_seed_argument", "ambiguous_internal_rng_state_requires_context_proof"}


def test_constructor_seed_inside_inert_rng_record_is_not_discarded_as_mt_word(tmp_path):
    _, families, _ = a.read_proposal(proposal(tmp_path))
    detail = a.analyze_literal({"numpy_rng_state": {"constructor_args": [6160],
        "literal_state": {"key": [1710000001]}}}, families)
    assert len(detail["candidate_matches"]) == 2
    assert detail["candidate_matches"][0]["semantics"] == "ambiguous_seed_argument"
    assert detail["candidate_matches"][0]["value"] == 6160
    assert not detail["ignored_internal_or_nonseed_matches"]


def test_include_new_v619_v620_and_fit_metadata_not_just_named_ledgers(tmp_path):
    root = tmp_path/"warehouse"
    for name in ("output/v619-run/fit.jsonl", "output/v620-run/result.json", "output/v620-resets.jsonl"):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    (root/"weights.npz").write_bytes(b"do not read model weights")
    found = list(a.candidates([root], [], registered_trace_paths=[root/"output/v619-run/fit.jsonl"]))
    assert len(found) == 3
    assert all(path.suffix != ".npz" for path in found)


def test_proposal_itself_and_own_same_candidate_declaration_are_not_prior_use(tmp_path):
    root = tmp_path/"warehouse"
    root.mkdir()
    pp = root/"proposal.json"
    a.write_new(pp, {"schema": "warehouse-seed-proposal.v1", "candidate_id": "warehouse-v620-01",
        "families": [{"name": "pilot", "kind": "init", "values": [6160]}]})
    a.write_new(root/"plan.json", {"candidate_id": "warehouse-v620-01", "pair_seed": 6160})
    report = a.audit(pp, [root], tmp_path/"audit")
    assert len(report["inventory"]) == 1
    row = report["summary"]["families"]["pilot"]
    assert not row["references_or_reservations"]
    assert len(row["own_candidate_declarations_not_use"]) == 1
    assert report["summary"]["authorization_granted"] is False


def test_failed_claim_counts_spent_even_for_same_candidate(tmp_path):
    pp = proposal(tmp_path)
    root = tmp_path/"warehouse"
    root.mkdir()
    a.write_new(root/"seed-use-claim.json", {"candidate_id": "warehouse-v620-01",
        "pair_seed": 6160, "status": "interrupted"})
    report = a.audit(pp, [root], tmp_path/"audit")
    row = report["summary"]["families"]["pilot"]
    assert row["status"].startswith("blocked")
    assert row["references_or_reservations"][0]["classification"].startswith("claim_spent")
    assert a.file_classification('/output/warehouse_sealed_claims/receipt.json', {}) == "claim_spent_even_if_failed_or_interrupted"


def test_v619_action_and_scene_references_block_actual_reuse(tmp_path):
    pp = proposal(tmp_path)
    root = tmp_path/"warehouse_v619"
    root.mkdir()
    (root/"actual.resets.jsonl").write_text(json.dumps({"seed": 1002670005, "action_seed": 1710000002})+'\n')
    report = a.audit(pp, [root], tmp_path/"audit")
    assert report["summary"]["families"]["fresh_env"]["status"].startswith("blocked")
    assert report["summary"]["families"]["actions"]["status"].startswith("blocked")


def test_safe_inert_pickle_does_not_execute_reduce(tmp_path):
    marker = tmp_path/"not-created"
    class Exploit:
        def __reduce__(self):
            return (subprocess.call, (["touch", str(marker)],))
    fields = t.inert_pickle_fields(pickle.dumps({"payload": Exploit(), "pair_seed": 6160}))
    assert not marker.exists()
    assert fields["pair_seed"] == 6160
    assert "constructor_args" in fields["payload"]


def test_gzip_yaml_and_large_json_complete_triage_is_parameterized(tmp_path):
    pp = proposal(tmp_path)
    root = tmp_path/"warehouse"
    root.mkdir()
    with gzip.open(root/"training_trajectory.json", "wb") as handle:
        handle.write(pickle.dumps({"pair_seed": 6160, "numpy_rng_state": [[1710000001]]}))
    (root/"old-config.yaml").write_text('action_seed: 1710000003\n')
    (root/"large-report.json").write_text(json.dumps({"payload": "x"*1000,
        "scenario_seed": 1002670008, "rng_state": [[1710000002]]}))
    audit = a.audit(pp, [root], tmp_path/"audit", max_bytes=100)
    assert len(audit["summary"]["unresolved_paths"]) == 3
    result = t.triage(tmp_path/"audit/inventory.json", pp, tmp_path/"triage")
    assert not result["summary"]["unresolved_paths"]
    assert result["summary"]["families"]["pilot"]["status"].startswith("blocked")
    assert result["summary"]["families"]["fresh_env"]["status"].startswith("blocked")
    assert result["summary"]["families"]["actions"]["status"].startswith("blocked")
    assert result["summary"]["authorization_granted"] is False


def test_full_stream_parse_rejects_trailing_invalid_json(tmp_path):
    pp = proposal(tmp_path)
    root = tmp_path/"warehouse"
    root.mkdir()
    (root/"bad-report.json").write_text('{"seed":1002670005,"padding":"'+'x'*1000+'"} BROKEN')
    a.audit(pp, [root], tmp_path/"audit", max_bytes=100)
    result = t.triage(tmp_path/"audit/inventory.json", pp, tmp_path/"triage")
    assert len(result["summary"]["unresolved_paths"]) == 1
    assert result["summary"]["families"]["pilot"]["status"].startswith("unresolved")


def test_hash_change_after_audit_and_proposal_drift_reject_before_triage(tmp_path):
    pp = proposal(tmp_path)
    root = tmp_path/"warehouse"
    root.mkdir()
    target = root/"report.json"
    target.write_text('{"pair_seed":6160}')
    a.audit(pp, [root], tmp_path/"audit")
    target.write_text('{"pair_seed":6161}')
    with pytest.raises(RuntimeError, match="changed since"):
        t.triage(tmp_path/"audit/inventory.json", pp, tmp_path/"triage")
    value = json.loads(pp.read_bytes())
    value["candidate_id"] = "different"
    pp.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="proposal differs"):
        t.triage(tmp_path/"audit/inventory.json", pp, tmp_path/"different-triage")


def test_missing_root_and_exclusive_output_never_grant_authority(tmp_path):
    pp = proposal(tmp_path)
    report = a.audit(pp, [tmp_path/"missing"], tmp_path/"audit")
    assert report["summary"]["unresolved_paths"]
    assert report["summary"]["authorization_granted"] is False
    with pytest.raises(FileExistsError):
        a.audit(pp, [], tmp_path/"audit")


def test_weight_directory_and_json_names_are_excluded_before_read(tmp_path, monkeypatch):
    pp = proposal(tmp_path)
    root = tmp_path/"warehouse"
    weight_names = ["actors/seed_receipt.json", "checkpoints/training_report.json",
                    "frozen_actor.json", "model_weights.json", "weights.npz"]
    forbidden = set()
    for name in weight_names:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"DO NOT READ WEIGHTS")
        forbidden.add(path)
    unknown = root/"frame_000001.json"
    unknown.write_bytes(b"DO NOT READ NONWHITE METADATA")
    forbidden.add(unknown)
    receipt = root/"actual-seed-receipt.json"
    receipt.write_text('{"pair_seed":6160}')
    real = Path.read_bytes
    def guarded(path):
        if path in forbidden: raise AssertionError("excluded content was read")
        return real(path)
    monkeypatch.setattr(Path,"read_bytes",guarded)
    report = a.audit(pp,[root],tmp_path/"audit")
    assert [row["path"] for row in report["inventory"]] == [str(receipt)]
    assert len(report["selection_stats"]["excluded_weights_name_stat_only"]) == 5
    assert report["read_model_weight_files"] is False
    assert report["requires_final_delta_inventory"] is True


def test_only_explicitly_registered_program_trace_bypasses_name_whitelist(tmp_path):
    root=tmp_path/"warehouse_v619";root.mkdir()
    trace=root/"fit.jsonl";trace.write_text('{}\n')
    other=root/"validation.jsonl";other.write_text('{}\n')
    assert list(a.candidates([root],[])) == []
    assert list(a.candidates([root],[],registered_trace_paths=[trace])) == [trace]


def test_inventory_only_reads_no_metadata_payloads(tmp_path, monkeypatch):
    pp=proposal(tmp_path)
    root=tmp_path/"warehouse";root.mkdir()
    one=root/"run-seed-receipt.json";one.write_bytes(b"DO NOT READ")
    large=root/"large-report.json";large.write_bytes(b"x"*200)
    real=Path.read_bytes
    def guarded(path):
        if path in {one,large}: raise AssertionError("stat-only content read")
        return real(path)
    monkeypatch.setattr(Path,"read_bytes",guarded)
    report=a.inventory_only(pp,[root],tmp_path/"selection",max_bytes=100)
    assert report["selected_metadata_files"]==2
    assert report["selected_metadata_bytes"]==211
    assert report["large_deferred_files"]==1
    assert report["metadata_content_read_count"]==report["model_weight_read_count"]==0
    assert report["authorization_granted"] is False


def test_claim_context_and_durable_v6_start_records_survive_narrow_selector(tmp_path):
    root=tmp_path/"warehouse_v620"
    names=("claims/abcdef.json","fit/started.json","fresh/independent.started.json")
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    found=list(a.candidates([root],[]))
    assert {p.relative_to(root).as_posix() for p in found}==set(names)

from experiments.warehouse import exact_seed_audit as e
from experiments.warehouse import seed_context as c


def exact_plan(tmp_path, uses=None):
    path=tmp_path/'exact-plan.json'
    if uses is None:
        uses=[{'kind':'init','purpose':'actor_initialization','argument':'torch.manual_seed',
               'value':6160,'group':'pilot/init','reuse_tag':'paired_gammas'},
              {'kind':'action_rng','purpose':'independent','argument':'action_seed',
               'value':1710000001,'group':'pilot/independent/0','reuse_tag':'paired_gammas'},
              {'kind':'classifier_rng','purpose':'classifier','argument':'random_state',
               'value':19,'group':'pilot/classifier/0','reuse_tag':'fixed_algorithm_argument_not_fresh'}]
    a.write_new(path,{'schema':'warehouse-exact-seed-plan.v1','candidate_id':'pilot',
                      'authorization_granted':False,'uses':uses})
    return path


def test_exact_plan_rejects_missing_required_field_even_with_optional_fields(tmp_path):
    path=exact_plan(tmp_path);value=json.loads(path.read_bytes())
    value['uses'][0].pop('value');value['uses'][0]['consumed_as_random_draw']=True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='malformed'):
        e.read_exact_plan(path)


def test_exact_plan_rejects_cross_purpose_fresh_seed_sharing(tmp_path):
    path=exact_plan(tmp_path);value=json.loads(path.read_bytes())
    value['uses'].append({**value['uses'][1],'kind':'critic','purpose':'different', 'group':'different'})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='across purposes'):
        e.read_exact_plan(path)


def test_exact_projection_preserves_gaps_and_reprojects_actual_reservations(tmp_path):
    pp=tmp_path/'wide-proposal.json'
    a.write_new(pp,{'schema':'warehouse-seed-proposal.v1','candidate_id':'pilot', 'families':[
        {'name':'wide_action','kind':'action_rng','start':1711000000,'end_exclusive':1711004000}]})
    root=tmp_path/'warehouse';root.mkdir()
    a.write_new(root/'old-training-report.json',{'training_episodes':[{'seed':1711001586}],
        'action_reservation':{'start':1711000050,'end_exclusive':1711000055}})
    wide=a.audit(pp,[root],tmp_path/'wide')
    uses=[{'kind':'action_rng','purpose':'independent','argument':'action_seed','value':1711000000+pair*1000+i,
           'group':f'pair-{pair}/action-{i}','reuse_tag':'paired_gammas'} for pair in range(3) for i in range(216)]
    plan,_=e.read_exact_plan(exact_plan(tmp_path,uses));families=e.compile_families(plan)
    assert families[0]['intervals']==((1711000000,1711000216),(1711001000,1711001216),(1711002000,1711002216))
    projected=e.project_inventory(wide,families)
    assert not any(row['value']==1711001586 for row in projected['summary']['families'][families[0]['name']]['references_or_reservations'])
    spans=projected['summary']['families'][families[0]['name']]['reservation_overlaps']
    assert spans[0]['historical_interval']=={'start':1711000050,'end_exclusive':1711000055}
    assert spans[0]['overlaps']==[[1711000050,1711000055]]
    assert projected['summary']['authorization_granted'] is False


def test_fresh_projection_outside_prior_scope_rejected_but_fixed_argument_not_claimed_fresh(tmp_path):
    pp=proposal(tmp_path);root=tmp_path/'warehouse';root.mkdir()
    wide=a.audit(pp,[root],tmp_path/'wide')
    plan,_=e.read_exact_plan(exact_plan(tmp_path));families=e.compile_families(plan)
    result=e.project_inventory(wide,families)
    assert result['unscanned_fixed_argument_values_not_claimed_fresh']==[
        {'family':'init:declared_fixed_rng_and_hash_arguments','value':19,'query_only':True}]
    plan['uses'][0]['value']=999999
    with pytest.raises(RuntimeError,match='outside'):
        e.project_inventory(wide,e.compile_families(plan))


def test_old_heuristic_ignored_rng_matches_are_restored_before_exact_review(tmp_path):
    pp=proposal(tmp_path);root=tmp_path/'warehouse';root.mkdir()
    a.write_new(root/'report.json',{'rng_state':{'seed':[[6160]]}})
    wide=a.audit(pp,[root],tmp_path/'wide')
    detail=wide['inventory'][0]['detail'];match=detail['candidate_matches'].pop()
    match['semantics']='internal_rng_state_word_not_initialization'
    detail['ignored_internal_or_nonseed_matches'].append(match)
    plan,_=e.read_exact_plan(exact_plan(tmp_path));result=e.project_inventory(wide,e.compile_families(plan))
    refs=result['summary']['families']['init:actor_initialization']['references_or_reservations']
    assert refs[0]['path']==['rng_state','seed',0,0]
    assert refs[0]['semantics']=='ambiguous_internal_rng_state_requires_context_proof'


def test_weight_path_pair_hint_blocks_initializer_without_content_reads(tmp_path,monkeypatch):
    pp=proposal(tmp_path);root=tmp_path/'warehouse';weight=root/'pair_6160_steps_200000/actors/raw.actor.npz'
    weight.parent.mkdir(parents=True);weight.write_bytes(b'not opened')
    real=Path.open
    def guard(path,*args,**kwargs):
        if path==weight:raise AssertionError('weight bytes opened')
        return real(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    report=a.audit(pp,[root],tmp_path/'wide')
    family=report['summary']['families']['pilot']
    assert not report['inventory'] and family['status'].startswith('blocked')
    assert family['excluded_weight_path_seed_hints'][0]['content_read'] is False


def test_full_python_mt_structure_required_and_state_index_not_a_word():
    state=[3,[0]*624+[40],None]
    evidence={'snapshot':{},'environment_random':{}}
    value={'snapshot':{'rng':state}}
    lookup=lambda p:c.get(value,p)
    assert c.classify_context(['snapshot','rng',1,10],0,lookup,evidence)[0].startswith('confirmed_python')
    assert c.classify_context(['snapshot','rng',1,624],40,lookup,evidence)[0]=='requires_context_review'
    for broken in ([2,[0]*624+[40],None],[3,[0]*623+[40],None],[3,[0]*624+[625],None]):
        assert c.python_mt_shape(broken) is None
    assert c.classify_context(['snapshot','rng',1,10],0,lookup,{})[0]=='requires_context_review'


def test_counter_exclusion_requires_source_and_reconciled_ack():
    op={'old_head':{'path':'a','sha256':'a','step':6163},'completion':{'actual_steps':120,
        'checkpoint':{'path':'b','sha256':'b','step':6283}},'status':'acknowledged',
        'request':{'kind':'evaluation','steps':120}}
    value={'operations':{'id':op}};lookup=lambda p:c.get(value,p);p=['operations','id','old_head','step']
    assert c.classify_context(p,6163,lookup,{'checkpoint_steps':{}})[0].startswith('confirmed_checkpoint')
    op['completion']['actual_steps']=119
    assert c.classify_context(p,6163,lookup,{'checkpoint_steps':{}})[0]=='requires_context_review'


def test_complete_stream_context_proves_full_parent_and_rejects_trailing_garbage(tmp_path):
    path=tmp_path/'large-report.json';state=[3,[0]*624+[40],None]
    path.write_text(json.dumps({'snapshot':{'rng':state},'unused':'x'*1000}))
    lookup,proof=c.stream_context(path,[['snapshot','rng']],tmp_path/'ok.stderr')
    assert c.python_mt_shape(lookup(['snapshot','rng']))['mt_word_count']==624
    assert proof['events']==627
    path.write_text(path.read_text()+' BROKEN')
    with pytest.raises(ValueError,match='parse failed'):
        c.stream_context(path,[['snapshot','rng']],tmp_path/'bad.stderr')


def test_jsonl_stream_metadata_paths_preserve_physical_line(tmp_path):
    pp=proposal(tmp_path);root=tmp_path/'warehouse';root.mkdir()
    path=root/'training.jsonl';path.write_text('{"seed":1002670005}\n{"seed":1002670006}\n')
    a.audit(pp,[root],tmp_path/'bounded',max_bytes=1)
    result=t.triage(tmp_path/'bounded/inventory.json',pp,tmp_path/'triaged')
    matches=result['inventory'][0]['detail']['candidate_matches']
    assert [m['path'] for m in matches]==[[0,'seed'],[1,'seed']]


def test_delta_includes_new_weight_paths_as_stat_only_receipt_review(tmp_path):
    pp=proposal(tmp_path);root=tmp_path/'warehouse';root.mkdir()
    baseline=a.audit(pp,[root],tmp_path/'baseline')
    weight=root/'pair_6160/actors/actor.npz';weight.parent.mkdir(parents=True);weight.write_bytes(b'opaque')
    (root/'new-receipt.json').write_text('{}')
    result=e.delta_inventory(baseline,pp,output_directory=tmp_path/'delta')
    assert result['requires_new_content_audit'] and result['requires_weight_path_receipt_review']
    assert result['new_opaque_weight_paths_name_stat_only']==[str(weight)]
    assert result['authorization_granted'] is False


def test_missing_roots_remain_unresolved_after_exact_projection_and_context(tmp_path):
    pp=proposal(tmp_path);report=a.audit(pp,[tmp_path/'missing-warehouse'],tmp_path/'baseline')
    plan,_=e.read_exact_plan(exact_plan(tmp_path));projected=e.project_inventory(report,e.compile_families(plan))
    assert projected['summary']['unresolved_paths']==[str(tmp_path/'missing-warehouse')]
    annotation={'input_inventory_digest':a.digest(projected),'context_producer_sha256':a.file_sha(c.__file__),
                'records':[],'source_evidence':{}}
    reapplied=e.apply_context(projected,annotation)
    assert reapplied['summary']['families']['init:actor_initialization']['status']=='missing_roots_prevent_negative_conclusion'


def test_opaque_pickle_scalar_bytes_block_negative_conclusion(tmp_path):
    pp=proposal(tmp_path);root=tmp_path/'warehouse';root.mkdir()
    with gzip.open(root/'old-training-report.json','wb') as f:
        f.write(pickle.dumps({'seed':b'opaque scalar representation'}))
    a.audit(pp,[root],tmp_path/'baseline')
    result=t.triage(tmp_path/'baseline/inventory.json',pp,tmp_path/'triage')
    assert result['summary']['unresolved_paths']
    assert result['inventory'][0]['opaque_pickle_fields'][0]['path']==['seed']
    assert result['summary']['families']['pilot']['status'].startswith('unresolved')


def test_older_ledger_references_are_context_review_not_clean_negative(tmp_path):
    pp=proposal(tmp_path);root=tmp_path/'warehouse';root.mkdir()
    a.write_new(root/'old-seed-ledger.json',{'schema':'legacy-seed-ledger.v1','claimed_seeds':[6160]})
    result=a.audit(pp,[root],tmp_path/'baseline')
    assert result['summary']['families']['pilot']['status']=='audit_references_require_original_use_context_review'
    assert result['summary']['families']['pilot']['audit_references_not_original_execution']
