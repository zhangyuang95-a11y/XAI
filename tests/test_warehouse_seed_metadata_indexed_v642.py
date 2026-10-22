"""Audit detail parity includes every matched, ignored and reserved event."""
import json
from pathlib import Path
from experiments.warehouse import seed_metadata_audit as old,seed_metadata_audit_indexed_v642 as new


def test_literal_and_range_reports_are_identical():
    families=[dict(name='environment',kind='env',intervals=((10,13),(21,22)),query_only=False),
              dict(name='query_overlap',kind='env',intervals=((11,23),),query_only=True),
              dict(name='init',kind='init',intervals=((100,102),),query_only=False)]
    value=dict(seed=11,seed_ranges=[[10,2]],reserved_intervals=[dict(start=20,end_exclusive=24)],
        initialization_seed=100,metadata=dict(count=11,hash=11),random_state={'state':[11,[[12]]]},other=[True,11.,12,22,1000])
    assert new.analyze_literal(value,families)==old.analyze_literal(value,families)


def test_complete_inventory_coverage_classification_and_flags_match(tmp_path):
    root=tmp_path/'warehouse-history';root.mkdir();(root/'seed_ledger.json').write_text(json.dumps(dict(seed=11,init_seed=101)))
    (root/'reset.jsonl').write_text('{"seed":12,"count":11}\n{"action_seed":22}\n')
    (root/'config.toml').write_text('seed=11\n')
    (root/'empty-result.json').write_text('')
    (root/'large-report.json').write_text(' '*200)
    (root/'model.npz').write_bytes(b'opaque weight bytes must remain unread')
    (root/'symlink-result.json').symlink_to(root/'seed_ledger.json')
    source=root/'backend/training/seed_ledger.py';source.parent.mkdir(parents=True);source.write_text('# 500_004 10_500_204\n')
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps(dict(schema='warehouse-seed-proposal.v1',candidate_id='synthetic-index-parity',
        families=[dict(name='scenes',kind='env',values=[11,12,22])],exclude_paths=[],registered_trace_paths=[])))
    a=old.audit(proposal,[root],tmp_path/'old',max_bytes=128);b=new.audit(proposal,[root],tmp_path/'new',max_bytes=128)
    assert a['inventory']==b['inventory'] and a['summary']==b['summary'] and a['selection_stats']==b['selection_stats']
    for r in (a,b):
        r.pop('auditor_sha256');r['scope_exclusions'][1]='output';r.pop('interval_index_source_sha256',None)
    assert a==b


def test_indexed_bounded_audit_triage_and_original_exact_projection(tmp_path):
    from experiments.warehouse import seed_triage_indexed_v642 as triage,exact_seed_audit as exact
    root=tmp_path/'warehouse-history';root.mkdir();(root/'receipt.json').write_text(json.dumps(dict(seed=2001,not_seed='x'*1024)))
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps(dict(schema='warehouse-seed-proposal.v1',candidate_id='synthetic-index-pipeline',
        families=[dict(name='scenes',kind='env',values=[2001])],exclude_paths=[],registered_trace_paths=[])))
    inv=new.audit(proposal,[root],tmp_path/'bounded',max_bytes=128)
    result=triage.triage(tmp_path/'bounded/inventory.json',proposal,tmp_path/'triaged')
    oldfamilies=old.read_proposal(proposal)[1]
    projected=exact.project_inventory(result,oldfamilies)
    family=projected['summary']['families']['scenes']
    assert family['references_or_reservations']
    assert family['status']=='blocked_by_recorded_reference_or_reservation'
