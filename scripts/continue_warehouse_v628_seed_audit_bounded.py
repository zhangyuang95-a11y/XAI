#!/usr/bin/env python3
"""Explicit engineering continuation before any physical experiment reset.

Original scientific source and plan remain immutable. Reuse only its completed
read-only bounded inventory after each source file is checked by triage.
The new parser hash and query-superset semantics have a separate admission.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v628_protocol as v,v628_data as d
from experiments.warehouse import v628_seed_triage_bounded as fast,exact_seed_audit as exact,seed_metadata_audit as meta
from scripts.run_warehouse_v625 import write_new


def run():
    path,plan=v.checked_plan();folder=v.OUTPUT/'seed-ledger'
    v.require(not (v.OUTPUT/'collection').exists() and not (v.OUTPUT/'jobs').exists(),'continuation only before any experimental reset')
    interruption=v.bound(v.pointer(v.OUTPUT/'audit-interruption-bounded.json'))
    v.require(interruption['scientific_resets']==0 and interruption['all_original_processes_observed_terminated'],'original audit must be stopped and recorded')
    source={str(Path(f).relative_to(ROOT)):v.file_sha(f) for f in (Path(__file__),Path(fast.__file__),ROOT/'tests/test_warehouse_v628_seed_triage_bounded.py')}
    admission=dict(schema='warehouse-v628-seed-audit-engineering-addendum.v1',original_plan=v.pointer(path),
        original_scientific_source_sha256=plan['source_sha256'],additional_source_files=source,
        interruption=v.pointer(v.OUTPUT/'audit-interruption-bounded.json'),
        original_bounded_inventory=v.pointer(folder/'bounded/inventory.json'),
        actual_seed_plan=v.pointer(folder/'exact-plan.json'),proposal=v.pointer(folder/'proposal.json'),
        query_semantics='bounded_complete_JSON_integer_evidence_prefilter_then_original_exact_interval_classification',plain_JSON_byte_bound=768*1024*1024,
        scene_allocation_changed=False,model_changed=False,scientific_budget_changed=False,
        evidence_reused='completed_read_only_metadata_inventory_only_no_experimental_measurements',
        environment_resets=0,actor_updates=0,PPO_joint_steps=0,pilot_opened=False)
    write_new(folder/'bounded-admission.json',admission)
    result=fast.triage(folder/'bounded/inventory.json',folder/'proposal.json',folder/'triage-bounded-01')
    print('FAST_TRIAGE_COMPLETE',flush=True)
    evidence=exact.assess(folder/'exact-plan.json',folder/'triage-bounded-01/result.json',folder/'exact-bounded-01',context_source_root=ROOT)
    v.require(not evidence['summary']['unresolved_paths'],'unresolved historical metadata')
    for row in evidence['summary']['families'].values():
        v.require(not any(row.get(k) for k in ('references_or_reservations','reservation_overlaps',
            'audit_references_not_original_execution','excluded_weight_path_seed_hints')),'historical seed conflict')
    roots=[ROOT.parent/name for name in d.HISTORY_NAMES];stats={};selection=[]
    traces=result['registered_trace_paths']
    for p in meta.candidates(roots,[v.OUTPUT],registered_trace_paths=traces,selection_stats=stats):
        s=p.stat();selection.append(dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in result['inventory'] if r['classification']!='source_template_environment_reservation'}
    v.require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in selection},'history changed during seed audit')
    for name,sha in source.items():v.require(v.file_sha(ROOT/name)==sha,'additional audit source changed')
    write_new(folder/'clearance.json',dict(schema='warehouse-v628-seed-clearance.v1',cases_sha256=v.digest(d.cases()),
        exact=v.pointer(folder/'exact-bounded-01/result.json'),selection=selection,weight_paths=stats['excluded_weights_name_stat_only'],
        registered_traces=traces,approved_finite_development_resets=1650,sealed_authority=False,training_authority=False,
        engineering_addendum=v.pointer(folder/'bounded-admission.json')))
    print('SEED_AUDIT_COMPLETE',flush=True)
    d.collect();print('COLLECTION_COMPLETE',flush=True)
    from scripts.run_warehouse_v628 import execute
    print(json.dumps(execute()),flush=True)


if __name__=='__main__':run()
