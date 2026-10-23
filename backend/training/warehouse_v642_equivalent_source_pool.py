"""Reuse a genuine physical source audit only after complete input equivalence.

This never shares final-NN targets, qualifies programs, or issues target-cache
certificates. Both distinct final NNs still use native checked_targets.
"""
from copy import deepcopy
from pathlib import Path
import gzip,json
from . import warehouse_v642_observed_pool as native
from experiments.warehouse.v626_protocol import require,pointer,bound,digest,file_sha
from scripts.run_warehouse_v625 import write_new

ROOT=Path(__file__).resolve().parents[2]
VERSION='warehouse-v642-trace-equivalent-source-pair-audit.v1'
FILES=('backend/training/warehouse_v642_equivalent_source_pool.py',
       'tests/test_warehouse_v642_equivalent_source_pool.py',
       'docs/warehouse_v642_equivalent_source_pool.md')


def source_files():
    return {**native.source_files(),**{name:file_sha(ROOT/name) for name in FILES}}


def _put(path,value):
    write_new(path,value);path.chmod(0o444);return pointer(path)


def _assets(refs):
    unique={}
    for ref in refs:
        path=Path(ref['path'])
        require(path.is_absolute() and path.resolve()==path and not path.is_symlink(), 'noncanonical source-audit input')
        require(str(path) not in unique or unique[str(path)]==ref,'conflicting source-audit SHA')
        unique[str(path)]=ref
    for ref in unique.values():
        require(pointer(ref['path'])==ref and not Path(ref['path']).stat().st_mode&0o222,
            'source-audit input missing, changed or writable')
    return tuple(deepcopy(unique[name]) for name in sorted(unique))


def equivalent_inputs(first_ref,second_ref):
    """All episodes, tensors/mode, private endpoints and event values; no sample."""
    first,second=bound(first_ref),bound(second_ref)
    for s in (first,second):
        require(set(s)=={'schema','scope','owner','inventory','cases'} and s['schema']==native.VERSION
            and s['scope']==native.SCOPE,'not an admitted program-only source')
        owner=bound(s['owner'])
        require(owner['program_evidence_inventory']==s['inventory'],'source owner/inventory mismatch')
    refs=[first_ref,second_ref,first['owner'],second['owner'],first['inventory'],second['inventory']]
    if first['inventory']!=second['inventory'] or first['cases'].keys()!=second['cases'].keys():return None
    inventory=bound(first['inventory']);cases={c['episode']:c for c in inventory['cases']}
    require(set(cases)==set(first['cases']),'source case inventory incomplete')
    actors={};proof=[]
    for episode,item in first['cases'].items():
        other=second['cases'][episode]
        for v in (item,other):
            require(set(v)=={'behavior_actor','trace','initial_environment','final_environment'},'unknown source item')
            refs.extend(v.values())
            ref=v['behavior_actor'];key=(ref['path'],ref['sha256'])
            if key not in actors:actors[key]=native.FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'])
        a,b=(actors[(v['behavior_actor']['path'],v['behavior_actor']['sha256'])] for v in (item,other))
        if a.model_version!=b.model_version or a.role_sha256!=b.role_sha256:return None
        if any(item[k]['sha256']!=other[k]['sha256'] for k in ('initial_environment','final_environment')):return None
        traces=[]
        for v in (item,other):
            path=Path(v['trace']['path'])
            if path.suffix=='.gz':
                with gzip.open(path,'rt') as handle:t=json.load(handle)
            else:t=bound(v['trace'])
            require(set(t)=={'schema','case','behavior_actor','events'} and t['schema']==native.EPISODE_VERSION
                and digest(t['case'])==digest(cases[episode]) and t['behavior_actor']==v['behavior_actor']
                and 0<len(t['events'])<=120,'source trace identity or complete frame inventory differs')
            traces.append(t)
        hashes=[digest(t['events']) for t in traces]
        if hashes[0]!=hashes[1]:return None
        proof.append(dict(episode=episode,case=cases[episode],physical_frames=len(traces[0]['events']),
            complete_events_sha256=hashes[0],role_sha256=a.role_sha256,model_version=a.model_version,
            initial_sha256=item['initial_environment']['sha256'],final_sha256=item['final_environment']['sha256']))
    inputs=_assets(refs)
    return dict(version=VERSION,sources=[first_ref,second_ref],inventory=first['inventory'],
        episodes=proof,input_asset_refs=list(inputs),complete_episodes=len(proof),
        joint_physical_frames=sum(v['physical_frames'] for v in proof),
        all_source_episodes_compared=True,no_final_NN_target_reuse=True,new_model_forwards=0,
        new_environment_resets=0,new_training_steps=0,training_authority=False,pilot_opened=False)


def audit_pair(source_refs,*,reference_directory,output):
    require(set(source_refs)=={'raw','regularized'},'both bound source pools required')
    sources=source_files();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    # This exact production path always invokes the real native full audit.
    raw=native.audit_observed_pool(source_refs['raw'],reference_directory=reference_directory)
    require(type(raw) is native.ObservedPool,'genuine native source audit required');raw.verify()
    first=_put(output/'raw-native-audit.json',dict(method='native_full_source_physics_and_behavior_NN_replay',
        audit=raw.audit,audit_sha256=raw.audit_sha256,input_asset_refs=list(raw.refs),source_files=native.source_files()))
    equivalence=equivalent_inputs(source_refs['raw'],source_refs['regularized'])
    if equivalence is None:
        regularized=native.audit_observed_pool(source_refs['regularized'],reference_directory=reference_directory)
        second=_put(output/'regularized-native-audit.json',dict(method='native_full_source_physics_and_behavior_NN_replay',
            audit=regularized.audit,audit_sha256=regularized.audit_sha256,input_asset_refs=list(regularized.refs),source_files=native.source_files()))
        refs=[first,second];mode='separate_native_full_audits'
    else:
        require(equivalence['complete_episodes']==raw.audit['complete_source_episodes']
            and equivalence['joint_physical_frames']==raw.audit['joint_physical_frames'],
            'equivalence omitted physically audited source episodes/frames')
        proof=_put(output/'complete-input-equivalence.json',dict(equivalence,native_preceding_audit=first,
            implementation_source_files=sources,implementation_source_sha256=digest(sources),
            source_data_sha256=raw.observed_data_sha256,second_source_physics_NN_simulation_recomputed=False))
        source=bound(source_refs['regularized'])
        refs=[source_refs['regularized'],source['owner'],source['inventory'],*raw.refs,first,proof]
        for item in source['cases'].values():refs.extend(item.values())
        refs=_assets(refs);audit=deepcopy(raw.audit)
        audit.update(source=source_refs['regularized'],owner=source['owner'],asset_manifest_sha256=digest(refs),
            implementation_provenance=proof,source_simulation_method=VERSION,
            identical_input_source_simulation_reused=True)
        for outcome in audit['outcomes']:
            outcome['behavior_actor']=deepcopy(source['cases'][outcome['case']['episode']]['behavior_actor'])
        regularized=native.ObservedPool(deepcopy(source_refs['regularized']),deepcopy(source['owner']),raw.parts,refs,
            raw.sources,raw.readiness,raw.observed_data_sha256,audit,digest(audit)).verify()
        refs=[first,proof];mode='native_first_audit_then_complete_input_equivalence'
    second_record=_put(output/'regularized-audit-record.json',dict(audit=regularized.audit,
        audit_sha256=regularized.audit_sha256,input_asset_refs=list(regularized.refs),source_files=sources,
        method=mode))
    refs.append(second_record)
    require(source_files()==sources,'source-audit implementation changed')
    receipt=_put(output/'result.json',dict(version=VERSION,source_files=sources,sources=source_refs,mode=mode,
        audit_sha256={'raw':raw.audit_sha256,'regularized':regularized.audit_sha256},proof_refs=refs,
        audit_records={'raw':first,'regularized':second_record},
        final_NN_target_replay_bypassed=False,program_qualification_changed=False,new_environment_resets=0,
        new_training_steps=0,training_authority=False,pilot_opened=False))
    return dict(pools={'raw':raw,'regularized':regularized},proof_refs=[receipt,*refs],mode=mode)


def verify_report(ref,source_refs):
    """Recheck all equivalence inputs/proofs, never claim another NN replay."""
    _assets([ref]);r=bound(ref)
    require(r['version']==VERSION and r['source_files']==source_files() and r['sources']==source_refs
        and r['final_NN_target_replay_bypassed'] is r['program_qualification_changed']
            is r['training_authority'] is r['pilot_opened'] is False
        and r['new_environment_resets']==r['new_training_steps']==0,'source-pair audit scope changed')
    inputs=[ref,*r['proof_refs']]
    require(set(r['audit_records'])==set(r['audit_sha256'])=={'raw','regularized'},
        'both source audit records required')
    records={v:bound(p) for v,p in r['audit_records'].items()}
    for variant,record in records.items():
        require(record['audit_sha256']==digest(record['audit'])==r['audit_sha256'][variant]
            and record['audit']['source']==source_refs[variant]
            and record['audit']['current_policy_task_quality_estimate'] is None
            and record['audit']['source_policy_outcomes_only'] is True,
            'source audit does not bind its actual data/owner or falsely claims final policy quality')
        inputs.extend([r['audit_records'][variant],*record['input_asset_refs']])
    require(records['raw']['method']=='native_full_source_physics_and_behavior_NN_replay'
        and records['raw']['source_files']==native.source_files(),'upstream full native source audit missing')
    if r['mode']=='native_first_audit_then_complete_input_equivalence':
        proof_ref=records['regularized']['audit']['implementation_provenance'];p=bound(proof_ref)
        expected=equivalent_inputs(source_refs['raw'],source_refs['regularized'])
        require(expected is not None and p==dict(expected,native_preceding_audit=r['audit_records']['raw'],
            implementation_source_files=source_files(),implementation_source_sha256=digest(source_files()),
            source_data_sha256=records['raw']['audit']['observed_data_sha256'],second_source_physics_NN_simulation_recomputed=False),
            'complete source equivalence proof no longer reproduces')
        require(p['joint_physical_frames']==records['raw']['audit']['joint_physical_frames']
            and p['complete_episodes']==records['raw']['audit']['complete_source_episodes']
            and records['regularized']['audit']['observed_data_sha256']==p['source_data_sha256'],
            'source equivalence omitted episodes or changed observed data')
        inputs.extend([proof_ref,*p['input_asset_refs']])
    else:require(r['mode']=='separate_native_full_audits','unknown source-audit method')
    return dict(audit_sha256=r['audit_sha256'],input_asset_refs=list(_assets(inputs)),mode=r['mode'])
