#!/usr/bin/env python3
"""Independent frozen-NN action/array/public-counter replay; no env reset."""
from pathlib import Path
from datetime import datetime
import gzip,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v629_confirmation as c
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,encode_actor_tree_features,FEATURE_NAMES
from backend.training.warehouse_v622_program import masked_probabilities
from env.warehouse.public_observation_v6 import rebuild_public_pre_action_state,encode_public_observation
from env.warehouse.pure_nn import AGENTS,action_masks
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v617 import ACTION_PHYSICS_CONFIG
from scripts.measure_warehouse_v628_collection import episode as public_episode
from scripts.run_warehouse_v625 import write_new


def verify_frame(actor,case,event,arrays,index):
    """Rebuild from pre-action public fields, then independent sampled action.

    No environment is instantiated or stepped. This checks both roles even
    when physics leaves a collided robot stationary or a role inactive.
    """
    state=rebuild_public_pre_action_state(event['pre_state_public'])
    c.v.require(event['frame']==index and state.frame==index,'frame identity differs')
    obs=encode_public_observation(state);m=action_masks(state)
    x=encode_actor_tree_features(obs,m);p=masked_probabilities(actor.logits(obs,m),m)
    c.v.require(np.array_equal(x,arrays['x'][index]) and np.array_equal(m,arrays['m'][index])
        and np.array_equal(p,arrays['p'][index]),'public reconstruction or complete-NN target array differs')
    indices=[]
    for i,role in enumerate(AGENTS):
        seed=independent_agent_seed(base_seed=case['action_seed'],episode_id=int(state.episode_id),frame=index,agent_id=role)
        action=int(np.random.default_rng(seed).choice(5,p=p[i]));indices.append(action)
        c.v.require(arrays['sampled'][index,i]==action and m[i,action]==1
            and event['proposed_actions'][role]==ACTIONS[action],'recorded action differs from bound frozen NN')
    c.v.require({a['agent_id']:a['last_action'] for a in event['post_state_public']['agents']}==event['proposed_actions'],
        'post-state submitted actions differ')
    return 2


def audit_collection(path):
    path,plan,bundle=c.checked_plan(path);folder=path.parent/'collection'
    result=c.v.bound(c.v.pointer(folder/'result.json'))
    c.v.require(json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=c.v.file_sha(folder/'result.json')),
        'collection missing or changed completion seal')
    c.v.require(result['schema']=='warehouse-v629-fixed-confirmation-collection.v1'
        and result['plan']==c.v.pointer(path) and result['actor']==plan['actor'] and result['actual_reset_count']==516
        and result['post_policy_interventions']==0 and result['actor_updates']==result['PPO_joint_steps']==0,
        'collection receipt plan/actor/count differs')
    clearance=c.checked_clearance(path,plan,execution=False)
    clearance_ref=c.v.pointer(path.parent/'seed-ledger/clearance.json')
    claim_ref=c.v.pointer(c.claim_path());claim=c.v.bound(claim_ref)
    started=c.v.bound(c.v.pointer(folder/'started.json'))
    binding=dict(clearance=clearance_ref,exact_evidence=clearance['exact'],expires_utc=clearance['expires_utc'])
    c.v.require(claim['plan']==c.v.pointer(path) and claim['cases_sha256']==c.v.digest(plan['cases'])
        and all(claim[k]==value for k,value in binding.items())
        and datetime.fromisoformat(clearance['issued_utc'])<=datetime.fromisoformat(claim['started_utc'])<datetime.fromisoformat(clearance['expires_utc'])
        and started==dict(plan=c.v.pointer(path),scheduled_resets=516,claim=claim_ref,**binding)
        and result['claim']==claim_ref and result['clearance']==clearance_ref and result['exact_evidence']==clearance['exact'],
        'single-use confirmation claim/clearance/start/collection differs')
    expected_ids={case['case_id'] for case in plan['cases']}
    c.v.require(set(result['cases'])==expected_ids,'missing or additional collected cases')
    actor=FrozenWarehouseV6Actor.load(plan['actor']['path'],expected_sha256=plan['actor']['sha256'],expected_role_sha256=plan['actor_role_sha256'])
    c.v.require(c.v.pointer(result['actual_resets']['path'])==result['actual_resets']
        and c.v.pointer(result['preauthorization']['path'])==result['preauthorization'],'journals changed')
    resets=[json.loads(line) for line in Path(result['actual_resets']['path']).read_bytes().splitlines()]
    auth=[json.loads(line) for line in Path(result['preauthorization']['path']).read_bytes().splitlines()]
    c.v.require(len(resets)==len(auth)==516,'reset/preauthorization count differs')
    counters=[];checks=0
    for i,case in enumerate(plan['cases']):
        authorized=auth[i]['authorized_utc']
        c.v.require(auth[i]==dict(index=i,case=case,plan_sha256=c.v.file_sha(path),clearance_sha256=clearance_ref['sha256'],authorized_utc=authorized)
            and datetime.fromisoformat(claim['started_utc'])<=datetime.fromisoformat(authorized)<datetime.fromisoformat(clearance['expires_utc'])
            and (not i or datetime.fromisoformat(auth[i-1]['authorized_utc'])<=datetime.fromisoformat(authorized))
            and resets[i]['seed']==case['scenario_seed'] and resets[i]['episode_id']==1
            and resets[i]['phase']==('pressure' if case['split']=='pressure' else 'program'),'reset/preauthorization order differs')
        c.v.require(all(resets[i][key]==ACTION_PHYSICS_CONFIG[key] for key in ('map_layout_id','move_battery_cost','charge_per_wait')),
            'reset used different physical environment')
        entry=result['cases'][case['case_id']]
        c.v.require(set(entry)=={'arrays','trace'},'unknown case assets')
        for key,suffix in (('arrays','.npz'),('trace','.jsonl.gz')):
            expected=folder/(case['case_id']+suffix)
            c.v.require(entry[key]==c.v.pointer(expected) and not expected.stat().st_mode&0o222,'case asset hash/location/immutability differs')
        with np.load(entry['arrays']['path'],allow_pickle=False) as z:
            c.v.require(set(z.files)=={'x','m','p','sampled'},'unknown trajectory arrays')
            a={key:z[key].copy() for key in z.files}
        n=len(a['x'])
        c.v.require(a['x'].shape==(n,2,len(FEATURE_NAMES)),'invalid joint observation dimensions')
        c.v.require(a['m'].shape==a['p'].shape==(n,2,5) and a['sampled'].shape==(n,2)
            and a['sampled'].dtype==np.int64 and all(a[key].dtype==np.float32 and np.isfinite(a[key]).all() for key in ('x','m','p')),
            'incomplete or invalid physical arrays')
        with gzip.open(entry['trace']['path'],'rt') as h:events=[json.loads(line) for line in h]
        counter=public_episode(case,events,n,plan['actor']['sha256']);counter['complete']=True
        for frame,event in enumerate(events):
            c.v.require(event.get('action_override_agents')==[],'missing/no longer empty action-override evidence')
            c.v.require(event['pre_state_public']['episode_id']==resets[i]['episode_id'],'public episode/reset differs')
            checks+=verify_frame(actor,case,event,a,frame)
        counters.append(counter)
    c.v.require(counters==result['outcomes'] and checks==result['NN_action_checks'],'public physical counters or NN action count differs')
    bundle.verify()
    audit=dict(schema='warehouse-v629-confirmation-collection-audit.v1',integrity_passed=True,
        plan=c.v.pointer(path),collection=c.v.pointer(folder/'result.json'),actual_reset_count=516,public_complete_episodes=516,
        NN_action_checks=checks,physical_counter_replay=True,public_feature_and_distribution_replay=True,
        actor=plan['actor'],source_sha256=plan['source_sha256'],audit_environment_resets=0,fit_calls=0,LP_calls=0,
        clearance=clearance_ref,exact_evidence=clearance['exact'],claim=claim_ref,
        actor_updates=0,PPO_joint_steps=0,sealed_audit_used=False,heldout_used=False,pilot_opened=False)
    write_new(path.parent/'collection-audit.json',audit);return audit


def load_datasets(path,plan):
    audit=c.v.bound(c.v.pointer(path.parent/'collection-audit.json'))
    receipt=c.v.bound(audit['collection'])
    c.v.require(audit['schema']=='warehouse-v629-confirmation-collection-audit.v1' and audit['integrity_passed'] is True
        and audit['plan']==c.v.pointer(path) and audit['source_sha256']==plan['source_sha256'] and audit['actor']==plan['actor']
        and audit['actual_reset_count']==audit['public_complete_episodes']==516,'no complete independent collection audit')
    for key in ('claim','clearance','exact_evidence'):
        c.v.require(audit[key]==receipt[key] and c.v.pointer(audit[key]['path'])==audit[key],
            'audited collection authority changed or missing')
    for case in plan['cases']:
        refs=receipt['cases'][case['case_id']]
        for key in ('arrays','trace'):
            c.v.require(c.v.pointer(refs[key]['path'])==refs[key],'audited physical asset changed or missing')
    datasets={}
    for split in ('validation','pressure'):
        joint={key:[] for key in ('x','m','p','sampled')};rows=[]
        for case in plan['cases']:
            if case['split']!=split:continue
            ref=receipt['cases'][case['case_id']]['arrays'];c.v.require(c.v.pointer(ref['path'])==ref,'audited arrays changed')
            with np.load(ref['path'],allow_pickle=False) as z:
                for key in joint:joint[key].append(z[key].copy())
                n=len(z['x'])
            rows.extend(dict(episode=case['case_id'],base_seed=str(case['base']),task=case['task']) for _ in range(n))
        joint={key:np.concatenate(values) for key,values in joint.items()}
        datasets[split]={role:{**{key:value[:,i].copy() for key,value in joint.items()},'rows':list(rows)} for i,role in enumerate(AGENTS)}
    return datasets,receipt,audit


def score(path):
    path,plan,bundle=c.checked_plan(path);datasets,receipt,audit=load_datasets(path,plan)
    value=c.metrics.evaluate_fixed(bundle,datasets,plan['cases'],receipt['outcomes'])
    value.update(plan=c.v.pointer(path),collection_audit=c.v.pointer(path.parent/'collection-audit.json'),collection=audit['collection'])
    write_new(path.parent/'metrics.json',value);return value


def finalize(path):
    path,plan,bundle=c.checked_plan(path)
    value=c.v.bound(c.v.pointer(path.parent/'metrics.json'));datasets,receipt,audit=load_datasets(path,plan)
    c.v.require(value['plan']==c.v.pointer(path) and value['collection_audit']==c.v.pointer(path.parent/'collection-audit.json')
        and value['collection']==audit['collection'] and value['fixed_assets']==bundle.provenance,'scoring inputs/locked assets differ')
    # Re-score from audited raw arrays, not from stored booleans or rounded tables.
    expected=c.metrics.evaluate_fixed(bundle,datasets,plan['cases'],receipt['outcomes'])
    for key in expected:c.v.require(value[key]==expected[key],'independent confirmation result differs: '+key)
    passed=expected['metrics_passed'] and audit['integrity_passed']
    result=dict(schema='warehouse-v629-fixed-confirmation-summary.v1',confirmation_passed=bool(passed),
        plan=c.v.pointer(path),metrics=c.v.pointer(path.parent/'metrics.json'),collection_audit=c.v.pointer(path.parent/'collection-audit.json'),
        method_and_six_assets_fixed=True,next_stage='paired_20k_training_calibration' if passed else 'candidate_closed_failed_confirmation',
        independent_final_audit_used=False,heldout_used=False,actor_updates=0,PPO_joint_steps=0,pilot_opened=False,publishable=False)
    write_new(path.parent/'summary.json',result)
    marker=c.v.OUTPUT/('confirmation.passed.json' if passed else 'confirmation.failed.json')
    write_new(marker,dict(summary=c.v.pointer(path.parent/'summary.json'),confirmation_passed=bool(passed)))
    return result
