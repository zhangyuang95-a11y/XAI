#!/usr/bin/env python3
"""Descriptive completed program-source outcomes, never quality admission.

The frozen behaviour policy is the15k snapshot. No program fitting, model
selection, NN/environment execution or new random draw occurs here.
"""
from pathlib import Path
import argparse,gzip,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse.v626_protocol import require,pointer,bound,digest,file_sha
from scripts.measure_warehouse_v628_collection import episode as public_episode
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v642-program-source-outcomes-descriptive.v1'


def summarize(rows):
    if not rows:return None
    n=len(rows)
    return dict(complete_episodes=n,physical_joint_frames=sum(r['physical_frames'] for r in rows),
        mean_deliveries=sum(r['deliveries'] for r in rows)/n,
        mean_robot_collisions=sum(r['robot_collisions'] for r in rows)/n,
        shutdown_episodes=sum(r['shutdown'] for r in rows),shutdown_rate=sum(r['shutdown'] for r in rows)/n)


def run(calibration,output):
    calibration=Path(calibration).resolve();descriptor_ref=pointer(calibration/'descriptor.json');d=bound(descriptor_ref)
    require(d['version']=='warehouse-v642-finite-paired-calibration-owner.v1'
        and d['stage']=='paired_20k_calibration_not_formal_training','actual admitted calibration descriptor required')
    inventory_ref=d['program_evidence_inventory'];inventory=bound(inventory_ref);variants={}
    require(inventory['expected_counts']==dict(fit=900,calibration=150,eligibility=300,validation=300),
        'program-source episode inventory differs from the locked calibration recipe')
    inputs=[descriptor_ref,inventory_ref]
    for variant in ('raw','regularized'):
        folder=calibration/variant/'program-source';actor=folder/'behavior-NN.npz';rows=[]
        actor_ref=pointer(actor) if actor.exists() and not actor.stat().st_mode&0o222 else None
        for case in inventory['cases']:
            trace=folder/(case['episode']+'-trace.json.gz');final=folder/(case['episode']+'-final.json')
            if not all(p.exists() and not p.stat().st_mode&0o222 for p in (trace,final)):continue
            trace_ref,final_ref=pointer(trace),pointer(final)
            with gzip.open(trace,'rt') as h:value=json.load(h)
            require(actor_ref is not None and value['schema']=='warehouse-v642-program-source-episode.v1'
                and value['case']==case and value['behavior_actor']==actor_ref,'source policy/episode binding differs')
            events=value['events'];require(events and all(e['action_override_agents']==[] for e in events)
                and all(not(e['terminated'] or e['truncated']) for e in events[:-1]),'partial or overridden source episode')
            adapted=[dict(case=case,frame=i,pre_state_public=e['pre_state_public'],post_state_public=e['post_state_public'],
                proposed_actions=e['proposed_actions'],actor_sha256=actor_ref['sha256']) for i,e in enumerate(events)]
            outcome=public_episode(case,adapted,len(events),actor_ref['sha256'])
            snapshot=bound(final_ref);require(snapshot['payload_sha256']==digest(snapshot['payload']),'final source snapshot changed')
            fields=snapshot['payload']['state']['fields']
            require(fields['frame']==outcome['physical_frames'] and fields['total_deliveries']==outcome['deliveries']
                and fields['robot_collision_events']==outcome['robot_collisions'] and bool(fields['shutdown_count'])==outcome['shutdown']
                and fields['terminated']==events[-1]['terminated']==outcome['shutdown']
                and fields['truncated']==events[-1]['truncated']==(not outcome['shutdown']),
                'public source outcome does not match the completed environment snapshot')
            inputs.extend([trace_ref,final_ref]);rows.append(outcome)
        if actor_ref:inputs.append(actor_ref)
        variants[variant]=dict(behavior_actor=actor_ref,behavior_joint_step=15000,descriptive=summarize(rows),
            by_split={s:summarize([r for r in rows if r['case']['split']==s]) for s in inventory['expected_counts']},
            by_task={str(t):summarize([r for r in rows if r['case']['task']==t]) for t in (1,2,3)})
    for ref in inputs:require(pointer(ref['path'])==ref,'program source changed during descriptive readback')
    result=dict(version=VERSION,descriptor=descriptor_ref,inventory=inventory_ref,variants=variants,input_asset_refs=inputs,
        source_sha256=file_sha(__file__),counter_reconstruction_source_sha256=file_sha(ROOT/'scripts/measure_warehouse_v628_collection.py'),
        scope='completed_program_extraction_source_episodes_only_not_independent_quality_or_final_NN',
        source_policy_outcomes_only=True,independent_NN_action_and_physics_replay_pending=True,
        selection_or_training_authority=False,model_acceptance_passed=False,pilot_opened=False,
        new_environment_resets=0,new_stochastic_draws=0,NN_updates=0)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);write_new(output,result);output.chmod(0o444)
    return pointer(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--calibration',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    print(json.dumps(run(args.calibration,args.output)),flush=True)
