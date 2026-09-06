"""Post-hoc RCPD extraction from a frozen kitchen Actor, without feedback.

All labels are actual Actor distributions. Fit/selection/audit scenes are
fingerprint-disjoint. The audit is collected only after the selected tree has
been serialized; it never selects a different tree.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from core import RCPD, RCPDConfig
from backend.cooperative_kitchen.policy import NumpyKitchenPolicy
from backend.cooperative_kitchen.calibration import atomic_json, reserved_fingerprints, unique_scenarios
from env.cooperative_kitchen import CooperativeKitchen, ACTIONS, OBSERVATION_FEATURES, program_decision

CRITICAL_GROUPS=('plating','handoff','blocked_counters','waiting')
INTERVENTIONS=('player_actions_three_steps','counter_clear_hypothetical')
DIRECTION_EPSILON=1e-5


def posthoc_program_payload(program, audit_result=None, *, selection_passes=False):
    """Replace generic core training-role defaults with truthful kitchen scope."""
    payload=program.to_dict()
    metadata=payload.setdefault('metadata',{})
    eligible=bool(selection_passes and audit_result and audit_result['passes'])
    metadata.update(training_role='posthoc_extraction_only',program_roles=['local_explanation_audit'],
                    feedback_enabled=False,feedback_eligible=False,feedback_weight=0.0,
                    regularization_version=False,explanation_eligible=eligible,
                    eligibility_source='kitchen_independent_selection_and_untouched_audit',
                    natural_language_validation='separate mandatory answer audit')
    metrics=metadata.setdefault('metrics',{})
    metrics.update(feedback_eligible=False,feedback_weight=0.0,
                   feedback_ineligibility_reasons=['posthoc_only_no_actor_feedback'],
                   explanation_eligible=eligible,
                   explanation_ineligibility_reasons=[] if eligible else
                       ['kitchen_held_out_extraction_gate_failed' if audit_result else 'kitchen_held_out_audit_pending'])
    if audit_result is not None:metadata['kitchen_held_out_audit']=audit_result
    metadata['program_structure_sha256']=hashlib.sha256(json.dumps(
        {k:payload[k] for k in ('action_names','feature_names','root')},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload


def _actor_row(policy,env,scene,*,pair=None,intervention=False,role=None,kind=None):
    observation=env.observations()['ai']
    actions,distribution=policy.act({'ai':observation})
    state=env.state;actor=next(a for a in state['actors'] if a['id']=='ai')
    groups=['ordinary']
    if state['pot']['ready'] or actor['holding']=='plate':groups.append('plating')
    if actor['holding']=='soup' or any(state['counters'].values()):groups.append('handoff')
    if all(state['counters'].values()):groups.append('blocked_counters')
    if actions['ai']=='WAIT' or state['pot']['remaining'] or (actor['holding'] is None and not any(state['counters'].values())):groups.append('waiting')
    dist=distribution['ai']
    return {'features':dict(zip(OBSERVATION_FEATURES,map(float,observation))),
        'probabilities':dict(zip(ACTIONS,map(float,dist['probabilities']))),'neural_action':actions['ai'],
        'groups':groups,'scene':scene['fingerprint'],'seed':scene['seed'],'frame':state['turn'],
        'pair':pair,'intervention':intervention,'intervention_role':role,'intervention_kind':kind,
        'source':'frozen_actor_trajectory' if not intervention else 'isolated_actor_intervention'}


def _advance_sequence(policy,env,sequence):
    branch=env.fork();executed=[]
    for action in sequence:
        if branch.state['done']:break
        ai=policy.act({'ai':branch.observations()['ai']})[0]['ai']
        result=branch.step({'human':action,'ai':ai},include_state=False)
        executed.append({'human':action,'ai':ai,'events':result['events']})
    return branch,executed


def intervention_pairs(policy,env,scene):
    """Paired real transitions and explicitly hypothetical inventory changes."""
    result=[]
    frame_id=f"{scene['fingerprint']}:{env.state['turn']}"
    # Equal-length branches control elapsed time. Future AI actions adapt freely.
    if env.state['turn']+3 < env.state['maxSteps']:
        baseline,baseline_actions=_advance_sequence(policy,env,['WAIT']*3)
        candidate_actions=[]
        for action in ACTIONS[:-1]:
            branch=env.fork()
            outcome=branch.step({'human':action,'ai':'WAIT'},include_state=False)
            valid=any(e.get('actor')=='human' and e['type'] not in ('invalid_interaction','blocked') for e in outcome['events'])
            if valid:candidate_actions.append(action)
        if candidate_actions:
            action=candidate_actions[(env.state['turn']//12) % len(candidate_actions)]
            altered,altered_actions=_advance_sequence(policy,env,[action]*3)
            if len(baseline_actions)==len(altered_actions)==3 and not baseline.state['done'] and not altered.state['done']:
                kind='player_actions_three_steps';pair=frame_id+':'+kind
                common={'assumption':{'baseline_player_actions':['WAIT']*3,'altered_player_actions':[action]*3,
                    'scope':'actual joint transitions; teammate adapts using the same frozen Actor'},
                    'origin_frame':env.state['turn']}
                a=_actor_row(policy,baseline,scene,pair=pair,intervention=True,role='baseline',kind=kind)
                b=_actor_row(policy,altered,scene,pair=pair,intervention=True,role='altered',kind=kind)
                result.append(({**a,**common},{**b,**common}))
    occupied=[key for key,item in env.state['counters'].items() if item is not None]
    if occupied:
        branch=env.fork();key=occupied[0]
        item_id=branch.state['_counter_item_ids'][key]
        branch.state['counters'][key]=None;branch.state['_counter_item_ids'][key]=None
        branch.state['_items'][item_id]['location']='hypothetical_removed'
        # Verify a fully consistent isolated snapshot; the live environment is untouched.
        branch.restore(branch.snapshot())
        kind='counter_clear_hypothetical';pair=frame_id+':'+kind
        common={'assumption':{'counter':key,'removed_item':env.state['counters'][key],
            'scope':'abstract state intervention at the same time; not a one-step player action or live deletion'},
            'origin_frame':env.state['turn']}
        a=_actor_row(policy,env,scene,pair=pair,intervention=True,role='baseline',kind=kind)
        b=_actor_row(policy,branch,scene,pair=pair,intervention=True,role='altered',kind=kind)
        result.append(({**a,**common},{**b,**common}))
    return result


def collect(policy,descriptors,*,sample_stride=2,intervention_stride=12):
    rows=[]
    for index,scene in enumerate(descriptors):
        env=CooperativeKitchen(seed=scene['seed'],scenario_id=scene['scenario_id'])
        rng=random.Random(scene['seed']+51)
        profile=('efficient','upper','lower','perturbed')[index % 4]
        while not env.state['done']:
            if env.state['turn'] % sample_stride==0:
                rows.append(_actor_row(policy,env,scene))
            if env.state['turn'] % intervention_stride==0:
                for pair in intervention_pairs(policy,env,scene):rows.extend(pair)
            ai=policy.act({'ai':env.observations()['ai']})[0]['ai']
            human=program_decision(env,'human',profile=profile,rng=rng)['action']
            env.step({'human':human,'ai':ai},include_state=False)
    return rows


def audit(program,rows,*,minimum_critical_samples=1):
    if not rows:raise ValueError('Cannot audit an empty extraction set')
    neural=np.asarray([[r['probabilities'][a] for a in ACTIONS] for r in rows])
    symbolic=np.asarray([[program.predict_proba(r['features'])[a] for a in ACTIONS] for r in rows])
    agreement=neural.argmax(1)==symbolic.argmax(1)
    ordinary=np.asarray([not r['intervention'] for r in rows])
    overall=float(agreement[ordinary].mean()) if ordinary.any() else None
    critical={}
    for kind in CRITICAL_GROUPS:
        indices=[i for i,r in enumerate(rows) if not r['intervention'] and kind in r['groups']]
        critical[kind]={'n':len(indices),'scenes':len({rows[i]['scene'] for i in indices}),
            'fidelity':float(agreement[indices].mean()) if indices else None}
    pair_indices={}
    for i,row in enumerate(rows):
        if row['intervention']:pair_indices.setdefault(row['pair'],[]).append(i)
    evidence=[]
    metrics={kind:{'pairs':0,'nontrivial_pairs':0,'direction_correct':0,'uninformative_pairs':0,'changed_action_pairs':0,'changed_action_correct':0} for kind in INTERVENTIONS}
    for pair,indices in pair_indices.items():
        if len(indices)!=2:raise ValueError(f'Incomplete intervention pair: {pair}')
        a,b=sorted(indices,key=lambda i:rows[i]['intervention_role']!='baseline')
        kind=rows[b]['intervention_kind'];m=metrics[kind];m['pairs']+=1
        delta=neural[b]-neural[a];symbolic_delta=symbolic[b]-symbolic[a]
        chosen=int(np.argmax(np.abs(delta)));nontrivial=bool(abs(delta[chosen])>DIRECTION_EPSILON)
        correct=bool(np.sign(delta[chosen])==np.sign(symbolic_delta[chosen])) if nontrivial else None
        if nontrivial:
            m['nontrivial_pairs']+=1;m['direction_correct']+=int(correct)
        else:m['uninformative_pairs']+=1
        changed=bool(neural[a].argmax()!=neural[b].argmax())
        if changed:m['changed_action_pairs']+=1;m['changed_action_correct']+=int(agreement[a] and agreement[b])
        evidence.append({'pair':pair,'kind':kind,'seed':rows[a]['seed'],'origin_frame':rows[a].get('origin_frame',rows[a]['frame']),
            'assumption':rows[a].get('assumption',{}),'baseline':{'neural':dict(zip(ACTIONS,map(float,neural[a]))),'program':dict(zip(ACTIONS,map(float,symbolic[a]))),'frame':rows[a]['frame']},
            'altered':{'neural':dict(zip(ACTIONS,map(float,neural[b]))),'program':dict(zip(ACTIONS,map(float,symbolic[b]))),'frame':rows[b]['frame']},
            'largest_change_action':ACTIONS[chosen],'actor_probability_change':float(delta[chosen]),'program_probability_change':float(symbolic_delta[chosen]),
            'nontrivial':nontrivial,'direction_correct':correct,'actor_action_changed':changed})
    for value in metrics.values():
        value['direction_fidelity']=value['direction_correct']/value['nontrivial_pairs'] if value['nontrivial_pairs'] else None
        value['changed_action_fidelity']=value['changed_action_correct']/value['changed_action_pairs'] if value['changed_action_pairs'] else None
    passes=overall is not None and overall>=.90
    passes=passes and all(value['n']>=minimum_critical_samples and value['fidelity'] is not None and value['fidelity']>=.85 for value in critical.values())
    passes=passes and all(value['nontrivial_pairs']>0 and value['direction_fidelity']>=.85 for value in metrics.values())
    return {'ordinary_samples':int(ordinary.sum()),'overall_fidelity':overall,'all_state_fidelity':float(agreement.mean()),
        'critical_states':critical,'counterfactual_by_mechanism':metrics,'passes':bool(passes),
        'depth':program.root.depth(),'leaves':program.root.leaf_count(),'predicates':len(program.root.used_predicates())},evidence


def extract(actor_path,output,fit_maps=60,selection_maps=30,test_maps=30,*,excluded_fingerprints=(),candidate_grid=None):
    policy=NumpyKitchenPolicy(actor_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    excluded=reserved_fingerprints(policy,excluded_fingerprints)
    descriptors={};rejected={}
    for split,count in (('extraction_fit',fit_maps),('extraction_selection',selection_maps),('extraction_test',test_maps)):
        rows,removed=unique_scenarios(split,count,excluded)
        descriptors[split]=rows;rejected[split]=removed;excluded.update(r['fingerprint'] for r in rows)
    train=collect(policy,descriptors['extraction_fit']);selection=collect(policy,descriptors['extraction_selection'])
    candidates=[]
    for depth,leaves in candidate_grid or [(d,l) for d in (4,6,8) for l in (16,32,64)]:
        config=RCPDConfig(regularization_lambda=0,max_depth=depth,max_leaf_nodes=leaves,max_predicates=len(OBSERVATION_FEATURES),
            min_samples_leaf=4,random_seed=2026,action_structure_weight=1.0)
        fitted=RCPD(config).fit(train,lambda r:r['probabilities'],lambda r:r['features'],validation_states=selection,
            group_provider=lambda r:r['groups'],counterfactual_pair_provider=lambda r:r['pair'],split_group_provider=lambda r:r['scene'],
            interaction_groups=CRITICAL_GROUPS,
            program_metadata={'domain':'cooperative_kitchen','actor_id':'ai','checkpoint_sha256':policy.checkpoint_id,
                'actor_sha256':policy.artifact_sha256,'feedback_enabled':False,'regularization_version':False,'label_source':'actual_frozen_actor_probabilities'})
        metrics,_=audit(fitted.program,selection)
        metrics.update(requested_depth=depth,requested_leaves=leaves)
        candidates.append((fitted.program,metrics))
        print(json.dumps({'candidate':metrics}),flush=True)
    passing=[c for c in candidates if c[1]['passes']]
    if passing:selected=min(passing,key=lambda c:(c[1]['leaves'],c[1]['depth'],c[1]['predicates']))
    else:selected=max(candidates,key=lambda c:(c[1]['overall_fidelity'] or 0,-c[1]['leaves'],-c[1]['depth']))
    # The serialized model is frozen before any audit rollout is collected.
    program_path=output/'program_ai.json'
    atomic_json(program_path,posthoc_program_payload(selected[0]))
    test=collect(policy,descriptors['extraction_test'])
    audited,evidence=audit(selected[0],test)
    # Only audit metadata changes after freezing; tree/predicates/probabilities
    # remain exactly the program chosen before collecting the held-out audit.
    atomic_json(program_path,posthoc_program_payload(selected[0],audited,selection_passes=selected[1]['passes']))
    with (output/'extraction_interventions.jsonl').open('w') as handle:
        for record in evidence:handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    report={'schema':'cooperative_kitchen_extraction_audit_v1','method':'posthoc_rcpd','feedback_enabled':False,'status':'candidate',
        'actor_sha256':policy.artifact_sha256,'checkpoint_sha256':policy.checkpoint_id,
        'program':'program_ai.json','program_sha256':hashlib.sha256(program_path.read_bytes()).hexdigest(),
        'selection':selected[1],'audit':audited,'candidates':[c[1] for c in candidates],
        'extraction_gate':bool(selected[1]['passes'] and audited['passes']),'explanation_gate':False,
        'thresholds':{'overall':.90,'critical':.85,'counterfactual_direction':.85,'nontrivial_epsilon':DIRECTION_EPSILON},
        'selection_protocol':'simplest passing tree on selection; if none passes save highest-fidelity diagnostic; no reselection after untouched audit',
        'split_scenarios':descriptors,'rejected_duplicate_seeds':rejected,
        'intervention_evidence':'extraction_interventions.jsonl','llm_audit':'separate mandatory gate; not performed by extraction'}
    atomic_json(output/'extraction_report.json',report)
    atomic_json(output/'extraction_split_ledger.json',{'splits':descriptors,'excluded_initial_fingerprints':sorted(excluded)})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('actor');p.add_argument('--output',required=True)
    p.add_argument('--fit-maps',type=int,default=60);p.add_argument('--selection-maps',type=int,default=30);p.add_argument('--test-maps',type=int,default=30)
    a=p.parse_args();r=extract(a.actor,a.output,a.fit_maps,a.selection_maps,a.test_maps)
    print(json.dumps({'extraction_gate':r['extraction_gate'],'overall_fidelity':r['audit']['overall_fidelity']}))

if __name__=='__main__':main()
