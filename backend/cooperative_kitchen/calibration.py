"""Development-only scenario pairing and frozen, private prediction questions.

Pair selection uses only deterministic program baselines, never group outcomes.
The actor is queried solely for questionnaire ground truth. All initial-state
fingerprints exclude arbitrary seeds/IDs so duplicate scenes cannot cross splits.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from env.cooperative_kitchen import CooperativeKitchen, ACTIONS, ACTOR_IDS, program_decision
from backend.cooperative_kitchen.splits import BASES, seeds, scenario_fingerprint
from backend.cooperative_kitchen.policy import NumpyKitchenPolicy

ACTION_LABELS = {
    'UP': {'zh':'向上','en':'Up'}, 'DOWN': {'zh':'向下','en':'Down'},
    'LEFT': {'zh':'向左','en':'Left'}, 'RIGHT': {'zh':'向右','en':'Right'},
    'INTERACT': {'zh':'交互','en':'Interact'}, 'WAIT': {'zh':'等待','en':'Wait'},
}


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def reserved_fingerprints(policy, extra=()):
    """Reserve geometry/inventory identities used by all three optimization seeds."""
    excluded=set(extra)
    ledger=Path(__file__).resolve().parents[2]/'output/cooperative_kitchen/v1/split_ledger.json'
    if ledger.exists():
        excluded.update(json.loads(ledger.read_text()).get('excluded_initial_fingerprints',[]))
    count=int(policy.metadata.get('config',{}).get('train_scenarios',512))
    for training_seed in range(3):
        for seed in seeds('train',count,training_seed):
            excluded.add(scenario_fingerprint(CooperativeKitchen(seed=seed,scenario_id='generated').public_view()))
    for split in ('validation','final_test'):
        for seed in seeds(split, max(60,int(policy.metadata.get('config',{}).get('validation_episodes',60)))):
            excluded.add(scenario_fingerprint(CooperativeKitchen(seed=seed,scenario_id='generated').public_view()))
        for seed in range(BASES[split]+50_000,BASES[split]+50_200):
            excluded.add(scenario_fingerprint(CooperativeKitchen(seed=seed,scenario_id='generated').public_view()))
    return excluded


def unique_scenarios(split, count, excluded=(), *, start_offset=0):
    if split not in BASES or count < 1: raise ValueError('Invalid scene split')
    seen=set(excluded); rows=[]; rejected=[]
    for seed in range(BASES[split]+start_offset,BASES[split]+99_999):
        env=CooperativeKitchen(seed=seed,scenario_id='generated')
        fp=scenario_fingerprint(env.public_view())
        if fp in seen:
            rejected.append(seed);continue
        rows.append({'scenario_id':'generated','seed':seed,'fingerprint':fp})
        seen.add(fp)
        if len(rows)==count:return rows,rejected
    raise RuntimeError('Unique kitchen scenario allocation exhausted')


def initial_kind(env):
    s=env.state
    if s['actors'][1]['holding']=='soup':return 'congestion'
    if s['pot']['ingredients']:return 'inprogress'
    return 'empty'


def baseline_episode(descriptor):
    env=CooperativeKitchen(seed=descriptor['seed'],scenario_id=descriptor['scenario_id'])
    category=initial_kind(env)
    while not env.state['done']:
        env.step({id:program_decision(env,id)['action'] for id in ACTOR_IDS},include_state=False)
    return {'initial_kind':category,'steps':env.state['turn'],'orders':env.state['orders'],'success':env.state['reason']=='success'}


def make_pairs(descriptors):
    """Choose closest baseline durations per initial category, with fixed ties."""
    evaluated=[{**d,'baseline':baseline_episode(d)} for d in descriptors]
    pairs=[]
    for kind in ('empty','inprogress','congestion'):
        candidates=[d for d in evaluated if d['baseline']['initial_kind']==kind and d['baseline']['success'] and d['baseline']['steps']<=160]
        choices=[]
        for i,a in enumerate(candidates):
            for b in candidates[i+1:]:
                gap=abs(a['baseline']['steps']-b['baseline']['steps'])
                if gap<=15:choices.append((gap,max(a['baseline']['steps'],b['baseline']['steps']),a['seed'],b['seed'],a,b))
        if not choices:raise RuntimeError(f'No calibrated pair for {kind}; increase development scene pool')
        *_,a,b=min(choices,key=lambda x:x[:4])
        number=len(pairs)+1
        descriptors=[]
        for letter,d in zip(('A','B'),(a,b)):
            descriptors.append({k:d[k] for k in ('scenario_id','seed','fingerprint')} | {
                'label':{'zh':f'厨房场景 {number}{letter}','en':f'Kitchen {number}{letter}'}})
        pairs.append({'id':f'pair_{number}','initial_kind':kind,'scenarios':descriptors,
                      'baseline_steps':[a['baseline']['steps'],b['baseline']['steps']],
                      'step_difference':abs(a['baseline']['steps']-b['baseline']['steps'])})
    return pairs,evaluated


def make_questionnaire(policy, descriptors):
    frames=[]
    seen=set()
    for descriptor in descriptors:
        env=CooperativeKitchen(seed=descriptor['seed'],scenario_id=descriptor['scenario_id'])
        while not env.state['done']:
            actions,dist=policy.act(env.observations())
            if env.state['turn'] % 9 == 0:
                public=env.public_view()
                key=json.dumps([public['actors'],public['pot'],public['counters']],sort_keys=True)
                if key not in seen:
                    seen.add(key);frames.append((env.fork(),descriptor,actions['ai']))
            human=program_decision(env,'human')['action']
            env.step({'human':human,'ai':actions['ai']},include_state=False)
    if len(frames)<8:raise RuntimeError('Insufficient independent questionnaire frames')
    # Prefer action diversity using only this dedicated questionnaire set.
    next_frames=[]
    for action in ACTIONS:
        candidate=next((f for f in frames if f[2]==action),None)
        if candidate:next_frames.append(candidate)
    for frame in frames:
        if len(next_frames)>=4:break
        if all(frame[0] is not f[0] for f in next_frames):next_frames.append(frame)
    next_frames=next_frames[:4]
    selected_ids={id(f[0]) for f in next_frames}
    cf_frames=[f for f in frames if id(f[0]) not in selected_ids and f[0].state['turn']<=176]
    if len(cf_frames)<4:raise RuntimeError('Insufficient three-step questionnaire frames')
    items=[]
    for env,descriptor,action in next_frames:
        items.append({'id':f'next_{len(items)+1}','kind':'next_action','type':'prediction','frame':env.state['turn'],
            'prompt':{'zh':'在这个画面下，队友下一步会选择什么动作？','en':'In this frame, which action will the teammate choose next?'},
            'state':env.public_view(),'options':[{'value':a,'label':ACTION_LABELS[a]} for a in ACTIONS],
            'correct_answer':action,'assumption':{'zh':'队友根据所示状态决策，尚未读取你的下一步输入。','en':'The teammate decides from the displayed state, before your next input.'},
            'source_seed':descriptor['seed'],'scenario_fingerprint':descriptor['fingerprint'],'actor_sha256':policy.artifact_sha256})
    for env,descriptor,_ in cf_frames:
        branch=env.fork();trajectory=[]
        for _ in range(3):
            if branch.state['done']:break
            actions,_=policy.act(branch.observations())
            trajectory.append(actions['ai'])
            branch.step({'human':'WAIT','ai':actions['ai']},include_state=False)
        if len(trajectory)!=3:continue
        items.append({'id':f'counterfactual_{len(items)-3}','kind':'counterfactual','type':'counterfactual','frame':env.state['turn'],
            'prompt':{'zh':'假设你连续等待三个联合步，队友在第三步会执行什么动作？','en':'If you wait for three joint steps, which action will the teammate execute on the third step?'},
            'state':env.public_view(),'options':[{'value':a,'label':ACTION_LABELS[a]} for a in ACTIONS],
            'correct_answer':trajectory[-1],'assumption':{'zh':'你连续等待三步；队友持续使用同一个固定神经策略。','en':'You wait for three steps; the teammate continues using the same frozen neural policy.'},
            'private_trajectory':trajectory,'source_seed':descriptor['seed'],'scenario_fingerprint':descriptor['fingerprint'],
            'actor_sha256':policy.artifact_sha256})
        if len(items)==8:break
    if len(items)!=8:raise RuntimeError('Insufficient full three-step questionnaire outcomes')
    return {'schema':'cooperative_kitchen_questionnaire_v1','status':'candidate','actor_sha256':policy.artifact_sha256,
            'items':items,'counts':{'next_action':4,'counterfactual':4},
            'answer_action_diversity':len(set(i['correct_answer'] for i in items)),
            'scales':[
                {'id':'cooperation_understanding','prompt':{'zh':'我理解如何与队友配合完成任务。','en':'I understand how to coordinate with the teammate.'}},
                {'id':'predictability','prompt':{'zh':'我能预测队友接下来会做什么。','en':'I can predict what the teammate will do next.'}},
                {'id':'difficulty','prompt':{'zh':'完成厨房任务对我来说很困难。','en':'I found the kitchen task difficult.'}},
            ],'scale_range':[1,7],'scale_anchors':{'zh':['非常不同意','非常同意'],'en':['Strongly disagree','Strongly agree']}}


def build_calibration(policy, output, *, candidate_count=90, questionnaire_maps=12, excluded_fingerprints=()):
    output=Path(output)
    excluded=reserved_fingerprints(policy,excluded_fingerprints)
    extraction_ledger=output/'extraction_split_ledger.json'
    if extraction_ledger.exists():
        excluded.update(json.loads(extraction_ledger.read_text()).get('excluded_initial_fingerprints',[]))
    descriptors,rejected=unique_scenarios('calibration',candidate_count,excluded)
    pairs,evaluated=make_pairs(descriptors)
    excluded.update(d['fingerprint'] for d in descriptors)
    questions,rejected_questions=unique_scenarios('questionnaire',questionnaire_maps,excluded)
    questionnaire=make_questionnaire(policy,questions)
    calibration={'schema':'cooperative_kitchen_calibration_v1','status':'candidate','pairs':pairs,
        'selection_basis':'program_partner_completion_and_step_matching_only','thresholds':{'maximum_steps':160,'maximum_pair_difference':15},
        'candidates':evaluated,'split':'calibration','rejected_duplicate_seeds':rejected,'calibration_gate':True}
    atomic_json(output/'calibration.json',calibration)
    atomic_json(output/'questionnaire.json',questionnaire)
    ledger={'calibration':descriptors,'questionnaire':questions,'excluded_initial_fingerprints':sorted(excluded),
        'rejected_questionnaire_duplicates':rejected_questions,'actor_sha256':policy.artifact_sha256}
    atomic_json(output/'calibration_split_ledger.json',ledger)
    return calibration,questionnaire


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('actor');p.add_argument('--output',required=True)
    p.add_argument('--candidate-count',type=int,default=90);p.add_argument('--questionnaire-maps',type=int,default=12)
    a=p.parse_args();calibration,questions=build_calibration(NumpyKitchenPolicy(a.actor),a.output,candidate_count=a.candidate_count,questionnaire_maps=a.questionnaire_maps)
    print(json.dumps({'pairs':len(calibration['pairs']),'questions':len(questions['items'])}))

if __name__=='__main__':main()
