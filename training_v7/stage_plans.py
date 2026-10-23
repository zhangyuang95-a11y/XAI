"""Finite, predeclared scenes and random streams for the later common stages."""
from .contracts import BC_STAGES,CANDIDATE,DOMAINS,MethodSpec,digest


def evaluation_cases(name,phase,pair,budget,purpose,base,count,*,policies,pairings,trajectories=1,checkpoint='final'):
    cases=[]
    for i in range(count):
        for trajectory in range(trajectories):
            for policy in policies:
                for pairing in pairings:
                    cases.append(dict(case_id=f'{name}:{phase}:{pair}:{budget}:{purpose}:{checkpoint}:{i}:{trajectory}:{policy}:{pairing}',
                        seed=base+i,task=1+i%3,trajectory=trajectory,
                        action_seed=1_000_000_000_000+(base+i)*4+trajectory,
                        purpose=purpose,policy=policy,pairing=pairing,checkpoint=checkpoint))
    return cases


def pair_plans(spec,*,phase,pair,budgets,has_proxy):
    method=MethodSpec()
    if phase not in ('development','formal') or pair not in (0,1,2):raise ValueError('unknown registered phase/pair')
    if (phase=='development' and tuple(budgets)!=method.budgets
            or phase=='formal' and (len(budgets)!=1 or budgets[0] not in method.budgets)):
        raise ValueError('all common development budgets or one locked formal budget required')
    di=DOMAINS.index(spec.name);phase_base=40_000_000_000 if phase=='development' else 50_000_000_000
    base=phase_base+di*1_000_000_000+pair*10_000_000
    init=(10_100_000_000 if phase=='development' else 10_200_000_000)+di*1_000_000+pair*1000
    header=dict(version='three-domain-unified-finite-stage-plan.v1',candidate_id=CANDIDATE,
        phase=phase,domain=spec.name,domain_sha256=spec.sha256,method_sha256=method.sha256,pair_id=pair,
        initialization_seed=init,critic_seed=init+60,optimizer_seed=init+50,bc_action_seed=init+25,
        action_seed=init+40,sealed_authority=False,pilot_opened=False)
    cases=[];ordinal=0
    for stage in BC_STAGES:
        for i in range(stage[1]):
            cases.append(dict(case_id=f'{spec.name}:{phase}:{pair}:bc:{ordinal}',seed=base+ordinal,
                task=1+ordinal%3,purpose='bc',stage=stage[0]));ordinal+=1
    plans={'bc':dict(header,cases=cases,budget=0)}
    for bi,budget in enumerate(budgets):
        # BC and the PPO/action/optimizer stream start identically for each
        # budget attempt; confirmation/program and pressure scenes are fresh.
        method_budget_index=method.budgets.index(budget)
        if phase=='formal':method_budget_index=0
        cases=[];offset=base+2_000_000+method_budget_index*100_000;ordinal=0
        for partition,count in zip(('fit','calibration','eligibility','validation'),method.evidence_counts,strict=True):
            for i in range(count):
                cases.append(dict(case_id=f'{spec.name}:{phase}:{pair}:{budget}:program:{ordinal}',
                    seed=offset+ordinal,task=1+ordinal%3,purpose='program',partition=partition));ordinal+=1
        for block in range((budget+8*spec.horizon-1)//(8*spec.horizon)):
            for slot in range(8):
                ordinal=block*8+slot
                cases.append(dict(case_id=f'{spec.name}:{phase}:{pair}:{budget}:ppo:{ordinal}',
                    seed=base+1_000_000+ordinal,task=1+ordinal%3,purpose='ppo',slot=slot,block=block,
                    maximum_resets=spec.horizon))
        pairings=('nn_nn','offline_human_proxy') if has_proxy else ('nn_nn',)
        checkpoints=('84','93','final') if phase=='formal' else ('final',)
        for checkpoint in checkpoints:
            # Formal selection deliberately shares one same-domain quality set
            # across every pair and checkpoint. Program results never rank it.
            quality_base=(phase_base+di*1_000_000_000+5_000_000 if phase=='formal' else
                          base+5_000_000+method_budget_index*100_000)
            cases+=evaluation_cases(spec.name,phase,pair,budget,'quality',quality_base,75,
                policies=('raw','regularized') if phase=='formal' else ('reference','raw','regularized'),
                pairings=pairings,checkpoint=checkpoint)
        if phase=='development':
            cases+=evaluation_cases(spec.name,phase,pair,budget,'pressure',base+6_000_000+method_budget_index*100_000,72,
                policies=('reference','raw','regularized'),pairings=pairings,trajectories=3)
        plans[str(budget)]=dict(header,cases=cases,budget=budget,evidence_seed=init+30+method_budget_index*100,
            scenario_sequence_sha256=digest([c for c in cases if c['purpose']=='ppo']))
    return plans


def stage_seed_proposal(plans,*,registered_traces=()):
    families=[]
    # Merge intentional same-scene comparisons into one scenario family per
    # stage. No duplicate raw/reg/checkpoint declarations masquerade as reuse.
    scenes=set();actions=set();initial=set();critics=set()
    for plan in plans.values():
        scenes.update(c['seed'] for c in plan['cases'])
        actions.update(c['action_seed'] for c in plan['cases'] if 'action_seed' in c)
        initial.add(plan['initialization_seed']);critics.add(plan['critic_seed'])
        actions.update(plan[k] for k in ('optimizer_seed','bc_action_seed','action_seed','evidence_seed') if k in plan)
    for name,kind,values in (('scenarios','scenario',scenes),('initializations','init',initial),
                            ('critics','critic',critics),('action-streams','action_rng',actions)):
        families.append(dict(name=name,kind=kind,values=sorted(values)))
    return dict(schema='warehouse-seed-proposal.v1',candidate_id=CANDIDATE,
                families=families,registered_trace_paths=list(registered_traces))
