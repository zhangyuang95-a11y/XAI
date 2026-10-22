"""Lock a uniformly passing depth arm, then score fixed evidence on new scenes.

No collection occurs here. The inherited original comparator, scene pairing,
bootstrap endpoints and absolute gates retain their historical definitions.
"""
from dataclasses import dataclass
from copy import deepcopy
from . import v642_protocol as v,v629_confirmation_metrics as arithmetic
from backend.training.warehouse_v642_program import Program
from backend.training.warehouse_v629_program import Program as ReferenceProgram
from scripts.audit_warehouse_v642_program import forward as deep_forward
from scripts.run_warehouse_v632 import strict

ROLES=arithmetic.ROLES
BASELINE='fixed_corresponding_v628_search_prefix_outer_qualified_in_v629_G128'


@dataclass(frozen=True)
class Bundle:
    units:tuple
    selected_method:str
    selected_depth:int
    actor:dict
    actor_role_sha256:dict
    provenance:dict
    asset_refs:tuple

    def verify(self):
        v.require(self.selected_depth in (9,12) and self.selected_method==f'depth-{self.selected_depth}' and len(self.units)==6 and
            {(u.role,u.fold) for u in self.units}=={(r,f) for r in ROLES for f in range(3)},'need one uniformly passing depth arm')
        for ref in (self.actor,*self.provenance.values(),*self.asset_refs):v.require(v.pointer(ref['path'])==ref,'fixed confirmation asset changed')
        for u in self.units:
            v.require(isinstance(u.program,Program) and isinstance(u.reference,ReferenceProgram) and
                u.program.role==u.reference.role==u.role and u.program.actor_role_sha256==u.reference.actor_role_sha256==self.actor_role_sha256[u.role] and
                u.program.target_actor_role_sha256==self.actor_role_sha256[u.role] and u.program.maximum_depth==self.selected_depth and
                u.program.maximum_leaves==256 and u.reference.maximum_leaves==128,'fixed role, Actor or capacity differs')
            v.require(v.pointer(u.program_ref['path'])==u.program_ref and v.pointer(u.reference_ref['path'])==u.reference_ref and
                v.digest(u.program.to_dict())==u.program_digest and v.digest(u.reference.to_dict())==u.reference_digest,'attempted program/qualification mutation')
        return self


def load_locked_bundle():
    # Reject partial development before source traversal, seeds or environments.
    for name in ('result.json','independent-replay.json','supervisor-exit.json','method-lock.json'):
        v.require((v.OUTPUT/name).is_file(),'v642 has no complete reaped and independently audited winner; confirmation remains closed')
    path,plan=v.checked_plan();refs={k:v.pointer(v.OUTPUT/name) for k,name in
        (('plan','plan.json'),('summary','result.json'),('independent_audit','independent-replay.json'),('execution','supervisor-exit.json'),('lock','method-lock.json'))}
    summary=v.bound(refs['summary']);audit=v.bound(refs['independent_audit']);execution=v.bound(refs['execution'])
    audit_plan_ref=v.pointer(v.OUTPUT/'audit-plan.json');ap=v.bound(audit_plan_ref)
    v.require(audit['schema']=='warehouse-v642-depth-matched-independent-replay.v1' and audit['audit_plan']==audit_plan_ref and
        ap['plan']==refs['plan'] and ap['summary']==refs['summary'] and ap['source']==v.pointer(v.ROOT/'scripts/audit_warehouse_v642.py') and
        ap['optimizer_refits']==0,'independent replay plan/source/summary binding differs')
    refs['audit_plan']=audit_plan_ref
    v.require(summary['plan']==refs['plan'] and summary['all_twelve_settled'] and len(summary['units'])==12 and
        audit['integrity_passed'] is True and len(audit['units'])==12 and all(r['integrity_passed'] for r in audit['units']),
        'partial or unaudited depth comparison cannot confirm')
    v.require(execution['schema']=='warehouse-v642-supervisor-terminal.v1' and execution['plan']==refs['plan'] and
        execution['summary']==refs['summary'] and execution['returncode']==0 and execution['observed_terminal_exit'] is True and
        execution['all_twelve_settled'] is True,'original driver has not been observed exiting successfully')
    checked={r['input']['path']:r for r in audit['units']};results=[];assets=[]
    for ref in summary['units']:
        r=v.bound(ref);spec=r['spec'];a=checked.get(ref['path'])
        v.require(r['status']=='completed' and r['plan']==refs['plan'] and spec in plan['jobs'] and a is not None and a['input']==ref and a['job']==spec['id'],'unit identity or independent replay differs')
        report=v.bound(r['search_report']);chosen=report['candidates'][report['selected']];counts=chosen['reliable_counts'];n=report['selection_frames']
        reliable=counts['E']/n>=.45 and counts['F']/n>=.45 and counts['expected_F']/n>=.40
        v.require(r['numerical_passed']==strict(r['metrics']['validation']) and report['reliable_selection_passed']==r['reliable_fit_only_passed']==reliable and
            r['development_passed']==(r['numerical_passed'] and reliable),'actual all-gate verdict differs')
        assets.extend((ref,r['program'],r['search_report'],spec['source_result']));results.append(r)
    passing=[depth for depth in (9,12) if all(r['development_passed'] for r in results if r['spec']['maximum_depth']==depth)]
    v.require(passing and summary['scientific_development_passed'] is True,'no uniform all-six winner; do not mix rolewise capacity results')
    for d in (9,12):
        rows=[r for r in results if r['spec']['maximum_depth']==d]
        expected=dict(all_six_passed=all(r['development_passed'] for r in rows),development_pass_count=sum(r['development_passed'] for r in rows),numerical_pass_count=sum(r['numerical_passed'] for r in rows))
        v.require(summary['arms'][str(d)]==expected,'arm aggregate differs from all six measured units')
    depth=passing[0]
    expected_lock=dict(schema='warehouse-v642-method-lock.v1',plan=refs['plan'],summary=refs['summary'],independent_audit=refs['independent_audit'],supervisor_exit=refs['execution'],
        selected_depth=depth,selection_rule='first_uniform_all_six_pass_depth9_then_depth12_no_rolewise_mixing',
        programs=[r['program'] for r in results if r['spec']['maximum_depth']==depth],scientific_development_precheck_passed=True,
        fresh_confirmation_required=True,training_authority=False,sealed_authority=False,model_acceptance_passed=False,pilot_opened=False)
    v.require(v.bound(refs['lock'])==expected_lock,'method lock differs from first all-six passing capacity')
    _,old=v.previous.checked_plan();quality=v.previous.common_quality(old)
    assets.extend((old['common_quality'],old['collection']))
    v.require(all(m['deliveries']>=5.5 and m['collisions']<=6 and m['shutdown_rate']<=.05 for m in quality.values()),'frozen NN historical development quality failed')
    old_plan=v.bound(old['previous_plan']);old_summary=v.bound(v.pointer(v.previous.previous.OUTPUT/'summary.json'));old_audit=v.bound(old_summary['independent_audit'])
    units=[]
    for r in results:
        spec=r['spec']
        if spec['maximum_depth']!=depth:continue
        role,fold=spec['role'],spec['fold'];folder=v.previous.previous.OUTPUT/'jobs'/f'G128-fold-{fold}-{role}'
        oldref=v.pointer(folder/'result.json');baseline=v.bound(oldref);seal=v.pointer(folder/'completed.json')
        original_job=next(j for j in old_plan['jobs'] if j['job_id']==folder.name);checked_old=next(a for a in old_audit['jobs'] if a['job']==original_job)
        v.require(v.bound(seal)==dict(result_sha256=oldref['sha256']) and baseline['job']==original_job and baseline['status']==checked_old['status']=='completed' and
            checked_old['independent_reference_metrics'] is True and baseline['plan_sha256']==old['previous_plan']['sha256'],'original comparator lacks audited provenance')
        program=Program.from_dict(v.bound(r['program']),plan['actor_role_sha256'][role]);reference=ReferenceProgram.from_dict(v.bound(baseline['reference']),plan['actor_role_sha256'][role])
        units.append(arithmetic.Unit(role,fold,program,reference,r['program'],baseline['reference'],v.digest(program.to_dict()),v.digest(reference.to_dict())))
        assets.extend((oldref,seal,baseline['reference'],program.protection['parent'],reference.protection['parent']))
    refs.update(v629_plan=old['previous_plan'],v629_summary=v.pointer(v.previous.previous.OUTPUT/'summary.json'),v629_independent_audit=old_summary['independent_audit'])
    return Bundle(tuple(units),f'depth-{depth}',depth,plan['actor'],deepcopy(plan['actor_role_sha256']),refs,tuple(assets)).verify()


def forward(program,x,m):
    if isinstance(program,Program):return deep_forward(program,x,m)
    return arithmetic.forward(program,x,m)


# Pure arithmetic is reused by explicit names; no old module globals change.
validate_cases=arithmetic.validate_cases
require=v.require
np=arithmetic.np
SPLITS=arithmetic.SPLITS
measure=arithmetic.measure
same_metrics=arithmetic.same_metrics
basic=arithmetic.basic
gate_metrics=arithmetic.gate_metrics
episode_endpoints=arithmetic.episode_endpoints
quality=arithmetic.quality
paired_bootstrap=arithmetic.paired_bootstrap


def evaluate_fixed(bundle, datasets, cases, outcomes):
    """No fit/calibrate/qualify API is imported or called in this stage."""
    require(isinstance(bundle,Bundle),'v642 confirmation requires its own locked depth-aware bundle')
    bundle.verify();validate_cases(cases)
    require(set(datasets)==set(SPLITS) and all(set(a)==set(ROLES) for a in datasets.values()), 'both roles in both fresh splits required')
    for split in SPLITS:
        require(datasets[split]['robot_1']['rows']==datasets[split]['robot_2']['rows'], 'joint physical frames differ between roles')
    reports=[];paired=[];byid={c['case_id']:c for c in cases}
    for unit in bundle.units:
        item=dict(role=unit.role,fold=unit.fold,program=unit.program_ref,reference=unit.reference_ref,metrics={},gates={})
        for split in SPLITS:
            a=datasets[split][unit.role]
            require(len(a['rows'])==len(a['x'])==len(a['m'])==len(a['p'])==len(a['sampled'])>0, 'all physical rows required')
            require(a['p'].shape==a['m'].shape==(len(a['x']),5) and np.isfinite(a['p']).all() and np.all((a['p']>=0)&(a['p']<=1))
                and np.allclose(a['p'].sum(1),1,atol=2e-6,rtol=0) and np.all(a['p'][a['m']==0]==0), 'invalid complete-NN target distribution')
            require(a['sampled'].shape==(len(a['x']),) and np.issubdtype(a['sampled'].dtype,np.integer)
                and np.all((a['sampled']>=0)&(a['sampled']<5))
                and np.all(a['m'][np.arange(len(a['x'])),a['sampled']]==1), 'invalid actually sampled physical action')
            ids={r['episode'] for r in a['rows']};expected={c['case_id'] for c in cases if c['split']==split}
            require(ids==expected and all(str(r['base_seed'])==str(byid[r['episode']]['base']) and r['task']==byid[r['episode']]['task']
                for r in a['rows']), 'fresh scene membership differs')
            pred=forward(unit.program,a['x'],a['m']);refpred=forward(unit.reference,a['x'],a['m'])
            for p,prediction in ((unit.program,pred),(unit.reference,refpred)):
                require(all(np.array_equal(x,y) for x,y in zip(prediction,p.distributions(a['x'],a['m']),strict=True)), 'independent frozen forward differs')
            m=measure(unit.program,a);b=measure(unit.reference,a)
            for p,prediction,metric in ((unit.program,pred,m),(unit.reference,refpred,b)):
                for purpose,selected in (('feedback',p.feedback_leaves),('explanation',p.explanation_leaves)):
                    same_metrics(basic(p,a,prediction,selected,purpose),metric[purpose])
            item['metrics'][split]=m;item['gates'][split]=gate_metrics(m,phase='development')
            item.setdefault('reference_metrics',{})[split]=b
            now,old=episode_endpoints(unit.program,a,pred),episode_endpoints(unit.reference,a,refpred)
            for case_id in sorted(expected):
                case=byid[case_id];paired.append(dict(case_id=case_id,base=case['base'],task=case['task'],
                    role=unit.role,fold=unit.fold,program_explanation=now[case_id][0],reference_explanation=old[case_id][0],
                    program_expected_feedback=now[case_id][1],reference_expected_feedback=old[case_id][1]))
        reports.append(item)
    q=quality(outcomes,cases);bootstrap={s:paired_bootstrap(paired,cases,s) for s in SPLITS}
    passed=all(g['passed'] for r in reports for g in r['gates'].values()) and all(r['passed'] for r in q.values()) and bootstrap['validation']['passed']
    bundle.verify()
    return dict(schema='warehouse-v642-fixed-confirmation-metrics.v1',selected_method=bundle.selected_method,selected_depth=bundle.selected_depth,
        primary_comparator=BASELINE,metrics_passed=bool(passed),
        scope='fixed_asset_metrics_only_requires_independent_collection_audit',full_confirmation_admission=False,
        units=reports,quality=q,bootstrap=bootstrap,primary_bootstrap_family='validation_four_endpoints',
        pressure_bootstrap='secondary_diagnostic_not_a_replacement_for_strict_pressure_gates',
        fixed_assets=bundle.provenance,all_qualification_and_temperatures_unchanged=True,
        scoring_environment_resets=0,fit_calls=0,LP_calls=0,actor_updates=0,PPO_joint_steps=0,
        sealed_audit_used=False,heldout_used=False,pilot_opened=False,publishable=False)
