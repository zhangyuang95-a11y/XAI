"""New-candidate final-program and matched-quality audit, never collection.

No reset, random NN initialization, optimizer or old 20k pass is authorized.
Only an actual new finite owner's completed pair and exact 75-scene inventory
can reach the collection auditor. The inherited task/fidelity arithmetic is
unchanged; its passing result is a calibration precheck, not model admission.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import re

import numpy as np

from . import v643_program_namespace as ns
from . import v643_program_provider as programs
from . import v643_calibration as numerical
from . import v642_calibration_quality as arithmetic
from . import v642_fresh_program_metrics as fresh
from .v642_update_receipt_audit import audit_updates
from .v643_risk_recipe import CANDIDATE, REFERENCE_SHA256
from .v626_protocol import require, pointer, bound, digest, file_sha
from backend.training import warehouse_v629_environment_replay as physics
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor, MODEL_VERSION, _tensor_names, role_sha256
from backend.training.warehouse_v6_actor import encode_actor_tree_features
from backend.training.warehouse_v622_program import masked_probabilities
from backend.training.warehouse_v642_program import Program, measure
from backend.training.warehouse_v642_training_program import complete_program, language_lock, limits
from backend.training.warehouse_v626_budget import Budget
from env.warehouse.decision_protocol import independent_agent_seed
from env.warehouse.domain import participant_study_config
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_action_features_v617 import ACTION_PHYSICS_CONFIG
from env.warehouse.public_observation_v6 import encode_public_observation, public_pre_action_snapshot
from env.warehouse.pure_nn import AGENTS, action_masks
from scripts.audit_warehouse_v642_program import forward
from scripts.audit_warehouse_v628 import ledger
from scripts.measure_warehouse_v628_collection import episode as public_episode

VERSION = 'warehouse-v643-matched-calibration-quality-audit.v1'
ASSET_VERSION = 'warehouse-v643-completed-calibration-assets.v1'
CHECKPOINT_VERSION = 'warehouse-v643-completed-checkpoint-wrapper.v1'
COLLECTION_VERSION = 'warehouse-v643-matched-quality-source.v1'
EPISODE_VERSION = 'warehouse-v643-quality-episode.v1'
FINAL_PROGRAM_VERSION = 'warehouse-v643-final-NN-programs.v1'
LEDGER_VERSION = 'warehouse-v643-owner-ledger-export.v1'
QUALITY_INVENTORY_VERSION = 'warehouse-v643-quality-case-inventory.v1'
VARIANTS = ('reference','raw','regularized')
ROLES = ('robot_1','robot_2')


def source_files():
    # Does not import the new seed-plan/owner source helper: that helper merges
    # this one into its closure, so doing so would create a closure cycle.
    files=programs.source_files()
    for name in ('experiments/warehouse/v643_calibration_quality.py',
                 'experiments/warehouse/v642_calibration_quality.py',
                 'experiments/warehouse/v642_fresh_program_metrics.py',
                 'experiments/warehouse/v642_update_receipt_audit.py',
                 'backend/training/warehouse_v642_checkpoint_export.py',
                 'scripts/measure_warehouse_v628_collection.py',
                 'tests/test_warehouse_v643_calibration_quality.py',
                 'docs/warehouse_v643_quality_bridge.md'):
        files[name]=file_sha(ns.ROOT/name)
    return dict(sorted(files.items()))


def recipe():
    return dict(arithmetic.recipe(),version=VERSION,candidate_id=CANDIDATE,
        original_quality_algebra=arithmetic.VERSION,final_programs_fixed_before_quality=True,
        old_20k_pass_required=False,quality_inventory_version=QUALITY_INVENTORY_VERSION,
        program_gate='each_variant_role_coverage45_expected40_agreement95_97_KL020_illegal_unproved_zero')


def validate_cases(cases):
    arithmetic.validate_cases(cases)
    require(all(c['case_id'].startswith(CANDIDATE+'-')
        and re.fullmatch(r'[A-Za-z0-9_.-]+',c['case_id']) for c in cases),
        'quality cases need the new candidate namespace')


def evaluate_outcomes(outcomes, cases, actors):
    """Pure inherited arithmetic; this alone cannot qualify an experiment."""
    validate_cases(cases);value=arithmetic.evaluate(outcomes,cases,actors)
    for variant in ('raw','regularized'):
        for task in ('1','2','3'):
            current=value['by_task'][variant][task];reference=value['by_task']['reference'][task]
            current['delivery_reference_fraction']=None if reference['deliveries']==0 else current['deliveries']/reference['deliveries']
            current['delivery_difference_from_reference']=current['deliveries']-reference['deliveries']
            current['collision_difference_from_reference']=current['collisions']-reference['collisions']
            current['shutdown_rate_difference_from_reference']=current['shutdown_rate']-reference['shutdown_rate']
    value.update(version=VERSION,candidate_id=CANDIDATE,recipe=recipe(),
        scope='pure_arithmetic_requires_new_owner_and_complete_saved_physics_NN_replay',
        independently_audited_collection=False,model_acceptance_passed=False,pilot_opened=False)
    return value


def _require_plan(plan):
    from .v643_training_seed_plan import SeedPlan
    require(type(plan) is SeedPlan, 'new source-closed finite v643 SeedPlan required')
    plan.verify();return plan


def _same(a,b):
    import torch
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and np.array_equal(a,b)
    if type(a) is dict or type(b) is dict:
        return type(a) is type(b) is dict and set(a)==set(b) and all(_same(a[k],b[k]) for k in a)
    if type(a) in (list,tuple) or type(b) in (list,tuple):
        return type(a) is type(b) and len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def checkpoint_role_hashes(numerical_value):
    """Same tensor-to-frozen arithmetic as v642 export, with no model/RNG."""
    state=numerical_value['model'];weights={}
    for role in ROLES:
        prefix='actors.'+role+'.'
        fixed=state[prefix+'tree.support_fixed'].detach().cpu().numpy()
        mask=state[prefix+'tree.support_mask'].detach().cpu().numpy()
        require(fixed.shape==(1,) and fixed[0]==1 and np.isin(mask,(0,1)).all(),
                'final checkpoint lost original fixed tree supports')
        values={name:state[prefix+name].detach().cpu().numpy().astype(np.float32,copy=True)
            for name in _tensor_names(MODEL_VERSION)}
        w=values['tree.split.weight'];require(mask.shape==w.shape,'final support mask shape differs')
        values['tree.split.weight']=np.add(np.multiply(w,mask.astype(np.float32),dtype=np.float32),np.float32(0.),dtype=np.float32)
        require(all(np.isfinite(v).all() for v in values.values()),'nonfinite final Actor checkpoint tensor')
        weights[role]=role_sha256(values,model_version=MODEL_VERSION)
    return weights


def _update_chain(refs, reference_roles, expected_roles):
    current=deepcopy(reference_roles)
    for ref in refs:
        row=ns.immutable(ref)
        require(row.get('version')==numerical.VERSION and row.get('separate_reward_and_cost_optimizers') is True,
                'old numerical update cannot stand in for the new risk method')
        for epoch in row['epochs']:
            for update in epoch['updates']:
                if update['actor_updated']:
                    require(update['role_sha256_before']==current,'final update role-hash chain is disconnected')
                    current=deepcopy(update['role_sha256_after'])
    require(current==expected_roles,'final frozen Actor differs from the last accepted update')


def audit_completed_assets(plan, ref):
    """Recount the actual completed new pair; no historical pass can replace it."""
    from . import v643_calibration_recovery as recovery
    _require_plan(plan);value=ns.immutable(ref)
    require(value.get('version')==ASSET_VERSION and value.get('candidate_id')==CANDIDATE
        and value.get('plan')==plan.ref and value.get('training_completed') is True
        and value.get('model_acceptance_passed') is value.get('pilot_opened') is False
        and value.get('source_files')==plan.value['source_files']
        and value.get('source_sha256')==plan.value['source_sha256']
        and all(value['source_files'].get(k)==v for k,v in source_files().items()),
        'quality requires the actual new source-bound 20k pair, not an old passed marker')
    require(set(value['actors'])==set(value['actor_role_sha256'])==set(VARIANTS)
        and set(value['owners'])==set(value['checkpoints'])==set(value['updates'])=={'raw','regularized'},
        'completed training must bind both variants and all three frozen comparisons')
    claim=ns.immutable(value['claim']);descriptor=ns.immutable(value['descriptor'])
    require(claim.get('schema')==ns.CLAIM_VERSION and claim.get('candidate_id')==CANDIDATE
        and claim.get('plan')==plan.ref and claim.get('descriptor')==value['descriptor']
        and descriptor.get('plan')==plan.ref and descriptor.get('candidate_id')==CANDIDATE,
        'completed pair uses an old or differently owned claim')
    reference_ref=plan.partition.value['reference_actor']
    require(value['actors']['reference']==reference_ref and reference_ref['sha256']==REFERENCE_SHA256,
            'matched reference is not the original complete v620 BC Actor')
    ns.immutable(reference_ref,decode=False)
    reference=FrozenWarehouseV6Actor.load(reference_ref['path'],expected_sha256=reference_ref['sha256'],
        expected_role_sha256=value['actor_role_sha256']['reference'])
    require(reference.model_version==MODEL_VERSION,'reference forward architecture changed')
    inputs=[ref,value['claim'],value['descriptor'],reference_ref];namespaces={};recounts={};boundaries={}
    for variant in ('raw','regularized'):
        namespace=ns.ProgramNamespace.from_owner(value['owners'][variant]);namespaces[variant]=namespace
        require(namespace.variant==variant and namespace.value['plan']==plan.ref
            and namespace.value['claim']==value['claim'],'completed variant owner differs')
        wrapper=ns.immutable(value['checkpoints'][variant])
        require(wrapper.get('version')==CHECKPOINT_VERSION and wrapper.get('candidate_id')==CANDIDATE
            and wrapper.get('owner')==namespace.owner_ref and wrapper.get('claim')==value['claim']
            and wrapper.get('updates')==value['updates'][variant] and wrapper.get('core_snapshot_is_projection') is True
            and wrapper['core'].get('version')==numerical.VERSION
            and wrapper['core'].get('risk_training_state_in_training_state') is True,
            'final checkpoint projection has another source, owner or incomplete core')
        numerical_ref=wrapper['training_state'];ns.immutable(numerical_ref,decode=False)
        numeric=numerical.read_boundary(numerical_ref['path'],numerical_ref['sha256'])
        full=ns.immutable(wrapper['full_owner_boundary'])
        recovery.validate_boundary(full,plan=plan,numerical_value=numeric)
        require(full['numerical_boundary']==numerical_ref and full['updates']==value['updates'][variant]
            and numeric['joint_steps']==numeric['core']['PPO']['processed_joint_steps']==20000
            and _same(wrapper['core']['PPO'],numeric['core']['PPO']),
            'quality needs the exact completed 20k numerical/reader/owner boundary')
        roles=checkpoint_role_hashes(numeric)
        require(roles==numeric['core']['PPO']['actor_role_sha256']==value['actor_role_sha256'][variant],
                'final role hashes differ from actual complete checkpoint tensors')
        actor=programs._current_actor(namespace,value['actors'][variant],roles)
        require(actor.role_sha256==roles,'actual final NN export differs')
        recounts[variant]=audit_updates(value['updates'][variant],variant=variant,expected_joint_steps=20000,
            core=numeric['core']['PPO'])
        _update_chain(value['updates'][variant],reference.role_sha256,roles)
        boundaries[variant]=numeric
        inputs.extend([namespace.owner_ref,value['checkpoints'][variant],numerical_ref,wrapper['full_owner_boundary'],
            value['actors'][variant],*value['updates'][variant]])
    require(len({r['sha256'] for r in value['actors'].values()})==3
        and all(len({value['actor_role_sha256'][v][role] for v in VARIANTS})==3 for role in ROLES),
        'three differently named archives cannot replace three actual distinct role policies')
    for item in inputs:ns.immutable(item,decode=False)
    return dict(value=value,ref=ref,namespaces=namespaces,recounts=recounts,boundaries=boundaries,inputs=inputs)


def _output(path, kind):
    path=Path(path).resolve();root=(ns.OWNER_ROOT/kind).resolve()
    require(path.is_relative_to(root) and path!=root,'v643 quality/final-program outputs need their own candidate namespace')
    return path


def finalize_programs(plan, completed_assets_ref, *, namespaces, states, observed, last_program_refs, output):
    """Final 20k fixed-language projection on TRAIN-program data only."""
    fixed=audit_completed_assets(plan,completed_assets_ref)
    require(set(namespaces)==set(states)==set(observed)==set(last_program_refs)=={'raw','regularized'},
            'final projection requires both complete actual variants')
    claim=ns.OWNER_ROOT/'final-programs.claim.json';require(not claim.exists(),'final program attempt is once-only even after failure')
    output=_output(output,'final-programs');output.mkdir(parents=True,exist_ok=False)
    claim_ref=programs.write_new(claim,dict(version=FINAL_PROGRAM_VERSION,candidate_id=CANDIDATE,
        plan=plan.ref,completed_assets=completed_assets_ref,single_use_even_if_interrupted=True,training_authority=False))
    programs.write_new(output/'started.json',dict(claim=claim_ref,completed_assets=completed_assets_ref))
    results={};all_programs={};assets=[*fixed['inputs'],claim_ref];targets={};observed_audits={}
    try:
        for variant in ('raw','regularized'):
            namespace=namespaces[variant];state=states[variant];pool=observed[variant]
            require(namespace.owner_ref==fixed['value']['owners'][variant]
                and namespace.payload_sha256==fixed['namespaces'][variant].payload_sha256
                and state.joint_steps==20000 and programs.training_fingerprint(state)==fixed['boundaries'][variant]['training_state_sha256']
                and pool.source_ref==fixed['value']['program_sources'][variant], 'final program input differs from actual completed training')
            pool.verify();before=programs.training_fingerprint(state)
            target=programs.relabel_current_NN(pool,fixed['value']['actors'][variant],namespace.output_root/'final-20k-targets')
            targets[variant]=target;observed_audits[variant]=pool.audit_sha256
            _,parts,refs=programs.checked_targets(pool,target);assets.extend(refs);all_programs[variant]={};results[variant]={}
            for role in ROLES:
                encoded=ns.immutable(last_program_refs[variant][role]);program=Program.from_dict(encoded,encoded['target_actor_role_sha256'])
                template_value=ns.immutable(namespace.value['owner']['program_templates'][role])
                template=Program.from_dict(template_value,template_value['target_actor_role_sha256'])
                require(language_lock(program)==language_lock(template),'final projection changed registered topology/support/protection')
                folder=output/variant/role;folder.mkdir(parents=True)
                identity=dict(version=FINAL_PROGRAM_VERSION,candidate_id=CANDIDATE,variant=variant,role=role,
                    owner=namespace.owner_ref,completed_assets=completed_assets_ref,target_actor=fixed['value']['actors'][variant])
                budget=Budget(folder/'checkpoint.json',identity,limits(program),1200);budget.phase='final'
                try:
                    with budget.session.hard_deadline():
                        fitted,metrics=complete_program(program,parts[role],budget,refit=True,
                            target_actor_role_sha256=fixed['value']['actor_role_sha256'][variant][role])
                        report=programs.numerical.legacy.current_report(fitted,parts[role])
                        for part in parts[role].values():
                            require(all(np.array_equal(a,b) for a,b in zip(fitted.distributions(part['x'],part['m']),
                                forward(fitted,part['x'],part['m']),strict=True)),'independent final-program forward differs')
                        ref=programs.write_new(folder/'program.json',fitted.to_dict())
                        budget.checkpoint(dict(program=fitted.to_dict(),current=report,metrics=metrics),status='completed')
                    solver=budget.summary();require(ledger(folder,dict(solver=solver))==solver['counts']
                        and solver['counts']['node']==0 and all(solver['counts'][k]<=solver['limits'][k] for k in solver['counts']),
                        'final program budget ledger differs')
                    all_programs[variant][role]=ref;results[variant][role]=dict(program=ref,current=report,solver=solver,
                        target_actor_role_sha256=fitted.target_actor_role_sha256,qualification_inherited=False)
                finally:budget.session.close()
                for name in ('checkpoint.json','checkpoint.json.ledger.jsonl'):
                    path=folder/name;path.chmod(0o444);assets.append(pointer(path))
                assets.extend([ref,last_program_refs[variant][role],fitted.protection['parent']])
            require(programs.training_fingerprint(state)==before,'final program projection changed NN/critic/optimizer/RNG state')
        value=dict(version=FINAL_PROGRAM_VERSION,candidate_id=CANDIDATE,plan=plan.ref,claim=claim_ref,
            completed_assets=completed_assets_ref,actors=fixed['value']['actors'],actor_role_sha256=fixed['value']['actor_role_sha256'],
            programs=all_programs,roles=results,targets=targets,observed_sources=fixed['value']['program_sources'],
            observed_audit_sha256=observed_audits,source_files=source_files(),asset_refs=assets,
            all_training_program_gates_passed=all(r['current']['gate']['passed'] for rs in results.values() for r in rs.values()),
            quality_frames_used_for_fitting=0,NN_updates=0,environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
        ref=programs.write_new(output/'result.json',value);programs.write_new(output/'completed.json',dict(result_sha256=ref['sha256']))
        return ref
    except BaseException as exc:
        programs.write_new(output/'failed.json',dict(version=FINAL_PROGRAM_VERSION,error_type=type(exc).__name__,
            single_use_claim_retained=True,automatic_retry=False,model_acceptance_passed=False,pilot_opened=False))
        raise


def _final_programs(plan, ref, fixed, observed=None):
    value=ns.immutable(ref)
    require(value.get('version')==FINAL_PROGRAM_VERSION and value.get('candidate_id')==CANDIDATE
        and value.get('plan')==plan.ref and value.get('completed_assets')==fixed['ref']
        and value.get('actors')==fixed['value']['actors'] and value.get('actor_role_sha256')==fixed['value']['actor_role_sha256']
        and value.get('source_files')==source_files() and value.get('quality_frames_used_for_fitting')==0
        and value.get('observed_sources')==fixed['value']['program_sources']
        and value.get('NN_updates')==value.get('environment_resets')==0
        and value.get('model_acceptance_passed') is value.get('pilot_opened') is False,
        'fresh quality needs frozen final programs from this exact new completed pair')
    require(ns.immutable(pointer(Path(ref['path']).with_name('completed.json')))==dict(result_sha256=ref['sha256']),
            'final programs lack a matching completion seal')
    claim=ns.immutable(value['claim'])
    require(Path(value['claim']['path'])==ns.OWNER_ROOT/'final-programs.claim.json'
        and claim.get('version')==FINAL_PROGRAM_VERSION and claim.get('candidate_id')==CANDIDATE
        and claim.get('plan')==plan.ref and claim.get('completed_assets')==fixed['ref']
        and claim.get('single_use_even_if_interrupted') is True,'final program claim differs')
    passed=[]
    for variant in ('raw','regularized'):
        require(set(value['programs'][variant])==set(ROLES),'all four final programs required')
        namespace=fixed['namespaces'][variant]
        pool=ns.audit_observed_pool(namespace,value['observed_sources'][variant]) if observed is None else observed[variant]
        pool.verify()
        require(pool.namespace.payload_sha256==namespace.payload_sha256
            and pool.source_ref==value['observed_sources'][variant]
            and pool.audit_sha256==value['observed_audit_sha256'][variant], 'final qualification has another physical program source')
        target,parts,_=programs.checked_targets(pool,value['targets'][variant])
        require(target['target_actor']==fixed['value']['actors'][variant]
            and target['target_actor_role_sha256']==fixed['value']['actor_role_sha256'][variant],
            'final program targets differ from actual 20k NN weights')
        for role in ROLES:
            encoded=ns.immutable(value['programs'][variant][role])
            program=Program.from_dict(encoded,fixed['value']['actor_role_sha256'][variant][role])
            require(program.role==role and program.report is not None,'final program is stale or unqualified')
            report=programs.numerical.legacy.current_report(program,parts[role])
            require(report==value['roles'][variant][role]['current']
                and set(report['feedback_leaves'])==set(program.feedback_leaves)
                and set(report['explanation_leaves'])==set(program.explanation_leaves),
                'final frozen qualifications do not replay from the disjoint final-NN program evidence')
            passed.append(report['gate']['passed'])
    for asset in value['asset_refs']:ns.immutable(asset,decode=False)
    require(value.get('all_training_program_gates_passed')==all(passed) and all(passed),
            'final program precheck failed; fresh quality reset is not admitted')
    return value


def validate_final_programs(plan, completed_assets_ref, final_programs_ref, *, observed=None):
    """Call before quality resets; data verification grants no reset authority."""
    fixed=audit_completed_assets(plan,completed_assets_ref)
    return _final_programs(plan,final_programs_ref,fixed,observed)


def _replay_episode(actor, actor_ref, case, entry):
    """Restore saved state, verify every NN action/physics result; never reset."""
    require(set(entry)=={'variant','case','behavior_actor','trace','initial_environment','final_environment','reset_binding'}
        and entry['case']==case and entry['behavior_actor']==actor_ref,'quality episode/model identity differs')
    for key in ('trace','initial_environment','final_environment'):ns.immutable(entry[key],decode=False)
    trace=ns.immutable(entry['trace'])
    require(set(trace)=={'schema','case','behavior_actor','events'} and trace['schema']==EPISODE_VERSION
        and trace['case']==case and trace['behavior_actor']==actor_ref and type(trace['events']) is list
        and 0<len(trace['events'])<=120,'quality trace identity or complete physical frame list differs')
    env=physics.restore_environment(ns.immutable(entry['initial_environment']))
    require(asdict(env.config)==asdict(participant_study_config(seed=case['scenario_seed'])) and env.state.frame==0
        and env.state.episode_id==1 and not(env.state.terminated or env.state.truncated)
        and env.state.participant_controlled_agent_id=='robot_1','quality initial physical configuration or episode differs')
    arrays={k:[] for k in ('x','m','p','sampled')};public_events=[]
    for frame,event in enumerate(trace['events']):
        require(not(env.state.terminated or env.state.truncated) and env.state.frame==frame
            and public_pre_action_snapshot(env.state)==event['pre_state_public']
            and type(event['terminated']) is type(event['truncated']) is bool,'quality trace public/terminal history disconnected')
        obs=encode_public_observation(env.state);m=action_masks(env.state);x=encode_actor_tree_features(obs,m)
        prob=masked_probabilities(actor.logits(obs,m),m);indices=[]
        for i,role in enumerate(ROLES):
            seed=independent_agent_seed(base_seed=case['action_seed'],episode_id=1,frame=frame,agent_id=role)
            indices.append(int(np.random.default_rng(seed).choice(5,p=prob[i])))
        proposed={role:ACTIONS[indices[i]] for i,role in enumerate(ROLES)}
        require(event['proposed_actions']==proposed and event['action_override_agents']==[],
                'quality action differs from exact current frozen NN or was overridden')
        _,rewards,terminated,truncated,info=env.step(proposed)
        require(event['rewards']==rewards and event['terminated']==terminated and event['truncated']==truncated
            and sorted(event['shutdowns'])==sorted(info['shutdowns']) and not info['action_override_agents']
            and public_pre_action_snapshot(env.state)==event['post_state_public'],'quality rewards/outcomes/physics differ')
        public_events.append(dict(case=case,frame=frame,pre_state_public=event['pre_state_public'],
            post_state_public=event['post_state_public'],actor_sha256=actor_ref['sha256']))
        for key,v in (('x',x),('m',m),('p',prob),('sampled',indices)):arrays[key].append(v)
    require((env.state.terminated or env.state.truncated) and physics.capture_environment(env)==ns.immutable(entry['final_environment']),
            'quality episode is partial or its final environment/RNG differs')
    outcome=public_episode(case,public_events,len(public_events),actor_ref['sha256'])
    require(outcome['deliveries']==env.state.total_deliveries and outcome['robot_collisions']==env.state.robot_collision_events
        and outcome['shutdown']==bool(env.state.shutdown_count),'independent public and private quality counters differ')
    outcome.update(variant=entry['variant'],actor=actor_ref,complete=True,NN_action_checks=2*len(public_events),action_overrides=0)
    return outcome,{key:np.asarray(v,np.int64 if key=='sampled' else np.float32) for key,v in arrays.items()}


def _quality_resets(plan, collection, cases):
    from . import v643_calibration_recovery as recovery
    ledger=ns.immutable(collection['owner_ledger_snapshot'])
    require(set(ledger)=={'version','candidate_id','plan','ledger','records'} and ledger['version']==LEDGER_VERSION
        and ledger['candidate_id']==CANDIDATE and ledger['plan']==plan.ref,'quality has another owner ledger')
    recovery.validate_ledger_snapshot(ledger['ledger'],ledger['records'],plan=plan)
    ns.immutable(collection['reset_journal'],decode=False)
    hooks=[json.loads(line) for line in Path(collection['reset_journal']['path']).read_bytes().splitlines()]
    hooks=[row for row in hooks if row.get('phase')=='quality']
    actual={};pending={};order=[]
    for record in ledger['records']:
        row=record['payload']
        if row.get('scope')!='quality':continue
        if row['kind']=='environment_constructor_and_reset_predebit':pending[record['sha256']]=row
        else:
            require(row['kind']=='actual_reset_bound' and row['predebit_sha256'] in pending,'quality reset has no unique predebit')
            debit=pending.pop(row['predebit_sha256']);key=row['variant']+':'+debit['case']['case_id']
            require(key not in actual,'quality reset was duplicated')
            actual[key]=record;order.append(row['reset'])
    require(not pending and len(actual)==len(hooks)==225 and hooks==order,
            'all 225 owner predebits and physical reset hooks must agree')
    expected={variant+':'+case['case_id'] for case in cases for variant in VARIANTS}
    require(set(actual)==expected,'quality reset scope is incomplete or outside the finite inventory')
    for key,record in actual.items():
        row=record['payload'];entry=collection['cases'][key];reset=row['reset']
        require(entry['reset_binding']==record and entry['initial_environment']==row['initial_environment']
            and reset['seed']==entry['case']['scenario_seed'] and reset['episode_id']==1 and reset['phase']=='quality'
            and all(reset[k]==ACTION_PHYSICS_CONFIG[k] for k in ('map_layout_id','move_battery_cost','charge_per_wait')),
            'quality source differs from actual physical reset and owner binding')
    return ledger


def audit_collection(plan, completed_assets_ref, collection_ref, final_programs_ref, *, output, observed=None):
    """Independent read-only 225-episode replay and frozen program scoring."""
    fixed=audit_completed_assets(plan,completed_assets_ref);final=_final_programs(plan,final_programs_ref,fixed,observed)
    collection=ns.immutable(collection_ref);inventory_ref=plan.partition.value['quality_case_inventory']
    inventory=ns.immutable(inventory_ref)
    require(set(inventory)=={'schema','candidate_id','cases'} and inventory['schema']==QUALITY_INVENTORY_VERSION
        and inventory['candidate_id']==CANDIDATE,'new finite 75-scene quality inventory required')
    cases=inventory['cases'];validate_cases(cases)
    require(set(collection)=={'schema','candidate_id','plan','claim','descriptor','inventory','actors','cases','reset_journal','owner_ledger_snapshot'}
        and collection['schema']==COLLECTION_VERSION and collection['candidate_id']==CANDIDATE
        and collection['plan']==plan.ref and collection['claim']==fixed['value']['claim']
        and collection['descriptor']==fixed['value']['descriptor'] and collection['inventory']==inventory_ref
        and collection['actors']==fixed['value']['actors']
        and set(collection['cases'])=={v+':'+c['case_id'] for c in cases for v in VARIANTS},
        'quality collection belongs to another candidate, model, scene inventory or incomplete policy set')
    _quality_resets(plan,collection,cases)
    output=_output(output,'quality');output.mkdir(parents=True,exist_ok=False)
    outcomes=[];entries=[];strata={};actors={}
    for variant in VARIANTS:
        ref=fixed['value']['actors'][variant]
        actors[variant]=FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],
            expected_role_sha256=fixed['value']['actor_role_sha256'][variant])
    for case in cases:
        for variant in VARIANTS:
            entry=collection['cases'][variant+':'+case['case_id']]
            require(entry['variant']==variant,'quality case variant differs from finite key')
            outcome,arrays=_replay_episode(actors[variant],fixed['value']['actors'][variant],case,entry)
            outcomes.append(outcome);file=output/(variant+'-'+case['case_id']+'.npz')
            with file.open('xb') as handle:np.savez_compressed(handle,**arrays)
            file.chmod(0o444);entries.append(dict(case=case,variant=variant,arrays=pointer(file)))
            if variant!='reference':
                strata[(variant,case['case_id'])]=arrays
    task=evaluate_outcomes(outcomes,cases,fixed['value']['actors'])
    program=fresh.evaluate(final,fixed['value']['actor_role_sha256'],entries,cases)
    by_task={}
    for variant in ('raw','regularized'):
        by_task[variant]={}
        for role_index,role in enumerate(ROLES):
            encoded=ns.immutable(final['programs'][variant][role]);p=Program.from_dict(encoded,fixed['value']['actor_role_sha256'][variant][role])
            before=p.to_dict();by_task[variant][role]={}
            for task_id in (1,2,3):
                selected=[case for case in cases if case['task']==task_id];part={}
                for key in ('x','m','p','sampled'):
                    part[key]=np.concatenate([strata[(variant,case['case_id'])][key][:,role_index] for case in selected])
                part['rows']=[dict(episode=case['case_id'],base_seed=str(case['base']),task=task_id,frame=i)
                    for case in selected for i in range(len(strata[(variant,case['case_id'])]['x']))]
                by_task[variant][role][str(task_id)]=measure(p,part)
            require(p.to_dict()==before,'task-stratified scoring changed frozen qualification')
    _require_plan(plan)
    for ref in [*fixed['inputs'],collection_ref,final_programs_ref]:ns.immutable(ref,decode=False)
    report=dict(version=VERSION,candidate_id=CANDIDATE,plan=plan.ref,claim=fixed['value']['claim'],
        completed_assets=completed_assets_ref,collection=collection_ref,final_programs=final_programs_ref,
        source_files=source_files(),quality_metrics=task,program_metrics=program,program_by_task=by_task,
        training_update_recounts=fixed['recounts'],complete_quality_episodes=225,
        NN_action_checks=sum(r['NN_action_checks'] for r in outcomes),
        calibration_precheck_passed=task['task_retention_passed'] and program['passed'],
        independently_replayed_saved_physics_and_current_NN=True,program_fit_calls=0,qualification_changes=0,
        new_environment_resets=0,NN_updates=0,subsequent_development_and_formal_audits_required=True,
        model_acceptance_passed=False,pilot_opened=False)
    ref=programs.write_new(output/'result.json',report);programs.write_new(output/'completed.json',dict(result_sha256=ref['sha256']))
    return ref
