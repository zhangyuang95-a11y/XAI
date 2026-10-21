"""Scheduled surrogate refit for a future admitted continuation owner.

This is not a training issuer/collector. It allocates no scenes, changes no
NN weights and chooses no environment actions. Ordinary calls fail at real
A/B readiness before any fit, output or checkpoint can be created.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .warehouse_v629_program import Program,complete_program,protected_indices
from .warehouse_v629_online_evidence import load_pool,current_report,gate
from .warehouse_v628_program import payload
from .warehouse_v626_budget import Budget
from .warehouse_v6_actor import FrozenWarehouseV6Actor,MODEL_VERSION
from .warehouse_v6_actor_training import role_hashes
from .warehouse_v622_actor_training import TrainingState
from .warehouse_v625_actor_training import PPOConfig
from .warehouse_v629_rollout_replay import training_fingerprint
from experiments.warehouse import v629_training_admission as admission
from experiments.warehouse.v626_protocol import pointer,bound,file_sha,digest,require
from scripts.audit_warehouse_v628 import forward,ledger
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v629-fixed-language-current-NN-refit.v1'
ROLES=('robot_1','robot_2')
EXTRACTION_ROOT=admission.v.ROOT/'output/warehouse_v629/training-extraction'


def source_files():
    sources=admission.source_files()
    sources['backend/training/warehouse_v629_program_refit.py']=file_sha(__file__)
    return dict(sorted(sources.items()))


def limits(program):
    n=len(program.leaves)-len(protected_indices(program))
    # Each fixed-support solve has at most 8 counterexample rounds; every
    # round and each of 5 calibration temperatures has <=4 forbidden-action
    # LPs. No topology/support/node search is permitted. A zero-work program
    # retains one unused ledger slot because the existing CAS requires >0.
    return dict(leaf=max(1,8*n),node=0,proof=52*n)


def refit_pair_at_boundary(*,reference_directory,owner_ref,state,target_actor_ref,program_refs,pool_refs,
        output,variant,epoch_id,config=PPOConfig()):
    readiness=admission.load_readiness(reference_directory)  # Must be first.
    readiness.verify()
    require(isinstance(state,TrainingState) and variant in ('raw','regularized') and type(epoch_id) is str and epoch_id
        and state.joint_steps in config.extraction_steps,'refit requires a registered 75/84/93 percent joint-step boundary')
    require(set(program_refs)==set(pool_refs)==set(ROLES) and pointer(owner_ref['path'])==owner_ref,
        'refit requires a bound owner and both program/data roles')
    require(readiness.value['actor']['sha256']==state.reference_sha256,'refit is not a continuation of the confirmed complete NN')
    require(pointer(target_actor_ref['path'])==target_actor_ref,'current NN asset differs')
    actor=FrozenWarehouseV6Actor.load(target_actor_ref['path'],expected_sha256=target_actor_ref['sha256'],
        expected_role_sha256=role_hashes(state.model))
    require(actor.model_version==MODEL_VERSION,'refit cannot change the original complete NN architecture')
    start_state=training_fingerprint(state);sources=source_files();programs={};parts={}
    refs=[deepcopy(owner_ref),deepcopy(target_actor_ref)]
    for role in ROLES:
        encoded=bound(program_refs[role]);p=Program.from_dict(encoded,encoded['actor_role_sha256'])
        require(p.role==role and admission.language_lock(p)==readiness.value['program_pair'][role]['language'],
            'refit changed the B-locked topology, supports, witnesses or protected contract')
        programs[role]=p;parts[role],pool_assets=load_pool(pool_refs[role],target_actor_ref,actor,role)
        refs.extend(deepcopy([program_refs[role],p.protection['parent'],*pool_assets]))
    require(all(parts[ROLES[0]][split]['rows']==parts[ROLES[1]][split]['rows'] for split in parts[ROLES[0]]),
        'paired role evidence must use the same physical episodes and partitions')
    for ref in refs:require(pointer(ref['path'])==ref,'refit prerequisite changed before fitting')
    readiness.verify()
    require(source_files()==sources and training_fingerprint(state)==start_state,'source or NN changed while preparing refit')
    output=Path(output).resolve();base=EXTRACTION_ROOT.resolve()
    require(output.is_relative_to(base) and output!=base,'refit must use a new isolated training-extraction directory')
    identity=dict(version=VERSION,owner=owner_ref,step=state.joint_steps,config=asdict(config),variant=variant,epoch_id=epoch_id,
        current_target_actor=target_actor_ref,reference_sha256=state.reference_sha256,programs=program_refs,pools=pool_refs,
        source_sha256=digest(sources),readiness_sha256=readiness.payload_sha256)
    claim=base/'claims'/(digest(dict(owner=owner_ref,step=state.joint_steps,variant=variant))+'.json')
    require(not claim.exists(),'scheduled owner/variant/boundary extraction attempt already consumed')
    output.mkdir(parents=True,exist_ok=False);claim.parent.mkdir(parents=True,exist_ok=True)
    write_new(claim,dict(identity=identity,output=str(output),single_use_even_if_failed=True,training_authority=False))
    refs.append(pointer(claim))
    write_new(output/'started.json',dict(identity=identity,claim=pointer(claim),training_authority=False,automatic_retry=False))
    results={}
    try:
        for name,sha in sources.items():
            file=admission.v.ROOT/name;destination=output/'source'/digest(sources)/name
            raw=file.read_bytes();require(file_sha(file)==sha,'refit source changed before snapshot')
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw);destination.chmod(0o444)
        for role in ROLES:
            folder=output/role;folder.mkdir();p=programs[role];old=deepcopy(p);protected=protected_indices(p)
            budget=Budget(folder/'checkpoint.json',{**identity,'role':role},limits(p),1200)
            budget.phase='final'
            try:
                with budget.session.hard_deadline():
                    p,measurements=complete_program(p,parts[role],budget,refit=True)
                    require(admission.language_lock(p)==readiness.value['program_pair'][role]['language']
                        and all(payload(p.leaves[i])==payload(old.leaves[i]) for i in protected),'refit altered its fixed language/protected output')
                    # Use strict current-NN gates, never the inherited generic
                    # development gate from complete_program's legacy report.
                    current=current_report(p,parts[role])
                    require(set(current['feedback_leaves'])==set(p.feedback_leaves)
                        and set(current['explanation_leaves'])==set(p.explanation_leaves),'current qualification differs from fitted envelope')
                    require(current['validation']=={purpose:measurements['validation'][purpose] for purpose in ('feedback','explanation')},
                        'independent current-NN validation disagrees with refit metrics')
                    for split,a in parts[role].items():
                        direct=p.distributions(a['x'],a['m']);independent=forward(p,a['x'],a['m'])
                        require(all(np.array_equal(x,y) for x,y in zip(direct,independent,strict=True)),
                            'independent refit routing/full-distribution forward differs')
                        measurements[split]['gate']=gate(measurements[split])
                    value=p.to_dict();write_new(folder/'program.json',value)
                    (folder/'program.json').chmod(0o444)
                    program_ref=pointer(folder/'program.json')
                    require(Program.from_dict(bound(program_ref),p.actor_role_sha256).to_dict()==value,'stored refit certificate or qualification differs')
                    budget.checkpoint(dict(program=value,current=current,measurements=measurements),status='completed')
                solver=budget.summary()
                require(ledger(folder,dict(solver=solver))==solver['counts'] and solver['counts']['node']==0
                    and all(solver['counts'][k]<=solver['limits'][k] for k in solver['counts']),'refit ledger or fixed budget differs')
                results[role]=dict(program=program_ref,current=current,metrics=measurements,solver=solver,
                    checkpoint=pointer(folder/'checkpoint.json'),ledger=pointer(folder/'checkpoint.json.ledger.jsonl'),
                    protected_leaves_unchanged=sorted(protected),anchor_actor_role_sha256=p.actor_role_sha256,
                    target_actor_role_sha256=actor.role_sha256[role],feedback_enabled=current['gate']['passed'])
                refs.extend([program_ref,results[role]['checkpoint'],results[role]['ledger']])
            finally:budget.session.close()
        readiness.verify()
        for ref in refs:require(pointer(ref['path'])==ref,'refit asset changed during fitting')
        require(training_fingerprint(state)==start_state and source_files()==sources,'scheduled refit changed the NN/critics/optimizers or source')
        for name,sha in sources.items():
            snapshot=output/'source'/digest(sources)/name
            require(file_sha(snapshot)==sha and not snapshot.stat().st_mode&0o222,'refit source snapshot changed or writable')
        result=dict(version=VERSION,identity=identity,source_files=sources,asset_refs=refs,roles=results,
            full_training_state_sha256=start_state,external_program_selects_execution_action=False,
            feedback_requires_fresh_current_NN_evidence_each_optimizer_update=True,
            raw_external_program_gradient=0,NN_updates=0,PPO_joint_steps=0,environment_resets=0,
            owner_must_check_finite_training_authority_and_pool_membership=True,training_authority=False,pilot_opened=False)
        write_new(output/'result.json',result);write_new(output/'completed.json',dict(result_sha256=file_sha(output/'result.json')))
        return result
    except BaseException as error:
        write_new(output/'failed.json',dict(version=VERSION,error_type=type(error).__name__,error=str(error),
            completed_roles=sorted(results),automatic_retry=False,training_authority=False,pilot_opened=False))
        raise
