"""Current-NN fidelity envelope, separate from immutable parent protection.

This module grants no training/collection capability and selects no runtime
action. A future admitted PPO runner must bind its training-only manifests.
Every envelope is specific to one complete frozen NN and one optimizer epoch;
any Actor update invalidates it before the next feedback-bearing update.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .warehouse_v629_program import Program
from .warehouse_v628_program import metrics
from .warehouse_v626_program import validate_rows, _eligible, digest, plain
from .warehouse_v622_program import MASK_INDICES, masked_probabilities
from .warehouse_v6_actor import FrozenWarehouseV6Actor, MODEL_VERSION, FEATURE_DIM
from env.warehouse.decision_protocol import independent_agent_seed
from experiments.warehouse.v626_protocol import pointer,bound,require

VERSION='warehouse-online-current-nn-evidence.v1'
MANIFEST_VERSION='warehouse-online-program-training-evidence-pool.v1'
SPLITS=('fit','calibration','eligibility','validation')


def runtime_probabilities(actor,role,x,m):
    """Canonical single physical-frame forward, matching FrozenActor.act.

    Batched GEMM can differ from runtime GEMV by several float32 ulps. Exact
    evidence replay must keep the original evaluation shape, not relax hashes
    or accept a batch-size-dependent target distribution.
    """
    x=np.asarray(x,np.float32);m=np.asarray(m,np.float32)
    require(x.ndim==2 and x.shape[1]==FEATURE_DIM and m.shape==(len(x),5) and len(x)>0,
        'joint evidence contains no physical rows')
    return np.stack([masked_probabilities(actor.roles[role].logits_encoded(row),mask)
        for row,mask in zip(x,m,strict=True)])


def gate(value):
    """Strict development gate; no pressure-phase relaxation in training."""
    f,e=value['feedback'],value['explanation']
    checks=dict(feedback_coverage=f['coverage']>=.45,explanation_coverage=e['coverage']>=.45,
        expected_feedback=f['expected_actionable_coverage']>=.40,
        feedback_agreement=f['agreement']>=.95,explanation_agreement=e['agreement']>=.97,
        feedback_KL=f['kl'] is not None and f['kl']<=.20,
        explanation_KL=e['kl'] is not None and e['kl']<=.20,
        feedback_validation_support=_eligible(f,.95),explanation_validation_support=_eligible(e,.97),
        raw_illegal=f['whole_tree_raw_illegal_count']==e['whole_tree_raw_illegal_count']==0,
        unproved=f['whole_tree_path_uncertified_count']==e['whole_tree_path_uncertified_count']==0)
    return dict(checks=checks,passed=all(checks.values()))


def load_pool(ref,actor_ref,actor,role):
    """Replay full-NN targets and sampled actions, with disjoint whole scenes.

    This verifies manifest contents, not historical freshness or authorization.
    Those remain mandatory in the owning source-bound training capability.
    No validation/confirmation pool can be relabeled here without changing its
    bound manifest; the owner must also verify manifest membership against its
    admitted scenario ledger before invoking this pure API.
    """
    manifest=bound(ref)
    require(set(manifest)=={'schema','scope','target_actor','role','parts'}
        and manifest['schema']==MANIFEST_VERSION and manifest['scope']=='online_program_training_evidence'
        and manifest['target_actor']==actor_ref and manifest['role']==role and set(manifest['parts'])==set(SPLITS),
        'online evidence pool schema, scope or current NN differs')
    parts={};refs=[ref];episodes={};scenes={};identities={}
    for split in SPLITS:
        source=manifest['parts'][split]
        require(set(source)=={'arrays','rows'},'unknown online evidence part')
        refs.extend([source['arrays'],source['rows']])
        require(pointer(source['arrays']['path'])==source['arrays'],'online arrays changed')
        with np.load(source['arrays']['path'],allow_pickle=False) as z:
            require(set(z.files)=={'x','m','p','sampled'},'unknown online evidence arrays')
            a={key:z[key].copy() for key in z.files}
        a['rows']=bound(source['rows']);n=len(a['x'])
        require(n>0 and a['x'].shape==(n,FEATURE_DIM) and a['m'].shape==a['p'].shape==(n,5)
            and a['sampled'].shape==(n,) and a['sampled'].dtype==np.int64 and len(a['rows'])==n
            and all(a[key].dtype==np.float32 and np.isfinite(a[key]).all() for key in ('x','m','p')),
            'invalid complete online evidence arrays')
        validate_rows(a['x'])
        require(np.isin(a['m'],(0,1)).all() and np.all(a['m'][:,4]==1)
            and np.array_equal(a['x'][:,MASK_INDICES],a['m']),'online evidence physical masks differ')
        p=runtime_probabilities(actor,role,a['x'],a['m'])
        require(np.array_equal(a['p'],p),'online targets are not the current complete masked NN')
        seen=set();case_frames={}
        for i,row in enumerate(a['rows']):
            require(set(row)=={'episode','base_seed','task','frame','action_seed','episode_id'}
                and all(isinstance(row[k],str) and row[k] for k in ('episode','base_seed'))
                and all(type(row[k]) is int for k in ('task','frame','action_seed','episode_id'))
                and row['task'] in (1,2,3) and 0<=row['frame']<120 and 0<=row['action_seed']<2**63 and row['episode_id']>=1,
                'online evidence requires public episode and action-stream provenance')
            key=(row['episode'],row['frame']);require(key not in seen,'repeated physical evidence frame');seen.add(key)
            identity=(row['base_seed'],row['task'],row['action_seed'],row['episode_id'])
            require(row['episode'] not in identities or identities[row['episode']]==identity,'episode provenance changes within a pool')
            identities[row['episode']]=identity;case_frames.setdefault(row['episode'],[]).append(row['frame'])
            seed=independent_agent_seed(base_seed=row['action_seed'],episode_id=row['episode_id'],frame=row['frame'],agent_id=role)
            proposed=int(np.random.default_rng(seed).choice(5,p=p[i]))
            require(a['sampled'][i]==proposed and a['m'][i,proposed]==1,'online sampled action differs from current NN stream')
        require(all(sorted(frames)==list(range(len(frames))) for frames in case_frames.values()),'partial/noncontiguous evidence episode')
        # A short episode needs its complete public-terminal proof in the owning
        # collector audit. This API does not infer "no shutdown" from its length.
        episodes[split]={r['episode'] for r in a['rows']};scenes[split]={r['base_seed'] for r in a['rows']}
        parts[split]=a
    require(all(not episodes[a]&episodes[b] and not scenes[a]&scenes[b]
        for a in SPLITS for b in SPLITS if a<b),'fit/calibration/qualification/validation scenes overlap')
    for item in refs:require(pointer(item['path'])==item,'online evidence changed during replay')
    return parts,refs


def score(program,a,selected,purpose,predictions):
    return metrics(program,a['rows'],(a['x'],a['m'],a['sampled']),a['p'],selected,purpose,predictions)


def current_report(program,parts):
    """Derive qualification on eligibility only; validation decides enablement."""
    counts=np.bincount(program.leaf_indices(parts['fit']['x']),minlength=len(program.leaves)).tolist()
    e=parts['eligibility'];pred=program.distributions(e['x'],e['m'])
    eligibility={str(i):{purpose:score(program,e,{i},purpose,pred) for purpose in ('feedback','explanation')}
        for i in range(len(program.leaves))}
    feedback=frozenset(i for i,count in enumerate(counts) if count and _eligible(eligibility[str(i)]['feedback'],.93))
    explanation=frozenset(i for i,count in enumerate(counts) if count and _eligible(eligibility[str(i)]['explanation'],.95))
    a=parts['validation'];pred=program.distributions(a['x'],a['m'])
    validation={purpose:score(program,a,selected,purpose,pred) for purpose,selected in
        (('feedback',feedback),('explanation',explanation))}
    return plain(dict(fit_leaf_counts=counts,eligibility=eligibility,feedback_leaves=sorted(feedback),
        explanation_leaves=sorted(explanation),validation=validation,gate=gate(validation),
        split_episode_ids={s:sorted({r['episode'] for r in a['rows']}) for s,a in parts.items()},
        split_scenario_ids={s:sorted({r['base_seed'] for r in a['rows']}) for s,a in parts.items()}))


@dataclass(frozen=True)
class OnlineEvidence:
    program: Program
    report: dict
    program_ref: dict
    target_actor_ref: dict
    target_role_sha256: str
    pool_ref: dict
    epoch_id: str
    asset_refs: tuple
    envelope_digest: str

    def payload(self):
        return plain(dict(version=VERSION,role=self.program.role,epoch_id=self.epoch_id,
            anchor_program=self.program_ref,anchor_actor_role_sha256=self.program.actor_role_sha256,
            current_target_actor=self.target_actor_ref,current_target_role_sha256=self.target_role_sha256,
            pool=self.pool_ref,report=self.report,execution_actions_from_program=False,
            authorizes_training=False,environment_resets=0,fit_calls=0,LP_calls=0))

    def verify(self,*,current_role_sha256,epoch_id):
        require(current_role_sha256==self.target_role_sha256 and epoch_id==self.epoch_id,
            'online evidence expired after Actor update or optimizer epoch change')
        require(digest(self.payload())==self.envelope_digest,'online evidence report or binding mutated')
        for ref in self.asset_refs:require(pointer(ref['path'])==ref,'online evidence asset changed or missing')
        require(self.program.to_dict()==bound(self.program_ref),'online surrogate mutated after qualification')
        return self

    def loss_targets(self,x,m,*,current_role_sha256,epoch_id):
        """Return detached loss data only; never an execution proposal."""
        self.verify(current_role_sha256=current_role_sha256,epoch_id=epoch_id)
        routes,z,q=self.program.distributions(x,m)
        qualified=np.isin(routes,self.report['feedback_leaves']) & bool(self.report['gate']['passed'])
        return dict(probabilities=q,qualified=qualified,raw_program_top=z.argmax(-1))


def build_evidence(program_ref,target_actor_ref,pool_ref,*,epoch_id):
    require(isinstance(epoch_id,str) and epoch_id and len(epoch_id)<=128,'explicit optimizer epoch ID required')
    require(pointer(target_actor_ref['path'])==target_actor_ref,'current target Actor pointer differs')
    encoded=bound(program_ref)
    program=Program.from_dict(encoded,encoded['actor_role_sha256'])
    # Its actor SHA belongs to the immutable parent protection contract. Never
    # replace it with the current NN hash or reinterpret old qualification.
    actor=FrozenWarehouseV6Actor.load(target_actor_ref['path'],expected_sha256=target_actor_ref['sha256'])
    require(actor.model_version==MODEL_VERSION,'online evidence requires unchanged complete Actor architecture')
    parts,refs=load_pool(pool_ref,target_actor_ref,actor,program.role)
    report=current_report(program,parts)
    out=OnlineEvidence(program,report,deepcopy(program_ref),deepcopy(target_actor_ref),actor.role_sha256[program.role],
        deepcopy(pool_ref),epoch_id,tuple(deepcopy([program_ref,target_actor_ref,*refs])), '')
    out=OnlineEvidence(**{**out.__dict__,'envelope_digest':digest(out.payload())})
    return out.verify(current_role_sha256=out.target_role_sha256,epoch_id=epoch_id)


def load_evidence(ref,*,current_role_sha256,epoch_id):
    """Recompute qualifications and full-NN target provenance; trust no booleans."""
    encoded=bound(ref)
    require(encoded.get('version')==VERSION,'unknown online evidence schema')
    out=build_evidence(encoded['anchor_program'],encoded['current_target_actor'],encoded['pool'],epoch_id=encoded['epoch_id'])
    require(encoded==out.payload(),'stored online evidence does not independently replay')
    return out.verify(current_role_sha256=current_role_sha256,epoch_id=epoch_id)
