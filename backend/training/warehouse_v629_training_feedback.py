"""Qualified program loss for a future admitted paired PPO update.

No optimizer step, environment, execution action or training authority exists
here. In particular, gradient diagnostics are not accepted-update counters.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .warehouse_v629_online_evidence import OnlineEvidence
from .warehouse_v622_continuation_safety import program_kl_loss
from .warehouse_v6_actor_training import role_hashes
from env.warehouse.pure_nn import AGENTS


def coefficient(joint_steps,total_steps,variant):
    if variant not in ('raw','regularized') or type(total_steps) is not int or total_steps<=4000 \
            or type(joint_steps) is not int or not 0<=joint_steps<=total_steps:
        raise ValueError('registered variant, full budget and valid joint-step progress required')
    if variant=='raw':return 0.
    # Integer boundaries avoid epsilon-sized feedback at exactly 80 percent.
    return .01*min(1.,max(0.,(100*joint_steps-80*total_steps)/(5*total_steps)))


@dataclass
class Feedback:
    logits: torch.Tensor
    program_loss: object
    epoch_id: str
    role_sha256_before: dict
    evidence: dict
    diagnostics: dict

    def verify_before_optimizer_step(self,model,*,current_epoch_id):
        if current_epoch_id!=self.epoch_id:
            raise RuntimeError('optimizer epoch changed after program loss construction')
        if role_hashes(model)!=self.role_sha256_before:
            raise RuntimeError('Actor changed after program loss construction; evidence must be rebuilt')
        for role,ev in self.evidence.items():
            ev.verify(current_role_sha256=self.role_sha256_before[role],epoch_id=self.epoch_id)
        return self


def feedback_loss(model,encoded,masks,sampled,*,joint_steps,total_steps,variant,epoch_id,evidence=None):
    """Program KL divided by all physical frames, with actual role gradients.

    The admitted owner supplies freshly verified epoch evidence and must call
    verify_before_optimizer_step immediately before a shared PPO Adam step.
    Each successful Adam step invalidates evidence for changed role weights;
    a minibatch cannot reuse a certificate for an earlier NN hash.
    Raw and pre-window regularized calls never inspect an evidence provider.
    """
    if not isinstance(epoch_id,str) or not epoch_id:raise ValueError('optimizer epoch identity required')
    if not isinstance(encoded,torch.Tensor) or encoded.ndim!=3 or encoded.shape[1]!=2 or not len(encoded):
        raise ValueError('feedback requires a nonempty batch of two-role physical frames')
    lam=coefficient(joint_steps,total_steps,variant)
    logits=model.outputs_encoded(encoded,masks);before=role_hashes(model);used={}
    if not lam:
        result=program_kl_loss(logits,None,masks,None,None,raw_program_top=None,variant='raw',coefficient=0.)
    else:
        if not isinstance(evidence,dict) or set(evidence)!=set(AGENTS):
            raise ValueError('regularized feedback requires both current role evidence objects')
        x=encoded.detach().cpu().numpy();m=masks.detach().cpu().numpy();targets=[]
        for i,role in enumerate(AGENTS):
            ev=evidence[role]
            if not isinstance(ev,OnlineEvidence) or ev.program.role!=role:
                raise ValueError('online evidence belongs to another role or schema')
            targets.append(ev.loss_targets(x[:,i],m[:,i],current_role_sha256=before[role],epoch_id=epoch_id));used[role]=ev
        tensors={key:torch.as_tensor(np.stack([a[key] for a in targets],axis=1),device=logits.device)
            for key in ('probabilities','qualified','raw_program_top')}
        result=program_kl_loss(logits,tensors['probabilities'],masks,sampled,tensors['qualified'],
            raw_program_top=tensors['raw_program_top'],variant='regularized',coefficient=lam)
    gradients={}
    for i,role in enumerate(AGENTS):
        parameters=tuple(p for p in model.actors[role].parameters() if p.requires_grad)
        grad=torch.autograd.grad(result.role_losses[i],parameters,retain_graph=True,allow_unused=True) \
            if parameters and result.role_losses[i].requires_grad else ()
        norm=sum((g.detach().double().square().sum() for g in grad if g is not None),torch.tensor(0.,dtype=torch.float64,device=logits.device)).sqrt()
        if not torch.isfinite(norm):raise RuntimeError('nonfinite actual program gradient')
        gradients[role]=float(norm)
    diagnostics=dict(variant=variant,coefficient=lam,physical_frames=result.physical_frames,
        selected_frames={r:int(result.selected_counts[i]) for i,r in enumerate(AGENTS)},
        role_program_gradient_norm_before_shared_clip=gradients,
        accepted_optimizer_updates='owned_by_admitted_PPO_runner_not_inferred_from_gradient',
        execution_actions_generated=0,environment_resets=0,optimizer_steps=0)
    return Feedback(logits,result,epoch_id,before,used,diagnostics).verify_before_optimizer_step(model,current_epoch_id=epoch_id)
