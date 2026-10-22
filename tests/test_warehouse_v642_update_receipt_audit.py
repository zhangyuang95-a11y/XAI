"""Tampered receipts and invented accepted gradients cannot pass a recount."""
from copy import deepcopy
import json
import pytest
from experiments.warehouse import v642_update_receipt_audit as a
from experiments.warehouse.v626_protocol import pointer


def example(tmp_path,variant):
    refs=[];start=0
    for end in (1024,2048,3072,4000,5024):
        active=start>=4000;positive=active and variant=='regularized'
        u=dict(actor_updated=active,program_gradient=dict(role_program_gradient_norm_before_shared_clip=
            {r:.1 if positive else 0. for r in a.ROLES}),
            accepted_positive_program_gradient={r:positive for r in a.ROLES})
        if active:u.update(role_weight_delta_l2={r:.01 for r in a.ROLES},common_gradient_clip_factor=.5,
            role_sha256_before={r:'before' for r in a.ROLES},role_sha256_after={r:'after' for r in a.ROLES},
            program_loss=.02 if positive else 0.)
        value=dict(start_joint_step=start,end_joint_step=end,partial_episodes_labeled_negative=0,
            complete_episodes=2,shutdown_episodes=0,epochs=[dict(updates=[u])],
            accepted_positive_program_gradient_updates={r:int(positive) for r in a.ROLES})
        p=tmp_path/f'{start}.json';p.write_text(json.dumps(value));refs.append(pointer(p));start=end
    core=dict(processed_joint_steps=5024,variant=variant,safe_batch_boundary=True,failed=False,
        actor_updates=1,critic_updates=5,accepted_program_updates={r:int(variant=='regularized') for r in a.ROLES})
    return refs,core


@pytest.mark.parametrize('variant',['raw','regularized'])
def test_recounts_actual_receipts_without_claiming_quality(tmp_path,variant):
    refs,core=example(tmp_path,variant)
    out=a.audit_updates(refs,variant=variant,expected_joint_steps=5024,core=core)
    assert out['joint_steps']==5024 and out['accepted_program_updates']==core['accepted_program_updates']
    assert not out['model_acceptance_passed'] and not out['environment_resets']


@pytest.mark.parametrize('change',['bytes','gap','false_gradient','warmup','partial','raw_gradient','core'])
def test_rejects_bad_receipt_evidence(tmp_path,change):
    variant='raw' if change=='raw_gradient' else 'regularized'
    refs,core=example(tmp_path,variant)
    i=0 if change in ('warmup','partial') else -1
    p=refs[i]['path'];value=json.loads(open(p).read())
    if change=='bytes':value['end_joint_step']-=8
    elif change=='gap':value['start_joint_step']+=8
    elif change=='false_gradient':value['epochs'][0]['updates'][0]['role_weight_delta_l2']['robot_1']=0.
    elif change=='warmup':value['epochs'][0]['updates'][0]['actor_updated']=True
    elif change=='partial':value['partial_episodes_labeled_negative']=1
    elif change=='raw_gradient':value['epochs'][0]['updates'][0]['program_gradient']['role_program_gradient_norm_before_shared_clip']['robot_1']=.01
    elif change=='core':core['accepted_program_updates']['robot_1']+=1
    with open(p,'w') as h:json.dump(value,h)
    if change!='bytes':refs[i]=pointer(p)
    with pytest.raises((RuntimeError,ValueError)):
        a.audit_updates(refs,variant=variant,expected_joint_steps=5024,core=core)
