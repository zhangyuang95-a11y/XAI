"""Recount accepted NN/program updates without constructing any model or RNG.

This checks the hashed training receipts. It is not a new physics replay,
task-quality evaluation, or proof that a combined update was caused solely
by the program objective.
"""
import math
from .v626_protocol import require,bound

ROLES=('robot_1','robot_2')
VERSION='warehouse-v642-update-receipts-recount.v1'


def audit_updates(receipt_refs,*,variant,expected_joint_steps,core):
    require(variant in ('raw','regularized') and type(expected_joint_steps) is int
        and expected_joint_steps>=4000,'explicit completed training boundary required')
    require(len({r['path'] for r in receipt_refs})==len(receipt_refs),
        'an optimizer receipt was repeated')
    cursor=0;actor_updates=critic_updates=completed=shutdowns=0
    accepted={r:0 for r in ROLES};gradient_sum={r:0. for r in ROLES}
    max_gradient={r:0. for r in ROLES};weight_delta={r:0. for r in ROLES}
    for ref in receipt_refs:
        row=bound(ref)
        require(row['start_joint_step']==cursor and cursor<row['end_joint_step']<=expected_joint_steps
            and 0<row['end_joint_step']-cursor<=1024
            and not(cursor<4000<row['end_joint_step']),
            'receipt steps are missing, duplicated or cross the critic-only boundary')
        require(row['partial_episodes_labeled_negative']==0
            and type(row['complete_episodes']) is int
            and 0<=row['shutdown_episodes']<=row['complete_episodes'],
            'partial episode labels or invalid complete-episode outcomes')
        completed+=row['complete_episodes'];shutdowns+=row['shutdown_episodes']
        subtotal={r:0 for r in ROLES}
        for epoch in row['epochs']:
            require(epoch['updates'],'empty optimizer epoch')
            for update in epoch['updates']:
                critic_updates+=1
                active=cursor>=4000
                require(update['actor_updated'] is active,
                    'Actor updated during warmup or skipped an accepted training update')
                actor_updates+=int(active)
                diagnostics=update['program_gradient']
                norms=diagnostics['role_program_gradient_norm_before_shared_clip']
                for role in ROLES:
                    norm=norms[role]
                    require(isinstance(norm,(int,float)) and math.isfinite(norm) and norm>=0,
                        'nonfinite or negative program gradient')
                    gradient_sum[role]+=norm;max_gradient[role]=max(max_gradient[role],norm)
                    positive=update['accepted_positive_program_gradient'][role]
                    require(type(positive) is bool,'accepted-gradient flag must be boolean')
                    if active:
                        delta=update['role_weight_delta_l2'][role]
                        clip=update['common_gradient_clip_factor']
                        require(math.isfinite(delta) and delta>=0 and math.isfinite(clip) and 0<clip<=1,
                            'invalid actual weight delta or shared gradient clipping')
                        expected=norm>0 and delta>0 and update['role_sha256_before'][role]!=update['role_sha256_after'][role]
                        require(positive==expected,'accepted-gradient flag differs from actual gradient/weight evidence')
                        weight_delta[role]+=delta
                    else:require(not positive and norm==0,'warmup supplied an Actor program gradient')
                    if variant=='raw':require(norm==0 and not positive,'raw external program gradient is nonzero')
                    subtotal[role]+=int(positive)
                if variant=='raw' and active:require(update['program_loss']==0,'raw external program loss is nonzero')
        require(subtotal==row['accepted_positive_program_gradient_updates'],
            'batch accepted-gradient total differs from individual updates')
        for role in ROLES:accepted[role]+=subtotal[role]
        cursor=row['end_joint_step']
    require(cursor==expected_joint_steps==core['processed_joint_steps'] and core['variant']==variant
        and core['safe_batch_boundary'] is True and core['failed'] is False,
        'incomplete or unsafe training boundary')
    require(actor_updates==core['actor_updates'] and critic_updates==core['critic_updates']
        and accepted==core['accepted_program_updates'],'core counters differ from hashed optimizer receipts')
    if expected_joint_steps==20000 and variant=='regularized':
        require(all(n>0 for n in accepted.values()),'both regularized Actors need actual accepted program gradients')
    return dict(version=VERSION,variant=variant,joint_steps=cursor,receipt_count=len(receipt_refs),
        actor_updates=actor_updates,critic_updates=critic_updates,accepted_program_updates=accepted,
        role_program_gradient_norm_sum=gradient_sum,role_program_gradient_norm_max=max_gradient,
        role_combined_weight_delta_path_length=weight_delta,
        complete_training_episodes=completed,shutdown_training_episodes=shutdowns,
        training_outcomes_are_not_independent_quality_evaluation=True,
        combined_update_not_causal_program_attribution=True,new_RNG_constructors=0,
        environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
