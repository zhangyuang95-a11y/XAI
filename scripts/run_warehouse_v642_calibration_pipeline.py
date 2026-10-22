#!/usr/bin/env python3
"""One admitted 20k pair: declare, audit, collect and PPO; no retry/resume."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_calibration_pipeline as pipeline
from experiments.warehouse import v642_training_partitions as partitions,v642_training_seed_plan as seeds
from backend.training.warehouse_v625_actor_training import PPOConfig
from scripts.run_warehouse_v625 import write_new

RECIPE=ROOT/'configs/warehouse_v642_calibration_recipe.json'


def recipe():
    return dict(schema='warehouse-v642-paired-calibration-recipe.v1',config=asdict(PPOConfig()),
        slots=8,reserved_training_cases_per_slot=128,program_evidence_counts=partitions.admission.PROGRAM_EVIDENCE_COUNTS,
        initial_actors='same_complete_frozen_v620_dual_Actor_for_both_variants',program_pair_fold=0,
        same_critic_and_optimizer_initialization_within_pair=True,reference_buffer='original_audited_300_argmax_common_fit',
        current_NN_targets_not_execution_controllers=True,program_evidence_excluded_from_NN_gradients=True,
        quality_and_program_evaluation_after_training_required=True,automatic_retry=False,automatic_resume=False,pilot_opened=False)


def inventories(args):
    config=recipe();training=dict(schema=partitions.TRAIN_VERSION,cases=[]);evidence=dict(schema=partitions.observed.INVENTORY_VERSION,
        scope=partitions.observed.SCOPE,expected_counts=dict(config['program_evidence_counts']),cases=[])
    for index in range(config['slots']*config['reserved_training_cases_per_slot']):
        task=index%3+1;base=args.train_base+index
        training['cases'].append(dict(episode_id=f'{args.run_id}-train-{index:04d}',scenario_seed=10*base+task,task=task,
            action_seed=args.train_action_base+index,slot=index%8,partition='train'))
    index=0
    for split,count in config['program_evidence_counts'].items():
        for j in range(count):
            task=j%3+1;base=args.evidence_base+index
            evidence['cases'].append(dict(episode=f'{args.run_id}-evidence-{split}-{j:04d}',base_seed=base,scenario_seed=10*base+task,
                task=task,action_seed=args.evidence_action_base+index,split=split));index+=1
    partitions.validate_inventories(training,evidence,initialization_seed=args.initialization_seed,optimizer_seed=args.optimizer_seed)
    return training,evidence


def run(args):
    # This must precede seed declarations, recipe I/O and directory writes.
    readiness=partitions.admission.load_readiness(args.reference_directory);readiness.verify()
    seeds.confirmation.run_name(args.run_id)
    partitions.require(json.loads(RECIPE.read_text())==recipe(),'registered calibration recipe differs')
    training,evidence=inventories(args)
    inputs=partitions.OUTPUT_ROOT/(args.run_id+'-inputs');inputs.mkdir(parents=True,exist_ok=False)
    for name,value in (('train-inventory.json',training),('program-evidence-inventory.json',evidence)):
        write_new(inputs/name,value);(inputs/name).chmod(0o444)
    proposal=partitions.prepare(reference_directory=args.reference_directory,training_inventory_ref=partitions.pointer(inputs/'train-inventory.json'),
        evidence_inventory_ref=partitions.pointer(inputs/'program-evidence-inventory.json'),initialization_seed=args.initialization_seed,
        optimizer_seed=args.optimizer_seed,output=partitions.OUTPUT_ROOT/args.run_id)
    plan=seeds.prepare(reference_directory=args.reference_directory,proposal_ref=proposal.ref,run_id=args.run_id)
    pipeline.prepare(reference_directory=args.reference_directory,plan_ref=plan.ref)
    print(json.dumps(dict(stage='complete_training_source_and_seed_plan_prepared',plan=plan.ref,
        reserved_training_episodes_per_variant=len(training['cases']),program_evidence_episodes_per_variant=len(evidence['cases']),
        actual_resets=0,Actor_updates=0,training_authority=False)),flush=True)
    clearance=pipeline.audit_seeds(reference_directory=args.reference_directory,plan_ref=plan.ref)
    print(json.dumps(dict(stage='exact_training_seed_audit_passed',clearance=clearance,actual_resets=0)),flush=True)
    result=pipeline.run(reference_directory=args.reference_directory,plan_ref=plan.ref)
    print(json.dumps(result,allow_nan=False),flush=True);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference-directory',required=True,type=Path);parser.add_argument('--run-id',required=True)
    for field in ('initialization-seed','optimizer-seed','train-base','train-action-base','evidence-base','evidence-action-base'):
        parser.add_argument('--'+field,required=True,type=int)
    return run(parser.parse_args(argv))

if __name__=='__main__':main()
