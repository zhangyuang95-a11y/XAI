"""Read-only evidence gate for the explicit-depth v642 experiment.

This grants no collection, training, RNG or environment capability. Historical
source contracts stay immutable; confirmation must bind a new source closure.
"""
from pathlib import Path
from . import v630_protocol as previous
from .v626_protocol import require,pointer,bound,digest,file_sha

ROOT=previous.ROOT
CANDIDATE='warehouse-v642-depth9-depth12-01'
OUTPUT=ROOT/'output/warehouse_v642'/CANDIDATE
VERSION='warehouse-v642-margin-context-rebuild-development.v1'


def checked_plan():
    path=OUTPUT/'plan.json';p=bound(pointer(path));_,old=previous.checked_plan()
    require(p['schema']==VERSION and p['actor']==old['actor'] and p['actor_role_sha256']==old['actor_role_sha256']
        and p['depth_arms']==[9,12] and p['maximum_leaves']==256 and p['user_depth12_authorized'] is True
        and p['equal_call_limits_per_pair'] and p['original_core_and_current_fit_reliable_E_paths_outputs_frozen']
        and p['selection_rule']=='one_uniform_depth_arm_all_six_pass_prefer_depth9_no_rolewise_mixing'
        and p['NN_updates']==p['PPO_joint_steps']==p['new_environment_resets']==0
        and p['training_authority'] is False and p['sealed_authority'] is False and p['pilot_opened'] is False,
        'unknown extraction contract or NN identity')
    expected={(d,f,r) for d in (9,12) for f in range(3) for r in ('robot_1','robot_2')}
    require(len(p['jobs'])==12 and {(j['maximum_depth'],j['fold'],j['role']) for j in p['jobs']}==expected,'incomplete paired capacity design')
    for fold in range(3):
        for role in ('robot_1','robot_2'):
            pair=[j for j in p['jobs'] if j['fold']==fold and j['role']==role]
            require(pair[0]['limits']==pair[1]['limits'] and pair[0]['source_result']==pair[1]['source_result'] and
                pair[0]['frozen_leaves']==pair[1]['frozen_leaves'] and pair[0]['structural_roots']==pair[1]['structural_roots'],'unmatched depth arms')
    for name,sha in p['source_files'].items():
        source=Path(name) if Path(name).is_absolute() else ROOT/name;snapshot=OUTPUT/'source'/source.relative_to(ROOT)
        require(file_sha(source)==file_sha(snapshot)==sha and not snapshot.stat().st_mode&0o222,'registered source or read-only snapshot changed')
    for ref in (p['actor'],p['parent_summary'],p['parent_audit']):require(pointer(ref['path'])==ref,'bound extraction input changed')
    require(bound(p['parent_audit'])['integrity_passed'],'missing audited parent closure')
    return path,p


def source_files():
    _,p=checked_plan();files={}
    for name,sha in p['source_files'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name;relative=str(path.relative_to(ROOT))
        require(relative not in files or files[relative]==sha,'conflicting source aliases');files[relative]=sha
    files['experiments/warehouse/v642_protocol.py']=file_sha(__file__)
    return dict(sorted(files.items()))


def recipe():
    return dict(schema='warehouse-v642-confirmation-gate-contract.v1',gate=previous.recipe()['gate'],
        maximum_leaves=256,allowed_depths=[9,12],selection='uniform_all_six_prefer_depth9',
        training_authority=False,sealed_authority=False,pilot_opened=False)
