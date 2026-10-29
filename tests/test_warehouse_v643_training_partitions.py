"""Synthetic inventory/source tests, not fresh-scene or model acceptance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from experiments.warehouse import v643_training_partitions as p


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)); path.chmod(0o444)
    return p.pointer(path)


def inventories():
    training=dict(schema=p.TRAIN_VERSION,cases=[])
    for i in range(168):
        task=i%3+1
        training['cases'].append(dict(episode_id=f'{p.CANDIDATE}-train-{i:03d}',scenario_seed=10*(10000+i)+task,
            task=task,action_seed=20000+i,slot=i%8,partition='train'))
    program=dict(schema=p.PROGRAM_VERSION,scope=p.PROGRAM_SCOPE,candidate_id=p.CANDIDATE,
                 expected_counts=dict(p.PROGRAM_COUNTS),cases=[])
    index=0
    for split, count in p.PROGRAM_COUNTS.items():
        for i in range(count):
            task=i%3+1
            program['cases'].append(dict(episode=f'{p.CANDIDATE}-program-{split}-{i:03d}',base_seed=30000+index,
                scenario_seed=10*(30000+index)+task,task=task,action_seed=40000+index,split=split))
            index+=1
    quality=dict(schema=p.QUALITY_VERSION,candidate_id=p.CANDIDATE,cases=[])
    for i in range(75):
        task=i%3+1
        quality['cases'].append(dict(case_id=f'{p.CANDIDATE}-quality-{i:03d}',base=50000+i,
            scenario_seed=10*(50000+i)+task,task=task,action_seed=60000+i))
    return training,program,quality


def checked(values):
    return p.validate_inventories(*values,initialization_seed=0,optimizer_seed=1)


def test_training_program_and_quality_are_three_finite_separate_namespaces():
    result=checked(inventories())
    assert result['training_episodes']==168 and result['program_counts']==p.PROGRAM_COUNTS
    assert result['quality_episodes']==75
    assert result['episodes_per_slot']=={str(i):21 for i in range(8)}
    assert len(result['episode_ids'])==1893
    assert len(result['all_seed_arguments'])==3*1893+2
    assert not result['history_freshness_verified']
    assert result['program_or_quality_frames_permitted_in_gradient_buffers']==0
    assert result['environment_authority'] is result['optimizer_authority'] is False


@pytest.mark.parametrize('change', ['quality_alias','program_alias','action_alias','initialization_alias',
    'task_alias','duplicate_id','old_candidate','old_program_schema','unbalanced_program',
    'unknown_field','bool_task','unsafe_id','missing_slot','short_slot','sealed','quality_count'])
def test_leakage_old_bindings_and_unsafe_or_insufficient_inventories_fail(change):
    train,program,quality=inventories()
    if change=='quality_alias':
        quality['cases'][0].update(base=10000,scenario_seed=100001)
    if change=='program_alias':
        program['cases'][0].update(base_seed=10000,scenario_seed=100001)
    if change=='action_alias':quality['cases'][0]['action_seed']=program['cases'][0]['scenario_seed']
    if change=='initialization_alias':program['cases'][0]['action_seed']=0
    if change=='task_alias':train['cases'][1].update(scenario_seed=100002,task=2)
    if change=='duplicate_id':quality['cases'][0]['case_id']=program['cases'][0]['episode']
    if change=='old_candidate':program['candidate_id']='warehouse-v642-depth9-depth12-01'
    if change=='old_program_schema':program['schema']='warehouse-v642-program-evidence-case-inventory.v1'
    if change=='unbalanced_program':program['cases'][0].update(task=2,scenario_seed=300002)
    if change=='unknown_field':program['cases'][0]['passed']=True
    if change=='bool_task':train['cases'][0]['task']=True
    if change=='unsafe_id':train['cases'][0]['episode_id']='../escape'
    if change=='missing_slot':train['cases']=[c for c in train['cases'] if c['slot']!=7]
    if change=='short_slot':train['cases']=train['cases'][:160]
    if change=='sealed':quality['cases'][0].update(base=240100,scenario_seed=2401001)
    if change=='quality_count':quality['cases'].pop()
    with pytest.raises((RuntimeError,ValueError)):
        checked((train,program,quality))


def test_no_development_budget_or_other_recipe_can_enter_initial_calibration():
    for budget in (200000,300000,400000):
        with pytest.raises(RuntimeError,match='fixed 20k'):
            p.validate_inventories(*inventories(),initialization_seed=0,optimizer_seed=1,
                config=replace(p.PPOConfig(),joint_step_budget=budget))


@pytest.fixture
def partition(tmp_path, monkeypatch):
    values=inventories()
    refs=[write(tmp_path/(kind+'.json'), value) for kind,value in zip(('train','program','quality'),values)]
    ready=SimpleNamespace(value=dict(actor=dict(path=str(tmp_path/'synthetic-not-real-BC.npz'),sha256=p.REFERENCE_SHA256),
        reference=dict(buffer=dict(path=str(tmp_path/'synthetic-reference.npz'),sha256='a'*64)),
        program_pair={role:dict(program=dict(path=str(tmp_path/(role+'-template.json')),sha256='c'*64))
                      for role in ('robot_1','robot_2')}),
        payload_sha256='b'*64,verify=lambda:None)
    # This fake is confined to source/inventory software tests; no claims,
    # Actors, environments, or real history are created from this object.
    monkeypatch.setattr(p.historical_reference,'load_readiness',lambda unused:ready)
    monkeypatch.setattr(p,'OUTPUT_ROOT',tmp_path/'partitions')
    plan=p.prepare(reference_directory=tmp_path/'reference',training_inventory_ref=refs[0],program_inventory_ref=refs[1],
        quality_inventory_ref=refs[2],initialization_seed=0,optimizer_seed=1,output=p.OUTPUT_ROOT/'synthetic')
    return plan


def test_partition_binds_original_reference_and_all_three_inventories_without_old20kpass(partition):
    partition.verify()
    assert partition.value['reference_actor']['sha256']==p.REFERENCE_SHA256
    assert partition.value['old_20k_pass_required'] is False
    assert partition.value['training_authority'] is False
    assert partition.value['historical_confirmation_is_not_v643_calibration_pass'] is True


def test_all_reference_reads_use_one_absolute_cache_key(partition,tmp_path,monkeypatch):
    calls=[];ready=partition.reference_evidence
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(p.historical_reference,'load_readiness',lambda directory:(calls.append(directory) or ready))
    p.load('reference',partition.ref)
    p.load(tmp_path/'reference',partition.ref)
    p.prepare(reference_directory='reference',training_inventory_ref=partition.value['train_case_inventory'],
        program_inventory_ref=partition.value['program_evidence_inventory'],quality_inventory_ref=partition.value['quality_case_inventory'],
        initialization_seed=0,optimizer_seed=1,output=p.OUTPUT_ROOT/'normalized-reference')
    assert calls==[(tmp_path/'reference').resolve()]*3


def test_mutated_inventory_or_falsepass_cannot_survive_partition_binding(partition):
    path=Path(partition.value['quality_case_inventory']['path'])
    value=json.loads(path.read_text());value['passed']=True
    path.chmod(0o644);path.write_text(json.dumps(value));path.chmod(0o444)
    with pytest.raises(RuntimeError,match='hash changed'):
        partition.verify()


def test_unavailable_original_reference_stops_before_new_inventory_reads_or_output(tmp_path, monkeypatch):
    def absent(_):raise RuntimeError('original reference not authenticated')
    def forbidden(_):raise AssertionError('must not inspect proposed inventories')
    monkeypatch.setattr(p.historical_reference,'load_readiness',absent)
    monkeypatch.setattr(p,'_immutable',forbidden)
    with pytest.raises(RuntimeError,match='original reference'):
        p.prepare(reference_directory=tmp_path/'reference',training_inventory_ref=None,program_inventory_ref=None,
            quality_inventory_ref=None,initialization_seed=None,optimizer_seed=None,output=tmp_path/'must-not-exist')
    assert not list(tmp_path.iterdir())
