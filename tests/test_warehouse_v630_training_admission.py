"""Readiness contracts, synthetic confirmation records; no training owner."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import numpy as np
import pytest

from experiments.warehouse import v630_training_admission as a


def test_missing_actual_winner_stops_before_buffer_initialization_rng_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(a.v,'OUTPUT',tmp_path/'no-candidate')
    def forbidden(*args,**kwargs):pytest.fail('closed development must not touch reference or RNG')
    monkeypatch.setattr(a.reference,'source_bundle',forbidden)
    monkeypatch.setattr(np.random,'default_rng',forbidden)
    with pytest.raises(RuntimeError,match='no complete locked audited winner'):
        a.load_readiness(tmp_path/'no-reference')
    assert list(tmp_path.iterdir())==[]


def test_training_source_and_recipe_keep_full_actor_reference_and_zero_raw_feedback():
    files=a.source_files()
    for name in (*a.FILES,'backend/training/warehouse_v629_reference_buffer.py',
                 'backend/training/warehouse_v629_ppo_update.py',
                 'experiments/warehouse/v629_training_admission.py'):
        assert files[name]==a.v.file_sha(a.v.ROOT/name)
    recipe=a.recipe()
    assert recipe['raw_external_program_gradient']==0 and recipe['program_pair_fold']==0
    assert recipe['initial_Actor']=='complete_frozen_v620_both_roles'
    assert recipe['current_target_NN_hash_distinct_from_immutable_anchor_hash']
    assert recipe['original_reference_buffer_reused_without_regeneration']
    assert not recipe['environment_authority'] and not recipe['optimizer_authority']


def test_training_role_pair_uses_registered_fold_zero_instead_of_best_score(monkeypatch):
    units=[]
    for fold in range(3):
        for role in ('robot_1','robot_2'):
            units.append(SimpleNamespace(fold=fold,role=role,program_ref={'chosen':fold},
                        reference_ref={'baseline':fold},program=SimpleNamespace(role=role)))
    monkeypatch.setattr(a.program_training,'language_lock',lambda p:dict(role=p.role))
    pair=a.program_pair(SimpleNamespace(units=units))
    assert set(pair)=={'robot_1','robot_2'} and all(r['program']==dict(chosen=0) for r in pair.values())


@pytest.fixture
def confirmation(tmp_path,monkeypatch):
    # Reuse the established synthetic evidence generator with this module's
    # schema and verifier. It never accesses actual A/B output or a new seed.
    spec=importlib.util.spec_from_file_location('v630_training_confirmation_fixture',
              Path(__file__).with_name('test_warehouse_v629_training_admission.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.a=a
    return helper.confirmation.__wrapped__(tmp_path,monkeypatch)


def test_new_confirmation_is_recomputed_before_readiness(confirmation):
    folder,path,winner,*_,calls=confirmation
    value,refs=a.confirmation_evidence(winner)
    assert len(calls)==1 and value['summary']==a.v.pointer(folder/'summary.json')
    assert {'allocation.json','allocation.claim.json','plan.claim.json','inventory.json'} <= {
        Path(ref['path']).name for ref in refs}


@pytest.mark.parametrize('fault',['current_failure','stored_metrics','publish_flag','foreign_winner'])
def test_boolean_pass_or_changed_summary_cannot_arm_training(confirmation,fault):
    import json
    folder,path,winner,metrics,stored,summary,calls=confirmation
    def write(file,value):file.write_text(json.dumps(value))
    if fault=='current_failure':metrics['metrics_passed']=False
    elif fault=='stored_metrics':
        stored['all_qualification_and_temperatures_unchanged']=False
        write(folder/'metrics.json',stored)
    elif fault=='publish_flag':
        summary['pilot_opened']=True;write(folder/'summary.json',summary)
        write(a.v.OUTPUT/'confirmation.passed.json',dict(summary=a.v.pointer(folder/'summary.json'),confirmation_passed=True))
    else:winner=SimpleNamespace(actor=winner.actor,provenance=dict(foreign=True))
    with pytest.raises(RuntimeError):a.confirmation_evidence(winner)
