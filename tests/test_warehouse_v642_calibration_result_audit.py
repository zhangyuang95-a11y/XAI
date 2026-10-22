"""Result export refuses unobserved/incomplete runs before any NN load."""
import json
import pytest
from scripts import audit_warehouse_v642_calibration_results as c


@pytest.mark.parametrize('change',['nonzero_exit','wrong_completion','not_observed','not_finished'])
def test_incomplete_or_unobserved_training_cannot_export(tmp_path,monkeypatch,change):
    value=dict(version=c.pipeline.VERSION,calibration_training_finished=True,model_acceptance_passed=False,
        publishable=False,pilot_opened=False)
    completed=tmp_path/'controller-completed.json';completed.write_text(json.dumps(value));ref=c.pointer(completed)
    record=dict(exit_code=0,controller_completed=ref,observed_by='exec_tool_terminal_result')
    if change=='nonzero_exit':record['exit_code']=1
    elif change=='wrong_completion':record['controller_completed']={'path':'elsewhere','sha256':'invalid'}
    elif change=='not_observed':record['observed_by']='process_disappeared'
    else:
        value['calibration_training_finished']=False;completed.write_text(json.dumps(value));ref=c.pointer(completed)
        record['controller_completed']=ref
    exit_path=tmp_path/'exit.json';exit_path.write_text(json.dumps(record))
    def forbidden(*args,**kwargs):pytest.fail('invalid run reached model/source loading or tensor export')
    monkeypatch.setattr(c.pipeline.owner,'load_seed_plan',forbidden)
    monkeypatch.setattr(c,'export_checkpoint_actor',forbidden)
    with pytest.raises(RuntimeError):c.run(reference_directory=tmp_path/'reference',completed_ref=ref,
        supervisor_exit_ref=c.pointer(exit_path),output=tmp_path/'assets')
    assert not (tmp_path/'assets').exists()
