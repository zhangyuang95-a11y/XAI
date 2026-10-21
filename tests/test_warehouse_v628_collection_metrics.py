"""Public-event counters must reject partial episodes and ambiguous removals."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('collector_metrics',Path(__file__).resolve().parents[1]/'scripts/measure_warehouse_v628_collection.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def events():
    case=dict(case_id='fixture',sampling_mode='sampled',task=1)
    state=dict(frame=0,agents=[dict(agent_id='robot_1',active=True,carrying_task_id='order-0',position=[0,0]),
        dict(agent_id='robot_2',active=True,carrying_task_id=None,position=[1,1])],
        tasks=[dict(task_id='order-0',delivery_position=[0,1])],last_robot_collision_event=False)
    out=[]
    for i in range(120):
        after=deepcopy(state);after['frame']=i+1;after['last_robot_collision_event']=i in (10,20)
        if i==0:
            after['tasks']=[dict(task_id='order-1',delivery_position=[1,2])]
            after['agents'][0].update(position=[0,1],carrying_task_id=None)
        out.append(dict(case=case,frame=i,actor_sha256='a'*64,pre_state_public=state,post_state_public=after));state=after
    return case,out


def test_replacement_delivery_and_collision_counts():
    case,rows=events();result=m.episode(case,rows,120,'a'*64)
    assert result['deliveries']==1 and result['robot_collisions']==2 and result['shutdown'] is False
    with pytest.raises(ValueError,match='partial episode'):m.episode(case,rows[:10],10,'a'*64)


def test_shutdown_is_complete_and_unphysical_removal_is_rejected():
    case,rows=events();rows=rows[:10];rows[-1]['post_state_public']['agents'][0]['active']=False
    assert m.episode(case,rows,10,'a'*64)['shutdown'] is True
    case,rows=events();rows[0]['post_state_public']['agents'][0]['position']=[7,7]
    with pytest.raises(ValueError,match='physically delivered'):m.episode(case,rows,120,'a'*64)
