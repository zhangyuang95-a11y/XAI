"""Actual JavaScript demo / Python authority parity, without modifying the demo."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest

from env.cooperative_kitchen import CooperativeKitchen, KitchenConfig, program_decision

ROOT=Path(__file__).resolve().parents[1]
NODE=shutil.which('node') or '/Users/zhangyuang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
BASE_KEYS=('schema','preset','turn','maxSteps','targetOrders','orders','done','reason','actors','pot','counters')


def js_trajectory(actions,auto):
    script="""
const E=require(process.argv[1]);
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
let s=E.reset('supply');const frames=[];
for(const action of input.actions){const r=E.step(s,action,{auto:input.auto});frames.push(r);s=r.state;}
process.stdout.write(JSON.stringify(frames));
"""
    result=subprocess.run([NODE,'-e',script,str(ROOT/'ui/cooperative_kitchen_demo/engine.js')],input=json.dumps(dict(actions=actions,auto=auto)),text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def normalize_event(event):
    e=dict(event)
    for key in ('item_id','batch_id','plate_id','facing_before'):e.pop(key,None)
    if e['type']=='turn_in_place':e['type']='blocked'
    return e


@pytest.mark.parametrize('auto,seed',[(True,0),(False,0),(False,1),(False,2),(False,3),(False,4)])
def test_real_js_python_trajectory_parity(auto,seed):
    actions=np.random.default_rng(seed).choice(['UP','DOWN','LEFT','RIGHT','INTERACT','WAIT'],120).tolist()
    js=js_trajectory(actions,auto)
    env=CooperativeKitchen(KitchenConfig(horizon=120))
    for player_action,expected in zip(actions,js):
        ai=program_decision(env,'ai')
        assert ai==expected['decisions']['ai']
        human=program_decision(env,'human') if auto else None
        if auto:assert human==expected['decisions']['human']
        result=env.step({'human':human['action'] if auto else player_action,'ai':ai['action']})
        view=env.public_view()
        assert {k:view[k] for k in BASE_KEYS}==expected['state']
        assert [normalize_event(e) for e in result['events']]==expected['events']
