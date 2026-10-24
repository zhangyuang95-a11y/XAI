"""Actual preview Actors and HTTP isolation; not scientific model evaluation."""
from hashlib import sha256
import ast
import json
from pathlib import Path
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from runtime_v642.contracts import RuntimeUnavailable
from ui.model_preview_server import Preview, make_server, ROOT


@pytest.fixture
def preview(tmp_path):
    result = Preview(tmp_path / 'manual.sqlite3')
    yield result
    result.close()


@pytest.mark.parametrize('domain,seed', [('warehouse',40786),('pong',880101),('kitchen',1000)])
@pytest.mark.parametrize('variant',['raw','regularized'])
@pytest.mark.parametrize('task',[1,2,3])
def test_actual_pair_demo_takeover_restore_replay(preview,domain,seed,variant,task):
    result = preview.start(dict(domain=domain,variant=variant,seed=seed,task=task))
    token = result['session_token']
    assert result['state']['turn'] == 0
    binding = preview.catalog.binding(domain,variant)
    adapter = preview.adapters[domain]
    for index,command in enumerate((None,'wait','left')):
        _,before,_,_ = preview._session(token)
        expected = adapter.decide(before,binding)
        payload = dict(command_id='actual-step-'+str(index),expected_turn=index,human_action=command)
        actual = preview.step(token,payload)
        assert actual['last_action']['ai'] == expected['action']
        assert actual['last_action']['ai_action_overrides'] == 0
        assert actual['state']['turn'] == index+1
        assert actual['binding'] == binding.to_dict()
        assert preview.step(token,payload) == actual  # Retry must not advance physics.
        assert preview.view(token)['state'] == actual['state']
        forged = dict(payload,human_action='right')
        with pytest.raises(RuntimeUnavailable):
            preview.step(token,forged)
    replay = preview.export(token)
    assert replay['independently_replayed'] is True
    assert replay['replay']['final_state']['turn'] == 3
    assert replay['human_study_sample'] is False


def test_exact_versions_hashes_and_no_pilot(preview):
    meta = preview.metadata()
    assert meta['pilot_opened'] is meta['study_ready'] is False
    assert meta['database_is_participant_database'] is False
    assert meta['models']['warehouse']['acceptance'] == 'calibration_pair_failed'
    expected = {'raw':'ca7bfddfca1eb89d3c73935cd22d3828600009053e80663cca58e66de9703677',
                'regularized':'2ff0ed2778ecc14b3cd170aef4eb0410ab6a30382e9109b94fb0531874c93ab1'}
    for variant,key in expected.items():
        assert preview.catalog.binding('warehouse',variant).actor_sha256 == key
    for domain in ('pong','kitchen'):
        assert meta['models'][domain]['acceptance'] == 'historical_v5_passed'
        for variant in ('raw','regularized'):
            assert preview.catalog.binding(domain,variant).policy_id == f'{domain}-{variant}-v5'
    with pytest.raises(RuntimeUnavailable):
        preview.catalog.require_pilot()


def test_unknown_seed_wrong_turn_and_missing_session_rejected(preview):
    with pytest.raises(RuntimeUnavailable):
        preview.start(dict(domain='kitchen',variant='raw',task=1,seed=50000))
    with pytest.raises(RuntimeUnavailable):
        preview.view('x'*43)
    result=preview.start(dict(domain='warehouse',variant='raw',task=1,seed=40786))
    with pytest.raises(RuntimeUnavailable):
        preview.step(result['session_token'],dict(command_id='stale-cmd',expected_turn=7,human_action='wait'))


def test_corrupt_asset_fails_before_actor_execution(tmp_path):
    from runtime_v642.catalog import AssetResolver
    key='a'*64
    (tmp_path/'bad.json').write_text('{}')
    (tmp_path/'bad.json').chmod(0o444)
    resolver=AssetResolver(tmp_path,{key:dict(path='bad.json',dependencies=[])})
    with pytest.raises(RuntimeUnavailable,match='hash mismatch'):
        resolver.path(key)


def test_frozen_warehouse_runtime_bodies_are_unchanged():
    def definitions(path, names):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse((ROOT/path).read_text()).body
                if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names}
    assert definitions('ui/model_preview_warehouse.py',{'WarehouseAdapter','_history','_seed'}) == definitions(
        'runtime_v642/warehouse.py',{'WarehouseAdapter','_history','_seed'})
    names={'encode','decode','capture_environment','restore_environment'}
    assert definitions('ui/model_preview_snapshot.py',names) == definitions('backend/training/warehouse_v629_environment_replay.py',names)
    names={'_plain','_events','_actor_view','_orders'}
    assert definitions('ui/model_preview_views.py',names) == definitions('domains/warehouse/turnbased.py',names)


@pytest.mark.parametrize('variant',['raw','regularized'])
def test_warehouse_numpy_adapter_matches_original_complete_frames(preview,variant):
    from runtime_v642.warehouse import WarehouseAdapter as Original
    original=Original(preview.catalog)
    carried=preview.adapters['warehouse']
    binding=preview.catalog.binding('warehouse',variant)
    state=original.initial_state(40786,2,binding)
    assert state == carried.initial_state(40786,2,binding)
    for command in (None,'left','wait',None,'up','right',None,'down'):
        assert original.decide(state,binding) == carried.decide(state,binding)
        expected=original._advance(state,binding,command)
        actual=carried._advance(state,binding,command)
        assert expected == actual
        state=actual[0]


def test_production_runtime_never_imports_torch():
    code='''
import sys
class DenyTorch:
    def find_spec(self,name,*args):
        if name == 'torch' or name.startswith('torch.'):
            raise RuntimeError('preview tried to import training runtime')
sys.meta_path.insert(0,DenyTorch())
from ui.model_preview_server import Preview
p=Preview(':memory:')
assert 'torch' not in sys.modules
p.close()
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_http_preview_isolated_from_pilot_and_private_sessions(preview):
    server=make_server(preview)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    try:
        for path in ('/','/warehouse/','/pong/','/kitchen/'):
            with urlopen(base+path) as response:
                assert response.status == 200
                assert '模型试玩' in response.read().decode()
        with urlopen(base+'/api/release') as response:
            meta=json.load(response);assert meta['pilot_opened'] is False
        with pytest.raises(HTTPError) as error:
            urlopen(base+'/manual-api/session')
        assert error.value.code == 409
        for path in ('/prolific/','/api/start','/etc/secrets','/preview-assets/../model_preview_server.py'):
            with pytest.raises(HTTPError) as error:
                urlopen(base+path)
            assert error.value.code == 404
        data=json.dumps(dict(domain='pong',variant='raw',task=1,seed=880101)).encode()
        req=Request(base+'/manual-api/start',data=data,headers={'Content-Type':'application/json','Origin':'https://other.example'})
        with pytest.raises(HTTPError):
            urlopen(req)
        with urlopen(Request(base+'/manual-api/start',data=data,headers={'Content-Type':'application/json'})) as response:
            result=json.load(response)
        with urlopen(Request(base+'/manual-api/session',headers={'Authorization':'Bearer '+result['session_token']})) as response:
            assert json.load(response)['state']['turn'] == 0
    finally:
        server.shutdown();server.server_close();thread.join()
