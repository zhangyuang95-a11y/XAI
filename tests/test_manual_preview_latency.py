"""Preview acceleration must preserve fail-closed binding and exact physics."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import shutil
import threading
from collections import Counter
from copy import deepcopy
import gzip
import json
from urllib.request import Request, urlopen

import pytest

from runtime_v642.contracts import RuntimeUnavailable
from runtime_v642.legacy_domain import PongAdapter, KitchenAdapter
from ui.model_preview_warehouse import WarehouseAdapter
from ui import model_preview_server as server


@pytest.fixture(scope='module')
def preview(tmp_path_factory):
    p=server.Preview(tmp_path_factory.mktemp('fast-preview')/'preview.sqlite3')
    yield p
    p.close()


@pytest.mark.parametrize('domain,old_class', [('warehouse',WarehouseAdapter),('pong',PongAdapter),('kitchen',KitchenAdapter)])
@pytest.mark.parametrize('variant',['raw','regularized'])
@pytest.mark.parametrize('task',[1,2,3])
def test_original_full_decisions_transition_restore_counterfactual(preview,domain,old_class,variant,task):
    binding=preview.catalog.binding(domain,variant)
    old=old_class(preview.catalog); new=preview.adapters[domain]
    state=old.initial_state(server.SEEDS[domain][0],task,binding)
    with preview.runtime.request():
        assert new.initial_state(server.SEEDS[domain][0],task,binding)==state
        for command in (None,'wait','left',None):
            assert old.decide(state,binding)==new.decide(state,binding)
            expected=old._advance(state,binding,command)
            actual=new._advance(state,binding,command)
            assert expected==actual
            state=actual[0]
        assert old.public_state(state,binding)==new.public_state(state,binding)
        assert old.counterfactual(state,['wait','left'],binding)==new.counterfactual(state,['wait','left'],binding)


def test_request_rehashes_every_file_once_and_duplicate_runs_no_nn(preview,monkeypatch):
    start=preview.start(dict(domain='warehouse',variant='raw',task=1,seed=40786)); token=start['session_token']
    counts=Counter(); original=Path.read_bytes
    def read(path):
        if path.is_relative_to(server.PACKAGE): counts[str(path)]+=1
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    payload=dict(command_id='one-full-hash-per-file',expected_turn=0,human_action=None)
    result=preview.step(token,payload)
    assert len(counts)==len(preview.catalog.assets.records)+1
    assert set(counts.values())=={1}
    counts.clear(); assert preview.step(token,payload)==result
    assert set(counts.values())=={1} and len(counts)==28
    assert preview.runtime.completed.metrics['inference']==0


@pytest.mark.parametrize('damage',['content_same_stat','missing','replacement','writable','catalog_writable','symlink','parent_symlink'])
def test_next_request_rejects_changed_package_even_after_warmup(tmp_path,monkeypatch,damage):
    package=tmp_path/'package'; shutil.copytree(server.PACKAGE,package)
    monkeypatch.setattr(server,'PACKAGE',package)
    p=server.Preview(':memory:')
    try:
        start=p.start(dict(domain='pong',variant='raw',task=1,seed=880101)); token=start['session_token']
        p.view(token)
        target=(package/'catalog.json') if damage=='catalog_writable' else p.catalog.assets.path(p.catalog.policies['pong']['raw'].actor_sha256)
        saved=target.read_bytes(); stat=target.stat()
        if damage=='missing': target.unlink()
        elif damage=='replacement':
            replacement=target.with_suffix('.replacement'); replacement.write_bytes(saved[:-1]+bytes([saved[-1]^1]));replacement.chmod(0o444);replacement.replace(target)
        elif damage in ('writable','catalog_writable'):target.chmod(0o644)
        elif damage=='symlink':
            kept=target.with_suffix('.kept');target.rename(kept);target.symlink_to(kept)
        elif damage=='parent_symlink':
            parent=target.parent;kept=parent.with_name(parent.name+'.kept');parent.rename(kept);parent.symlink_to(kept,target_is_directory=True)
        else:
            target.chmod(0o644);target.write_bytes(saved[:-1]+bytes([saved[-1]^1]));target.chmod(0o444)
            os.utime(target,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with pytest.raises(RuntimeUnavailable):p.view(token)
    finally:p.close()


def test_same_session_duplicate_is_atomic_and_export_never_holds_compute_lock(preview,monkeypatch):
    one=preview.start(dict(domain='pong',variant='raw',task=1,seed=880101))['session_token']
    two=preview.start(dict(domain='kitchen',variant='raw',task=1,seed=1000))['session_token']
    payload=dict(command_id='duplicate-concurrent',expected_turn=0,human_action='wait')
    with ThreadPoolExecutor(3) as pool:
        a=pool.submit(preview.step,one,payload); b=pool.submit(preview.step,one,payload)
        assert a.result(timeout=3)==b.result(timeout=3)
        entered,release=threading.Event(),threading.Event()
        adapter=preview.adapters['pong']; replay=adapter.replay
        def slow(*args):
            entered.set();assert release.wait(5);return replay(*args)
        monkeypatch.setattr(adapter,'replay',slow)
        export=pool.submit(preview.export,one);assert entered.wait(3)
        try:
            # Both another session and this same session advance while the
            # consistent prefix is independently replayed outside all locks.
            assert pool.submit(preview.step,two,dict(payload,command_id='other-session')).result(timeout=2)['state']['turn']==1
            assert pool.submit(preview.step,one,dict(payload,command_id='same-during-export',expected_turn=1)).result(timeout=2)['state']['turn']==2
        finally:release.set()
        result=export.result(timeout=3)
        assert result['replay']['final_state']['turn']==1


def test_restore_reuse_never_trusts_supplied_digest_or_shares_mutable_environment(preview):
    binding=preview.catalog.binding('warehouse','raw')
    adapter=preview.adapters['warehouse'];state=adapter.initial_state(40786,1,binding)
    with preview.runtime.request():
        restored=adapter.restore(state,binding);restored.state.frame+=1
        assert adapter.restore(state,binding).state.frame==0
        forged=deepcopy(state);forged['turn']=1  # retains original supplied hash
        with pytest.raises(RuntimeUnavailable):adapter.restore(forged,binding)
        adapter.decide(state,binding)
        with pytest.raises(RuntimeUnavailable):adapter.decide(forged,binding)


def test_static_json_reuses_already_hashed_bytes_inside_request(preview,monkeypatch):
    counts=Counter();original=Path.read_bytes
    def read(path):
        if path.is_relative_to(server.PACKAGE):counts[str(path)]+=1
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read)
    key='012af827e8416f293bec97670bb3aec1b752de58dc0155051e98d33c56eabc9a'
    with preview.runtime.request():
        assert preview.catalog.assets.json(key) is preview.catalog.assets.json(key)
    assert len(counts)==28 and set(counts.values())=={1}


def test_gzip_preserves_complete_response_and_timing_headers(preview):
    http=server.make_server(preview);thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:
        with urlopen(Request(f'http://127.0.0.1:{http.server_port}/api/release',headers={'Accept-Encoding':'gzip'})) as response:
            raw=response.read();assert response.headers['Content-Encoding']=='gzip'
            assert json.loads(gzip.decompress(raw))==preview.metadata()
            assert len(raw)<len(gzip.decompress(raw))*.5
            assert all(name+';dur=' in response.headers['Server-Timing'] for name in
                ('integrity','inference','physics','storage','serialization','total'))
    finally:http.shutdown();http.server_close();thread.join()
