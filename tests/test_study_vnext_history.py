"""Synthetic call-count/integrity tests; no real models or scientific scenes."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import json
from dataclasses import FrozenInstanceError

import pytest

from runtime_v642 import RuntimeRegistry, RuntimeUnavailable
from runtime_v642.contracts import digest
from runtime_v642.store import Store as CoreStore, _all, _one, encode
from study_vnext.history import HistoryCertificates, full_history_verification, raw_row_fingerprint
from study_vnext.store import StudyStore, CONSENT_VERSION


@pytest.fixture
def prepared(tmp_path):
    spec = importlib.util.spec_from_file_location('vnext_history_toy_core',
        Path(__file__).with_name('test_runtime_v642_store.py'))
    helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    class LongAdapter(helper.FakeAdapter):
        def wrap(self, seed, task, turn, binding, total=0):
            value = super().wrap(seed, task, turn, binding, total)
            value['max_turns'] = 12;value['terminal'] = turn == 12
            value['state_sha256'] = digest({k:v for k,v in value.items() if k != 'state_sha256'})
            return value
    package, _, catalog = helper.package_at(tmp_path/'package')
    adapters = {d:LongAdapter(d) for d in ('warehouse','pong','kitchen')}
    store = StudyStore(tmp_path/'history.sqlite',RuntimeRegistry(catalog,adapters=adapters),
        scenario_pools={d:(11,) for d in adapters},demonstration_seeds={d:10 for d in adapters},mode='preview')
    enrolled = store.enroll(domain='warehouse',participant_reference='synthetic-history',
        group='A',admin=True,consent=True,consent_version=CONSENT_VERSION)
    token, instance = enrolled['session_token'],enrolled['instance']['id']
    store.complete_preparation(token,instance,kind='controls',
        observed_codes=['KeyW','KeyA','KeyS','KeyD','Space'])
    demo = store.demonstration(token,instance)
    store.complete_preparation(token,instance,kind='demonstration',
        bundle_sha256=digest(demo),final_frame_count=len(demo['records']))
    store.acknowledge_demonstration(token,instance);store.start_task(token,instance)
    for turn in range(8):
        store.action(token,instance,command_id='first-'+str(turn),expected_turn=turn,human_action='wait')
    store.view(token,instance)
    yield store,token,instance,adapters['warehouse'],package
    store.close()


def run_row(store,db,instance):
    return _one(db,'SELECT * FROM v642_runs WHERE instance_id=?',(instance,))


def decide_count(adapter):
    return sum(method == 'decide' for method,_ in adapter.calls)


def drop_frame_mutation_guard(store):
    store.db.execute('DROP TRIGGER v642_frames_update_immutable')


def rewrite_valid_hash_chain(store,run, *, public_change_at):
    rows = _all(store.db,'SELECT * FROM v642_frames WHERE run_id=? ORDER BY turn',(run['id'],))
    previous = None
    for row in rows:
        frame = store._saved_frame(row)
        if row['turn'] == public_change_at:
            frame['public']['synthetic_not_a_model_evaluation'] = False
        frame['previous_frame_sha256'] = previous
        frame['frame_sha256'] = digest({k:v for k,v in frame.items() if k != 'frame_sha256'})
        store.db.execute('UPDATE v642_frames SET public_json=?,previous_frame_sha256=?,frame_sha256=? '
            'WHERE run_id=? AND turn=?',(encode(frame['public']),previous,frame['frame_sha256'],run['id'],row['turn']))
        previous = frame['frame_sha256']


def test_warm_history_matches_original_with_constant_fresh_NN_work(prepared):
    store,token,instance,adapter,_ = prepared
    adapter.calls.clear();fast = store.view(token,instance);warm_calls = decide_count(adapter)
    adapter.calls.clear()
    with full_history_verification():full = store.view(token,instance)
    full_calls = decide_count(adapter)
    assert fast == full
    assert warm_calls == 3  # live decision + incoming replay + independent step
    assert full_calls >= 25
    assert store._history.counts()['prefixes'] == 8
    # The cache contains only proof identities, never current decision/state.
    assert all(type(key) is tuple and len(key)==3 and value is None
               for key,value in store._history._prefixes.items())
    assert not any(key[1] == 8 for key in store._history._prefixes)


def test_even_historical_through_request_recomputes_real_head_and_last_transition(prepared):
    store,token,instance,adapter,_ = prepared
    called=[];original=adapter.decide
    def decide(state,binding):
        called.append(state['turn']);return original(state,binding)
    adapter.decide=decide
    frame=store.frame(token,instance,task=1,turn=0)
    assert frame['turn']==0
    assert set(called)=={0,7,8}
    assert called.count(8)==1 and called.count(7)==2


@pytest.mark.parametrize('location',[0,3,7,8])
def test_modified_prefix_with_recomputed_hashes_is_rejected_by_original_checks(prepared,monkeypatch,location):
    store,token,instance,_,_ = prepared
    row=run_row(store,store.db,instance)
    calls=[];original=CoreStore._verified_run
    def verify(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(CoreStore,'_verified_run',verify)
    drop_frame_mutation_guard(store)
    rewrite_valid_hash_chain(store,row,public_change_at=location)
    before=store._history.counts()
    with pytest.raises(RuntimeUnavailable,match='saved state/public/NN decision differs'):
        store.view(token,instance)
    if location < 8:assert calls  # changed certified prefix forces full verifier
    assert store._history.counts()==before


def test_raw_bytes_change_misses_certificate_even_when_decoded_value_is_identical(prepared,monkeypatch):
    store,token,instance,adapter,_ = prepared
    drop_frame_mutation_guard(store)
    row=_one(store.db,'SELECT * FROM v642_frames WHERE turn=2')
    pretty=json.dumps(json.loads(row['state_json']),indent=2)
    store.db.execute('UPDATE v642_frames SET state_json=? WHERE run_id=? AND turn=2',(pretty,row['run_id']))
    adapter.calls.clear();fast=store.view(token,instance)
    assert decide_count(adapter)>=25
    with full_history_verification():assert store.view(token,instance)==fast


def test_run_origin_change_cannot_reuse_old_prefix_and_binding_change_is_refused(prepared):
    store,token,instance,adapter,_ = prepared
    with store._transaction() as db:
        row,bound=store._instance(db,token,instance);run=run_row(store,db,instance)
        changed=dict(run,created=run['created']+1)
        adapter.calls.clear();store._verified_run(db,bound,changed)
        assert decide_count(adapter)>=25
        different=store.registry.bind('warehouse','C',mode='preview')
        with pytest.raises(RuntimeUnavailable,match='binding changed'):
            store._verified_run(db,different,run)
        with pytest.raises(RuntimeUnavailable,match='run origin differs'):
            store._verified_run(db,bound,dict(run,seed=run['seed']+1))


def test_changed_NN_asset_is_rejected_before_static_history_reuse(prepared):
    store,token,instance,adapter,package = prepared
    binding=store.registry.catalog.binding('warehouse','regularized')
    path=package.root/package.records[binding.actor_sha256]['path']
    original=path.read_bytes();path.chmod(0o644);path.write_bytes(original+b' ');path.chmod(0o444)
    adapter.calls.clear()
    try:
        with pytest.raises(RuntimeUnavailable):store.view(token,instance)
        assert decide_count(adapter)==0
    finally:
        path.chmod(0o644);path.write_bytes(original);path.chmod(0o444)


def test_changed_current_NN_output_is_never_satisfied_by_prefix_cache(prepared):
    store,token,instance,adapter,_ = prepared
    original=adapter.decide
    def wrong(state,binding):
        result=original(state,binding)
        if state['turn']==8:result['action']='forged-current-action'
        return result
    adapter.decide=wrong
    with pytest.raises(RuntimeUnavailable,match='saved state/public/NN decision differs'):
        store.view(token,instance)


def test_rollback_discards_staged_prefix_and_demo_parse_certificates(prepared):
    store,token,instance,_,_ = prepared
    store._history=HistoryCertificates()
    with pytest.raises(RuntimeError,match='deliberate rollback'):
        with store._transaction(write=True) as db:
            row,bound=store._instance(db,token,instance);run=run_row(store,db,instance)
            store._verified_run(db,bound,run)
            store._saved_demonstration(db,row,bound)
            assert store._history.counts()==dict(prefixes=0,demonstrations=0)
            raise RuntimeError('deliberate rollback')
    assert store._history.counts()==dict(prefixes=0,demonstrations=0)
    store.view(token,instance)
    assert store._history.counts()==dict(prefixes=8,demonstrations=1)


def test_demo_parse_is_exact_byte_bound_and_returned_mutation_cannot_poison_cache(prepared):
    store,token,instance,adapter,_ = prepared
    with store._transaction() as db:
        row,bound=store._instance(db,token,instance)
        _,bundle=store._saved_demonstration(db,row,bound)
        expected=deepcopy(bundle);bundle['final_state']['synthetic_counter']=999
    with store._transaction() as db:
        row,bound=store._instance(db,token,instance)
        _,again=store._saved_demonstration(db,row,bound)
        assert again==expected
        adapter.calls.clear();store._saved_demonstration(db,row,bound,replay=True)
        assert decide_count(adapter)==len(expected['records'])
    store.db.execute('DROP TRIGGER v642_demonstrations_update_immutable')
    saved=_one(store.db,'SELECT * FROM v642_demonstrations')
    broken=json.loads(saved['bundle_json']);broken['final_state']['synthetic_counter']=999
    store.db.execute('UPDATE v642_demonstrations SET bundle_json=?',(encode(broken),))
    with pytest.raises(RuntimeUnavailable,match='saved preparation demonstration differs'):
        store.view(token,instance)


def test_stage_boundary_exports_and_after_task3_use_full_original_verification(prepared):
    store,token,instance,adapter,_ = prepared
    replay=store.replay(token,instance,task=1)
    assert replay['all_AI_actions_recomputed'] is False
    adapter.calls.clear();exported=store.export_instance(token,instance,admin=True)
    assert exported['all_AI_frames_recomputed'] is True
    assert decide_count(adapter)>=25
    for turn in range(8,12):
        store.action(token,instance,command_id='end-'+str(turn),expected_turn=turn,human_action='wait')
    adapter.calls.clear();store.start_task(token,instance)
    assert decide_count(adapter)>=25
    for task in (2,3):
        if task==3:store.start_task(token,instance)
        for turn in range(12):
            store.action(token,instance,command_id=f'{task}-{turn}',expected_turn=turn,human_action='wait')
    adapter.calls.clear();store.understanding(token,instance)
    assert decide_count(adapter)>=3*25


def test_LRU_is_bounded_and_uncertain_transaction_does_not_publish():
    cache=HistoryCertificates(maximum_prefixes=2,maximum_demonstrations=1)
    with cache.transaction():cache.stage_prefixes([('run',0,'a'),('run',1,'b')])
    with cache.transaction():cache.stage_prefixes([('run',2,'c')])
    assert cache.counts()['prefixes']==2
    assert sum(cache.contains(k) for k in [('run',0,'a'),('run',1,'b'),('run',2,'c')])==2
    before=set(cache._prefixes)
    with pytest.raises(RuntimeError):
        with cache.transaction():
            cache.stage_prefixes([('run',3,'d')])
            raise RuntimeError('uncertain commit')
    assert set(cache._prefixes)==before


def test_demo_cache_count_and_total_bytes_are_bounded_including_staging(prepared):
    store,token,instance,_,_=prepared
    saved=_one(store.db,'SELECT * FROM v642_demonstrations')
    binding=store.registry.bind('warehouse','A',mode='preview').binding
    rows=[dict(saved,instance_id=f'row-{index}') for index in range(3)]
    size=max(raw_row_fingerprint(rows[0])[1],len(encode(json.loads(saved['bundle_json'])).encode('utf-8')))
    cache=HistoryCertificates(maximum_demonstrations=16,maximum_demonstration_bytes=2*size)
    with cache.transaction():
        for row in rows:
            assert cache.parsed_demonstration(row,binding)['binding']==binding.to_dict()
        assert cache._pending()['demonstration_bytes']<=2*size
        assert len(cache._pending()['demonstrations'])==2
    assert cache.counts()['demonstrations']==2
    assert cache.demonstration_bytes==2*size
    too_small=HistoryCertificates(maximum_demonstration_bytes=1)
    with too_small.transaction():
        bundle=too_small.parsed_demonstration(rows[0],binding)
        assert bundle['binding']==binding.to_dict()
        assert not too_small._pending()['demonstrations']
    assert too_small.counts()['demonstrations']==too_small.demonstration_bytes==0


def test_warm_private_view_parses_only_head_and_never_caches_head_record(prepared,monkeypatch):
    store,token,instance,_,_=prepared
    called=[];original=store._saved_frame
    def saved(row):called.append(row['turn']);return original(row)
    monkeypatch.setattr(store,'_saved_frame',saved)
    store.view(token,instance)
    assert called==[8]
    head=_one(store.db,'SELECT * FROM v642_frames WHERE turn=8')
    raw_sha,_=raw_row_fingerprint(head)
    assert not any(key[1]==raw_sha for key in store._history._frames)
    assert store._history.frame_bytes>0
    called.clear()
    with full_history_verification():store.view(token,instance)
    assert called==list(range(9))


def test_lazy_history_and_public_replay_have_no_mutable_alias_to_parse_cache(prepared):
    store,token,instance,_,_=prepared
    replay=store.replay(token,instance,task=1)
    assert type(replay['frames']) is list
    replay['frames'][3]['state']['synthetic_counter']=999
    replay['frames'][0]['decision']['action']='mutated-record'
    replay['frames'][-1]['public']['state_sha256']='changed-head'
    again=store.replay(token,instance,task=1)
    assert again['frames'][3]['state']['synthetic_counter']==3
    assert again['frames'][0]['decision']['action']!='mutated-record'
    with full_history_verification():
        full=store.replay(token,instance,task=1)
    assert full['frames']==again['frames']
    assert all(type(payload) is bytes for payload in store._history._frames.values())


def test_ordinary_preparation_reads_typed_summary_without_cloning_a_bundle(prepared,monkeypatch):
    store,token,instance,adapter,_=prepared
    with store._transaction() as db:
        row,bound=store._instance(db,token,instance)
        saved=_one(db,'SELECT * FROM v642_demonstrations')
        summary=store._history.demonstration_summary(saved,bound.binding)
        assert summary.frame_count==2 and summary.bundle_sha256==saved['bundle_sha256']
        assert not hasattr(summary,'records')
        with pytest.raises(FrozenInstanceError):summary.frame_count=999
    def forbidden(*_):raise AssertionError('ordinary preparation cloned an entire demo')
    monkeypatch.setattr(store._history,'parsed_demonstration',forbidden)
    with store._transaction() as db:
        row,bound=store._instance(db,token,instance)
        assert store._preparation(db,row,bound)['demonstration_frame_count']==2


def test_frame_parse_cache_count_and_actual_bytes_are_bounded_and_rollback_safe(prepared):
    store,_,_,_,_=prepared
    row=_one(store.db,'SELECT * FROM v642_frames WHERE turn=0')
    frame=store._saved_frame(row);source_size=raw_row_fingerprint(row)[1]
    size=max(source_size,len(encode(frame).encode('utf-8')))
    cache=HistoryCertificates(maximum_frames=16,maximum_frame_bytes=2*size)
    with cache.transaction():
        for index in range(3):cache.stage_frame(('origin',str(index)),frame,source_size)
        assert cache._pending()['frame_bytes']<=2*size
        assert len(cache._pending()['frames'])==2
    assert cache.frame_bytes==2*size and len(cache._frames)==2
    copy=cache.parsed_frame(('origin','2'));copy['state']['seed']=999
    assert cache.parsed_frame(('origin','2'))['state']['seed']!=999
    before=set(cache._frames)
    with pytest.raises(RuntimeError):
        with cache.transaction():
            cache.stage_frame(('origin','rollback'),frame,source_size)
            raise RuntimeError('rollback after successful parsing')
    assert set(cache._frames)==before
    too_small=HistoryCertificates(maximum_frame_bytes=1)
    with too_small.transaction():too_small.stage_frame(('origin','big'),frame,source_size)
    assert not too_small._frames and too_small.frame_bytes==0


@pytest.mark.parametrize('field',['run_id','turn','binding_json','binding_sha256','state_json',
    'public_json','decision_json','transition_json','previous_frame_sha256','frame_sha256','created'])
def test_streaming_fingerprint_includes_every_raw_SQL_field_and_exact_type(field):
    row=dict(run_id='r',turn=1,binding_json='{"b":1}',binding_sha256='a'*64,state_json='{"s":1}',
        public_json='{"p":1}',decision_json='{"d":1}',transition_json='{"t":1}',
        previous_frame_sha256='b'*64,frame_sha256='c'*64,created=1.)
    original=raw_row_fingerprint(row)[0];changed=dict(row)
    value=row[field];changed[field]=value+' ' if type(value) is str else value+1
    assert raw_row_fingerprint(changed)[0]!=original
    assert raw_row_fingerprint(dict(reversed(list(row.items()))))[0]==original
    assert len({raw_row_fingerprint({'x':v})[0] for v in (None,False,0,0.,'0',b'0')})==6
