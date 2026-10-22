"""Fixed admission reuse checks every byte-bound asset and source again."""
from copy import deepcopy
import pytest
from experiments.warehouse import v642_training_admission as a

@pytest.fixture
def ready(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('# fixed synthetic source\n')
    asset=tmp_path/'asset.json';asset.write_text('{"synthetic":true}')
    sources={'source.py':a.v.file_sha(source)};recipe=dict(synthetic_only=True)
    monkeypatch.setattr(a,'source_files',lambda:dict(sources));monkeypatch.setattr(a,'recipe',lambda:dict(recipe))
    monkeypatch.setattr(a,'_READINESS_CACHE',{});monkeypatch.setattr(a.v,'OUTPUT',tmp_path/'candidate')
    refs=(a.v.pointer(asset),);value=dict(source_files=dict(sources),recipe=dict(recipe),asset_manifest_sha256=a.v.digest(refs),prerequisites_passed=True)
    r=a.TrainingReadiness(value,refs,a.v.digest(value));calls=[]
    def build(path):calls.append(path);return r.verify()
    monkeypatch.setattr(a,'_load_readiness_uncached',build)
    return r,calls,asset,sources,recipe


def test_math_once_but_bound_inputs_verified_on_every_reuse(ready):
    r,calls,asset,*_=ready
    assert a.load_readiness('synthetic-reference') is r
    assert a.load_readiness('synthetic-reference') is r and calls==['synthetic-reference']
    asset.write_text('{"synthetic":false}')
    with pytest.raises(RuntimeError,match='asset changed'):a.load_readiness('synthetic-reference')
    assert calls==['synthetic-reference']


def test_source_change_and_mutated_result_fail_without_recomputing(ready):
    r,calls,asset,sources,recipe=ready;a.load_readiness('one')
    sources['source.py']='0'*64
    with pytest.raises(RuntimeError):a.load_readiness('one')
    assert calls==['one']


def test_different_candidate_or_reference_cannot_share_cache(ready):
    r,calls,*_=ready;a.load_readiness('one');a.load_readiness('two')
    a.v.OUTPUT=a.v.OUTPUT.parent/'other-candidate';a.load_readiness('one')
    assert calls==['one','two','one']
