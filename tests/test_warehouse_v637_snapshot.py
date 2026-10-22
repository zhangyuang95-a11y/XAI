from pathlib import Path
import hashlib
import pytest
from scripts.run_warehouse_v637 import snapshot_source

def test_identical_source_aliases_are_verified_without_rewriting(tmp_path):
    source=tmp_path/'source.py';source.write_bytes(b'constant=1\n');sha=hashlib.sha256(source.read_bytes()).hexdigest()
    target=tmp_path/'snapshots/source.py';snapshot_source(source,target,sha);snapshot_source(source,target,sha)
    assert target.read_bytes()==source.read_bytes() and target.stat().st_mode&0o222==0
    source.write_bytes(b'constant=2\n');changed=hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError,match='collides'):snapshot_source(source,target,changed)
