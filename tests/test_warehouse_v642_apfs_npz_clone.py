"""Kernel clone storage tests; synthetic arrays are not Warehouse scenes."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import errno
import os
import stat
import sys

import numpy as np
import pytest

from experiments.warehouse import v642_apfs_npz_clone as clone


def ref(path): return dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def npzs(tmp_path):
    rng = np.random.default_rng(642)  # synthetic byte fixture, no scene/NN work
    x = rng.standard_normal((4096, 32)).astype(np.float32)
    m = np.ones((4096, 5), np.float32)
    paths = []
    for index in range(2):
        p = rng.random((4096, 5), dtype=np.float32); p /= p.sum(axis=1, keepdims=True)
        path = tmp_path / f'{index}.npz'
        np.savez_compressed(path, x=x, m=m, p=p, sampled=np.full(4096, index, np.int64))
        path.chmod(0o444); paths.append(path)
    return paths


def actual_APFS():
    if sys.platform != 'darwin': pytest.skip('real APFS clone tests require Darwin')


def test_kernel_clone_really_shares_prefix_and_arrays_are_bitwise_identical(npzs, tmp_path):
    actual_APFS(); base, target = npzs; before = [(ref(p), stat.S_IMODE(p.stat().st_mode)) for p in npzs]
    output = tmp_path / 'new.npz'
    result = clone.clone_patch_npz(ref(base), ref(target), output)
    assert ref(output)['sha256'] == ref(target)['sha256']
    assert output.read_bytes() == target.read_bytes()
    assert result['kernel_clone_succeeded'] and not result['ordinary_copy_fallback']
    assert result['shared_physical_extent_bytes'] > 300000
    assert result['changed_block_bytes_written'] < result['logical_bytes'] / 2
    assert all(a['bitwise_equal'] for a in result['npz_arrays'])
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert [(ref(p), stat.S_IMODE(p.stat().st_mode)) for p in npzs] == before
    assert output.stat().st_ino not in {p.stat().st_ino for p in npzs}


def test_same_target_uses_zero_data_writes(npzs, tmp_path):
    actual_APFS(); base = npzs[0]
    result = clone.clone_patch_npz(ref(base), ref(base), tmp_path / 'same.npz')
    assert result['changed_blocks'] == result['changed_block_bytes_written'] == 0
    assert result['shared_physical_extent_bytes'] == base.stat().st_size


@pytest.mark.parametrize('growth', [False, True])
def test_shorter_and_longer_target_truncation_are_exact(npzs, tmp_path, growth):
    actual_APFS(); path = tmp_path / 'different.npz'
    data = np.arange(150000 if growth else 100, dtype=np.uint32)
    np.savez(path, v=data); path.chmod(0o444)
    result = clone.clone_patch_npz(ref(npzs[0]), ref(path), tmp_path / 'patched.npz')
    assert result['logical_bytes'] == path.stat().st_size
    assert (tmp_path / 'patched.npz').read_bytes() == path.read_bytes()


@pytest.mark.parametrize('fault', ['source_symlink', 'parent_symlink', 'destination_exists', 'writable_source', 'wrong_hash'])
def test_unsafe_or_mutable_inputs_are_rejected(npzs, tmp_path, fault):
    actual_APFS(); base, target = npzs; output = tmp_path / 'new.npz'; base_ref = ref(base)
    if fault == 'source_symlink':
        link = tmp_path / 'source-link.npz'; link.symlink_to(base); base_ref['path'] = str(link)
    elif fault == 'parent_symlink':
        directory = tmp_path / 'real'; directory.mkdir(); link = tmp_path / 'link'; link.symlink_to(directory)
        output = link / 'new.npz'
    elif fault == 'destination_exists': output.write_bytes(b'keep')
    elif fault == 'writable_source': base.chmod(0o644)
    else: base_ref['sha256'] = 'f' * 64
    with pytest.raises(clone.CloneUnavailable): clone.clone_patch_npz(base_ref, ref(target), output)
    assert not list(tmp_path.rglob('*.clone-*'))
    if fault == 'destination_exists': assert output.read_bytes() == b'keep'


def test_unavailable_kernel_clone_does_not_fall_back_to_copy(npzs, tmp_path, monkeypatch):
    class AbsentKernelClone:
        def clonefile(self, *a):
            import ctypes
            ctypes.set_errno(errno.EOPNOTSUPP); return -1
    monkeypatch.setattr(clone, '_system', lambda: AbsentKernelClone())
    monkeypatch.setattr(clone, '_filesystem', lambda *a: 'apfs')
    output = tmp_path / 'no-copy.npz'
    with pytest.raises(clone.CloneUnavailable, match='ordinary-copy fallback is forbidden'):
        clone.clone_patch_npz(ref(npzs[0]), ref(npzs[1]), output)
    assert not output.exists() and not list(tmp_path.glob('.*clone-*'))


def test_non_APFS_is_rejected_even_if_clone_symbol_exists(npzs, tmp_path, monkeypatch):
    actual_APFS(); monkeypatch.setattr(clone, '_filesystem', lambda *a: 'hfs')
    with pytest.raises(clone.CloneUnavailable, match='APFS is required'):
        clone.clone_patch_npz(ref(npzs[0]), ref(npzs[1]), tmp_path / 'no.npz')


def test_cross_filesystem_is_rejected_before_clone(npzs, tmp_path, monkeypatch):
    actual_APFS(); native = os.fstat
    def changed_parent(fd):
        s = native(fd)
        return SimpleNamespace(st_dev=-1) if stat.S_ISDIR(s.st_mode) else s
    monkeypatch.setattr(clone.os, 'fstat', changed_parent)
    with pytest.raises(clone.CloneUnavailable, match='cross-filesystem'):
        clone.clone_patch_npz(ref(npzs[0]), ref(npzs[1]), tmp_path / 'no.npz')


def test_failed_patch_hash_verification_removes_only_new_staging_file(npzs, tmp_path, monkeypatch):
    actual_APFS(); before = [ref(p) for p in npzs]
    monkeypatch.setattr(clone.os, 'pwrite', lambda fd, data, offset: len(data))
    output = tmp_path / 'not-published.npz'
    with pytest.raises(clone.CloneUnavailable, match='not byte-identical'):
        clone.clone_patch_npz(*before, output)
    assert not output.exists() and not list(tmp_path.glob('.*clone-*'))
    assert [ref(p) for p in npzs] == before
