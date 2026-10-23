"""Experimental, byte-identical APFS clone + changed-block NPZ writer.

No pipeline calls this helper. Inputs remain immutable; the only publication
is a new path. Kernel clone failure is fatal and has no ordinary-copy fallback.
"""
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import ctypes
import fcntl
import os
import stat
import struct
import sys
import uuid

import numpy as np

VERSION = 'warehouse-v642-APFS-byte-identical-NPZ-clone.v1'
CLONE_NOFOLLOW_ANY = 0x0008
F_LOG2PHYS_EXT = 65
DEFAULT_BLOCK_BYTES = 65536


class CloneUnavailable(RuntimeError):
    pass


def require(ok, message):
    if not ok: raise CloneUnavailable(message)


class _StatFS(ctypes.Structure):
    _fields_ = [('bsize', ctypes.c_uint32), ('iosize', ctypes.c_int32),
        *[(k, ctypes.c_uint64) for k in ('blocks', 'bfree', 'bavail', 'files', 'ffree')],
        ('fsid', ctypes.c_int32 * 2),
        *[(k, ctypes.c_uint32) for k in ('owner', 'type', 'flags', 'subtype')],
        ('fstype', ctypes.c_char * 16), ('mnton', ctypes.c_char * 1024), ('mntfrom', ctypes.c_char * 1024),
        ('flags_ext', ctypes.c_uint32), ('reserved', ctypes.c_uint32 * 7)]


def _system():
    require(sys.platform == 'darwin', 'APFS clonefile is only supported on Darwin')
    try:
        lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        lib.clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32)
        lib.clonefile.restype = ctypes.c_int
        lib.fstatfs.argtypes = (ctypes.c_int, ctypes.POINTER(_StatFS))
        lib.fstatfs.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError) as exc:
        raise CloneUnavailable('native clonefile/fstatfs unavailable; copying is forbidden') from exc


def _filesystem(lib, fd):
    info = _StatFS()
    require(lib.fstatfs(fd, ctypes.byref(info)) == 0, 'fstatfs failed')
    return bytes(info.fstype).decode('ascii')


def _path(path, *, exists=True):
    path = Path(os.path.abspath(os.fspath(path)))
    for part in (path, *path.parents):
        if os.path.lexists(part): require(not stat.S_ISLNK(os.lstat(part).st_mode), 'symlink paths are forbidden')
    require(not exists or path.is_file(), 'immutable input is missing')
    return path


def _fingerprint(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink)


def _sha(fd):
    digest = sha256(); offset = 0
    while True:
        chunk = os.pread(fd, 1024 * 1024, offset)
        if not chunk: break
        digest.update(chunk); offset += len(chunk)
    return digest.hexdigest()


def _open_source(stack, ref):
    require(type(ref) is dict and set(ref) == {'path', 'sha256'} and type(ref['sha256']) is str
        and len(ref['sha256']) == 64, 'explicit immutable input SHA pointer required')
    path = _path(ref['path']); fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    stack.callback(os.close, fd); before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode) and not before.st_mode & 0o222, 'clone inputs must be read-only regular files')
    require(_sha(fd) == ref['sha256'], 'clone input SHA mismatch')
    return path, fd, before


def _unchanged(path, fd, before, expected_sha):
    require(_fingerprint(os.fstat(fd)) == _fingerprint(before)
        and _fingerprint(os.stat(path, follow_symlinks=False)) == _fingerprint(before)
        and _sha(fd) == expected_sha, 'immutable source bytes, mode or identity changed')


def extent_map(fd):
    """Darwin physical extent observations; no writes or device access."""
    size, offset, result = os.fstat(fd).st_size, 0, []
    while offset < size:
        try:
            raw = fcntl.fcntl(fd, F_LOG2PHYS_EXT, struct.pack('=Iqq', 0, size - offset, offset))
            flags, length, device = struct.unpack('=Iqq', raw)
        except OSError as exc:
            raise CloneUnavailable('cannot verify actual APFS extent sharing') from exc
        length = min(length, size - offset)
        require(length > 0 and device >= 0 and flags == 0, 'unknown or sparse APFS physical extent')
        result.append(dict(offset=offset, length=length, device_offset=device)); offset += length
        require(len(result) <= 1000000, 'physical extent enumeration exceeded bound')
    return result


def shared_extent_bytes(left, right):
    shared, i, j = 0, 0, 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        start, end = max(a['offset'], b['offset']), min(a['offset'] + a['length'], b['offset'] + b['length'])
        if end > start and a['device_offset'] - a['offset'] == b['device_offset'] - b['offset']:
            shared += end - start
        if a['offset'] + a['length'] <= b['offset'] + b['length']: i += 1
        else: j += 1
    return shared


def _npz_equal(left, right):
    checked = []
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(a.files == b.files, 'NPZ array inventory differs')
        for name in a.files:
            x, y = a[name], b[name]
            require(x.dtype == y.dtype and x.shape == y.shape
                and x.tobytes(order='C') == y.tobytes(order='C'), 'NPZ array bytes differ')
            checked.append(dict(name=name, dtype=x.dtype.str, shape=list(x.shape), bitwise_equal=True))
    return checked


def clone_patch_npz(base_ref, target_ref, output_path, *, block_bytes=DEFAULT_BLOCK_BYTES):
    """Create a new immutable NPZ with target bytes and shared base extents.

    No existing destination is replaced. Intended future integration is after
    writing a fresh unsealed staging target, before its final manifest/seal;
    the caller would explicitly publish this new verified inode. This helper
    itself cannot rewrite an already bound scientific asset.
    """
    started = perf_counter(); lib = _system()
    require(type(block_bytes) is int and 4096 <= block_bytes <= 1024 * 1024 and block_bytes % 4096 == 0,
        'comparison blocks must be 4KiB aligned and between 4KiB and 1MiB')
    output = _path(output_path, exists=False)
    require(output.parent.is_dir() and not os.path.lexists(output), 'output must be a new path in an existing directory')
    temporary = output.with_name('.' + output.name + '.clone-' + uuid.uuid4().hex)
    published = created = False
    with ExitStack() as stack:
        base, base_fd, before_base = _open_source(stack, base_ref)
        target, target_fd, before_target = _open_source(stack, target_ref)
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stack.callback(os.close, parent_fd)
        parent_stat = os.fstat(parent_fd)
        require(before_base.st_dev == before_target.st_dev == parent_stat.st_dev, 'cross-filesystem clone is forbidden')
        require(all(_filesystem(lib, fd) == 'apfs' for fd in (base_fd, target_fd, parent_fd)), 'APFS is required; copying is forbidden')
        free_before = os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize
        clone_started = perf_counter()
        try:
            result = lib.clonefile(os.fsencode(base), os.fsencode(temporary), CLONE_NOFOLLOW_ANY)
            if result != 0:
                error = ctypes.get_errno()
                raise CloneUnavailable(f'kernel clonefile failed errno={error}; ordinary-copy fallback is forbidden')
            created = True
            clone_seconds = perf_counter() - clone_started
            require(temporary.is_file() and temporary.stat().st_ino not in (before_base.st_ino, before_target.st_ino),
                'clone did not produce an independent inode')
            temporary.chmod(0o600)
            out_fd = os.open(temporary, os.O_RDWR | os.O_NOFOLLOW); stack.callback(os.close, out_fd)
            offset = changed_blocks = written = skipped = 0
            patch_started = perf_counter()
            while offset < before_target.st_size:
                desired = os.pread(target_fd, min(block_bytes, before_target.st_size - offset), offset)
                require(desired, 'target ended while comparing blocks')
                previous = os.pread(base_fd, len(desired), offset)
                if desired != previous:
                    done = 0
                    while done < len(desired):
                        count = os.pwrite(out_fd, desired[done:], offset + done)
                        require(count > 0, 'changed-block write made no progress'); done += count
                    written += len(desired); changed_blocks += 1
                else: skipped += len(desired)
                offset += len(desired)
            os.ftruncate(out_fd, before_target.st_size); os.fsync(out_fd)
            patch_seconds = perf_counter() - patch_started
            require(_sha(out_fd) == target_ref['sha256'], 'patched file is not byte-identical to target NPZ')
            arrays = _npz_equal(temporary, target)
            shared = shared_extent_bytes(extent_map(base_fd), extent_map(out_fd))
            st = os.fstat(out_fd)
            _unchanged(base, base_fd, before_base, base_ref['sha256'])
            _unchanged(target, target_fd, before_target, target_ref['sha256'])
            os.fchmod(out_fd, stat.S_IMODE(before_target.st_mode)); os.fsync(out_fd)
            # link is only atomic no-overwrite publication of our new clone,
            # never a hard link to either immutable scientific input.
            os.link(temporary, output, follow_symlinks=False); published = True
            temporary.unlink(); os.fsync(parent_fd)
            free_after = os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize
            return dict(version=VERSION, base=base_ref.copy(), target=target_ref.copy(),
                output=dict(path=str(output), sha256=target_ref['sha256']), kernel='APFS clonefile',
                clone_flags=CLONE_NOFOLLOW_ANY, kernel_clone_succeeded=True, ordinary_copy_fallback=False,
                logical_bytes=st.st_size, allocated_bytes=st.st_blocks * 512,
                shared_physical_extent_bytes=shared,
                exclusive_data_allocation_upper_estimate_bytes=max(0, st.st_blocks * 512 - shared),
                allocation_measurement='F_LOG2PHYS_EXT matched device extents; st_blocks includes shared extents',
                comparison_block_bytes=block_bytes, changed_blocks=changed_blocks,
                changed_block_bytes_written=written, identical_bytes_not_written=skipped,
                target_truncated_from_bytes=before_base.st_size, target_truncated_to_bytes=before_target.st_size,
                npz_arrays=arrays, source_bytes_and_modes_unchanged=True,
                volume_free_delta_bytes=free_before - free_after,
                volume_free_delta_isolation_guaranteed=False,
                clone_seconds=clone_seconds, patch_seconds=patch_seconds, total_seconds=perf_counter() - started,
                scientific_asset_replaced=False, pipeline_integrated=False, scientific_acceptance_claimed=False)
        except BaseException:
            if created and temporary.exists(): temporary.unlink()
            if published and output.exists(): output.unlink()
            raise
