"""Read-only source/artifact check for public monorepo or legacy private builds."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from package_release import ALLOWED, selected_files, sha256


def verify(directory=None, *, descriptor=None):
    public_descriptor = ROOT/'deployment/cooperative_kitchen/release.json'
    if descriptor is None and directory is None and public_descriptor.exists(): descriptor = public_descriptor
    document = None
    if descriptor is not None:
        from materialize_release import load_descriptor, no_symlinks, require, verify_materialized
        document = load_descriptor(descriptor)
        target = ROOT/document['release_path']
        if directory is not None: require(Path(directory).resolve() == target, 'verification_output_mismatch')
        if os.environ.get('KITCHEN_OUTPUT'):
            configured = Path(os.environ['KITCHEN_OUTPUT'])
            if not configured.is_absolute(): configured = ROOT/configured
            require(no_symlinks(configured) == target, 'configured_output_mismatch')
        directory = target
        verify_materialized(directory, document)
    directory = Path(directory or os.environ.get('KITCHEN_OUTPUT', ROOT/'output/cooperative_kitchen/v3-id-pilot')).resolve()
    if not directory.is_relative_to(ROOT): raise ValueError('KITCHEN_OUTPUT must stay inside the deployment root.')
    manifest = json.loads((directory/'manifest.json').read_text())
    files = selected_files(directory, manifest)
    if set(kind for kind, _, _ in files) - {'manifest'} != set(ALLOWED):
        raise ValueError('The deployment must contain all twelve selected release artifacts.')
    from backend.cooperative_kitchen.artifacts import runtime_hash
    actual = runtime_hash()
    if manifest.get('runtime_sha256') != actual: raise ValueError('Deployment source does not match the release runtime hash.')
    if document is not None and document['runtime_sha256'] != actual: raise ValueError('Public descriptor runtime mismatch.')
    audit_path = ROOT/'DEPLOYMENT_AUDIT.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for name, digest in audit['files'].items():
            path = ROOT/name
            if path.is_symlink() or not path.is_file() or sha256(path) != digest: raise ValueError('Deployment audit mismatch: '+name)
    return {'passed': True, 'runtime_sha256': actual, 'manifest_sha256': sha256(directory/'manifest.json'),
        'release_status': manifest.get('status'), 'runtime_artifacts': len(files),
        'recruitment_gate_changed': False, 'credentials_printed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=None, help='Explicit legacy release path without a monorepo descriptor.')
    parser.add_argument('--descriptor', type=Path, default=None)
    args = parser.parse_args()
    try: print(json.dumps(verify(args.release, descriptor=args.descriptor), indent=2))
    except (ValueError, OSError, KeyError, TypeError):
        parser.exit(1, 'Kitchen deployment verification failed; no private contents were logged.\n')
