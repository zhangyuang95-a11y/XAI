"""Create a new private kitchen deployment directory from an explicit whitelist.

Never edits the source repository, release manifest, Git state or cloud service.
The output contains private model/questionnaire artifacts; it must only be pushed
to a private repository. Existing directories are never replaced or updated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from package_release import ALLOWED, selected_files, sha256

RUNTIME_TREES = {'backend/cooperative_kitchen': {'.py'}, 'env/cooperative_kitchen': {'.py'},
                 'ui/cooperative_kitchen_web': {'.js', '.html', '.css', '.svg'}}
SUPPORT = ('backend/__init__.py', 'env/__init__.py', 'ui/__init__.py',
    'ui/cooperative_kitchen_server.py', 'ui/cooperative_kitchen_store.py',
    'backend/adapters/__init__.py', 'backend/adapters/base.py',
    'core/__init__.py', 'core/program.py', 'core/policy_contracts.py', 'core/rcpd.py',
    'core/rcpd_config.py', 'core/rcpd_tree.py', 'core/policy_program_regularizer.py',
    'requirements-kitchen.txt', 'render-kitchen.yaml', '.env.kitchen.example',
    'docs/cooperative_kitchen_research.md',
    'scripts/cooperative_kitchen/package_release.py', 'scripts/cooperative_kitchen/verify_deployment.py')
SECRET_PATTERNS = (re.compile(rb'sk-[A-Za-z0-9_-]{20,}'),
    re.compile(rb'-----BEGIN(?: RSA| EC| OPENSSH)? PRIVATE KEY-----'),
    re.compile(rb'(?:postgres(?:ql)?(?:\+psycopg)?|https?)://[^/\s:\'\"]+:[^@\s\'\"]+@'))


def sources(root):
    paths = set(SUPPORT)
    for folder, suffixes in RUNTIME_TREES.items():
        paths.update(p.relative_to(root).as_posix() for p in (root/folder).rglob('*') if p.is_file() and p.suffix in suffixes)
    return sorted(paths)


def create(release, destination, *, allow_candidate=False):
    release = Path(release).resolve(); destination = Path(destination).resolve()
    if not release.is_relative_to(ROOT): raise ValueError('Release must be inside this source checkout.')
    if destination.exists(): raise ValueError('Destination exists; use a fresh private staging directory.')
    if destination.is_relative_to(ROOT): raise ValueError('Keep the independent deployment outside the source repository.')
    manifest = json.loads((release/'manifest.json').read_text())
    if manifest.get('status') not in {'pilot_ready', 'formal_ready'} and not allow_candidate:
        raise ValueError('A candidate deployment requires --allow-candidate; recruitment remains gated.')
    from backend.cooperative_kitchen.artifacts import runtime_hash
    source_runtime = runtime_hash()
    if manifest.get('runtime_sha256') != source_runtime: raise ValueError('Freeze the matching manifest before staging source.')
    artifacts = selected_files(release, manifest)
    if {kind for kind, _, _ in artifacts} - {'manifest'} != set(ALLOWED): raise ValueError('All twelve runtime artifacts are required.')
    include = {name: ROOT/name for name in sources(ROOT)}
    for _, _, file in artifacts: include[file.relative_to(ROOT).as_posix()] = file
    for name, path in include.items():
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT): raise ValueError('Unsafe source path: '+name)
        if any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(ROOT)): raise ValueError('Symlinked source directory: '+name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.kitchen-private-stage-', dir=destination.parent)); stage.chmod(0o700)
    try:
        digests = {}; total = 0
        for name, source in sorted(include.items()):
            contents = source.read_bytes()
            if any(pattern.search(contents) for pattern in SECRET_PATTERNS): raise ValueError('Potential credential detected in selected file: '+name)
            target = stage/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(contents); target.chmod(0o600)
            digests[name] = hashlib.sha256(contents).hexdigest(); total += len(contents)
        # Detect mutable source/artifact writes while the snapshot was copied.
        if any(sha256(source) != digests[name] for name, source in include.items()) or runtime_hash() != source_runtime:
            raise ValueError('Source or release changed while staging. Freeze and retry into a new directory.')
        readme = ('# PolicyLens Cooperative Kitchen — private deployment\n\n'
            'This repository contains a neural Actor, extracted program and questionnaire answer keys. '
            'Keep the repository PRIVATE. Do not upload these artifacts to a public repository or static asset host.\n\n'
            'Configure a new Render Free service with render-kitchen.yaml. The existing warehouse service is separate. '
            'Store DATABASE_URL, DEEPSEEK_API_KEY and KITCHEN_ADMIN_KEY only in Render environment variables. '
            'The DeepSeek model name is a rolling alias, not a frozen snapshot.\n\n'
            f'Release directory: `{release.relative_to(ROOT).as_posix()}`. '
            'Candidate deployments allow inspection but do not override participant recruitment gates.\n\n'
            'Build: `python -m pip install -r requirements-kitchen.txt && python scripts/cooperative_kitchen/verify_deployment.py`\n\n'
            'Start: `python -m ui.cooperative_kitchen_server --host 0.0.0.0 --port $PORT`\n')
        (stage/'README.md').write_text(readme); (stage/'README.md').chmod(0o600)
        (stage/'.gitattributes').write_text('# Preserve exact audited source/model bytes across Git checkouts.\n* -text\n')
        (stage/'.gitattributes').chmod(0o600)
        names = sorted([*include, 'README.md', '.gitignore', '.gitattributes', 'DEPLOYMENT_AUDIT.json'])
        allow = set()
        for name in names:
            p = Path(name)
            for parent in p.parents:
                if str(parent) != '.': allow.add('!'+parent.as_posix()+'/')
            allow.add('!'+name)
        (stage/'.gitignore').write_text('# Deny by default; only the audited deployment whitelist may be committed.\n*\n'+ '\n'.join(sorted(allow))+'\n')
        (stage/'.gitignore').chmod(0o600)
        for name in ('README.md', '.gitignore', '.gitattributes'): digests[name] = sha256(stage/name)
        audit = {'schema': 'cooperative_kitchen_private_deployment_v1', 'private_repository_required': True,
            'runtime_sha256': source_runtime, 'manifest_sha256': sha256(release/'manifest.json'),
            'release_path': release.relative_to(ROOT).as_posix(), 'release_status': manifest.get('status'),
            'artifact_files_including_manifest': len(artifacts), 'files': digests,
            'source_and_artifact_bytes': total, 'credential_pattern_scan_passed': True,
            'excluded': ['original .git history', 'warehouse/candy/demo code', 'real env files', 'API keys', 'databases', 'logs', 'checkpoints', 'trajectories', 'participant data'],
            'git_initialized': False, 'pushed': False, 'deployed': False}
        (stage/'DEPLOYMENT_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n'); (stage/'DEPLOYMENT_AUDIT.json').chmod(0o600)
        os.rename(stage, destination)
        return {'directory': str(destination), 'files': len(digests)+1, 'artifact_files': len(artifacts),
            'bytes': total, 'runtime_sha256': source_runtime, 'manifest_sha256': audit['manifest_sha256'],
            'private_repository_required': True, 'pushed': False, 'deployed': False}
    finally:
        if stage.exists(): shutil.rmtree(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=ROOT/'output/cooperative_kitchen/v2-deepseek')
    parser.add_argument('--destination', type=Path, default=ROOT.parent/'policylens-kitchen-study-deploy')
    parser.add_argument('--allow-candidate', action='store_true')
    args = parser.parse_args()
    try: print(json.dumps(create(args.release, args.destination, allow_candidate=args.allow_candidate), indent=2))
    except (ValueError, OSError, KeyError) as error: parser.exit(1, str(error)+'\n')
