"""Exercise the selected kitchen runtime in a clean CPU-only Python environment.

No external database, LLM call, participant session or file database is used.
--root may point at the assembled private deployment to verify import closure.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import uuid

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--release', default='output/cooperative_kitchen/v3-id-pilot')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--require-matching-manifest', action='store_true')
    args = parser.parse_args(); root = args.root.resolve(); sys.path.insert(0, str(root))
    for key in ('DATABASE_URL', 'DEEPSEEK_API_KEY', 'DASHSCOPE_API_KEY', 'KITCHEN_ADMIN_KEY'):
        os.environ.pop(key, None)
    assert importlib.util.find_spec('torch') is None, 'Run this check in an environment without PyTorch.'
    from fastapi.testclient import TestClient
    from backend.cooperative_kitchen.artifacts import load_release
    from backend.cooperative_kitchen.policy import NumpyKitchenPolicy
    from backend.cooperative_kitchen.explanations import ExplanationEngine
    from backend.cooperative_kitchen.llm import KitchenLLMClient
    from backend.cooperative_kitchen.study import KitchenStudy
    from ui.cooperative_kitchen_server import create_app
    from env.cooperative_kitchen import CooperativeKitchen

    output = root/args.release; release = load_release(output)
    matches = release['versions']['runtime_sha256'] == release['manifest'].get('runtime_sha256')
    if args.require_matching_manifest: assert matches, 'Manifest and isolated runtime hash differ.'
    policy = NumpyKitchenPolicy(release['actor_path'])
    explainer = ExplanationEngine(policy, release['program_path'], client=KitchenLLMClient(api_key=''), extraction_report=release['reports']['extraction'])
    study = KitchenStudy(output, 'sqlite:///:memory:', namespace='development', allow_sqlite=True,
                         policy=policy, explainer=explainer, release=release)
    with TestClient(create_app(study, start_workers=False, secure_cookie=False)) as client:
        status = client.get('/api/status').json(); assert status['policy_kind'] == 'neural'; assert not status['study_ready']
        assert client.get('/').status_code == 200
        response = client.post('/api/session', json={'operation_id': str(uuid.uuid4()), 'mode': 'freeplay', 'language': 'zh'})
        assert response.status_code == 200; view = response.json()
        for command, fields in [('next', {}), ('action', {'action': 'UP'}), ('action', {'action': 'INTERACT'}), ('auto_step', {})]:
            response = client.post('/api/command', json={'operation_id': str(uuid.uuid4()), 'version': view['run']['version'], 'command': command, **fields})
            assert response.status_code == 200, (command, response.status_code); view = response.json()
        assert view['state']['turn'] == 3
    env = CooperativeKitchen(); env.reset(); before = env.snapshot()
    answer = explainer.generate(before, kind='counterfactual', language='zh')
    assert answer.get('verified') is True and before == env.snapshot()
    modules = {}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] in {'backend', 'core', 'env', 'ui'} and getattr(module, '__file__', None):
            file = Path(module.__file__).resolve(); assert file.is_relative_to(root), f'Import escaped isolated source root: {name}'
            modules[name] = file.relative_to(root).as_posix()
    assert 'torch' not in sys.modules
    report = {'schema': 'cooperative_kitchen_clean_cpu_runtime_v1', 'passed': True,
        'scope': 'Local clean CPU environment, in-memory development namespace, no cloud LLM or remote load claim.',
        'source_root': str(root), 'platform': platform.platform(), 'python': platform.python_version(),
        'runtime_sha256': release['versions']['runtime_sha256'], 'runtime_matches_manifest': matches,
        'actor_sha256': policy.artifact_sha256, 'program_loaded': explainer.program is not None,
        'http_acknowledged_steps': view['state']['turn'], 'offline_counterfactual_verified': True,
        'torch_installed': False, 'torch_imported': False, 'external_database_used': False,
        'peak_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024), 2),
        'project_imports': modules}
    study.store.engine.dispose(); args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='project_imports'}, indent=2))


if __name__ == '__main__': main()
