"""Local-only full-physics browser fixture; never a recruitment server.

Uses the selected real Actor, program, eight-question bank and six scenarios.
Only release gates and the first A/B block ordering are fixed, explicitly within
namespace=test and a new PostgreSQL schema. No dynamics, episode limits or
participant APIs are patched; the browser performs the first ID registration.
The schema is removed when this process exits normally or receives Ctrl-C.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def configure_test_assignments(study):
    """Seed one test-only block without registering IDs or minting cookies."""
    if not study.test_mode or study.store.namespace != 'test':
        raise ValueError('Fixed assignments are restricted to an isolated test namespace.')
    from sqlalchemy import insert, select
    from ui.cooperative_kitchen_store import blocks, participants, runs, encode
    with study.store.transaction() as db:
        study.store.namespace_lock(db)
        if db.execute(select(participants.c.id).where(participants.c.namespace=='test')).first() or db.execute(
            select(runs.c.id).where(runs.c.namespace=='test')).first():
            raise ValueError('Browser fixture requires a fresh schema without participants or runs.')
        if db.execute(select(blocks.c.block_index).where(blocks.c.namespace=='test')).first():
            raise ValueError('Browser fixture requires an unused assignment block.')
        db.execute(insert(blocks).values(namespace='test', block_index=0,
            cells=encode([['A','XY'],['B','YX'],['A','YX'],['B','XY']])))
    suffix = uuid.uuid4().hex[:8]
    # The full browser runner joins A then B. Remaining two cells stay available
    # for dedicated assignment tests; production assignment remains random.
    return {'A': 'fixture_a_' + suffix, 'B': 'fixture_b_' + suffix}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isolated-test-only', action='store_true', required=True)
    parser.add_argument('--release', type=Path, default=ROOT/'output/cooperative_kitchen/v3-id-pilot')
    parser.add_argument('--metadata', type=Path, default=ROOT/'output/cooperative_kitchen/v3-id-pilot/browser-full/fixture.json')
    parser.add_argument('--port', type=int, default=8006)
    args = parser.parse_args()
    from sqlalchemy import create_engine, text, select
    from sqlalchemy.engine import make_url
    import uvicorn
    from backend.cooperative_kitchen.artifacts import load_release
    from backend.cooperative_kitchen.policy import NumpyKitchenPolicy
    from backend.cooperative_kitchen.explanations import ExplanationEngine
    from backend.cooperative_kitchen.llm import KitchenLLMClient
    from backend.cooperative_kitchen.study import KitchenStudy
    from ui.cooperative_kitchen_server import create_app
    from ui.cooperative_kitchen_store import episodes, frames, runs

    dsn = os.environ.get('KITCHEN_TEST_DATABASE_URL', '')
    url = make_url(dsn)
    if url.get_backend_name() != 'postgresql': parser.error('KITCHEN_TEST_DATABASE_URL must select PostgreSQL.')
    host = url.host or url.query.get('host', '')
    if host not in {'127.0.0.1', 'localhost', '::1'} and not str(host).startswith('/tmp/'):
        parser.error('This fixture only accepts a local PostgreSQL server.')
    release = load_release(args.release)
    if not release.get('actor_path') or not release.get('program_path') or len(release.get('question_bank', [])) != 8:
        parser.error('A hashed manifest-selected Actor, program and eight-item bank are required.')
    if sorted(map(len, release.get('scenarios', {}).values())) != [3, 3]:
        parser.error('The release must contain two groups of three real scenarios.')
    policy = NumpyKitchenPolicy(release['actor_path'])
    if not policy.trained: parser.error('The Actor must be an actual trained artifact.')
    original_gates = release.get('missing_configuration', [])
    release['study_ready'] = True  # Explicit test-only injection, never written to manifest.
    explainer = ExplanationEngine(policy, release['program_path'], client=KitchenLLMClient(api_key=''),
        extraction_report=release.get('reports', {}).get('extraction', {}))
    schema = 'kitchen_browser_' + uuid.uuid4().hex
    admin = create_engine(dsn)
    study = None
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'schema': 'cooperative_kitchen_full_browser_fixture_v1', 'test_only': True,
        'url': f'http://127.0.0.1:{args.port}', 'postgres_schema': schema,
        'versions': release['versions'], 'release_gate_bypass': True,
        'original_missing_gates': original_gates, 'cloud_model_validation': False,
        'dynamics_patched': False, 'max_steps': 180, 'target_orders': 2,
        'enrollment_mode': 'internal_pilot', 'assignment_block_fixture': True,
        'browser_registers_unused_ids': True, 'participant_id_by_condition': {}, 'runs': {}}
    try:
        with admin.begin() as db: db.execute(text(f'CREATE SCHEMA "{schema}"'))
        isolated = url.update_query_dict({'options': f'-csearch_path={schema}'}).render_as_string(hide_password=False)
        study = KitchenStudy(args.release, isolated, namespace='test', test_mode=True,
            policy=policy, explainer=explainer, release=release, enrollment_mode='internal_pilot',
            qa_limits={'min_interval_seconds': 0})
        metadata['participant_id_by_condition'] = configure_test_assignments(study)
        args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
        args.metadata.chmod(0o600)
        print(json.dumps({'ready': True, 'url': metadata['url'], 'metadata': str(args.metadata), 'test_only': True}), flush=True)
        uvicorn.run(create_app(study, admin_key='isolated-browser-fixture-only', secure_cookie=False), host='127.0.0.1', port=args.port, access_log=False)
    finally:
        try:
            if study:
                study.stop_workers()
                audit = {'schema': 'cooperative_kitchen_full_browser_database_audit_v1', 'test_only': True, 'conditions': {}}
                with study.store.transaction() as db:
                    all_runs = [json.loads(row[0]) for row in db.execute(select(runs.c.document).where(runs.c.namespace=='test'))]
                    for condition, participant_id in metadata['participant_id_by_condition'].items():
                        matching = [run for run in all_runs if run['participant_id']==participant_id]
                        if not matching:
                            audit['conditions'][condition] = {'registered': False}
                            continue
                        if len(matching) != 1 or matching[0]['condition'] != condition:
                            raise AssertionError('Actual assignment did not match the isolated test block.')
                        metadata['runs'][condition] = {'id': matching[0]['id'], 'task_order': matching[0]['task_order']}
                    for condition, info in metadata['runs'].items():
                        run = study.store.run_by_id(db, info['id'])
                        eps = [json.loads(row[0]) for row in db.execute(select(episodes.c.document).where(episodes.c.run_id==info['id']).order_by(episodes.c.episode_index))]
                        audit['conditions'][condition] = {'registered': True, 'participant_id': run['participant_id'],
                            'condition': run['condition'], 'task_order': run['task_order'], 'phase': run['phase'], 'episodes': [
                            {'id': ep['id'], 'index': ep['index'], 'phase': ep['phase'], 'done': ep['done'], 'summary': ep['summary'],
                             'saved_frames': len(list(db.execute(select(frames.c.turn).where(frames.c.episode_id==ep['id']))))} for ep in eps]}
                args.metadata.with_name('database_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
                args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
        finally:
            if study: study.store.engine.dispose()
            try:
                with admin.begin() as db: db.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            finally: admin.dispose()


if __name__ == '__main__': main()
