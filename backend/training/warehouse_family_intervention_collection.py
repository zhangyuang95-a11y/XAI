"""Finite development-only intervention queries from one frozen genuine Actor.

This producer does not fit a tree, update an NN, or grant explanation eligibility.
Its receipt is intentionally different from every older trajectory collector.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import argparse
import fcntl
import json
from pathlib import Path
import re

import numpy as np

from backend import warehouse_runtime_family as registry
from backend.warehouse_shutdown_runtime import ShutdownRuntime
from backend.warehouse_family_explanation import FamilyExplainer, actor_parameter_sha256
from backend.training import warehouse_family_explanation_audit as audit
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training.warehouse_native_shutdown_feedback_run import _physical_fingerprint
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS
from ui import warehouse_family_release as release

VERSION = 'warehouse-family-development-intervention-collection.v1'
CANDIDATE = ROOT / 'output/warehouse_native/native_cycle_500k_candidate_20260909'
CANDIDATE_SHA256 = 'ae52ad4adde3c99604cf00fc631c5b2f842348286bcf5d59ad5d97ab17519bcd'
POOLS = ROOT / 'output/warehouse_native/native_cycle_500k_tree_20260909/pools.json'
POOLS_SHA256 = 'd79ff211b5cfa179436f52db52bbc6c9bdfefe05441eb09c6384e76655c8ad04'
COUNTS = {'train': 64, 'selection': 32}
FLAGS = {'qualification_evaluated': False, 'explanation_eligible': False,
         'release_ready': False, 'participant_enabled': False}


def contract():
    return {'version': VERSION, 'purpose': 'development_RCPD_training_and_selection_queries',
        'pools': COUNTS, 'source_splits': {'train': 'train', 'selection': 'extraction'},
        'partners': list(audit.PARTNERS), 'horizon': 120, 'seed': '17000+global_scenario_index',
        'anchor': 'action-before frame % 10 == 0 and robot_2 critical_groups nonempty',
        'intervention_actions': list(ACTIONS), 'branch_steps': 1,
        'base_label': 'robot_2 actual frozen NN command submitted in the base trajectory',
        'counterfactual_label': 'robot_2 next_decision queried after one isolated step; NOT submitted or executed',
        'terminal_branch_label': None, 'unchanged_or_incorrect_predictions_removed': False,
        'tree_predictions': 'diagnostic only; never labels or runtime actions',
        'physical_projection': audit.contract()['physical_projection'],
        'base_step_cap': 34560, 'counterfactual_step_cap': 17280, 'total_step_cap': 51840,
        'ppo_steps': 0, 'tree_fits': 0, 'checkpoint_decodes': 0,
        'automatic_retry': False, 'refund_allowed': False, **FLAGS}


def sources():
    result = {**release.release_sources(), **audit.execution_sources()}
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _same(a, b, reason):
    if digest(a) != digest(b): raise ValueError(reason)


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Explicit lowercase external SHA256 required')
    return value


def _root(value):
    value = Path(value).expanduser().absolute()
    if value.resolve() != value or value.is_symlink(): raise ValueError('Canonical non-symlink path required')
    return value


def _json(path, expected):
    _sha(expected)
    if path.is_symlink() or file_hash(path) != expected: raise ValueError('External artifact hash differs')
    return audit._read(path)


def _selected_pools(scenarios, original, fixture):
    if set(original) != {'train', 'selection'}: raise ValueError('Exactly two original development pools required')
    if not fixture and {k: len(v) for k, v in original.items()} != {'train': 192, 'selection': 32}:
        raise ValueError('Original 192/32 pools required')
    counts = {k: min(1, len(v)) for k, v in original.items()} if fixture else COUNTS
    selected = {k: deepcopy(original[k][:counts[k]]) for k in COUNTS}
    if any(not v for v in selected.values()): raise ValueError('Nonempty train and selection pools required')
    fingerprints = {}
    for pool, split in (('train', 'train'), ('selection', 'extraction')):
        _same(original[pool], scenarios['splits'][split][:len(original[pool])], 'Original pool prefix/order differs')
        values = [_physical_fingerprint(s, scenarios['configuration']) for s in selected[pool]]
        if len(values) != len(set(values)): raise ValueError('Repeated selected physical initial state')
        fingerprints[pool] = set(values)
        forbidden = {s['fingerprint'] for name, entries in scenarios['splits'].items()
                     if name != split for s in entries}
        if fingerprints[pool] & forbidden: raise ValueError('Development prefix overlaps another registered pool')
        if any(not s['id'].startswith(split + '_') for s in selected[pool]):
            raise ValueError('Development source IDs differ')
    if fingerprints['train'] & fingerprints['selection']: raise ValueError('Train/selection overlap')
    return selected


def _bindings(runtime, program, scenarios):
    identity = registry.verify(runtime, allow_test_fixture=runtime.test_fixture, expected_family='retained_beta197')
    return {'runtime_signature': runtime.signature, 'runtime_family': identity['family'],
        'runtime_version': identity['runtime_version'], 'actor_sha256': runtime.actor_sha256,
        'actor_parameters_sha256': actor_parameter_sha256(runtime.actor),
        'actor_metadata_sha256': digest(runtime.actor.metadata), 'protocol_sha256': runtime.protocol_sha256,
        'scenario_manifest_sha256': digest(scenarios), 'program_sha256': file_hash(program),
        'contract_sha256': digest(contract()), 'sources_sha256': digest(sources())}


def _contexts(pools, horizon):
    contexts = []; offset = 0
    for pool in ('train', 'selection'):
        for partner in audit.PARTNERS:
            for index, scene in enumerate(pools[pool]):
                contexts.append({'pool': pool, 'pool_scene_index': index, 'scenario_index': offset + index,
                    'scenario_id': scene['id'], 'initial_fingerprint': scene['fingerprint'],
                    'partner': partner, 'seed': 17000 + offset + index, 'horizon': horizon})
        offset += len(pools[pool])
    return contexts


def _production_source(candidate, expected_manifest_sha256, pools, expected_pools_sha256):
    candidate, pools = _root(candidate), _root(pools)
    if (candidate != CANDIDATE or expected_manifest_sha256 != CANDIDATE_SHA256
            or pools != POOLS or expected_pools_sha256 != POOLS_SHA256):
        raise ValueError('This collection is registered for the fixed 500k candidate and original pools')
    context = release._candidate({'directory': str(candidate), 'manifest_sha256': _sha(expected_manifest_sha256)})
    try:
        binding = context.candidate['manifest']['original_input_bindings'][str(pools)]
        if binding['sha256'] != expected_pools_sha256 or pools.stat().st_size != binding['size']:
            raise ValueError('Pool bytes are not anchored by the frozen candidate')
        original = _json(pools, expected_pools_sha256)
        return context, original
    except BaseException:
        context.close(); raise


def _write_preparation(runtime, program, scenarios, original, *, source, pool_source, output, execution_id, fixture):
    if type(execution_id) is not str or not re.fullmatch('[A-Za-z0-9_-]{3,80}', execution_id):
        raise ValueError('Unique execution ID required')
    if scenarios.get('test_fixture', False) is not fixture or runtime.test_fixture is not fixture:
        raise ValueError('Fixture identity differs')
    _same(asdict(runtime.config), scenarios['configuration'], 'Public physical configuration differs')
    horizon = runtime.config.horizon
    if (not fixture and horizon != 120) or (fixture and not 1 <= horizon <= 3):
        raise ValueError('Production horizon120 or at most three-step fixture required')
    selected = _selected_pools(scenarios, original, fixture)
    bindings = _bindings(runtime, program, scenarios)
    FamilyExplainer(program, expected_program_sha256=bindings['program_sha256'], runtime=runtime, allow_test_fixture=fixture)
    contexts = _contexts(selected, horizon)
    caps = {'base': len(contexts) * horizon, 'counterfactual': len(contexts) * ((horizon + 9)//10) * 5}
    if not fixture: _same(caps, {'base': 34560, 'counterfactual': 17280}, 'Fixed auxiliary budget differs')
    output = _root(output)
    for parent in (CANDIDATE, POOLS.parent, Path(runtime._actor_path).parent):
        if output == parent or parent in output.parents: raise ValueError('Output must be separate from immutable inputs')
    if output.exists(): raise ValueError('Preparation output already exists; never overwrite or re-prepare')
    plan = {'version': VERSION, 'contract': contract(), 'test_fixture': fixture, 'execution_id': execution_id,
        'output_identity': str(output), 'source': source, 'pool_source': pool_source,
        'bindings': bindings, 'actor_metadata': deepcopy(runtime.actor.metadata),
        'feature_names': list(runtime.actor.metadata['feature_names']), 'configuration': asdict(runtime.config),
        'pools_sha256': digest(selected), 'contexts': contexts, 'caps': caps,
        'maximum_environment_steps': sum(caps.values()), 'sources': sources(), **FLAGS}
    output.mkdir(parents=True, exist_ok=False); audit._sync(output.parent)
    for name, value in (('plan.json', plan), ('pools.json', selected), ('scenarios.json', scenarios), ('protocol.json', runtime.protocol)):
        audit._put(output/name, value)
    # A separate copy makes fixture reload and prepared-input byte checks explicit.
    for name, source_path in (('actor.npz', runtime._actor_path), ('program.json', program)):
        with (output/name).open('xb') as stream:
            stream.write(Path(source_path).read_bytes()); stream.flush(); audit.os.fsync(stream.fileno())
    with (output/'collection.lock').open('xb'): pass
    audit._sync(output)
    return {'plan': plan, 'plan_sha256': file_hash(output/'plan.json'), 'output': str(output), **FLAGS}


def prepare(candidate=CANDIDATE, *, expected_manifest_sha256=CANDIDATE_SHA256,
            pools_path=POOLS, expected_pools_sha256=POOLS_SHA256, output, execution_id):
    context, original = _production_source(candidate, expected_manifest_sha256, pools_path, expected_pools_sha256)
    try:
        return _write_preparation(context.runtime, Path(candidate)/'program.json', context.scenarios, original,
            source={'kind': 'genuine_frozen_candidate', 'directory': str(_root(candidate)), 'manifest_sha256': expected_manifest_sha256},
            pool_source={'path': str(_root(pools_path)), 'sha256': expected_pools_sha256, 'size': Path(pools_path).stat().st_size},
            output=output, execution_id=execution_id, fixture=False)
    finally: context.close()


def prepare_fixture(runtime, program, scenarios, original_pools, *, output, execution_id):
    """Explicit tiny synthetic NumPy path; unavailable in the production CLI."""
    if runtime.test_fixture is not True: raise ValueError('Synthetic runtime required')
    return _write_preparation(runtime, program, scenarios, original_pools,
        source={'kind': 'synthetic_fixture_not_production'}, pool_source={'synthetic': True},
        output=output, execution_id=execution_id, fixture=True)


class _Journal:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.state = {'version': VERSION, 'status': 'running', 'plan_sha256': file_hash(root.parent/'plan.json'),
            'reserved': {p: 0 for p in plan['caps']}, 'acknowledged': {p: 0 for p in plan['caps']}, 'pending': None}
        self.commit(self.state)

    def commit(self, state):
        audit._put(self.root/'state.json', state, replace=(self.root/'state.json').exists())
        self.state = deepcopy(state)

    def before(self, context):
        _same(audit._read(self.root/'state.json'), self.state, 'Journal changed before reservation')
        state = deepcopy(self.state); phase = context['phase']; index = sum(state['reserved'].values())
        if (state['status'] != 'running' or state['pending'] is not None or state['reserved'] != state['acknowledged']
                or state['reserved'][phase] >= self.plan['caps'][phase]
                or context['operation_id'] != f"{self.plan['execution_id']}:{phase}:{index:06d}"):
            raise ValueError('Interrupted/unconfirmed/exhausted development collection')
        reservation = {'version': VERSION, 'context': deepcopy(context), 'permanent': True, 'refund_allowed': False}
        audit._put(self.root/'reservations'/f'{index:06d}.json', reservation)
        state['reserved'][phase] += 1; state['pending'] = reservation
        self.commit(state); return True

    def after(self, completed):
        state = deepcopy(self.state); reserved = state['pending']
        if reserved is None: raise ValueError('Completed step lacks reservation')
        for key, value in reserved['context'].items(): _same(completed[key], value, 'Completed operation differs')
        index = sum(state['acknowledged'].values()); path = self.root/'steps'/f'{index:06d}.json.gz'
        if (Path(completed['record_path']) != path or file_hash(path) != completed['record_sha256']
                or completed['actual_steps'] != 1): raise ValueError('Durable raw record differs')
        raw = audit._read(path)
        _same(raw['before_sha256'], reserved['context']['before_sha256'], 'Raw before binding differs')
        _same(raw['info']['requested_actions'], reserved['context']['submitted_actions'], 'Actual NN submission differs')
        audit._put(self.root/'confirmations'/f'{index:06d}.json', {'version': VERSION, 'reservation': reserved, 'completion': completed})
        state['acknowledged'][completed['phase']] += 1; state['pending'] = None
        self.commit(state); return True


@contextmanager
def _locked(root):
    with (root/'collection.lock').open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def collect(prepared, *, expected_plan_sha256, execution_permitted=False, allow_test_fixture=False):
    if execution_permitted is not True: raise ValueError('Explicit execution permission required')
    root = _root(prepared); plan = _json(root/'plan.json', expected_plan_sha256)
    _same(plan['version'], VERSION, 'Producer version differs')
    _same(plan['output_identity'], str(root), 'Prepared output identity differs')
    _same(plan['test_fixture'], allow_test_fixture, 'Fixture scope differs')
    _same(plan['contract'], contract(), 'Development contract changed')
    _same(plan['sources'], sources(), 'Frozen source changed')
    for key, value in FLAGS.items(): _same(plan[key], value, 'Collection cannot grant eligibility')
    with _locked(root):
        output = root/'collection'
        if output.exists(): raise ValueError('Existing collection cannot be retried, even after partial failure')
        pools = audit._read(root/'pools.json'); scenarios = audit._read(root/'scenarios.json')
        _same(digest(pools), plan['pools_sha256'], 'Prepared pools differ')
        _same(digest(scenarios), plan['bindings']['scenario_manifest_sha256'], 'Prepared scenario manifest differs')
        _same(plan['configuration'], scenarios['configuration'], 'Prepared public configuration differs')
        _same(file_hash(root/'actor.npz'), plan['bindings']['actor_sha256'], 'Prepared Actor differs')
        _same(file_hash(root/'program.json'), plan['bindings']['program_sha256'], 'Prepared program differs')
        protocol = audit._read(root/'protocol.json')
        _same(digest(protocol), plan['bindings']['protocol_sha256'], 'Prepared protocol differs')
        horizon = plan['configuration']['horizon']
        _same(_contexts(pools, horizon), plan['contexts'], 'Fixed context matrix/order/seed differs')
        n = len(plan['contexts'])
        expected_caps = {'base': n*horizon, 'counterfactual': n*((horizon+9)//10)*5}
        _same(plan['caps'], expected_caps, 'Context budget differs')
        _same(plan['maximum_environment_steps'], sum(expected_caps.values()), 'Total budget differs')
        if allow_test_fixture:
            if not 1 <= horizon <= 3 or any(len(pools[k]) != 1 for k in COUNTS):
                raise ValueError('Fixture collection exceeds two short initial states')
            _same(_selected_pools(scenarios, pools, True), pools, 'Fixture pool membership differs')
        else:
            _same(expected_caps, {'base': 34560, 'counterfactual': 17280}, 'Production cap differs')
        context = None
        if allow_test_fixture:
            runtime = ShutdownRuntime(root/'actor.npz', protocol=protocol, expected_protocol_sha256=digest(protocol),
                expected_actor_sha256=plan['bindings']['actor_sha256'],
                expected_bindings={**plan['actor_metadata'], 'actor_sha256': plan['bindings']['actor_sha256']},
                allow_test_fixture=True, config=collaborative_study_config(horizon=plan['configuration']['horizon']))
        else:
            source, pool_source = plan['source'], plan['pool_source']
            context, original = _production_source(source['directory'], source['manifest_sha256'], pool_source['path'], pool_source['sha256'])
            runtime = context.runtime
            _same(_selected_pools(context.scenarios, original, False), pools, 'Frozen development prefix changed')
        try:
            _same(asdict(runtime.config), plan['configuration'], 'Actual public configuration differs')
            _same(_bindings(runtime, root/'program.json', scenarios), plan['bindings'], 'Current runtime bindings differ')
            _same(runtime.actor.metadata, plan['actor_metadata'], 'Current Actor metadata differs')
            _same(list(runtime.actor.metadata['feature_names']), plan['feature_names'], 'Observation order differs')
            program = FamilyExplainer(root/'program.json', expected_program_sha256=plan['bindings']['program_sha256'],
                runtime=runtime, allow_test_fixture=allow_test_fixture).program
            output.mkdir()
            for name in ('steps', 'acks', 'reservations', 'confirmations'): (output/name).mkdir()
            audit._sync(root); journal = _Journal(output, plan)
            meter = audit._Meter(output, plan['execution_id'], journal.before, journal.after, plan['caps'])
            episode_records = []; labels = []
            try:
                private = registry.fresh_instance(runtime, allow_test_fixture=allow_test_fixture,
                    expected_family='retained_beta197', expected_signature=runtime.signature)
                for item in plan['contexts']:
                    scene = pools[item['pool']][item['pool_scene_index']]
                    env = private.environment(scene); rng = np.random.default_rng(item['seed']); first = len(meter.entries)
                    while not env.done:
                        snapshot = env.snapshot(); before_decision = audit._decision(env, private, program, meter)
                        groups = audit._groups(snapshot)
                        player = partner_action(env, 'robot_1', item['partner'], rng)
                        _same(snapshot, env.snapshot(), 'Program partner mutated prestate')
                        base_context = {**item, 'producer_version': VERSION, 'phase': 'base', 'frame': env.state.frame,
                            'label_semantics': 'actual_frozen_NN_command_submitted'}
                        row = meter.step(env, private, program, before_decision, player, base_context)
                        labels.append(_label(meter.entries[-1], row))
                        live = digest(env.snapshot())
                        if not snapshot['state']['frame'] % 10 and groups:
                            for action in ACTIONS:
                                branch = private.from_snapshot(snapshot)
                                row = meter.step(branch, private, program, before_decision, action,
                                    {**base_context, 'phase': 'counterfactual', 'intervention_action': action,
                                     'label_semantics': 'next_frozen_NN_query_NOT_submitted_or_executed'})
                                labels.append(_label(meter.entries[-1], row))
                                _same(digest(env.snapshot()), live, 'Isolated branch mutated live state/RNG')
                        if env.state.frame > item['horizon']: raise ValueError('Declared horizon exceeded')
                    episode_records.append({**item, 'first_entry': first, 'end_entry_exclusive': len(meter.entries),
                        'base_steps': env.state.frame, 'terminal': True})
                _same(_bindings(runtime, root/'program.json', scenarios), plan['bindings'], 'Collection inputs changed')
                _same(plan['sources'], sources(), 'Collection sources changed')
                if journal.state['pending'] is not None or meter.pending is not None: raise ValueError('Unconfirmed terminal collection')
                for phase in plan['caps']:
                    _same(journal.state['acknowledged'][phase], meter.counts[phase+'_steps'], 'Actual global counts differ')
                audit._put(output/'label_index.json', labels)
                report = {'version': VERSION, 'scope': 'development_queries_not_independent_explanation_validation',
                    'episodes': len(episode_records), 'execution': meter.report(),
                    'base_labels': sum(x['field'] == 'decision' for x in labels),
                    'counterfactual_query_labels': sum(x['field'] == 'next_decision' for x in labels),
                    'terminal_branches_without_next_label': sum(x['field'] is None for x in labels),
                    'program_teacher_labels': 0, 'nn_command_overrides': 0, **FLAGS}
                audit._put(output/'report.json', report)
                manifest = {'version': VERSION, 'status': 'completed', 'plan_sha256': expected_plan_sha256,
                    'bindings': plan['bindings'], 'contexts': plan['contexts'], 'episodes': episode_records,
                    'entries': meter.entries, 'execution': meter.report(), 'test_fixture': allow_test_fixture,
                    'label_index_sha256': file_hash(output/'label_index.json'), 'report_sha256': file_hash(output/'report.json'), **FLAGS}
                audit._put(output/'manifest.json', manifest)
                done = deepcopy(journal.state); done.update(status='completed', manifest_sha256=file_hash(output/'manifest.json'))
                journal.commit(done)
                return {'manifest_sha256': done['manifest_sha256'], 'report': report, 'output': str(output), **FLAGS}
            except BaseException as error:
                failed = deepcopy(journal.state); failed['status'] = 'failed'
                try:
                    journal.commit(failed)
                    audit._put(output/'failure.json', {'version': VERSION, 'reason': str(error),
                        'execution': meter.report(), 'journal': failed, 'automatic_retry': False, **FLAGS})
                except BaseException as persistence: error.persistence_failure = str(persistence)
                raise
        finally:
            if context is not None: context.close()


def _label(entry, row):
    base = row['phase'] == 'base'
    field = 'decision' if base else 'next_decision' if row['next_decision'] is not None else None
    return {'producer_version': VERSION, 'record_path': entry['path'], 'record_sha256': entry['sha256'],
        'pool': row['pool'], 'pool_scene_index': row['pool_scene_index'], 'scenario_index': row['scenario_index'],
        'partner': row['partner'], 'source_frame': row['frame'], 'field': field, 'role': 'robot_2',
        'groups': row['groups'] if base else audit._groups(row['after']) if field else [],
        'executed': base, 'teacher': 'actual_frozen_neural_query', 'program_teacher': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--collect', action='store_true')
    parser.add_argument('--output', type=Path); parser.add_argument('--execution-id')
    parser.add_argument('--prepared', type=Path); parser.add_argument('--expected-plan-sha256')
    parser.add_argument('--execution-permitted', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare:
        if not args.output or not args.execution_id: parser.error('--prepare requires --output and --execution-id')
        value = prepare(output=args.output, execution_id=args.execution_id)
    elif args.collect:
        if not args.prepared or not args.expected_plan_sha256: parser.error('--collect requires --prepared and --expected-plan-sha256')
        value = collect(args.prepared, expected_plan_sha256=args.expected_plan_sha256, execution_permitted=args.execution_permitted)
    else:
        value = {'execute': False, 'candidate': str(CANDIDATE), 'manifest_sha256': CANDIDATE_SHA256,
            'pool_source': str(POOLS), 'pool_sha256': POOLS_SHA256, 'contract': contract()}
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__': main()
