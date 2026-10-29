"""Persistent core study flow with explicit frozen-domain dependencies.

No study_v3 engine, policy lookup, release constant or mutable legacy Store is
imported.  This API is dormant until constructed by a caller.  Software flow
completion is not scientific admission; pilot always checks the catalog.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import json
import secrets
import sqlite3
import threading
import time
import uuid

from .contracts import DOMAINS, GROUP_VARIANTS, PolicyBinding, RuntimeUnavailable, digest, require
from .registry import RuntimeRegistry

VERSION = 'runtime-v642-persistent-core-study.v2-committed-history'
FRAME_VERSION = 'runtime-v642-bound-study-frame.v1'
EXPORT_VERSION = 'runtime-v642-bound-core-export.v2-committed-history'
_GROUPS = tuple(GROUP_VARIANTS)
_TABLES = ('instances', 'demonstrations', 'runs', 'frames', 'questions', 'commands')
_SCHEMA = '''
CREATE TABLE IF NOT EXISTS v642_metadata(version TEXT PRIMARY KEY,release_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS v642_instances(
 id TEXT PRIMARY KEY,participant_key TEXT NOT NULL,token_hash TEXT NOT NULL UNIQUE,
 release_id TEXT NOT NULL,domain TEXT NOT NULL,mode TEXT NOT NULL,group_code TEXT NOT NULL,
 scenario_seed INTEGER NOT NULL,binding_json TEXT NOT NULL,binding_sha256 TEXT NOT NULL,
 stage TEXT NOT NULL,created REAL NOT NULL,UNIQUE(participant_key,domain,mode));
CREATE TABLE IF NOT EXISTS v642_demonstrations(
 instance_id TEXT PRIMARY KEY REFERENCES v642_instances(id),binding_json TEXT NOT NULL,
 binding_sha256 TEXT NOT NULL,bundle_json TEXT NOT NULL,bundle_sha256 TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS v642_runs(
 id TEXT PRIMARY KEY,instance_id TEXT NOT NULL REFERENCES v642_instances(id),task INTEGER NOT NULL,
 seed INTEGER NOT NULL,initial_state_json TEXT NOT NULL,initial_state_sha256 TEXT NOT NULL,
 state_json TEXT NOT NULL,status TEXT NOT NULL,created REAL NOT NULL,ended REAL,
 UNIQUE(instance_id,task));
CREATE TABLE IF NOT EXISTS v642_frames(
 run_id TEXT NOT NULL REFERENCES v642_runs(id),turn INTEGER NOT NULL,
 binding_json TEXT NOT NULL,binding_sha256 TEXT NOT NULL,state_json TEXT NOT NULL,
 public_json TEXT NOT NULL,decision_json TEXT NOT NULL,transition_json TEXT,
 previous_frame_sha256 TEXT,frame_sha256 TEXT NOT NULL,created REAL NOT NULL,
 PRIMARY KEY(run_id,turn));
CREATE TABLE IF NOT EXISTS v642_questions(
 id TEXT PRIMARY KEY,instance_id TEXT NOT NULL REFERENCES v642_instances(id),run_id TEXT NOT NULL,
 turn INTEGER NOT NULL,binding_json TEXT NOT NULL,binding_sha256 TEXT NOT NULL,
 question TEXT NOT NULL,language TEXT NOT NULL,result_json TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS v642_commands(
 instance_id TEXT NOT NULL REFERENCES v642_instances(id),command_id TEXT NOT NULL,
 request_sha256 TEXT NOT NULL,response_json TEXT NOT NULL,created REAL NOT NULL,
 PRIMARY KEY(instance_id,command_id));
CREATE TRIGGER IF NOT EXISTS v642_immutable_binding BEFORE UPDATE OF participant_key,token_hash,
 release_id,domain,mode,group_code,scenario_seed,binding_json,binding_sha256 ON v642_instances
 BEGIN SELECT RAISE(ABORT,'immutable v642 enrollment binding');END;
CREATE TRIGGER IF NOT EXISTS v642_immutable_run_origin BEFORE UPDATE OF instance_id,task,seed,
 initial_state_json,initial_state_sha256 ON v642_runs
 BEGIN SELECT RAISE(ABORT,'immutable v642 run origin');END;
'''


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def decode(value):
    from .catalog import decode as strict_decode
    try:
        return strict_decode(value)
    except (ValueError, TypeError) as exc:
        raise RuntimeUnavailable('persisted core JSON is malformed') from exc


class StudyError(RuntimeError):
    def __init__(self, code, status=400):
        self.code, self.status = code, status
        super().__init__(code)


def check(condition, code, status=400):
    if not condition:
        raise StudyError(code, status)


def _token_hash(token):
    check(type(token) is str and 16 <= len(token) <= 256, 'invalid_session', 401)
    return sha256(token.encode()).hexdigest()


def _seed(value):
    return type(value) is int and 0 <= value <= (2**63-4)//10


def _one(db, sql, values=()):
    row = db.execute(sql, values).fetchone()
    return dict(row) if row is not None else None


def _all(db, sql, values=()):
    return [dict(row) for row in db.execute(sql, values).fetchall()]


class Store:
    """SQLite core API; group never enters any Actor/state/adapter call."""
    def __init__(self, database, registry, *, scenario_pools, demonstration_seeds,
                 mode='pilot', choose=secrets.choice, clock=time.time):
        require(type(registry) is RuntimeRegistry and mode in ('pilot', 'preview', 'test'),
                'explicit runtime registry and fixed service mode required')
        require(set(scenario_pools) == set(demonstration_seeds) == set(DOMAINS)
            and all(type(rows) in (list, tuple) and rows and len(set(rows)) == len(rows)
                and all(_seed(seed) for seed in rows) for rows in scenario_pools.values())
            and all(_seed(seed) for seed in demonstration_seeds.values()),
            'explicit finite participant scenes and demo scenes required')
        require(callable(choose) and callable(clock), 'explicit assignment/clock dependencies required')
        self.registry, self.mode, self.choose, self.clock = registry, mode, choose, clock
        self.scenarios = {domain: tuple(rows) for domain, rows in scenario_pools.items()}
        self.demo_seeds = dict(demonstration_seeds)
        self.database = Path(database)
        require(not self.database.is_symlink(), 'new study DB cannot be symlinked')
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.database), timeout=30, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        existing = {row[0] for row in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if any(name.startswith('pl3_') for name in existing):
            self.db.close()
            raise RuntimeUnavailable('legacy database must use the read-only legacy reader')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript(_SCHEMA)
        for table in ('demonstrations', 'frames', 'questions', 'commands'):
            for operation in ('UPDATE', 'DELETE'):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS v642_{table}_{operation.lower()}_immutable "
                    f"BEFORE {operation} ON v642_{table} BEGIN SELECT RAISE(ABORT,'append-only v642 evidence');END")
        for table in ('instances', 'runs'):
            self.db.execute(f"CREATE TRIGGER IF NOT EXISTS v642_{table}_delete_immutable BEFORE DELETE ON v642_{table} "
                            "BEGIN SELECT RAISE(ABORT,'immutable v642 history');END")
        row = _one(self.db, 'SELECT * FROM v642_metadata')
        if row is None:
            self.db.execute('INSERT INTO v642_metadata VALUES(?,?)', (VERSION, registry.catalog.release_id))
        else:
            require(row == dict(version=VERSION, release_id=registry.catalog.release_id),
                    'DB belongs to another core schema or release')

    def close(self):
        with self.lock:
            self.db.close()

    @contextmanager
    def _transaction(self, *, write=False):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            try:
                yield self.db
            except BaseException:
                self.db.rollback()
                raise
            else:
                self.db.commit()

    def _gate(self):
        if self.mode == 'pilot':
            self.registry.catalog.require_pilot()
        else:
            self.registry.catalog.verify_assets()

    def status(self):
        self._gate()
        return dict(version=VERSION, release_id=self.registry.catalog.release_id, mode=self.mode,
            pilot_available=self.mode == 'pilot', core_api=True, old_UI_migrated=False,
            human_collaboration_benefit='not_measured')

    def _bound(self, row):
        require(row['mode'] == self.mode and row['release_id'] == self.registry.catalog.release_id,
                'saved runtime mode/release differs')
        saved = decode(row['binding_json'])
        require(digest(saved) == row['binding_sha256'], 'immutable enrollment binding hash differs')
        try:
            binding = PolicyBinding.from_dict(saved)
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeUnavailable('persisted policy binding is malformed') from exc
        bound = self.registry.bind(row['domain'], row['group_code'], mode=self.mode)
        require(binding == bound.verify() and binding.to_dict() == decode(row['binding_json'])
                and digest(binding.to_dict()) == row['binding_sha256'], 'immutable enrollment binding differs')
        return bound

    def _instance(self, db, token, instance_id):
        row = _one(db, 'SELECT * FROM v642_instances WHERE id=? AND token_hash=?',
                   (instance_id, _token_hash(token)))
        check(row is not None, 'instance_not_found', 404)
        return row, self._bound(row)

    def enroll(self, *, domain, participant_reference, group=None, scenario_seed=None, admin=False):
        self._gate()  # Before random assignment or any write.
        check(domain in DOMAINS and type(participant_reference) is str
              and 1 <= len(participant_reference.strip()) <= 160, 'invalid_enrollment')
        check(type(admin) is bool, 'invalid_admin_authority', 403)
        if self.mode == 'pilot':
            check(group is None and scenario_seed is None, 'pilot_assignment_is_randomized', 403)
        elif group is not None or scenario_seed is not None:
            check(admin, 'admin_required_for_assignment_override', 403)
        check(group is None or group in _GROUPS, 'invalid_group')
        check(scenario_seed is None or _seed(scenario_seed) and scenario_seed in self.scenarios[domain], 'unknown_participant_scene')
        participant = sha256(participant_reference.strip().encode()).hexdigest()
        with self._transaction(write=True) as db:
            check(_one(db, 'SELECT id FROM v642_instances WHERE participant_key=? AND domain=? AND mode=?',
                  (participant, domain, self.mode)) is None, 'participant_already_enrolled', 409)
            counts = _all(db, 'SELECT scenario_seed,group_code,COUNT(*) AS n FROM v642_instances '
                'WHERE domain=? AND mode=? GROUP BY scenario_seed,group_code', (domain, self.mode))
            count = {(row['scenario_seed'], row['group_code']): row['n'] for row in counts}
            totals = {seed: sum(count.get((seed, code), 0) for code in _GROUPS) for seed in self.scenarios[domain]}
            if scenario_seed is None:
                candidates = tuple(seed for seed in self.scenarios[domain] if totals[seed] == min(totals.values()))
                scenario_seed = self.choose(candidates)
                require(scenario_seed in candidates, 'assignment chooser violated scene balance')
            require(scenario_seed in self.scenarios[domain], 'assignment chooser returned an unregistered scene')
            if group is None:
                minimum = min(count.get((scenario_seed, code), 0) for code in _GROUPS)
                candidates = tuple(code for code in _GROUPS if count.get((scenario_seed, code), 0) == minimum)
                group = self.choose(candidates)
                require(group in candidates, 'assignment chooser violated group balance')
            require(group in _GROUPS, 'assignment chooser returned an invalid group')
            bound = self.registry.bind(domain, group, mode=self.mode)
            binding = bound.verify()
            token, instance_id = secrets.token_urlsafe(32), uuid.uuid4().hex
            db.execute('INSERT INTO v642_instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', (
                instance_id, participant, _token_hash(token), binding.release_id, domain, self.mode, group,
                scenario_seed, encode(binding.to_dict()), digest(binding.to_dict()), 'enrolled', self.clock()))
            row = _one(db, 'SELECT * FROM v642_instances WHERE id=?', (instance_id,))
            return dict(session_token=token, instance=self._identity(row))

    @staticmethod
    def _identity(row):
        return dict(id=row['id'], release_id=row['release_id'], domain=row['domain'], mode=row['mode'],
            group=row['group_code'], scenario_seed=row['scenario_seed'], stage=row['stage'],
            binding=decode(row['binding_json']), binding_sha256=row['binding_sha256'])

    @staticmethod
    def _state(bound, state):
        b = bound.verify()
        require(type(state) is dict and state.get('binding') == b.to_dict()
            and state.get('binding_identity_sha256') == b.identity_sha256
            and type(state.get('task')) is int and state['task'] in (1, 2, 3)
            and type(state.get('turn')) is int and state['turn'] >= 0
            and type(state.get('terminal')) is bool, 'runtime state is not exactly model-bound')
        try:
            bound.adapter.restore(state, b)
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeUnavailable('saved bound domain state is malformed') from exc
        return state

    def _decision(self, bound, state):
        self._state(bound, state)
        if state['terminal']:
            return {}
        decision = bound.adapter.decide(state, bound.binding)
        require(type(decision) is dict and decision.get('binding_identity_sha256') == bound.binding.identity_sha256
            and decision.get('state_sha256') == state['state_sha256']
            and decision.get('ai_action_overrides') == 0
            and decision.get('execution_source') == 'bound_frozen_NN_with_physical_mask'
            and type(decision.get('action')) is str and decision['action'], 'decision is not the bound NN action')
        return decision

    def _frame(self, bound, state, *, transition=None, previous=None):
        decision = self._decision(bound, state)
        public = bound.adapter.public_state(state, bound.binding)
        require(public.get('binding') == bound.binding.to_dict()
            and public.get('state_sha256') == state['state_sha256'], 'public frame lost its frozen binding')
        item = dict(version=FRAME_VERSION, turn=state['turn'], binding=bound.binding.to_dict(),
            binding_sha256=digest(bound.binding.to_dict()), state=deepcopy(state), public=public,
            decision=decision, transition=deepcopy(transition), previous_frame_sha256=previous)
        return dict(**item, frame_sha256=digest(item))

    @staticmethod
    def _saved_frame(row):
        item = dict(version=FRAME_VERSION, turn=row['turn'], binding=decode(row['binding_json']),
            binding_sha256=row['binding_sha256'], state=decode(row['state_json']), public=decode(row['public_json']),
            decision=decode(row['decision_json']), transition=None if row['transition_json'] is None
                else decode(row['transition_json']), previous_frame_sha256=row['previous_frame_sha256'])
        require(digest(item) == row['frame_sha256'], 'stored frame hash chain differs')
        return dict(**item, frame_sha256=row['frame_sha256'])

    def _insert_frame(self, db, run_id, frame):
        db.execute('INSERT INTO v642_frames VALUES(?,?,?,?,?,?,?,?,?,?,?)', (run_id, frame['turn'],
            encode(frame['binding']), frame['binding_sha256'], encode(frame['state']), encode(frame['public']),
            encode(frame['decision']), None if frame['transition'] is None else encode(frame['transition']),
            frame['previous_frame_sha256'], frame['frame_sha256'], self.clock()))

    def _verified_run(self, db, bound, run, *, through=None):
        require(run['task'] in (1, 2, 3), 'invalid saved task')
        rows = _all(db, 'SELECT * FROM v642_frames WHERE run_id=? ORDER BY turn', (run['id'],))
        require(rows and all(row['turn'] == i for i, row in enumerate(rows)), 'noncontiguous persisted run')
        check(through is None or type(through) is int and 0 <= through < len(rows), 'frame_not_found', 404)
        initial = decode(run['initial_state_json'])
        require(digest(initial) == run['initial_state_sha256'] and initial['turn'] == 0
            and initial['task'] == run['task']
            and initial.get('seed', initial.get('snapshot', {}).get('seed')) == run['seed'], 'run origin differs')
        previous = None
        frames = []
        for row in rows[:through+1 if through is not None else len(rows)]:
            actual = self._saved_frame(row)
            require(actual['state']['task'] == run['task'], 'task changed in persisted frame')
            expected = self._frame(bound, actual['state'], transition=actual['transition'],
                                   previous=None if previous is None else previous['frame_sha256'])
            require(actual == expected, 'saved state/public/NN decision differs from bound runtime')
            if previous is None:
                require(actual['state'] == initial and actual['transition'] is None, 'initial frame differs')
            else:
                bundle = actual['transition']
                require(type(bundle) is dict and bundle.get('binding') == bound.binding.to_dict()
                    and bundle.get('two_frozen_NN') is False and bundle['initial_state'] == previous['state']
                    and bundle['final_state'] == actual['state'] and len(bundle['records']) == 1
                    and bundle['records'][0]['decision'] == previous['decision'], 'stored human transition differs')
                record = bundle['records'][0]
                require(type(record.get('human_action')) is str, 'missing recorded human replacement')
                replayed = bound.adapter.replay(bundle, bound.binding)
                stepped = bound.adapter.step(previous['state'], record['human_action'], bound.binding,
                                             decision=previous['decision'])
                require(replayed == stepped == actual['state'], 'bound NN/physics transition replay differs')
            previous = actual
            frames.append(actual)
        if through is None:
            require(previous['state'] == decode(run['state_json'])
                and (run['status'] == 'complete') == previous['state']['terminal'], 'run head/status differs')
        return frames

    def _view(self, db, row, bound):
        runs = _all(db, 'SELECT * FROM v642_runs WHERE instance_id=? ORDER BY task', (row['id'],))
        summaries = []
        current = None
        for run in runs:
            frames = self._verified_run(db, bound, run)
            last = frames[-1]
            summaries.append(dict(id=run['id'], task=run['task'], status=run['status'], turns=last['turn']))
            current = dict(run_id=run['id'], **last)
        require([run['task'] for run in runs] == list(range(1, len(runs)+1)) and len(runs) <= 3
                and all(run['status'] == 'complete' for run in runs[:-1]), 'saved task order differs')
        if runs:
            expected = ('complete' if runs[-1]['task'] == 3 else f"task_{runs[-1]['task']}_complete") \
                if runs[-1]['status'] == 'complete' else f"task_{runs[-1]['task']}"
            require(row['stage'] == expected, 'saved stage and physical task completion differ')
        else:
            require(row['stage'] in ('enrolled', 'demo_complete'), 'saved pre-task stage differs')
            if row['stage'] == 'demo_complete':
                require(_one(db, 'SELECT instance_id FROM v642_demonstrations WHERE instance_id=?', (row['id'],)) is not None,
                        'demonstration completion has no saved bound demo')
        return dict(version=VERSION, instance=self._identity(row), runs=summaries, current=current,
            explanation_questions_allowed=row['group_code'] == 'A' and row['stage'] == 'task_2',
            old_UI_migrated=False, human_collaboration_benefit='not_measured')

    def view(self, token, instance_id):
        self._gate()
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            return self._view(db, row, bound)

    resume = view

    def demonstration(self, token, instance_id):
        self._gate()
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            saved = _one(db, 'SELECT * FROM v642_demonstrations WHERE instance_id=?', (instance_id,))
            if saved is None:
                check(row['stage'] == 'enrolled', 'demonstration_stage_required', 409)
                bundle = bound.adapter.demonstration(bound.binding, seed=self.demo_seeds[row['domain']], task=1,
                    steps=360 if row['domain'] == 'kitchen' else 120)
                require(bundle.get('binding') == bound.binding.to_dict() and bundle.get('two_frozen_NN') is True,
                        'demonstration did not use both bound NNs')
                require(bound.adapter.replay(bundle, bound.binding) == bundle['final_state'], 'demonstration replay differs')
                db.execute('INSERT INTO v642_demonstrations VALUES(?,?,?,?,?,?)', (instance_id,
                    encode(bound.binding.to_dict()), digest(bound.binding.to_dict()), encode(bundle), digest(bundle), self.clock()))
                return deepcopy(bundle)
            require(decode(saved['binding_json']) == bound.binding.to_dict()
                and saved['binding_sha256'] == digest(bound.binding.to_dict()), 'saved demo binding differs')
            bundle = decode(saved['bundle_json'])
            require(digest(bundle) == saved['bundle_sha256'] and bundle.get('two_frozen_NN') is True
                and bundle.get('binding') == bound.binding.to_dict()
                and bound.adapter.replay(bundle, bound.binding) == bundle['final_state'], 'saved NN demo differs')
            return bundle

    def acknowledge_demonstration(self, token, instance_id):
        # Revalidate outside this separate transaction; no nested DB transaction.
        self.demonstration(token, instance_id)
        self._gate()
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            if row['stage'] == 'enrolled':
                db.execute("UPDATE v642_instances SET stage='demo_complete' WHERE id=?", (instance_id,))
                row['stage'] = 'demo_complete'
            return self._view(db, row, bound)

    def start_task(self, token, instance_id):
        self._gate()
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            runs = _all(db, 'SELECT * FROM v642_runs WHERE instance_id=? ORDER BY task', (instance_id,))
            if runs:
                self._verified_run(db, bound, runs[-1])
            task = len(runs)+1
            check(task <= 3 and (row['stage'] == 'demo_complete' if task == 1 else
                row['stage'] == f'task_{task-1}_complete' and runs[-1]['status'] == 'complete'),
                'previous_stage_incomplete', 409)
            state = bound.adapter.initial_state(row['scenario_seed'], task, bound.binding)
            require(state['task'] == task and state['turn'] == 0 and not state['terminal'], 'invalid bound task origin')
            frame = self._frame(bound, state)
            run_id = uuid.uuid4().hex
            db.execute('INSERT INTO v642_runs VALUES(?,?,?,?,?,?,?,?,?,?)', (run_id, instance_id, task,
                row['scenario_seed'], encode(state), digest(state), encode(state), 'active', self.clock(), None))
            self._insert_frame(db, run_id, frame)
            row['stage'] = f'task_{task}'
            db.execute('UPDATE v642_instances SET stage=? WHERE id=?', (row['stage'], instance_id))
            return self._view(db, row, bound)

    def action(self, token, instance_id, *, command_id, expected_turn, human_action):
        self._gate()
        check(type(command_id) is str and 1 <= len(command_id) <= 100
            and type(expected_turn) is int and expected_turn >= 0 and type(human_action) is str
            and 1 <= len(human_action) <= 100, 'invalid_action_request')
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            run = _one(db, 'SELECT * FROM v642_runs WHERE instance_id=? ORDER BY task DESC LIMIT 1', (instance_id,))
            check(run is not None, 'task_not_started', 409)
            request_hash = digest(dict(run_id=run['id'], task=run['task'], expected_turn=expected_turn,
                                       human_action=human_action))
            frames = self._verified_run(db, bound, run)
            old = _one(db, 'SELECT * FROM v642_commands WHERE instance_id=? AND command_id=?', (instance_id, command_id))
            if old is not None:
                check(old['request_sha256'] == request_hash, 'command_id_reused', 409)
                return decode(old['response_json'])
            last = frames[-1]
            check(run['status'] == 'active' and row['stage'] == f"task_{run['task']}"
                  and not last['state']['terminal'], 'task_not_active', 409)
            check(last['turn'] == expected_turn, 'stale_turn', 409)
            nxt = bound.adapter.step(last['state'], human_action, bound.binding, decision=last['decision'])
            bundle = bound.adapter.counterfactual(last['state'], [human_action], bound.binding)
            require(bundle.get('binding') == bound.binding.to_dict() and bundle.get('two_frozen_NN') is False
                and bundle['initial_state'] == last['state'] and bundle['final_state'] == nxt
                and len(bundle['records']) == 1 and bundle['records'][0]['human_action'] == human_action
                and bundle['records'][0]['decision'] == last['decision']
                and bound.adapter.replay(bundle, bound.binding) == nxt,
                'human takeover changed the NN action or bound transition')
            require(nxt['turn'] == last['turn']+1 and nxt['task'] == run['task'], 'physical turn skipped or task changed')
            frame = self._frame(bound, nxt, transition=bundle, previous=last['frame_sha256'])
            self._insert_frame(db, run['id'], frame)
            complete = nxt['terminal']
            db.execute('UPDATE v642_runs SET state_json=?,status=?,ended=? WHERE id=?',
                (encode(nxt), 'complete' if complete else 'active', self.clock() if complete else None, run['id']))
            if complete:
                row['stage'] = 'complete' if run['task'] == 3 else f"task_{run['task']}_complete"
                db.execute('UPDATE v642_instances SET stage=? WHERE id=?', (row['stage'], instance_id))
            response = self._view(db, row, bound)
            db.execute('INSERT INTO v642_commands VALUES(?,?,?,?,?)',
                (instance_id, command_id, request_hash, encode(response), self.clock()))
            return response

    def replay(self, token, instance_id, *, task):
        self._gate()
        check(type(task) is int and task in (1, 2, 3), 'invalid_task')
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            run = _one(db, 'SELECT * FROM v642_runs WHERE instance_id=? AND task=?', (instance_id, task))
            check(run is not None, 'run_not_found', 404)
            return dict(version=FRAME_VERSION, binding=bound.binding.to_dict(), task=task,
                frames=self._verified_run(db, bound, run), all_AI_actions_recomputed=True)

    def frame(self, token, instance_id, *, task, turn):
        self._gate()
        check(type(task) is int and task in (1, 2, 3), 'invalid_task')
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            run = _one(db, 'SELECT * FROM v642_runs WHERE instance_id=? AND task=?', (instance_id, task))
            check(run is not None, 'run_not_found', 404)
            return self._verified_run(db, bound, run, through=turn)[-1]

    @staticmethod
    def _qa_allowed(row):
        check(row['group_code'] == 'A' and row['stage'] == 'task_2', 'explanations_only_A_active_Task2', 403)

    @staticmethod
    def _evidence(bound, decision, evidence):
        if evidence is None:
            return None
        program = next(p for p in bound.binding.programs if p.role == 'ai')
        require(type(evidence) is dict and evidence.get('eligible') is True
            and program.independent_audit_sha256 is not None
            and evidence.get('target_actor_role_sha256') == bound.binding.ai_actor_sha256
            and evidence.get('nn_action') == decision['action']
            and evidence.get('source_program_sha256', evidence.get('program_sha256')) == program.source_program_sha256
            and evidence.get('independent_audit_sha256', evidence.get('historical_acceptance_sha256')) == program.independent_audit_sha256
            and evidence.get('external_program_selects_execution_action') is False
            and evidence.get('unique_causal_motive_established') is False,
            'explanation is not audited evidence matching the current NN action')
        return evidence

    @staticmethod
    def _answer(action, evidence, language):
        if evidence is None:
            return ('该次已执行动作没有通过审计且与 NN 动作一致的程序证据，无法推断动机。' if language == 'zh' else
                    'No audited program evidence matches this recorded NN action, so its motive cannot be inferred.')
        return (f'该次已执行决策中，冻结 NN 选择了 {action}。所附经审计的程序路径与该动作一致，描述的是模型行为模式；无法据此确定唯一动机。' if language == 'zh' else
            f'In this recorded decision, the frozen NN selected {action}. The attached audited program path matches this action and describes a policy pattern; it does not establish a unique motive.')

    @staticmethod
    def _question_coordinate(turn, decision_turn):
        check(turn is None or decision_turn is None, 'ambiguous_question_turn')
        selected = turn if turn is not None else decision_turn
        check(selected is None or type(selected) is int and selected >= 0, 'invalid_question_turn')
        return selected

    def _question_target(self, db, bound, instance, *, task, decision_turn=None):
        check(type(task) is int and task in (1, 2), 'question_target_must_be_Task1_or_Task2')
        run = _one(db, 'SELECT * FROM v642_runs WHERE instance_id=? AND task=?', (instance['id'], task))
        check(run is not None, 'run_not_found', 404)
        frames = self._verified_run(db, bound, run)
        head = frames[-1]['turn']
        if decision_turn is None:
            decision_turn = head - 1
        check(type(decision_turn) is int and 0 <= decision_turn < head,
              'question_requires_executed_action', 409)
        before, after = frames[decision_turn:decision_turn+2]
        # _verified_run independently replays the actual outgoing transition.
        require(before['decision'] and not before['state']['terminal']
            and after['transition']['records'][0]['decision'] == before['decision'],
            'question lacks a committed NN transition')
        return run, frames, before, after

    def ask(self, token, instance_id, *, question, turn=None, decision_turn=None,
            target_task=2, language='zh', human_actions=None):
        """Explain a committed decision only. ``turn`` is the legacy decision-turn alias.

        An omitted coordinate selects the latest executed decision of target_task,
        never the head state's not-yet-executed NN output. Historical Task1 is
        accessible only while this instance is in A's active Task2.
        """
        self._gate()
        check(type(question) is str and 1 <= len(question.strip()) <= 1000
            and language in ('zh', 'en'), 'invalid_question')
        selected = self._question_coordinate(turn, decision_turn)
        if human_actions is not None:
            check(type(human_actions) is list and 0 < len(human_actions) <= 12
                and all(type(a) is str and 1 <= len(a) <= 100 for a in human_actions), 'invalid_counterfactual')
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            self._qa_allowed(row)
            run, frames, target, outcome = self._question_target(db, bound, row,
                task=target_task, decision_turn=selected)
            head, selected = frames[-1]['turn'], target['turn']
            if human_actions is not None:
                check(len(human_actions) <= head-selected, 'counterfactual_exceeds_committed_history', 409)
            # This context grants access to audited historical Task1 paths only;
            # the actual instance authorization was checked above in this transaction.
            question_bound = replace(bound, question_context_task=2)
            evidence = None
            explanation = getattr(bound.adapter, 'explanation', None)
            if callable(explanation):
                evidence = explanation(target['state'], question_bound)
            self._evidence(bound, target['decision'], evidence)
            action = target['decision']['action']
            result = dict(version='runtime-v642-committed-question.v1',
                status='grounded_pattern' if evidence is not None else 'insufficient_evidence',
                answer=self._answer(action, evidence, language), language=language, binding=bound.binding.to_dict(),
                frame_sha256=target['frame_sha256'], result_frame_sha256=outcome['frame_sha256'],
                target_task=target_task, decision_turn=selected, result_turn=selected+1,
                through_turn_at_question=head, history_head_sha256=frames[-1]['frame_sha256'],
                human_action=outcome['transition']['records'][0]['human_action'],
                nn_action=action, evidence=evidence, unique_causal_motive_established=False,
                semantic_freeform_interpretation=False, counterfactual=None)
            if human_actions is not None:
                bundle = bound.adapter.counterfactual(target['state'], human_actions, bound.binding)
                require(bundle.get('binding') == bound.binding.to_dict() and bundle.get('two_frozen_NN') is False
                    and bundle['initial_state'] == target['state']
                    and 0 < len(bundle['records']) <= len(human_actions)
                    and bundle['final_state']['turn'] <= head
                    and bound.adapter.replay(bundle, bound.binding) == bundle['final_state'], 'unbound QA counterfactual')
                result['counterfactual'] = bundle
            qid = uuid.uuid4().hex
            result['question_id'] = qid
            db.execute('INSERT INTO v642_questions VALUES(?,?,?,?,?,?,?,?,?,?)', (qid, instance_id, run['id'],
                selected, encode(bound.binding.to_dict()), digest(bound.binding.to_dict()), question.strip(),
                language, encode(result), self.clock()))
            return result

    def _verified_question(self, db, bound, instance, row):
        require(instance['group_code'] == 'A' and row['instance_id'] == instance['id']
            and decode(row['binding_json']) == bound.binding.to_dict()
            and row['binding_sha256'] == digest(bound.binding.to_dict()), 'exported QA binding or group differs')
        result = decode(row['result_json'])
        require(set(result) == {'version', 'status', 'answer', 'language', 'binding', 'frame_sha256',
            'result_frame_sha256', 'target_task', 'decision_turn', 'result_turn', 'through_turn_at_question',
            'history_head_sha256', 'human_action', 'nn_action', 'evidence', 'unique_causal_motive_established',
            'semantic_freeform_interpretation', 'counterfactual', 'question_id'}
            and result['version'] == 'runtime-v642-committed-question.v1', 'unknown or legacy QA temporal scope')
        run, frames, target, outcome = self._question_target(db, bound, instance,
            task=result['target_task'], decision_turn=row['turn'])
        recorded_head = result['through_turn_at_question']
        require(run['id'] == row['run_id'] and type(recorded_head) is int
            and row['turn'] < recorded_head <= frames[-1]['turn']
            and result['history_head_sha256'] == frames[recorded_head]['frame_sha256']
            and result['decision_turn'] == row['turn'] and result['result_turn'] == row['turn']+1
            and result['result_frame_sha256'] == outcome['frame_sha256']
            and result['human_action'] == outcome['transition']['records'][0]['human_action']
            and result.get('binding') == bound.binding.to_dict()
            and result.get('frame_sha256') == target['frame_sha256'] and result.get('question_id') == row['id']
            and result.get('nn_action') == target['decision']['action'] and result.get('language') == row['language']
            and result.get('unique_causal_motive_established') is False
            and result.get('semantic_freeform_interpretation') is False, 'stored QA/frame identity differs')
        evidence = self._evidence(bound, target['decision'], result.get('evidence'))
        if evidence is not None:
            explanation = getattr(bound.adapter, 'explanation', None)
            require(callable(explanation) and explanation(target['state'], replace(bound, question_context_task=2)) == evidence,
                    'stored QA evidence does not replay from the same audited reader')
        require(result.get('answer') == self._answer(target['decision']['action'], evidence, row['language'])
            and result.get('status') == ('grounded_pattern' if evidence is not None else 'insufficient_evidence'),
            'stored QA answer exceeds its model/program evidence')
        bundle = result.get('counterfactual')
        if bundle is not None:
            require(bundle.get('binding') == bound.binding.to_dict() and bundle.get('two_frozen_NN') is False
                and bundle['initial_state'] == target['state']
                and 0 < len(bundle['records']) <= recorded_head-row['turn']
                and bundle['final_state']['turn'] <= recorded_head
                and bound.adapter.replay(bundle, bound.binding) == bundle['final_state'], 'stored QA counterfactual differs')
        return result

    def answers(self, token, instance_id):
        """Authorized answer history; active Task2 authorization is rechecked."""
        self._gate()
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            self._qa_allowed(row)
            questions = _all(db, 'SELECT * FROM v642_questions WHERE instance_id=? ORDER BY created,id', (instance_id,))
            return dict(questions=[dict(question=q['question'], **self._verified_question(db, bound, row, q))
                                  for q in questions])

    def kitchen_practice_metadata(self, token, instance_id):
        self._gate()
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            check(row['domain'] == 'kitchen', 'practice_only_kitchen')
            return dict(kind='neutral_kitchen_teaching_practice', labelled_exception=True,
                frozen_teammate_experiment=False, model_evaluation=False, scientific_model_evidence=False,
                interactive_practice_available=False, reason='legacy_tutorial_UI_not_migrated')

    def participant_view(self, token, instance_id, value):
        """Project verified responses without next-action or private-state leakage.

        Stored head decisions remain private for deterministic replay. An action
        is public only after its physical result has been committed to this run.
        """
        self._gate()
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            heads = {run['task']: decode(run['state_json'])['turn'] for run in
                     _all(db, 'SELECT task,state_json FROM v642_runs WHERE instance_id=?', (instance_id,))}

            def public_decision(decision, *, both=False):
                if not decision:
                    return {}
                result = {key: decision[key] for key in ('action', 'binding_identity_sha256', 'state_sha256')}
                if both:
                    result['NN_pair_actions'] = deepcopy(decision['policy_actions'])
                return result

            def project(item):
                if isinstance(item, (list, tuple)):
                    return [project(x) for x in item]
                if type(item) is not dict:
                    return item
                if item.get('version') == 'runtime-v642-committed-question.v1':
                    # An answer captured before a concurrent Task2 completion
                    # cannot be displayed after its authorization has expired.
                    self._qa_allowed(row)
                if {'initial_state', 'final_state', 'records', 'two_frozen_NN'} <= set(item):
                    require(item.get('binding') == bound.binding.to_dict(), 'public replay binding differs')
                    return dict(version='runtime-v642-public-recorded-actions.v1', binding=item['binding'],
                        two_frozen_NN=item['two_frozen_NN'], replay_verified_by_server=True,
                        initial_public=bound.adapter.public_state(item['initial_state'], bound.binding),
                        final_public=bound.adapter.public_state(item['final_state'], bound.binding),
                        records=[dict(human_action=r.get('human_action'), NN_decision=public_decision(r['decision'],
                            both=item['two_frozen_NN']),
                            public_after=bound.adapter.public_state(r['state'], bound.binding)) for r in item['records']],
                        contains_restorable_private_snapshot=False)
                if item.get('version') == FRAME_VERSION and 'state' in item:
                    task, turn = item['state']['task'], item['turn']
                    require(task in heads and 0 <= turn <= heads[task], 'public frame exceeds committed run')
                    result = {key: project(entry) for key, entry in item.items()
                              if key not in ('state', 'transition', 'decision')}
                    committed = bool(item['decision']) and turn < heads[task]
                    result['decision'] = public_decision(item['decision']) if committed else {}
                    result['decision_committed'] = committed
                    incoming = item.get('transition')
                    result['last_executed_action'] = None if incoming is None else dict(
                        decision_turn=turn-1, result_turn=turn,
                        human_action=incoming['records'][0]['human_action'],
                        ai_action=incoming['records'][0]['decision']['action'])
                    return result
                return {key: project(entry) for key, entry in item.items()}

            return project(value)

    def export_instance(self, token, instance_id, *, admin=False):
        self._gate()
        check(admin is True, 'admin_required_for_research_export', 403)
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            return self._export_rows(db, row, bound)

    def research_index(self, *, admin=False):
        self._gate()
        check(admin is True, 'admin_required', 403)
        with self._transaction() as db:
            rows = _all(db, 'SELECT * FROM v642_instances WHERE mode=? ORDER BY created,id', (self.mode,))
            for row in rows:
                self._bound(row)
            return dict(instances=[dict(**self._identity(row), participant_key=row['participant_key']) for row in rows],
                        session_secrets_included=False)

    def research_export(self, instance_id, *, admin=False):
        self._gate()
        check(admin is True, 'admin_required', 403)
        with self._transaction() as db:
            row = _one(db, 'SELECT * FROM v642_instances WHERE id=? AND mode=?', (instance_id, self.mode))
            check(row is not None, 'instance_not_found', 404)
            return self._export_rows(db, row, self._bound(row))

    def _export_rows(self, db, row, bound):
        view = self._view(db, row, bound)
        records = dict(instances=[dict(**self._identity(row), participant_key=row['participant_key'])])
        # SQL row order is unspecified, and dump/restore can change which index
        # PostgreSQL scans. Explicit stable ordering makes verified exports
        # identical across SQLite/PostgreSQL, restarts and restore drills.
        ordering = dict(demonstrations='instance_id', runs='task,id',
                        questions='created,id', commands='created,command_id')
        for table, order in ordering.items():
            records[table] = _all(db, f'SELECT * FROM v642_{table} WHERE instance_id=? ORDER BY {order}', (row['id'],))
        records['frames'] = []
        for run in records['runs']:
            records['frames'].extend(dict(run_id=run['id'], **frame)
                for frame in self._verified_run(db, bound, run))
        for demo in records['demonstrations']:
            bundle = decode(demo['bundle_json'])
            require(demo['binding_sha256'] == digest(bound.binding.to_dict())
                and decode(demo['binding_json']) == bound.binding.to_dict()
                and digest(bundle) == demo['bundle_sha256'] and bundle.get('two_frozen_NN') is True
                and bound.adapter.replay(bundle, bound.binding) == bundle['final_state'], 'export demo replay differs')
        for question in records['questions']:
            self._verified_question(db, bound, row, question)
        frames = {(f['run_id'], f['turn']): f for f in records['frames']}
        for command in records['commands']:
            response = decode(command['response_json'])
            current = response.get('current')
            require(type(current) is dict and current.get('turn', 0) >= 1
                and response['instance']['binding'] == bound.binding.to_dict(), 'saved command binding differs')
            frame = frames.get((current.get('run_id'), current['turn']))
            require(frame == current and current['transition'] is not None,
                    'saved command response is not a verified bound frame')
            incoming = current['transition']['records'][0]
            require(command['request_sha256'] == digest(dict(run_id=current['run_id'], task=frame['state']['task'],
                expected_turn=current['turn']-1, human_action=incoming['human_action'])),
                'saved command request/transition differs')
        return dict(version=EXPORT_VERSION, binding=bound.binding.to_dict(), instance=view['instance'],
            records=records, all_AI_frames_recomputed=True, scientific_acceptance_claimed=False,
            human_collaboration_benefit='not_measured')

    def archived_instance(self, instance_id, *, export=False, admin=False):
        check(admin is True, 'admin_required', 403)
        return self.registry.archived_instance(instance_id, export=export)


class ReadOnlyLegacyReader:
    """Read archived SQLite pl3 data only, never initialize the legacy Store.

    Requires a quiescent snapshot: pending WAL/journal is refused. SQLite
    immutable+mode=ro prevents database, journal and shared-memory writes.
    """
    def __init__(self, database):
        self.path = Path(database)
        require(self.path.is_file() and not self.path.is_symlink(), 'legacy snapshot is missing or symlinked')
        self._check_snapshot()
        self.source_sha256 = sha256(self.path.read_bytes()).hexdigest()

    def _check_snapshot(self):
        require(self.path.is_file() and not self.path.is_symlink()
            and all(not Path(str(self.path)+suffix).exists() or Path(str(self.path)+suffix).stat().st_size == 0
                    for suffix in ('-wal', '-journal')), 'legacy DB requires a quiescent read-only snapshot')

    @contextmanager
    def _read(self):
        self._check_snapshot()
        require(sha256(self.path.read_bytes()).hexdigest() == self.source_sha256,
                'legacy archived snapshot changed after reader binding')
        connection = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        try:
            connection.execute('BEGIN')
            yield connection
        finally:
            connection.close()
            self._check_snapshot()
            require(sha256(self.path.read_bytes()).hexdigest() == self.source_sha256,
                    'legacy archived snapshot changed during read/export')

    def read_instance(self, instance_id):
        with self._read() as db:
            row = _one(db, 'SELECT * FROM pl3_instances WHERE id=?', (instance_id,))
            check(row is not None, 'legacy_instance_not_found', 404)
            return dict(archived=True, read_only=True, resumable_with_v642=False, instance=row,
                        source_sha256=self.source_sha256)

    def export_instance(self, instance_id):
        with self._read() as db:
            row = _one(db, 'SELECT * FROM pl3_instances WHERE id=?', (instance_id,))
            check(row is not None, 'legacy_instance_not_found', 404)
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            records = {'instances': [row]}
            for table in ('enrollments', 'policy_bindings', 'policy_binding_metadata', 'runs', 'questions', 'questionnaires',
                          'timings', 'tutorials', 'tutorial_events', 'auto_explanation_settings', 'auto_explanations',
                          'auto_explanation_confirmations', 'understanding_settings', 'understanding_ratings',
                          'reward_settings', 'prompt_settings', 'prompt_skips'):
                if 'pl3_'+table in tables:
                    records[table] = _all(db, f'SELECT * FROM pl3_{table} WHERE instance_id=?', (instance_id,))
            if 'pl3_frames' in tables and 'pl3_runs' in tables:
                records['frames'] = _all(db, 'SELECT * FROM pl3_frames WHERE run_id IN '
                    '(SELECT id FROM pl3_runs WHERE instance_id=?) ORDER BY run_id,turn', (instance_id,))
            if 'pl3_releases' in tables:
                records['releases'] = _all(db, 'SELECT * FROM pl3_releases WHERE id=?', (row.get('release_id'),))
            if 'pl3_participants' in tables:
                records['participants'] = _all(db, 'SELECT id,group_code,created FROM pl3_participants WHERE id=?',
                                                (row.get('participant_id'),))
            return dict(archived=True, read_only=True, resumable_with_v642=False, release_id=row.get('release_id'),
                        records=records, source_sha256=self.source_sha256)
