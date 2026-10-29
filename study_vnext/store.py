"""Consent, single-domain enrollment, surveys and timing over the bound core.

The core remains the sole owner of actor, action, replay and question evidence.
This extension never makes a catalog scientifically acceptable.
"""
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import threading

from runtime_v642.contracts import RuntimeUnavailable, require
from runtime_v642.store import Store as CoreStore, check, encode, _one, _all
from .protocol import PROTOCOL_ID, CONSENT_VERSION, protocol_document, protocol_sha256
from .history import (HistoryCertificates, verified_run, full_verification_enabled,
    full_history_verification, original_full_verification)

EXTRA_SCHEMA = '''
CREATE TABLE IF NOT EXISTS vnext_protocol(id TEXT PRIMARY KEY,sha256 TEXT NOT NULL,document_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS vnext_enrollments(instance_id TEXT PRIMARY KEY REFERENCES v642_instances(id),
 protocol_id TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,consent_version TEXT NOT NULL,
 language TEXT NOT NULL,question_transmission_consent INTEGER NOT NULL CHECK(question_transmission_consent IN (0,1)),
 created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS vnext_surveys(instance_id TEXT PRIMARY KEY REFERENCES v642_instances(id),
 answers_json TEXT NOT NULL,feedback TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS vnext_understanding(instance_id TEXT PRIMARY KEY REFERENCES v642_instances(id),
 version TEXT NOT NULL,sha256 TEXT NOT NULL,answers_json TEXT NOT NULL,result_json TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS vnext_answer_renderings(question_id TEXT PRIMARY KEY REFERENCES v642_questions(id),
 core_sha256 TEXT NOT NULL,rendering_json TEXT NOT NULL,metrics_json TEXT NOT NULL,created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS vnext_preparation(instance_id TEXT NOT NULL REFERENCES v642_instances(id),
 kind TEXT NOT NULL CHECK(kind IN ('controls','demonstration')),details_json TEXT NOT NULL,created REAL NOT NULL,
 PRIMARY KEY(instance_id,kind));
CREATE TABLE IF NOT EXISTS vnext_events(instance_id TEXT NOT NULL REFERENCES v642_instances(id),
 event_id TEXT NOT NULL,task INTEGER NOT NULL,kind TEXT NOT NULL,seconds REAL NOT NULL,
 details_json TEXT NOT NULL,created REAL NOT NULL,PRIMARY KEY(instance_id,event_id));
CREATE UNIQUE INDEX IF NOT EXISTS vnext_one_domain_per_participant ON v642_instances(participant_key,mode);
'''
RATINGS = ('coordination', 'understanding', 'predictability', 'workload')
# reading is an explicit answer-panel mode duration; question_entry is time
# spent in the question input. Neither is a measure of participant attention.
EVENTS = ('active', 'reading', 'question_entry', 'replay', 'explanation_wait', 'answer_displayed', 'technical_fault')
_ENROLLMENT_COLUMNS = ('instance_id', 'protocol_id', 'protocol_sha256', 'consent_version',
    'language', 'question_transmission_consent', 'created')
_CONTROL_CODES = dict(warehouse=('KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space'),
    pong=('KeyA', 'KeyD', 'Space'), kitchen=('KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space', 'KeyE'))


class StudyStore(CoreStore):
    def __init__(self, *args, question_composer=None, **kwargs):
        self.question_composer = question_composer
        self._backend = None
        self._consenting = threading.local()
        self._history = HistoryCertificates()
        super().__init__(*args, **kwargs)
        columns = tuple(item['name'] for item in _all(self.db, 'PRAGMA table_info(vnext_enrollments)'))
        require(not columns or columns == _ENROLLMENT_COLUMNS,
                'study enrollment schema differs; explicit per-instance question consent is required')
        self.db.executescript(EXTRA_SCHEMA)
        for table in ('protocol', 'enrollments', 'surveys', 'understanding', 'events', 'answer_renderings', 'preparation'):
            for operation in ('UPDATE', 'DELETE'):
                self.db.execute(f"CREATE TRIGGER IF NOT EXISTS vnext_{table}_{operation.lower()} BEFORE {operation} "
                    f"ON vnext_{table} BEGIN SELECT RAISE(ABORT,'immutable study protocol/evidence');END")
        row = _one(self.db, 'SELECT * FROM vnext_protocol WHERE id=?', (PROTOCOL_ID,))
        expected = dict(id=PROTOCOL_ID, sha256=protocol_sha256(), document_json=encode(protocol_document()))
        if row is None:
            self.db.execute('INSERT INTO vnext_protocol VALUES(?,?,?)', tuple(expected.values()))
        else:
            require(row == expected, 'study protocol bytes changed for an existing database')

    @classmethod
    def from_postgres(cls, *, dsn, schema, registry, scenario_pools, demonstration_seeds, mode='preview',
                      connector=None, question_composer=None):
        from .storage import PostgresStorage
        from runtime_v642.store import _SCHEMA, VERSION as CORE_VERSION
        store=cls(':memory:',registry,scenario_pools=scenario_pools,demonstration_seeds=demonstration_seeds,
            mode=mode,question_composer=question_composer)
        backend=None
        try:
            backend=PostgresStorage(dsn=dsn,schema=schema,schemas=[_SCHEMA,EXTRA_SCHEMA],
                protocol_sha256=protocol_sha256(),connector=connector,
                append_only_tables=('v642_demonstrations','v642_frames','v642_questions','v642_commands',
                    'vnext_protocol','vnext_enrollments','vnext_surveys','vnext_understanding',
                    'vnext_events','vnext_answer_renderings','vnext_preparation'),
                nondeletable_tables=('v642_instances','v642_runs'))
            with backend.transaction(write=True) as db:
                expected=dict(version=CORE_VERSION,release_id=registry.catalog.release_id)
                rows=_all(db,'SELECT * FROM v642_metadata')
                if not rows:db.execute('INSERT INTO v642_metadata VALUES(?,?)',tuple(expected.values()))
                else:require(rows==[expected],'PostgreSQL core/release identity differs')
                protocol=dict(id=PROTOCOL_ID,sha256=protocol_sha256(),document_json=encode(protocol_document()))
                rows=_all(db,'SELECT * FROM vnext_protocol')
                if not rows:db.execute('INSERT INTO vnext_protocol VALUES(?,?,?)',tuple(protocol.values()))
                else:require(rows==[protocol],'PostgreSQL protocol identity differs')
            store.db.close();store.db=None;store._backend=backend;store.database=None
            return store
        except BaseException:
            if backend is not None:backend.close()
            store.close();raise

    def close(self):
        if self._backend is not None:self._backend.close()
        else:super().close()

    def _gate(self):
        super()._gate()
        # Institutional, power, persistence and production-provider receipts are
        # not present yet. A catalog alone cannot authorize this new protocol.
        require(self.mode != 'pilot', 'new study operational admission is not registered; recruitment closed')

    @contextmanager
    def _transaction(self, *, write=False):
        transaction=self._backend.transaction(write=write) if self._backend is not None else super()._transaction(write=write)
        # Publish static history certificates only after a confirmed DB commit.
        # Failed/uncertain PostgreSQL commits and SQLite rollbacks discard them.
        with self._history.transaction():
            with transaction as db:
                consent = getattr(self._consenting, 'value', None) if write else None
                if consent:
                    check(_one(db, 'SELECT id FROM v642_instances WHERE participant_key=? AND mode=?',
                        (consent['participant_key'], self.mode)) is None, 'one_domain_per_participant', 409)
                yield db
                if consent:
                    row = _one(db, 'SELECT id FROM v642_instances WHERE participant_key=? AND mode=?',
                        (consent['participant_key'], self.mode))
                    require(row is not None, 'enrollment did not produce an atomic binding')
                    db.execute('INSERT INTO vnext_enrollments VALUES(?,?,?,?,?,?,?)',
                        (row['id'], PROTOCOL_ID, protocol_sha256(), CONSENT_VERSION, consent['language'],
                         int(consent['question_transmission_consent']), self.clock()))

    def enroll(self, *, consent, consent_version, language='zh', question_transmission_consent=False, **selection):
        self._gate()
        check(consent is True and consent_version == CONSENT_VERSION and language in ('zh', 'en'),
              'current_explicit_consent_required')
        check(type(question_transmission_consent) is bool, 'invalid_question_transmission_consent')
        reference = selection.get('participant_reference')
        check(type(reference) is str and 1 <= len(reference.strip()) <= 160, 'invalid_enrollment')
        self._consenting.value = dict(participant_key=sha256(reference.strip().encode()).hexdigest(), language=language,
            question_transmission_consent=question_transmission_consent)
        try:
            return super().enroll(**selection)
        finally:
            self._consenting.value = None

    def _enrollment(self, db, instance_id):
        consent = _one(db, 'SELECT * FROM vnext_enrollments WHERE instance_id=?', (instance_id,))
        require(consent is not None and consent['protocol_sha256'] == protocol_sha256()
                and consent['protocol_id'] == PROTOCOL_ID and consent['consent_version'] == CONSENT_VERSION
                and type(consent.get('question_transmission_consent')) is int
                and consent['question_transmission_consent'] in (0, 1), 'study enrollment/protocol binding missing')
        return consent

    def _view(self, db, row, bound):
        if row['stage'] in ('demo_complete', 'task_1_complete', 'task_2_complete', 'complete') \
                and not full_verification_enabled():
            with full_history_verification():
                return self._view(db, row, bound)
        from runtime_v642.store import decode
        view = super()._view(db, row, bound)
        consent = self._enrollment(db, row['id'])
        survey = _one(db, 'SELECT * FROM vnext_surveys WHERE instance_id=?', (row['id'],))
        understanding = self._verified_understanding(db, row)
        require(not survey or row['stage'] == 'complete', 'questionnaire exists before task completion')
        after_tasks = row['stage'] == 'complete'
        finished = survey is not None and understanding is not None
        view['study'] = dict(protocol_id=PROTOCOL_ID, protocol_sha256=protocol_sha256(),
            phase='finished' if finished else 'questionnaire' if after_tasks else row['stage'],
            questionnaire_required=after_tasks and not finished,
            survey_required=after_tasks and survey is None,
            understanding_required=after_tasks and understanding is None,
            survey_answers=decode(survey['answers_json']) if after_tasks and survey else None,
            survey_feedback=survey['feedback'] if after_tasks and survey else None,
            understanding_answers=decode(understanding['answers_json']) if after_tasks and understanding else None,
            question_transmission_consent=bool(consent['question_transmission_consent']),
            preparation=self._preparation(db, row, bound),
            scientific_sample=False, recruitment_opened=False)
        return view

    def _verified_run(self, db, bound, run, *, through=None):
        return verified_run(self, self._history, db, bound, run, through=through)

    @original_full_verification
    def start_task(self, token, instance_id):
        return super().start_task(token, instance_id)

    def replay(self, token, instance_id, *, task):
        value = super().replay(token, instance_id, task=task)
        # Materialize independent mutable records only for the replay API.
        # Normal views keep historical records lazy and return the fresh head.
        value['frames'] = list(value['frames'])
        # Integrity and live-head replay remain fresh; historical prefix NN
        # proofs may be reused. Only a strict export claims every AI frame.
        value['all_AI_actions_recomputed'] = False
        value['immutable_history_validation_cache'] = not full_verification_enabled()
        return value

    def _verified_understanding(self, db, row):
        from runtime_v642.store import decode
        from .understanding import VERSION, instrument_sha256, score, UnderstandingError
        saved = _one(db, 'SELECT * FROM vnext_understanding WHERE instance_id=?', (row['id'],))
        if saved is None:
            return None
        require(row['stage'] == 'complete' and saved['version'] == VERSION
            and saved['sha256'] == instrument_sha256(), 'understanding instrument/task identity differs')
        try:
            result = score(row['domain'], decode(saved['answers_json']))
        except UnderstandingError as exc:
            raise RuntimeUnavailable('saved understanding responses are malformed') from exc
        require(decode(saved['result_json']) == result, 'saved understanding score differs from instrument')
        return saved

    @original_full_verification
    def _after_three_tasks(self, db, row, bound):
        check(row['stage'] == 'complete', 'finish_three_tasks_before_understanding', 409)
        # Validate the actual complete task history, not only a mutable stage
        # string, before any instrument delivery or private scoring.
        super()._view(db, row, bound)
        self._enrollment(db, row['id'])
        self._preparation(db, row, bound)

    def understanding(self, token, instance_id):
        from .understanding import bank_for_participant
        self._gate()
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            self._after_three_tasks(db, row, bound)
            consent = self._enrollment(db, row['id'])
            return bank_for_participant(row['domain'], consent['language'])

    def submit_understanding(self, token, instance_id, *, answers, version, sha256):
        from .understanding import VERSION, instrument_sha256, score, UnderstandingError
        self._gate()
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            self._after_three_tasks(db, row, bound)
            check(version == VERSION and sha256 == instrument_sha256(), 'current_understanding_instrument_required')
            try:
                result = score(row['domain'], answers)
            except UnderstandingError:
                check(False, 'invalid_understanding_answers')
            old = self._verified_understanding(db, row)
            if old is not None:
                check(old['answers_json'] == encode(answers), 'understanding_already_submitted', 409)
            else:
                db.execute('INSERT INTO vnext_understanding VALUES(?,?,?,?,?,?)',
                    (row['id'], VERSION, instrument_sha256(), encode(answers), encode(result), self.clock()))
            return self._view(db, row, bound)

    def _saved_demonstration(self, db, row, bound, *, replay=False):
        saved = _one(db, 'SELECT * FROM v642_demonstrations WHERE instance_id=?', (row['id'],))
        if saved is None:
            return None, None
        bound.verify()
        bundle = self._history.parsed_demonstration(saved, bound.binding)
        if replay or full_verification_enabled():
            require(bound.adapter.replay(bundle, bound.binding) == bundle['final_state'],
                    'saved preparation demonstration replay differs')
        return saved, bundle

    def _preparation(self, db, row, bound, *, replay=False):
        from runtime_v642.store import decode
        receipts = _all(db, 'SELECT * FROM vnext_preparation WHERE instance_id=?', (row['id'],))
        require(all(r['kind'] in ('controls', 'demonstration') for r in receipts), 'unknown preparation evidence')
        by_kind = {r['kind']: decode(r['details_json']) for r in receipts}
        require(len(by_kind) == len(receipts), 'duplicate preparation evidence')
        controls = by_kind.get('controls')
        if controls is not None:
            require(controls == dict(observed_codes=sorted(_CONTROL_CODES[row['domain']])),
                    'saved control practice differs from registered input codes')
        if replay or full_verification_enabled():
            saved,bundle=self._saved_demonstration(db,row,bound,replay=replay)
            summary=None if saved is None else dict(bundle_sha256=saved['bundle_sha256'],
                frame_count=len(bundle['records']))
        else:
            saved=_one(db,'SELECT * FROM v642_demonstrations WHERE instance_id=?',(row['id'],))
            summary=None
            if saved is not None:
                bound.verify()
                parsed=self._history.demonstration_summary(saved,bound.binding)
                summary=dict(bundle_sha256=parsed.bundle_sha256,frame_count=parsed.frame_count)
        demonstration = by_kind.get('demonstration')
        if demonstration is not None:
            require(controls is not None and saved is not None
                and demonstration == dict(bundle_sha256=summary['bundle_sha256'],final_frame_count=summary['frame_count']),
                'saved playback receipt differs from bound demonstration')
        if row['stage'] != 'enrolled':
            require(controls is not None and demonstration is not None,
                    'task progression has no complete preparation evidence')
        return dict(controls=controls is not None, demonstration=demonstration is not None,
            demonstration_sha256=None if summary is None else summary['bundle_sha256'],
            demonstration_frame_count=None if summary is None else summary['frame_count'])

    def complete_preparation(self, token, instance_id, *, kind, observed_codes=None,
                             bundle_sha256=None, final_frame_count=None):
        """Save client-reported practice/playback, not evidence of human attention."""
        self._gate()
        check(kind in ('controls', 'demonstration'), 'invalid_preparation_kind')
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            self._enrollment(db, row['id'])
            check(row['stage'] == 'enrolled', 'preparation_enrollment_stage_required', 409)
            if kind == 'controls':
                expected = _CONTROL_CODES[row['domain']]
                check(type(observed_codes) is list and len(observed_codes) == len(expected)
                    and all(type(code) is str for code in observed_codes)
                    and set(observed_codes) == set(expected)
                    and bundle_sha256 is None and final_frame_count is None,
                    'complete_registered_control_practice_required')
                details = dict(observed_codes=sorted(expected))
            else:
                check(observed_codes is None, 'invalid_demonstration_preparation')
                preparation = self._preparation(db, row, bound, replay=True)
                check(preparation['controls'], 'controls_practice_required', 409)
                check(preparation['demonstration_sha256'] is not None, 'generate_bound_demonstration_first', 409)
                check(type(bundle_sha256) is str and bundle_sha256 == preparation['demonstration_sha256']
                    and type(final_frame_count) is int and final_frame_count == preparation['demonstration_frame_count'],
                    'complete_bound_demonstration_playback_required')
                details = dict(bundle_sha256=bundle_sha256,final_frame_count=final_frame_count)
            old = _one(db, 'SELECT * FROM vnext_preparation WHERE instance_id=? AND kind=?', (row['id'], kind))
            if old is None:
                db.execute('INSERT INTO vnext_preparation VALUES(?,?,?,?)', (row['id'],kind,encode(details),self.clock()))
            else:
                require(old['details_json'] == encode(details), 'preparation receipt differs')
            return self._view(db, row, bound)

    def demonstration(self, token, instance_id):
        self._gate()
        with self._transaction() as db:
            row, bound = self._instance(db, token, instance_id)
            self._enrollment(db, row['id'])
            check(self._preparation(db, row, bound)['controls'], 'controls_practice_required', 409)
        return super().demonstration(token, instance_id)

    @original_full_verification
    def acknowledge_demonstration(self, token, instance_id):
        # Unlike the legacy core convenience API, acknowledgement never creates
        # a demonstration. The saved, replayed bundle and playback receipt must
        # already exist, along with the control-practice receipt.
        self._gate()
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            self._enrollment(db, row['id'])
            preparation = self._preparation(db, row, bound, replay=True)
            check(preparation['controls'] and preparation['demonstration'], 'preparation_incomplete', 409)
            if row['stage'] == 'enrolled':
                db.execute("UPDATE v642_instances SET stage='demo_complete' WHERE id=?", (instance_id,))
                row['stage'] = 'demo_complete'
            return self._view(db, row, bound)

    def ask(self, token, instance_id, **request):
        result=super().ask(token,instance_id,**request)
        if self.question_composer is None:return result
        from .qa import EvidenceQuestionComposer
        from runtime_v642.contracts import digest
        composer = self.question_composer
        require(type(composer) is EvidenceQuestionComposer,'explicit committed-evidence composer required')
        metrics=[]
        with self._transaction() as db:
            row,bound=self._instance(db,token,instance_id);self._qa_allowed(row)
            consent = self._enrollment(db, row['id'])
            # A globally configured transport cannot authorize this person's
            # question transmission. Nonconsenting enrollments stay local.
            if composer.provider is not None and consent['question_transmission_consent'] != 1:
                composer = EvidenceQuestionComposer()
            q=_one(db,'SELECT * FROM v642_questions WHERE id=?',(result['question_id'],))
            verified=self._verified_question(db,bound,row,q)
            bundle=verified.get('counterfactual')
            if bundle is not None and len(bundle['records'])==1:
                run,frames,before,after=self._question_target(db,bound,row,task=result['target_task'],decision_turn=result['decision_turn'])
                public=bound.adapter.public_state(bundle['final_state'],bound.binding)
                actual=after['public']
                for metric,a,b in [('team_score',actual.get('score',{}).get('task_score'),public.get('score',{}).get('task_score'))]:
                    if type(a) in (int,float) and type(b) in (int,float):metrics.append(dict(metric=metric,factual=a,alternative=b,
                        factual_frame_sha256=result['result_frame_sha256'],counterfactual_state_sha256=bundle['final_state']['state_sha256']))
        # Provider work occurs outside the DB lock. Both the explicit transport
        # opt-in and immutable enrollment opt-in are required before transmission.
        rendering=composer.compose(request['question'],result,request.get('language','zh'),counterfactual_metrics=metrics)
        with self._transaction(write=True) as db:
            row,bound=self._instance(db,token,instance_id);self._qa_allowed(row)
            consent = self._enrollment(db, row['id'])
            require(composer.provider is None or consent['question_transmission_consent'] == 1,
                    'question transmission was not consented by this enrollment')
            q=_one(db,'SELECT * FROM v642_questions WHERE id=?',(result['question_id'],))
            require(self._verified_question(db,bound,row,q)==result,'question changed during language selection')
            db.execute('INSERT INTO vnext_answer_renderings VALUES(?,?,?,?,?)',(result['question_id'],digest(result),encode(rendering),encode(metrics),self.clock()))
        return dict(result,language_rendering=rendering)

    def answers(self, token, instance_id):
        value=super().answers(token,instance_id)
        with self._transaction() as db:
            row,bound=self._instance(db,token,instance_id);self._qa_allowed(row)
            for item in value['questions']:
                q=_one(db,'SELECT * FROM v642_questions WHERE id=?',(item['question_id'],))
                rendering=self._rendering(db,bound,row,q)
                if rendering is not None:item['language_rendering']=rendering
        return value

    def _rendering(self,db,bound,instance,question):
        from .qa import EvidenceQuestionComposer
        from runtime_v642.contracts import digest
        from runtime_v642.store import decode
        saved=_one(db,'SELECT * FROM vnext_answer_renderings WHERE question_id=?',(question['id'],))
        if saved is None:return None
        core=self._verified_question(db,bound,instance,question)
        require(saved['core_sha256']==digest(core),'rendering core evidence differs')
        rendering,metrics=decode(saved['rendering_json']),json.loads(saved['metrics_json'])
        consent = self._enrollment(db, instance['id'])
        require(rendering.get('audit', {}).get('provider_used') is not True
                or consent['question_transmission_consent'] == 1,
                'stored external question rendering lacks enrollment consent')
        if metrics:
            bundle=core.get('counterfactual')
            require(bundle is not None and len(bundle['records'])==1,'stored comparison has no one-step counterfactual')
            _,_,_,after=self._question_target(db,bound,instance,task=core['target_task'],decision_turn=core['decision_turn'])
            public=bound.adapter.public_state(bundle['final_state'],bound.binding)
            a=after['public'].get('score',{}).get('task_score');b=public.get('score',{}).get('task_score')
            expected=[]
            if type(a) in (int,float) and type(b) in (int,float):
                expected=[dict(metric='team_score',factual=a,alternative=b,factual_frame_sha256=core['result_frame_sha256'],
                    counterfactual_state_sha256=bundle['final_state']['state_sha256'])]
            require(metrics==expected,'stored comparison metrics differ from verified physical outcome')
        class RecordedSelection:
            configured=True;consented=True
            def select(self,_):return dict(intent=rendering['intent'],fact_ids=rendering['fact_ids'])
        recomputed=EvidenceQuestionComposer(RecordedSelection()).compose(question['question'],core,question['language'],counterfactual_metrics=metrics)
        require(rendering['answer']==recomputed['answer'] and rendering['fact_ids']==recomputed['fact_ids']
            and rendering['audit']['canonical_basis_sha256']==recomputed['audit']['canonical_basis_sha256']
            and rendering.get('production_provider_verified') is False and rendering.get('free_generated_policy_claims') is False,
            'rendered answer exceeds committed evidence')
        return rendering

    def submit_survey(self, token, instance_id, *, answers, feedback=''):
        self._gate()
        check(type(answers) is dict and set(answers) == set(RATINGS)
            and all(type(x) is int and 1 <= x <= 7 for x in answers.values())
            and type(feedback) is str and len(feedback) <= 2000, 'invalid_questionnaire')
        with self._transaction(write=True) as db:
            row, bound = self._instance(db, token, instance_id)
            check(row['stage'] == 'complete', 'finish_three_tasks_before_questionnaire', 409)
            old = _one(db, 'SELECT * FROM vnext_surveys WHERE instance_id=?', (instance_id,))
            if old:
                check(old['answers_json'] == encode(answers) and old['feedback'] == feedback,
                      'questionnaire_already_submitted', 409)
            else:
                db.execute('INSERT INTO vnext_surveys VALUES(?,?,?,?)',
                    (instance_id, encode(answers), feedback, self.clock()))
            return self._view(db, row, bound)

    def event(self, token, instance_id, *, event_id, task, kind, seconds=0, details=None):
        self._gate()
        check(type(event_id) is str and 8 <= len(event_id) <= 100 and type(task) is int and task in (1, 2, 3)
            and kind in EVENTS and type(seconds) in (int, float) and math.isfinite(seconds)
            and 0 <= seconds <= 3600 and (details is None or type(details) is dict), 'invalid_study_event')
        details = details or {}
        check(set(details) <= {'question_id', 'fault_code'} and all(type(v) is str and len(v) <= 120 for v in details.values()),
              'invalid_event_details')
        with self._transaction(write=True) as db:
            row, _ = self._instance(db, token, instance_id)
            check(_one(db, 'SELECT id FROM v642_runs WHERE instance_id=? AND task=?', (instance_id, task)), 'task_not_started')
            if kind == 'answer_displayed':
                self._qa_allowed(row)
                check(_one(db, 'SELECT id FROM v642_questions WHERE id=? AND instance_id=?',
                    (details.get('question_id'), instance_id)), 'unknown_displayed_answer')
            values = (instance_id, event_id, task, kind, float(seconds), encode(details))
            old = _one(db, 'SELECT * FROM vnext_events WHERE instance_id=? AND event_id=?', (instance_id, event_id))
            if old:
                check(tuple(old[k] for k in ('instance_id','event_id','task','kind','seconds','details_json')) == values,
                      'event_id_reused', 409)
            else:
                db.execute('INSERT INTO vnext_events VALUES(?,?,?,?,?,?,?)', (*values, self.clock()))
            return dict(saved=True, event_id=event_id)

    @original_full_verification
    def _export_rows(self, db, row, bound):
        result = super()._export_rows(db, row, bound)
        result['study_protocol'] = protocol_document()
        result['protocol_sha256'] = protocol_sha256()
        result['records']['study_answer_renderings']=[]
        for question in result['records']['questions']:
            rendering=self._rendering(db,bound,row,question)
            if rendering is not None:result['records']['study_answer_renderings'].append(rendering)
        self._preparation(db, row, bound, replay=True)
        self._verified_understanding(db, row)
        ordering = dict(enrollments='instance_id', surveys='instance_id', understanding='instance_id',
                        events='created,event_id', preparation='kind')
        for table, order in ordering.items():
            result['records']['study_'+table] = _all(db,
                f'SELECT * FROM vnext_{table} WHERE instance_id=? ORDER BY {order}', (row['id'],))
        return result
