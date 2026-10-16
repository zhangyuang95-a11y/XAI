"""Authoritative study flow and Task-2-only explanation authorization."""
import hashlib
import hmac
import json
from collections import OrderedDict
from contextlib import closing, nullcontext
import re
import secrets
import threading
import time
import uuid

from . import RELEASE_ID, SUPPORTED_RELEASE_IDS, KITCHEN_SUPPORTED_RELEASE_IDS
from .database import Database
from .registry import engine, demonstration, MODULES, scenario_config
from . import kitchen_tutorial, automatic_explanations, understanding, rewards
from .policies import (GROUP_VARIANT, PolicyUnavailable, all_policies_accepted,
                       binding_for_group, policy_id, policy_manifest, verify_binding)
from .release_catalog import catalog_sha256, ReleaseAssetUnavailable

def encode(value): return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(value.encode()).hexdigest()
def uid(): return uuid.uuid4().hex
def compatible_instance(row):
    return (row["release_id"] in SUPPORTED_RELEASE_IDS
            and (row["domain"] not in ("kitchen", "pong", "warehouse") or row["release_id"] == RELEASE_ID))

def public_answer(answer):
    return {k:answer[k] for k in ('status','answer','evidence_ids','language') if k in answer}

class StudyError(Exception):
    def __init__(self, code, status=400):
        self.code, self.status = code, status
        super().__init__(code)

COMMON_ITEMS = [
 ('predictable','I could predict what my teammate would do.','我能预测队友接下来会做什么。'),
 ('understood','I understood when my teammate waited or changed its plan.','我理解队友何时等待或改变计划。'),
 ('coordinate','I knew how to coordinate my actions with my teammate.','我知道怎样安排自己的动作来配合队友。'),
 ('workload','The task required a lot of mental effort.','任务需要我投入很多思考。'),
 ('smooth','Working together felt smooth.','协作过程感觉顺畅。'),
]
EXPLANATION_ITEMS = [
 ('relevant','The Task 2 answers addressed my questions.','Task 2 的回答切中了我的问题。'),
 ('clear','The Task 2 answers were easy to understand.','Task 2 的回答容易理解。'),
 ('helpful','The Task 2 answers helped me choose my next action.','Task 2 的回答帮助我选择下一步动作。'),
]

EXPORT_TABLES = ('participants','enrollments','policy_bindings','policy_binding_metadata',
                 'instances','runs','frames','questions','questionnaires','timings','releases',
                 'tutorials','tutorial_events','auto_explanation_settings','auto_explanations',
                 'auto_explanation_confirmations','understanding_settings','understanding_ratings',
                 'reward_settings','prompt_settings','prompt_skips')

class Store:
    def __init__(self, settings, explainer=None):
        self.settings, self.db, self.explainer = settings, Database(settings.database), explainer
        self._qa_health_lock = threading.Lock()
        self._qa_healthy, self._qa_health_version = settings.verified, 0
        self._frame_cache_lock = threading.Lock()
        self._verified_frame_prefixes = OrderedDict()

    @property
    def qa_healthy(self):
        with self._qa_health_lock:
            return self._qa_healthy

    @qa_healthy.setter
    def qa_healthy(self, healthy):
        self.set_qa_health(healthy)

    def qa_health_snapshot(self):
        with self._qa_health_lock:
            return self._qa_healthy, self._qa_health_version

    def set_qa_health(self, healthy, *, expected_version=None):
        """An older probe cannot overwrite a newer participant QA result."""
        with self._qa_health_lock:
            if expected_version is not None and expected_version != self._qa_health_version:
                return False
            self._qa_healthy = bool(healthy)
            self._qa_health_version += 1
            return True

    @property
    def ready(self):
        if not (self.settings.ready and self.qa_healthy and all_policies_accepted()):
            return False
        # A long-running preview server must not admit a pilot merely because
        # training files appeared after startup. The frozen release manifest is
        # written only during a verified server start.
        with self.db.transaction(read_only=True) as db:
            return self._release_policies_match(db)

    @staticmethod
    def _release_policies_match(db):
        row = db.one('SELECT manifest_json FROM pl3_releases WHERE id=?', (RELEASE_ID,))
        if row is None:
            return False
        try:
            release = json.loads(row['manifest_json'])
        except (TypeError, ValueError):
            return False
        try:
            return (release.get('policy_assets_accepted') is True
                    and release.get('release_catalog_sha256') == catalog_sha256()
                    and release.get('policies') == policy_manifest())
        except ReleaseAssetUnavailable:
            return False

    def _binding(self, db, instance):
        if (instance['mode']=='pilot'
                and (not all_policies_accepted() or not self._release_policies_match(db))):
            raise StudyError('study_not_ready', 503)
        row = db.one('SELECT * FROM pl3_policy_bindings WHERE instance_id=?', (instance['id'],))
        if row is None:
            raise StudyError('policy_binding_missing', 503)
        if instance['group_code'] not in GROUP_VARIANT:
            raise StudyError('policy_binding_mismatch', 503)
        try:
            expected_policy_id = policy_id(instance['domain'], instance['group_code'])
        except PolicyUnavailable as exc:
            raise StudyError('policy_unavailable', 503) from exc
        if (row['domain'] != instance['domain']
                or row['variant'] != GROUP_VARIANT[instance['group_code']]
                or row['policy_id'] != expected_policy_id):
            raise StudyError('policy_binding_mismatch', 503)
        binding = {key: row[key] for key in ('policy_id','domain','variant','actor_path',
                   'actor_sha256','human_actor_sha256','ai_actor_sha256')}
        try:
            binding['program_hashes'] = json.loads(row['program_hashes_json'])
        except (TypeError, ValueError) as exc:
            raise StudyError('policy_binding_mismatch', 503) from exc
        if row['domain'] == 'warehouse':
            metadata = db.one('SELECT actor_model_version,feature_schema_sha256 '
                              'FROM pl3_policy_binding_metadata WHERE instance_id=?',
                              (instance['id'],))
            if (metadata is None or not metadata['actor_model_version']
                    or not metadata['feature_schema_sha256']):
                raise StudyError('policy_binding_metadata_missing', 503)
            binding.update(metadata)
        try:
            return verify_binding(binding)
        except PolicyUnavailable as exc:
            raise StudyError('policy_unavailable', 503) from exc

    @staticmethod
    def _verified_decision(eng, state, binding):
        Store._assert_state_binding(state, binding)
        decision = eng.decide(state)
        actor_hash = decision.get('actor_sha256', decision.get('model_sha256'))
        if (decision.get('policy_id') != binding['policy_id']
                or actor_hash != binding['actor_sha256']):
            raise StudyError('policy_decision_mismatch', 503)
        return decision

    @staticmethod
    def _assert_state_binding(state, binding):
        roles = state.get('role_sha256')
        expected_roles = {'human': binding['human_actor_sha256'],
                          'ai': binding['ai_actor_sha256']}
        if (state.get('policy_id') != binding['policy_id']
                or state.get('release_id') != RELEASE_ID
                or state.get('actor_sha256', state.get('policy_sha256')) != binding['actor_sha256']
                or roles != expected_roles):
            raise StudyError('policy_binding_mismatch', 503)

    @staticmethod
    def _recorded_frame(eng, row, binding):
        """Check stored replay and QA evidence against its frozen Actor pair."""
        state = json.loads(row['state_json'])
        if state.get('turn') != row['turn']:
            raise StudyError('recorded_frame_mismatch', 503)
        Store._assert_state_binding(state, binding)
        public = json.loads(row['public_json'])
        if public != eng.public_state(state):
            raise StudyError('recorded_frame_mismatch', 503)
        decision = json.loads(row['decision_json'])
        if state['terminal']:
            if decision:
                raise StudyError('recorded_decision_mismatch', 503)
        elif decision != Store._verified_decision(eng, state, binding):
            raise StudyError('recorded_decision_mismatch', 503)
        return state, decision, public

    @staticmethod
    def _recorded_transition(eng, previous, current, binding):
        before, decision, _ = Store._recorded_frame(eng, previous, binding)
        after, _, _ = Store._recorded_frame(eng, current, binding)
        action = previous.get('human_action')
        if (not isinstance(action, str) or before['terminal']
                or action not in eng.legal_actions(before)):
            raise StudyError('recorded_transition_mismatch', 503)
        try:
            replayed = eng.step(before, action, decision)
        except (RuntimeError, ValueError, KeyError) as exc:
            raise StudyError('recorded_transition_mismatch', 503) from exc
        if replayed != after:
            raise StudyError('recorded_transition_mismatch', 503)

    def _verified_run_prefix(self, db, eng, run, binding, turn, scenario_seed):
        """Replay a bounded v5 history from its seeded, frozen NN start state."""
        if (type(turn) is not int or turn < 0 or run['seed'] != scenario_seed):
            raise StudyError('recorded_initial_mismatch', 503)
        rows = db.all('SELECT turn,state_json,public_json,decision_json,human_action '
                      'FROM pl3_frames WHERE run_id=? AND turn<=? ORDER BY turn',
                      (run['id'], turn))
        if len(rows) != turn + 1 or any(row['turn'] != index for index, row in enumerate(rows)):
            raise StudyError('recorded_transition_mismatch', 503)
        identity = (run['id'], run['seed'], run['task'], scenario_seed,
                    binding['policy_id'], binding['actor_sha256'],
                    binding['human_actor_sha256'], binding['ai_actor_sha256'])
        fingerprints = tuple(digest(encode((row['turn'], row['state_json'], row['public_json'],
                                            row['decision_json'], row['human_action']))) for row in rows)
        with self._frame_cache_lock:
            cached = self._verified_frame_prefixes.get(run['id'])
        if cached is not None and cached['identity'] == identity:
            old = cached['fingerprints']
            unchanged = next((index for index in range(min(len(old), len(fingerprints)))
                              if old[index] != fingerprints[index]), min(len(old), len(fingerprints)))
            max_turns = cached['max_turns']
        else:
            unchanged, max_turns = 0, None
        if unchanged == 0:
            try:
                initial = json.loads(encode(eng.initial_state(
                    run['seed'], run['task'], binding['policy_id'], release_id=RELEASE_ID)))
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                raise StudyError('policy_unavailable', 503) from exc
            max_turns = initial['max_turns']
            first, _, _ = self._recorded_frame(eng, rows[0], binding)
            if first != initial:
                raise StudyError('recorded_initial_mismatch', 503)
        if turn > max_turns:
            raise StudyError('recorded_transition_mismatch', 503)
        for index in range(max(1, unchanged), len(rows)):
            self._recorded_transition(eng, rows[index - 1], rows[index], binding)
        with self._frame_cache_lock:
            previous = self._verified_frame_prefixes.get(run['id'])
            if previous is None or len(previous['fingerprints']) <= len(fingerprints) or previous['identity'] != identity:
                self._verified_frame_prefixes[run['id']] = {
                    'identity': identity, 'fingerprints': fingerprints, 'max_turns': max_turns}
            self._verified_frame_prefixes.move_to_end(run['id'])
            if len(self._verified_frame_prefixes) > 128:
                self._verified_frame_prefixes.popitem(last=False)
        return rows

    def _participant(self, db, token):
        if not token: raise StudyError('session_required',401)
        row=db.one('SELECT p.* FROM pl3_sessions s JOIN pl3_participants p ON p.id=s.participant_id WHERE s.token_hash=?',(digest(token),))
        if row is None: raise StudyError('session_required',401)
        return row

    def _instance(self, db, token, instance_id):
        row=self._owned_instance(db,token,instance_id)
        if not compatible_instance(row): raise StudyError('release_changed',409)
        if db.one('SELECT instance_id FROM pl3_prolific_releases WHERE instance_id=?',(row['id'],)):
            raise StudyError('prolific_submission_closed',409)
        return row

    def _owned_instance(self,db,token,instance_id):
        participant=self._participant(db,token)
        row=db.one('SELECT * FROM pl3_instances WHERE id=? AND participant_id=?',(instance_id,participant['id']))
        if row is None: raise StudyError('study_not_found',404)
        # Historic instances remain visible through the read-only archive view.
        return row

    def _ask_allowed(self, db, instance):
        if instance['group_code']!='A' or instance['stage']!='task2' or not instance['current_run']: return False
        row=db.one('SELECT status FROM pl3_runs WHERE id=?',(instance['current_run'],))
        return bool(row and row['status']=='active')

    def create(self, payload, token=None, admin=False, *, _db=None, _prolific=False):
        if payload.get('consent') is not True: raise StudyError('consent_required')
        domain=payload.get('domain')
        if domain not in MODULES: raise StudyError('unknown_domain')
        name=str(payload.get('participant_id','')).strip().casefold()
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{2,63}',name): raise StudyError('invalid_participant_id')
        mode=payload.get('mode','pilot')
        if mode not in ('pilot','preview','test'): raise StudyError('invalid_mode')
        if mode!='pilot' and not admin: raise StudyError('researcher_access_required',403)
        if mode=='pilot' and not self.ready: raise StudyError('study_not_ready',503)
        if mode=='pilot' and self.settings.prolific_study_id and not _prolific:
            raise StudyError('use_prolific_entry',403)
        requested_group=payload.get('group')
        if requested_group is not None and requested_group not in ('A','B','C'):
            raise StudyError('invalid_group')
        if mode=='pilot' and requested_group is not None and not _prolific:
            raise StudyError('group_override_forbidden',403)
        language='zh' if payload.get('language')=='zh' else 'en'
        recovery=None
        with (nullcontext(_db) if _db is not None else self.db.transaction()) as db:
            if not _prolific:
                linked=db.one('SELECT participant_id FROM pl3_prolific_links WHERE participant_id=?',(name,))
                if token:
                    linked=linked or db.one('SELECT l.participant_id FROM pl3_prolific_links l JOIN pl3_sessions s ON s.participant_id=l.participant_id WHERE s.token_hash=?',(digest(token),))
                if linked: raise StudyError('prolific_assignment_locked',409)
            existing=db.one('SELECT * FROM pl3_participants WHERE id=?',(name,))
            authenticated=None
            if token:
                try: authenticated=self._participant(db,token)
                except StudyError: pass
            if existing:
                code=str(payload.get('recovery_code',''))
                if not (authenticated and authenticated['id']==name) and not (code and hmac.compare_digest(digest(code),existing['recovery_hash'])):
                    raise StudyError('participant_exists_use_recovery',409)
                if db.one('SELECT id FROM pl3_instances WHERE participant_id=? AND mode!=?',(name,mode)):
                    raise StudyError('participant_mode_conflict',409)
            if not existing:
                recovery=secrets.token_urlsafe(18)
                db.execute('INSERT INTO pl3_participants VALUES(?,?,?,?)',(name,'unassigned',digest(recovery),time.time()))
            if not authenticated or authenticated['id']!=name:
                token=secrets.token_urlsafe(32)
                db.execute('INSERT INTO pl3_sessions VALUES(?,?,?)',(digest(token),name,time.time()))
            # Switching domains never rewrites or finishes another instance.
            # A rules revision starts a separate enrollment. Old runs, frames,
            # answers and questionnaires remain intact under their release ID.
            candidates=db.all('SELECT * FROM pl3_instances WHERE participant_id=? AND domain=? ORDER BY created DESC,id DESC',(name,domain))
            instance=next((r for r in candidates if compatible_instance(r)),None)
            if instance and requested_group and requested_group != instance['group_code']:
                raise StudyError('group_binding_immutable',409)
            if not instance:
                iid=uid()
                config=scenario_config(domain)
                # A pilot allocation is distinct from a one-use model test set.
                # Historical domain configs retain their older key.
                seeds=(config.get('pilot_scenario_seeds') or
                       config.get('heldout_seeds') or config['held_out_seeds'])
                allocations=db.all('SELECT scenario_seed,group_code,COUNT(*) AS n FROM pl3_instances WHERE domain=? AND mode=? AND release_id=? GROUP BY scenario_seed,group_code',(domain,mode,RELEASE_ID))
                usage={(r['scenario_seed'],r['group_code']):r['n'] for r in allocations}
                groups=('A','B','C')
                group_totals={group:sum(usage.get((seed,group),0) for seed in seeds)
                              for group in groups}
                seed_totals={seed:sum(usage.get((seed,group),0) for group in groups)
                             for seed in seeds}
                if requested_group:
                    group=requested_group
                    seed=secrets.choice([candidate for candidate in seeds
                        if (seed_totals[candidate],usage.get((candidate,group),0))
                        == min((seed_totals[s],usage.get((s,group),0)) for s in seeds)])
                else:
                    # Keep every scenario seed's A/B/C cells within one of one
                    # another, then balance domain-wide groups and seed exposure.
                    eligible=[(seed,group) for seed in seeds for group in groups
                        if usage.get((seed,group),0)
                        == min(usage.get((seed,g),0) for g in groups)]
                    best=min((group_totals[group],seed_totals[seed])
                             for seed,group in eligible)
                    seed,group=secrets.choice([(seed,group) for seed,group in eligible
                        if (group_totals[group],seed_totals[seed])==best])
                try:
                    binding=binding_for_group(domain,group)
                except PolicyUnavailable as exc:
                    raise StudyError('policy_unavailable',503) from exc
                if not existing:
                    db.execute('UPDATE pl3_participants SET group_code=? WHERE id=?',(group,name))
                db.execute('INSERT INTO pl3_instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (iid,name,domain,RELEASE_ID,group,mode,'demo',0,None,0,language,seed,time.time(),None))
                db.execute('INSERT INTO pl3_policy_bindings VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (iid,binding['policy_id'],domain,binding['variant'],binding['actor_path'],
                     binding['actor_sha256'],binding['human_actor_sha256'],
                     binding['ai_actor_sha256'],encode(binding['program_hashes']),time.time()))
                if domain == 'warehouse':
                    model_version = binding.get('actor_model_version')
                    schema = binding.get('feature_schema_sha256')
                    if not model_version or not schema:
                        raise StudyError('policy_binding_metadata_missing',503)
                    db.execute('INSERT INTO pl3_policy_binding_metadata VALUES(?,?,?,?)',
                        (iid,model_version,schema,time.time()))
                assignment='researcher_override' if requested_group else 'randomized_balanced'
                db.execute('INSERT INTO pl3_enrollments VALUES(?,?,?,?,?,?)',(iid,1,language,assignment,config.get('scenario_version',config['version']),time.time()))
                db.execute('INSERT INTO pl3_reward_settings VALUES(?,?)',(iid,encode(rewards.policy(domain))))
                if self.settings.optional_prompts:
                    db.execute('INSERT INTO pl3_prompt_settings VALUES(?,?)',(iid,'optional-explanations-required-ratings-v2'))
                if self.settings.automatic_explanations:
                    db.execute('INSERT INTO pl3_auto_explanation_settings VALUES(?,?,?)',
                        (iid,automatic_explanations.VERSION,time.time()))
                if self.settings.understanding_ratings:
                    db.execute('INSERT INTO pl3_understanding_settings VALUES(?,?,?,?,?)',
                        (iid,understanding.VERSION,encode(understanding.TASKS),encode(understanding.GROUPS),time.time()))
                instance=db.one('SELECT * FROM pl3_instances WHERE id=?',(iid,))
                if domain=='kitchen':
                    now=time.time()
                    db.execute('INSERT INTO pl3_tutorials VALUES(?,?,?,?,?)',
                        (iid,kitchen_tutorial.VERSION,encode(kitchen_tutorial.initial()),now,now))
            result=self._view(db,instance)
            result['recovery_code']=recovery
        return token,result

    def recover_view(self, token, domain=None):
        with self.db.transaction(read_only=True) as db:
            p=self._participant(db,token)
            rows=db.all('SELECT * FROM pl3_instances WHERE participant_id=? ORDER BY created DESC,id DESC',(p['id'],))
            if domain:
                instance=next((r for r in rows if r['domain']==domain and compatible_instance(r)),None)
            else:
                compatible=[r for r in rows if compatible_instance(r)]
                instance=next((r for r in compatible if r['stage']!='completed'),compatible[0] if compatible else None)
            if not instance: return {'participant_id':p['id'],'stage':'welcome',
                'previous_version_saved':any(not compatible_instance(r) and (not domain or r['domain']==domain) for r in rows),
                'archived_instances':[{'id':r['id'],'release_id':r['release_id'],'stage':r['stage']}
                    for r in rows if r['domain'] in ('warehouse','kitchen','pong') and not compatible_instance(r)
                    and (not domain or domain==r['domain'])]}
            if not compatible_instance(instance): raise StudyError('release_changed',409)
            return self._view(db,self._instance(db,token,instance['id']))

    def view(self, token, instance_id):
        with self.db.transaction(read_only=True) as db:
            instance=self._owned_instance(db,token,instance_id)
            if instance['domain'] in ('warehouse','kitchen','pong') and not compatible_instance(instance):
                return self._archive_view(db,instance)
            if not compatible_instance(instance):raise StudyError('release_changed',409)
            return self._view(db,instance)

    def _archive_view(self,db,instance):
        """Read stored historical frames without applying the current engine."""
        runs=db.all('SELECT id,task,status,score_json FROM pl3_runs WHERE instance_id=? ORDER BY task',(instance['id'],))
        task_runs=[]
        latest=None
        for run in runs:
            frame=db.one('SELECT turn,public_json FROM pl3_frames WHERE run_id=? ORDER BY turn DESC LIMIT 1',(run['id'],))
            task_runs.append({'id':run['id'],'task':run['task'],'status':run['status'],
                              'score':json.loads(run['score_json']),'turn':frame['turn'] if frame else 0})
            if frame:latest=(run,frame)
        questions=db.all('SELECT id,question,language,result_json,status,target_turn,target_run FROM pl3_questions WHERE instance_id=? ORDER BY requested',(instance['id'],))
        return {'id':instance['id'],'instance_id':instance['id'],
                'participant_id':instance['participant_id'],'domain':instance['domain'],
                'stage':'archived','previous_stage':instance['stage'],
                'revision':instance['revision'],'language':instance['language'],
                'mode':instance['mode'],'release_id':instance['release_id'],
                'group':instance['group_code'],'archived':True,'can_ask':False,
                'actions':[],'run_id':latest[0]['id'] if latest else None,
                'state':json.loads(latest[1]['public_json']) if latest else None,
                'task_runs':task_runs,'questions':[{'id':q['id'],'question':q['question'],
                    'language':q['language'],'result':public_answer(json.loads(q['result_json'])),
                    'status':q['status'],'target_turn':q['target_turn'],'target_run':q['target_run']}
                    for q in questions]}

    def _view(self,db,instance):
        eng=engine(instance['domain'])
        binding=self._binding(db,instance)
        language=instance['language']
        enrollment=db.one('SELECT assignment_source FROM pl3_enrollments WHERE instance_id=?',(instance['id'],))
        result={k:instance[k] for k in ('id','domain','stage','revision','language','mode','release_id')}
        result.update(instance_id=instance['id'],participant_id=instance['participant_id'],
            can_ask=self._ask_allowed(db,instance),rules=eng.rules(language),
            task2_explanation_notice=instance['stage']=='task2' and instance['group_code']=='A' and not self._ask_allowed(db,instance),
            task_runs=[],questions=[],automatic_explanations=[],state=None,run_id=instance['current_run'],
            automatic_explanations_enabled=bool(db.one('SELECT instance_id FROM pl3_auto_explanation_settings WHERE instance_id=?',(instance['id'],))),
            group=instance['group_code'],group_selection_locked=True,
            group_assignment_source=enrollment['assignment_source'] if enrollment else None,
            policy_id=binding['policy_id'],actor_sha256=binding['actor_sha256'],
            ai_actor_sha256=binding['ai_actor_sha256'],human_actor_sha256=binding['human_actor_sha256'])
        runs=db.all('SELECT id,task,status,score_json,state_json FROM pl3_runs WHERE instance_id=? ORDER BY task',(instance['id'],))
        result['task_runs']=[]
        for run in runs:
            recorded_state=json.loads(run['state_json'])
            self._assert_state_binding(recorded_state,binding)
            result['task_runs'].append({'id':run['id'],'task':run['task'],'status':run['status'],
                'score':eng.public_state(recorded_state)['score'],'turn':recorded_state['turn']})
        result['optional_prompts']=self._optional_prompts(db,instance)
        result['rewards']=rewards.summary(db,instance['id'])
        if instance['stage']=='demo':
            demo=demonstration(instance['domain'],binding['policy_id'])
            result['demo']={'index':instance['demo_index'],'captions':demo['captions'],'frames':demo['frames']}
        if instance['stage']=='practice':
            row=db.one('SELECT state_json FROM pl3_tutorials WHERE instance_id=?',(instance['id'],))
            practice=json.loads(row['state_json'])
            result['tutorial']=kitchen_tutorial.view(practice)
            result['state']=result['tutorial']['state']
            result['actions']=eng.legal_actions(practice['state'])
        if instance['domain']=='kitchen':
            result['public_help']=kitchen_tutorial.public_help()
            result['rule_metadata']=eng.rule_metadata(binding['policy_id'])
            result['rules']=[row[language] for row in result['public_help']]
        if instance['current_run']:
            run=db.one('SELECT * FROM pl3_runs WHERE id=?',(instance['current_run'],))
            state=json.loads(run['state_json'])
            self._assert_state_binding(state,binding)
            history=self._verified_run_prefix(db,eng,run,binding,state['turn'],instance['scenario_seed'])
            if self._recorded_frame(eng,history[-1],binding)[0] != state:
                raise StudyError('recorded_frame_mismatch',503)
            result.update(state=eng.public_state(state),actions=eng.legal_actions(state) if not state['terminal'] else [],run_status=run['status'])
            if hasattr(eng,'action_label'):
                result['action_labels']={a:eng.action_label(a,language) for a in result['actions']}
        protocol=db.one('SELECT version FROM pl3_auto_explanation_settings WHERE instance_id=?',(instance['id'],))
        result['automatic_explanation_protocol']=protocol['version'] if protocol else None
        rating_config=db.one('SELECT * FROM pl3_understanding_settings WHERE instance_id=?',(instance['id'],))
        result['understanding_rating_enabled']=bool(rating_config and instance['stage'].startswith('task')
            and int(instance['stage'][-1]) in json.loads(rating_config['tasks_json'])
            and instance['group_code'] in json.loads(rating_config['groups_json']))
        result['understanding_rating']=understanding.public(self._pending_understanding(db,instance),language)
        if result['can_ask']:
            result['automatic_explanations']=[automatic_explanations.public_card(r,language)
                for r in db.all('SELECT a.*,c.confirmed,s.skipped FROM pl3_auto_explanations a LEFT JOIN pl3_auto_explanation_confirmations c ON c.explanation_id=a.id LEFT JOIN pl3_prompt_skips s ON s.prompt_id=a.id WHERE a.run_id=? ORDER BY a.turn,a.id',(instance['current_run'],))]
            result['questions']=[{'id':q['id'],'question':q['question'],'language':q['language'],
                'result':public_answer(json.loads(q['result_json'])),'status':q['status'],'target_turn':q['target_turn'],'target_run':q['target_run']}
                for q in db.all("SELECT * FROM pl3_questions WHERE authorized_run=? AND status IN ('answered','clarification','unavailable') ORDER BY requested",(instance['current_run'],))]
        if instance['stage']=='questionnaire':
            items=COMMON_ITEMS+(EXPLANATION_ITEMS if instance['group_code']=='A' else [])
            if result['automatic_explanations_enabled'] and instance['group_code']=='A':
                items=COMMON_ITEMS+[
                    ('relevant','The Task 2 explanations were relevant to the situation.','Task 2 的解释与当时的情境相关。'),
                    ('clear','The Task 2 explanations were easy to understand.','Task 2 的解释容易理解。'),
                    ('helpful','The Task 2 explanations helped me choose my next action.','Task 2 的解释帮助我选择下一步动作。')]
            result['questionnaire']={'items':[{'id':i[0],'text':i[2 if language=='zh' else 1],'allow_na':i[0] in {q[0] for q in EXPLANATION_ITEMS}} for i in items],
                'comprehension':[]}
        link=db.one('SELECT study_id FROM pl3_prolific_links WHERE instance_id=?',(instance['id'],))
        if instance['mode']=='preview' and instance['participant_id'].startswith('review-'):
            result['full_study_preview']=True
            result['questionnaire_optional']=True
        if link:
            result['prolific']=True
            result['questionnaire_optional']=True
            if instance['stage']=='completed':
                from .prolific import cohort
                try:code,_=cohort(db,self.settings,link['study_id'])
                except StudyError:code=''
                if re.fullmatch(r'[A-Za-z0-9]{4,64}',code):
                    result['completion_code']=code
                    result['completion_url']='https://app.prolific.com/submissions/complete?cc='+code
        return result

    def command(self,token,kind,payload):
        command_id=payload.get('command_id')
        if not isinstance(command_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,100}',command_id): raise StudyError('command_id_required')
        fingerprint=digest(encode({'kind':kind,'payload':payload}))
        with self.db.transaction(scope=payload.get('instance_id')) as db:
            instance=self._instance(db,token,payload.get('instance_id'))
            prior=db.one('SELECT * FROM pl3_commands WHERE session_hash=? AND command_id=?',(digest(token),command_id))
            if prior:
                if prior['request_hash']!=fingerprint: raise StudyError('idempotency_conflict',409)
                # Never return a cached response containing now-revoked answers.
                return self._view(db,instance)
            if payload.get('revision')!=instance['revision']: raise StudyError('stale_state',409)
            if self._pending_understanding(db,instance) and kind not in ('understanding_rating','language','timing'):
                raise StudyError('understanding_rating_required',409)
            timings=payload.get('timings',[])
            if not isinstance(timings,list) or len(timings)>4:raise StudyError('invalid_timing')
            for measurement in timings:
                if not isinstance(measurement,dict):raise StudyError('invalid_timing')
                seconds=measurement.get('seconds');time_kind=measurement.get('kind')
                if time_kind not in ('active','reading','replay','understanding_rating') or type(seconds) not in (int,float) or not 0<=seconds<=86400:raise StudyError('invalid_timing')
                task=int(instance['stage'][-1]) if instance['stage'].startswith('task') else None
                db.execute('INSERT INTO pl3_timings VALUES(?,?,?,?,?,?)',(uid(),instance['id'],task,time_kind,seconds,time.time()))
            if kind=='demo_next':
                if instance['stage']!='demo': raise StudyError('wrong_stage',409)
                binding=self._binding(db,instance)
                end=len(demonstration(instance['domain'],binding['policy_id'])['captions'])
                db.execute('UPDATE pl3_instances SET demo_index=? WHERE id=?',(min(instance['demo_index']+1,end),instance['id']))
            elif kind in ('demo_finish','demo_skip'):
                if instance['stage']!='demo': raise StudyError('wrong_stage',409)
                binding=self._binding(db,instance)
                end=len(demonstration(instance['domain'],binding['policy_id'])['captions'])
                db.execute('UPDATE pl3_instances SET demo_index=? WHERE id=?',(end,instance['id']))
                if kind=='demo_skip':
                    # Skip is an explicit participant choice, recorded as its
                    # own idempotent command, never a simulated task action.
                    self._next(db,{**instance,'demo_index':end})
            elif kind=='practice_skip':
                if instance['domain']!='kitchen' or instance['stage']!='practice':raise StudyError('wrong_stage',409)
                self._tutorial(db,instance,{'command':'skip'})
                self._next(db,instance)
            elif kind=='next':
                if self._optional_prompts(db,instance):self._skip_prompts(db,instance,'next_task')
                self._next(db,instance)
            elif kind=='skip_prompts':
                if not self._optional_prompts(db,instance):raise StudyError('wrong_stage',409)
                self._skip_prompts(db,instance,'skip_button')
            elif kind=='tutorial': self._tutorial(db,instance,payload)
            elif kind=='action': self._action(db,instance,payload)
            elif kind=='understanding_rating': self._submit_understanding(db,instance,payload)
            elif kind=='confirm_explanation': self._confirm_automatic(db,instance,payload)
            elif kind=='open_question_guide': self._open_question_guide(db,instance,payload)
            elif kind=='request_explanation': self._request_automatic(db,instance,payload)
            elif kind=='language':
                if payload.get('language') not in ('en','zh'): raise StudyError('invalid_language')
                db.execute('UPDATE pl3_instances SET language=? WHERE id=?',(payload['language'],instance['id']))
            elif kind=='questionnaire': self._questionnaire(db,instance,payload)
            elif kind=='timing':
                seconds=payload.get('seconds')
                if not isinstance(seconds,(int,float)) or not 0<=seconds<=86400: raise StudyError('invalid_timing')
                timing_kind=payload.get('kind')
                if timing_kind not in ('reading','replay','active','explanation_wait','understanding_rating'):raise StudyError('invalid_timing')
                task=int(instance['stage'][-1]) if instance['stage'].startswith('task') else None
                db.execute('INSERT INTO pl3_timings VALUES(?,?,?,?,?,?)',(uid(),instance['id'],task,timing_kind,seconds,time.time()))
            else: raise StudyError('unknown_command',404)
            db.execute('UPDATE pl3_instances SET revision=revision+1 WHERE id=?',(instance['id'],))
            updated=db.one('SELECT * FROM pl3_instances WHERE id=?',(instance['id'],))
            db.execute('INSERT INTO pl3_commands VALUES(?,?,?,?,?)',(digest(token),command_id,fingerprint,'{}',time.time()))
            return self._view(db,updated)

    def _tutorial(self,db,instance,payload):
        if instance['domain']!='kitchen' or instance['stage']!='practice':raise StudyError('wrong_stage',409)
        row=db.one('SELECT state_json FROM pl3_tutorials WHERE instance_id=?',(instance['id'],))
        before=json.loads(row['state_json'])
        try:after=kitchen_tutorial.apply(before,payload.get('command'),payload.get('action'))
        except ValueError as exc:raise StudyError(str(exc)) from exc
        now=time.time()
        db.execute('UPDATE pl3_tutorials SET version=?,state_json=?,updated=? WHERE instance_id=?',(after['version'],encode(after),now,instance['id']))
        db.execute('INSERT INTO pl3_tutorial_events VALUES(?,?,?,?,?,?,?,?)',
            (uid(),instance['id'],kitchen_tutorial.VERSION,payload.get('command'),
             encode({'action':payload.get('action'),'language':instance['language']}),encode(before),encode(after),now))
        if after['completed'] or after['skipped']:
            db.execute('UPDATE pl3_instances SET demo_index=? WHERE id=?',(kitchen_tutorial.COUNT,instance['id']))
        else:
            db.execute('UPDATE pl3_instances SET demo_index=? WHERE id=?',(after['index'],instance['id']))

    def _next(self,db,instance):
        stage=instance['stage']
        if stage=='demo':
            binding=self._binding(db,instance)
            if instance['demo_index']<len(demonstration(instance['domain'],binding['policy_id'])['captions']):raise StudyError('finish_demo',409)
            if instance['domain']=='kitchen':
                db.execute("UPDATE pl3_instances SET stage='practice',demo_index=0 WHERE id=?",(instance['id'],))
                return
            task=1
        elif stage=='practice':
            if instance['domain']!='kitchen':raise StudyError('wrong_stage',409)
            row=db.one('SELECT state_json FROM pl3_tutorials WHERE instance_id=?',(instance['id'],))
            before=json.loads(row['state_json']);practice=kitchen_tutorial.normalize(before)
            if not (practice['completed'] or practice['skipped']):raise StudyError('finish_demo',409)
            if practice!=before:self._tutorial(db,instance,{'command':'finish'})
            task=1
        elif stage in ('task1','task2','task3'):
            run=db.one('SELECT * FROM pl3_runs WHERE id=?',(instance['current_run'],))
            if run['status']!='completed': raise StudyError('finish_task',409)
            if stage=='task3':
                db.execute("UPDATE pl3_instances SET stage='questionnaire' WHERE id=?",(instance['id'],))
                return
            task=int(stage[-1])+1
        else: raise StudyError('wrong_stage',409)
        eng=engine(instance['domain'])
        binding=self._binding(db,instance)
        try:
            state=eng.initial_state(instance['scenario_seed'],task,binding['policy_id'],
                                    release_id=RELEASE_ID)
        except (OSError, RuntimeError) as error:
            raise StudyError('policy_unavailable',503) from error
        self._assert_state_binding(state,binding)
        rid=uid(); now=time.time()
        db.execute('INSERT INTO pl3_runs VALUES(?,?,?,?,?,?,?,?,?)',
            (rid,instance['id'],task,instance['scenario_seed'],encode(state),'active',encode(eng.score(state)),now,None))
        decision=self._verified_decision(eng,state,binding)
        db.execute('INSERT INTO pl3_frames VALUES(?,?,?,?,?,?,?)',
            (rid,0,encode(state),encode(eng.public_state(state)),encode(decision),None,now))
        if task==2:self._record_automatic(db,instance,rid,state,decision)
        db.execute('UPDATE pl3_instances SET stage=?,current_run=? WHERE id=?',('task'+str(task),rid,instance['id']))

    def _action(self,db,instance,payload):
        if instance['stage'] not in ('task1','task2','task3'): raise StudyError('wrong_stage',409)
        if payload.get('run_id')!=instance['current_run']:raise StudyError('wrong_run',409)
        run=db.one('SELECT * FROM pl3_runs WHERE id=?',(instance['current_run'],))
        if run['status']!='active':raise StudyError('task_finished',409)
        if self._pending_automatic(db,instance) and not self._optional_prompts(db,instance):raise StudyError('explanation_confirmation_required',409)
        state=json.loads(run['state_json']); eng=engine(instance['domain'])
        binding=self._binding(db,instance)
        self._assert_state_binding(state,binding)
        history=self._verified_run_prefix(db,eng,run,binding,state['turn'],instance['scenario_seed'])
        if self._recorded_frame(eng,history[-1],binding)[0] != state:
            raise StudyError('recorded_frame_mismatch',503)
        if payload.get('turn')!=state['turn']:raise StudyError('stale_state',409)
        action=payload.get('action')
        if action not in eng.legal_actions(state):raise StudyError('illegal_action')
        if self._optional_prompts(db,instance):self._skip_prompts(db,instance,'continued_playing')
        decision=self._verified_decision(eng,state,binding)
        nxt=eng.step(state,action,decision)
        self._assert_state_binding(nxt,binding)
        db.execute('UPDATE pl3_frames SET human_action=?,decision_json=? WHERE run_id=? AND turn=?',
            (action,encode(decision),run['id'],state['turn']))
        now=time.time(); terminal=nxt['terminal']
        db.execute('UPDATE pl3_runs SET state_json=?,status=?,score_json=?,ended=? WHERE id=?',
            (encode(nxt),'completed' if terminal else 'active',encode(eng.score(nxt)),now if terminal else None,run['id']))
        next_decision=self._verified_decision(eng,nxt,binding) if not terminal else {}
        db.execute('INSERT INTO pl3_frames VALUES(?,?,?,?,?,?,?)',
            (run['id'],nxt['turn'],encode(nxt),encode(eng.public_state(nxt)),encode(next_decision),None,now))
        if run['task']==2:self._record_automatic(db,instance,run['id'],nxt,next_decision,decision)
        self._record_understanding(db,instance,run,nxt)
        if terminal:
            db.execute("UPDATE pl3_questions SET status='revoked' WHERE authorized_run=? AND status='pending'",(run['id'],))

    def _optional_prompts(self,db,instance):
        return bool(db.one('SELECT instance_id FROM pl3_prompt_settings WHERE instance_id=?',(instance['id'],)))

    def _skip_prompts(self,db,instance,reason):
        # Missing responses remain missing. Never insert a confirmation or rating.
        if self._required_question_guide(db,instance):raise StudyError('explanation_confirmation_required',409)
        for kind,row in [('explanation',self._pending_automatic(db,instance))]:
            if row:
                db.execute('INSERT INTO pl3_prompt_skips VALUES(?,?,?,?,?,?)',
                    (row['id'],instance['id'],instance['current_run'],kind,reason,time.time()))

    def _pending_understanding(self,db,instance):
        return db.one('SELECT r.* FROM pl3_understanding_ratings r WHERE instance_id=? AND run_id=? AND submitted IS NULL AND NOT EXISTS (SELECT 1 FROM pl3_prompt_skips s WHERE s.prompt_id=r.id) ORDER BY checkpoint LIMIT 1',
            (instance['id'],instance['current_run']))

    def _record_understanding(self,db,instance,run,state):
        config=db.one('SELECT * FROM pl3_understanding_settings WHERE instance_id=?',(instance['id'],))
        if not config or config['version']!=understanding.VERSION:return
        if run['task'] not in json.loads(config['tasks_json']) or instance['group_code'] not in json.loads(config['groups_json']):return
        checkpoint=understanding.checkpoint(state)
        if checkpoint is None:return
        if db.one('SELECT id FROM pl3_understanding_ratings WHERE run_id=? AND checkpoint=?',(run['id'],checkpoint)):return
        db.execute('INSERT INTO pl3_understanding_ratings VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (uid(),instance['id'],run['id'],run['task'],checkpoint,state['turn'],state['max_turns'],config['version'],time.time(),None,None,None))

    def _submit_understanding(self,db,instance,payload):
        row=self._pending_understanding(db,instance)
        if not row or row['id']!=payload.get('rating_id'):raise StudyError('understanding_rating_unavailable',409)
        rating=payload.get('rating')
        if type(rating) is not int or not 1<=rating<=5:raise StudyError('invalid_understanding_rating')
        db.execute('UPDATE pl3_understanding_ratings SET rating=?,language=?,submitted=? WHERE id=?',
            (rating,instance['language'],time.time(),row['id']))

    def _record_automatic(self,db,instance,run_id,state,decision,previous=None):
        if instance['group_code']!='A':return
        config=db.one('SELECT version FROM pl3_auto_explanation_settings WHERE instance_id=?',(instance['id'],))
        if not config or config['version'] not in automatic_explanations.SUPPORTED_VERSIONS:return
        card=automatic_explanations.candidate(instance['domain'],state,decision,previous,early_collision=config['version'] in ('sidebar-early-question-guide-v5',automatic_explanations.GUIDED_VERSION))
        if config['version']=='guided-question-choice-v3' and state['turn']==0 and not state['terminal']:
            card=card or {'trigger_key':'turn:0','trigger_types':[], 'turn':0,
                         'body':{'en':decision['reason_en'],'zh':decision['reason_zh']},
                         'reason_code':decision.get('reason_code')}
            card['onboarding']=True
            card['trigger_types'].append('guided_question')
        if not card:return
        if config['version'] in automatic_explanations.FIRST_NODE_VERSIONS:
            card['onboarding']=not bool(db.one('SELECT id FROM pl3_auto_explanations WHERE run_id=?',(run_id,)))
        if 'collision_risk' in card['trigger_types']:
            prior=db.all('SELECT content_json FROM pl3_auto_explanations WHERE run_id=?',(run_id,))
            if any('collision_risk' in json.loads(r['content_json'])['trigger_types'] for r in prior):
                card['trigger_types'].remove('collision_risk')
                if not card['trigger_types']:return
        if instance['domain']=='warehouse' and any(t.startswith('charge_') for t in card['trigger_types']):
            prior=db.all('SELECT content_json FROM pl3_auto_explanations WHERE run_id=?',(run_id,))
            charge_seen=any(any(t.startswith('charge_') for t in json.loads(r['content_json'])['trigger_types']) for r in prior)
            if charge_seen:
                # One charging explanation per Task 2, persisted across reloads.
                # A simultaneous collision or detour remains an independent node.
                card['trigger_types']=[t for t in card['trigger_types'] if not t.startswith('charge_')]
                if not card['trigger_types']:return
        if db.one('SELECT id FROM pl3_auto_explanations WHERE run_id=? AND trigger_key=?',(run_id,card['trigger_key'])):return
        # Stored with the same transaction as the action: retries cannot duplicate
        # a node, and refreshing never creates an explanation or advances play.
        action_zh={'up':'向上移动','down':'向下移动','left':'向左移动','right':'向右移动','wait':'等待','interact':'交互'}.get(decision.get('action'),decision.get('action','等待'))
        card['next_body']={lang:(f"My next action is {decision.get('action', 'wait')}. " if lang=='en' else f"我下一步会{action_zh}。")+card['body'][lang] for lang in ('en','zh')}
        card['version']=config['version']
        card['decision_sha256']=digest(encode(decision))
        card['state_sha256']=digest(encode(state))
        db.execute('INSERT INTO pl3_auto_explanations VALUES(?,?,?,?,?,?,?,?)',
            (uid(),instance['id'],run_id,state['turn'],card['trigger_key'],encode(card),time.time(),None))

    def _pending_automatic(self,db,instance):
        if not self._ask_allowed(db,instance):return None
        config=db.one('SELECT version FROM pl3_auto_explanation_settings WHERE instance_id=?',(instance['id'],))
        if not config or config['version'] not in automatic_explanations.CONFIRM_VERSIONS:return None
        return db.one('SELECT a.* FROM pl3_auto_explanations a LEFT JOIN pl3_auto_explanation_confirmations c ON c.explanation_id=a.id WHERE a.run_id=? AND c.explanation_id IS NULL AND NOT EXISTS (SELECT 1 FROM pl3_prompt_skips s WHERE s.prompt_id=a.id) ORDER BY a.turn,a.id LIMIT 1',(instance['current_run'],))

    def _required_question_guide(self,db,instance):
        row=self._pending_automatic(db,instance)
        content=json.loads(row['content_json']) if row else {}
        return content.get('version')==automatic_explanations.GUIDED_VERSION and content.get('onboarding',False)

    def _open_question_guide(self,db,instance,payload):
        row=self._pending_automatic(db,instance)
        if not row or row['id']!=payload.get('explanation_id') or not self._required_question_guide(db,instance):raise StudyError('answer_unavailable',403)
        content=json.loads(row['content_json'])
        content.setdefault('guide_opened_at',time.time())
        db.execute('UPDATE pl3_auto_explanations SET content_json=? WHERE id=?',(encode(content),row['id']))

    def _request_automatic(self,db,instance,payload):
        if not self._ask_allowed(db,instance):raise StudyError('explanations_unavailable',403)
        row=self._pending_automatic(db,instance)
        if not row or row['id']!=payload.get('explanation_id'):raise StudyError('answer_unavailable',403)
        content=json.loads(row['content_json'])
        if content.get('version') not in automatic_explanations.GUIDED_VERSIONS:raise StudyError('answer_unavailable',403)
        if payload.get('question_id') not in automatic_explanations.QUESTIONS or (payload.get('question_id')!='why' and content.get('version')!=automatic_explanations.GUIDED_VERSION):raise StudyError('invalid_question')
        if content.get('requested_at') is None:
            language=instance['language']
            content.update(requested_at=time.time(),question_id=payload['question_id'],
                           question_source='ai_question_button' if payload.get('source')=='ai_question_button' else 'question_panel',
                           question=automatic_explanations.QUESTIONS[payload['question_id']][language],question_language=language)
            db.execute('UPDATE pl3_auto_explanations SET content_json=? WHERE id=?',(encode(content),row['id']))

    def _confirm_automatic(self,db,instance,payload):
        if not self._ask_allowed(db,instance):raise StudyError('explanations_unavailable',403)
        identifier=payload.get('explanation_id')
        row=db.one('SELECT a.*,c.confirmed FROM pl3_auto_explanations a LEFT JOIN pl3_auto_explanation_confirmations c ON c.explanation_id=a.id WHERE a.id=? AND a.instance_id=? AND a.run_id=?',
            (identifier,instance['id'],instance['current_run']))
        if not row or json.loads(row['content_json']).get('version') not in automatic_explanations.CONFIRM_VERSIONS:
            raise StudyError('answer_unavailable',403)
        if row['confirmed'] is not None:return
        pending=self._pending_automatic(db,instance)
        if not pending or pending['id']!=identifier:raise StudyError('answer_unavailable',403)
        now=time.time()
        content=json.loads(row['content_json'])
        guided=content.get('version') in automatic_explanations.GUIDED_VERSIONS
        if guided:
            choice=payload.get('choice')
            if content.get('requested_at') is not None:
                if choice!='explanation':raise StudyError('invalid_explanation_choice',409)
            elif content.get('onboarding') or choice!='understood':
                raise StudyError('guided_question_required',409)
            content.update(response='read_explanation' if choice=='explanation' else 'self_reported_understood',responded_at=now)
            db.execute('UPDATE pl3_auto_explanations SET content_json=? WHERE id=?',(encode(content),identifier))
        db.execute('INSERT INTO pl3_auto_explanation_confirmations VALUES(?,?,?,?)',(identifier,instance['id'],instance['current_run'],now))
        if not guided or content.get('requested_at') is not None:
            db.execute('UPDATE pl3_auto_explanations SET displayed=COALESCE(displayed,?) WHERE id=?',(now,identifier))

    def acknowledge_automatic(self,token,instance_id,explanation_id):
        with self.db.transaction(scope=instance_id) as db:
            instance=self._instance(db,token,instance_id)
            if not self._ask_allowed(db,instance):raise StudyError('explanations_unavailable',403)
            row=db.one('SELECT id,content_json FROM pl3_auto_explanations WHERE id=? AND instance_id=? AND run_id=?',
                (explanation_id,instance_id,instance['current_run']))
            if not row:raise StudyError('answer_unavailable',403)
            content=json.loads(row['content_json'])
            if content.get('version') in automatic_explanations.GUIDED_VERSIONS and content.get('requested_at') is None:
                raise StudyError('guided_question_required',409)
            db.execute('UPDATE pl3_auto_explanations SET displayed=COALESCE(displayed,?) WHERE id=?',(time.time(),explanation_id))
        return {'ok':True}

    def _questionnaire(self,db,instance,payload):
        if instance['stage']!='questionnaire':raise StudyError('wrong_stage',409)
        answers=payload.get('answers',{})
        if not isinstance(answers,dict):raise StudyError('invalid_questionnaire')
        optional=bool(db.one('SELECT instance_id FROM pl3_prolific_links WHERE instance_id=?',(instance['id'],))) or (instance['mode']=='preview' and instance['participant_id'].startswith('review-'))
        for q in COMMON_ITEMS+(EXPLANATION_ITEMS if instance['group_code']=='A' else []):
            value=answers.get(q[0])
            if optional and value is None:
                answers[q[0]]=None
                continue
            if q in EXPLANATION_ITEMS and value=='na':continue
            if type(value) is not int or not 1<=value<=7:raise StudyError('incomplete_questionnaire')
        graded=[]
        feedback=str(payload.get('feedback',''))[:4000]
        db.execute('INSERT INTO pl3_questionnaires VALUES(?,?,?,?)',(instance['id'],encode({'ratings':answers,'feedback':feedback}),encode(graded),time.time()))
        db.execute("UPDATE pl3_instances SET stage='completed',completed=? WHERE id=?",(time.time(),instance['id']))

    def frame(self,token,instance_id,run_id,turn):
        with self.db.transaction(read_only=True) as db:
            instance=self._owned_instance(db,token,instance_id)
            if not compatible_instance(instance) and instance['domain'] not in ('warehouse','kitchen','pong'):
                raise StudyError('release_changed',409)
            row=db.one('SELECT f.* FROM pl3_frames f JOIN pl3_runs r ON r.id=f.run_id WHERE r.instance_id=? AND f.run_id=? AND f.turn=?',(instance['id'],run_id,turn))
            if not row:raise StudyError('frame_not_found',404)
            if compatible_instance(instance):
                binding=self._binding(db,instance)
                run=db.one('SELECT * FROM pl3_runs WHERE id=? AND instance_id=?',(run_id,instance['id']))
                history=self._verified_run_prefix(db,engine(instance['domain']),run,binding,turn,instance['scenario_seed'])
                public=self._recorded_frame(engine(instance['domain']),history[-1],binding)[2]
            else:
                public=json.loads(row['public_json'])
            return {'state':public,
                    'can_ask':compatible_instance(instance) and self._ask_allowed(db,instance)}

    def ask(self,token,payload):
        question=str(payload.get('question','')).strip()
        if not question or len(question)>2000:raise StudyError('invalid_question')
        qid=payload.get('question_id') or uid()
        if not isinstance(qid,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,100}',qid):raise StudyError('invalid_question_id')
        with self.db.transaction(scope=payload.get('instance_id')) as db:
            instance=self._instance(db,token,payload.get('instance_id'))
            if self._pending_understanding(db,instance):raise StudyError('understanding_rating_required',409)
            if not self._ask_allowed(db,instance):raise StudyError('explanations_unavailable',403)
            binding=self._binding(db,instance)
            if payload.get('authorized_run')!=instance['current_run']:raise StudyError('wrong_run',403)
            target=payload.get('target_run',instance['current_run']);turn=payload.get('turn')
            row=db.one('SELECT f.*,r.task FROM pl3_frames f JOIN pl3_runs r ON r.id=f.run_id WHERE r.instance_id=? AND f.run_id=? AND f.turn=? AND r.task<=2',(instance['id'],target,turn))
            if not row:raise StudyError('frame_not_found',404)
            eng=engine(instance['domain'])
            target_run=db.one('SELECT * FROM pl3_runs WHERE id=? AND instance_id=?',(target,instance['id']))
            verified_frames=self._verified_run_prefix(db,eng,target_run,binding,turn,instance['scenario_seed'])
            state,decision,_=self._recorded_frame(eng,verified_frames[-1],binding)
            # The recorded incoming decision is evidence for any question
            # about the action that produced the displayed frame.
            incoming=verified_frames[-2] if turn>0 else None
            incoming_state,incoming_decision,_=(self._recorded_frame(eng,incoming,binding)
                if incoming else (None,None,None))
            # Expire interrupted requests after both provider attempts plus a margin.
            for expired_language in ('en','zh'):
                expired={'status':'unavailable','answer':'上一次请求已中断，请重新提问。' if expired_language=='zh' else 'The previous request was interrupted. Please ask again.','evidence_ids':[],
                    'audit':{'error':'request_lease_expired','displayed':False}}
                db.execute("UPDATE pl3_questions SET status='unavailable',result_json=?,finished=? WHERE authorized_run=? AND status='pending' AND requested<? AND language=?",
                    (encode(expired),time.time(),instance['current_run'],time.time()-120,expired_language))
            prior=db.one('SELECT * FROM pl3_questions WHERE id=?',(qid,))
            if prior:
                if prior['instance_id']!=instance['id'] or prior['question']!=question or prior['target_run']!=target or prior['target_turn']!=turn:raise StudyError('idempotency_conflict',409)
                return {'id':qid,'status':prior['status'],'result':public_answer(json.loads(prior['result_json']))}
            if db.one("SELECT id FROM pl3_questions WHERE authorized_run=? AND status='pending'",(instance['current_run'],)):
                raise StudyError('question_busy',409)
            language=payload.get('language',instance['language'])
            if language not in ('en','zh'):raise StudyError('invalid_language')
            history=db.all("SELECT id,question,result_json FROM pl3_questions WHERE authorized_run=? AND status IN ('answered','clarification') ORDER BY requested DESC LIMIT 6",(instance['current_run'],))
            previous=[{'question':h['question'],'answer':json.loads(h['result_json']).get('answer','')} for h in reversed(history)]
            db.execute('INSERT INTO pl3_questions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (qid,instance['id'],instance['current_run'],target,turn,question,language,'{}','pending',time.time(),None,None))
            # A displayed frame is an AFTER-action snapshot. Its own stored
            # decision predicts the next transition; the previous row contains
            # the actual simultaneous actions that produced the displayed one.
            chronological=verified_frames[-16:]
            public_history=[self._recorded_frame(eng,frame,binding)[2]
                            for frame in chronological]
            action_context={'state':incoming_state,
                'decision':incoming_decision,
                'human_action':incoming['human_action'] if incoming else None}
        try:
            if not self.explainer:raise RuntimeError('model_unavailable')
            answer=self.explainer.answer(engine(instance['domain']),state,decision,question,language,previous,
                public_history,action_context=action_context)
            status=answer.get('status','answered')
            if status not in ('answered','clarification','unavailable'):status='unavailable'
        except Exception:
            status='unavailable';answer={'status':status,'answer':'问答暂时不可用，请稍后重试。' if language=='zh' else 'Questions are temporarily unavailable. Please try again.','evidence_ids':[]}
        answer.setdefault('audit',{}).update(context_question_ids=[h['id'] for h in reversed(history)],context_sha256=digest(encode(previous)),authorized_run=instance['current_run'],target_run=target,target_turn=turn,target_display_turn=turn,
            session_release_id=instance['release_id'],answer_release_id=RELEASE_ID,
            policy_id=binding['policy_id'],actor_sha256=binding['actor_sha256'],
            human_actor_sha256=binding['human_actor_sha256'],
            ai_actor_sha256=binding['ai_actor_sha256'],
            performed_decision_turn=turn-1 if incoming else None,
            incoming_transition_sha256=digest(encode(action_context)),authorization_at_request=True)
        # A participant response can revoke readiness when the provider fails,
        # but only the complete three-domain probe can certify it again.
        if status == 'unavailable':
            self.qa_healthy = False
        with self.db.transaction(scope=payload.get('instance_id')) as db:
            current=self._instance(db,token,instance['id'])
            self._binding(db,current)
            permitted=self._ask_allowed(db,current) and current['current_run']==instance['current_run']
            answer['audit']['authorization_at_completion']=permitted
            db.execute('UPDATE pl3_questions SET result_json=?,status=?,finished=? WHERE id=?',
                (encode(answer),status if permitted else 'revoked',time.time(),qid))
        if not permitted:raise StudyError('explanations_unavailable',403)
        return {'id':qid,'status':status,'result':public_answer(answer),'target_run':target,'target_turn':turn}

    def acknowledge_answer(self,token,instance_id,qid):
        with self.db.transaction(scope=instance_id) as db:
            instance=self._instance(db,token,instance_id)
            if not self._ask_allowed(db,instance):raise StudyError('explanations_unavailable',403)
            self._binding(db,instance)
            row=db.one('SELECT * FROM pl3_questions WHERE id=? AND instance_id=? AND authorized_run=?',(qid,instance_id,instance['current_run']))
            if not row or row['status'] not in ('answered','clarification','unavailable'):raise StudyError('answer_unavailable',403)
            db.execute('UPDATE pl3_questions SET displayed=? WHERE id=?',(time.time(),qid))
        return {'ok':True}

    def iter_export(self, release_id=None, mode=None, batch_size=64):
        """Read one consistent export without loading unrelated historical rows.

        SQL predicates select related rows before transfer. Each query uses a
        bounded cursor; callers must consume/close this iterator before sending
        a slow client response so no database transaction follows that client.
        """
        clauses, values = [], []
        if release_id:
            clauses.append('i.release_id=?'); values.append(release_id)
        if mode:
            clauses.append('i.mode=?'); values.append(mode)
        selected = ' AND '.join(clauses)
        relationships = {
            'participants': 'i.participant_id=r.id',
            'enrollments': 'i.id=r.instance_id',
            'policy_bindings': 'i.id=r.instance_id',
            'policy_binding_metadata': 'i.id=r.instance_id',
            'instances': 'i.id=r.id',
            'runs': 'i.id=r.instance_id',
            'questions': 'i.id=r.instance_id',
            'questionnaires': 'i.id=r.instance_id',
            'timings': 'i.id=r.instance_id',
            'releases': 'i.release_id=r.id',
            'tutorials': 'i.id=r.instance_id',
            'tutorial_events': 'i.id=r.instance_id',
            'auto_explanation_settings': 'i.id=r.instance_id',
            'auto_explanations': 'i.id=r.instance_id',
            'auto_explanation_confirmations': 'i.id=r.instance_id',
            'prompt_settings': 'i.id=r.instance_id',
            'prompt_skips': 'i.id=r.instance_id',
            'reward_settings': 'i.id=r.instance_id',
            'understanding_settings': 'i.id=r.instance_id',
            'understanding_ratings': 'i.id=r.instance_id',
        }
        ordering = {'prompt_settings':'r.instance_id','prompt_skips':'r.prompt_id',
                    'reward_settings':'r.instance_id','participants':'r.id',
                    'enrollments':'r.instance_id','policy_bindings':'r.instance_id',
                    'policy_binding_metadata':'r.instance_id','instances':'r.id',
                    'runs':'r.id','frames':'r.run_id,r.turn','questions':'r.id',
                    'questionnaires':'r.instance_id','timings':'r.id','releases':'r.id',
                    'tutorials':'r.instance_id','tutorial_events':'r.id',
                    'auto_explanation_settings':'r.instance_id','auto_explanations':'r.run_id,r.turn,r.id','auto_explanation_confirmations':'r.explanation_id',
                    'understanding_settings':'r.instance_id','understanding_ratings':'r.run_id,r.checkpoint'}
        with self.db.transaction(read_only=True) as db:
            for name in EXPORT_TABLES:
                # Recovery/session hashes never enter the export cursor.
                fields = 'r.id,r.group_code,r.created' if name == 'participants' else 'r.*'
                query = 'SELECT ' + fields + ' FROM pl3_' + name + ' r'
                if selected:
                    if name == 'frames':
                        query += (' WHERE EXISTS (SELECT 1 FROM pl3_runs u JOIN pl3_instances i'
                                  ' ON i.id=u.instance_id WHERE u.id=r.run_id AND ' + selected + ')')
                    else:
                        query += (' WHERE EXISTS (SELECT 1 FROM pl3_instances i WHERE '
                                  + relationships[name] + ' AND ' + selected + ')')
                query += ' ORDER BY ' + ordering[name]
                with closing(db.iterate(query, tuple(values), batch_size=batch_size)) as rows:
                    for row in rows:
                        yield {'table':name,'record':row}

    def export(self, release_id=None, mode=None):
        """Compatibility for local callers that explicitly request a dictionary.

        The HTTP endpoint uses iter_export and a disk file instead, so this
        materialized convenience API cannot grow production response memory.
        """
        result = {name:[] for name in EXPORT_TABLES}
        for item in self.iter_export(release_id, mode):
            result[item['table']].append(item['record'])
        return result
