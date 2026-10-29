"""Actual frozen-model software flows on existing public preview scenes only.

This is not task-quality evaluation, a study sample, a new scientific seed
allocation, or scientific/operational admission. All participant persistence is
temporary. No server, external question provider, or participant DB is used.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime_v642.contracts import digest
from runtime_v642.store import StudyError
from study_vnext.server import compose_preview
from study_vnext.store import CONSENT_VERSION, RATINGS
from ui.model_preview_server import SEEDS

VERSION = 'study-vnext-actual-frozen-model-software-check.v3-explicit-model-scope'
CODES = dict(warehouse=['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space'],
             pong=['KeyA', 'KeyD', 'Space'],
             kitchen=['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space', 'KeyE'])


# The runtime intentionally sanitizes OSError/TimeoutError. A software-check
# deadline must still unwind its transactions and remain distinguishable from
# a model/asset rejection, rather than being sanitized as an asset error.
class FlowDeadline(BaseException):
    pass


@contextmanager
def deadline(seconds):
    """Bound only temporary, single-threaded software transactions."""
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_signal, _frame):
        raise FlowDeadline('software flow time cap')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Check:
    def __init__(self, store):
        self.store = store
        self.calls = 0
        self.counts = {}
        self.elapsed = {}

    def call(self, method, *args, **kwargs):
        self.calls += 1
        name = getattr(method, '__name__', 'adapter_check')
        self.counts[name] = self.counts.get(name, 0) + 1
        began = perf_counter()
        try:
            with self.store.request_context():
                return method(*args, **kwargs)
        finally:
            self.elapsed[name] = self.elapsed.get(name, 0.) + perf_counter()-began

    def public(self, token, iid, private):
        return self.call(self.store.participant_view, token, iid, private)

    def no_next_action(self, token, iid, private):
        public = self.public(token, iid, private)
        if public['current'] is not None:
            assert public['current']['decision'] == {}
            assert public['current']['decision_committed'] is False
        return public

    def action(self, token, iid, private, *, label):
        current = private['current']
        before = current['decision']
        assert not current['state']['terminal']
        value = self.call(self.store.action, token, iid,
            command_id=f'{label}-{current["state"]["task"]}-{current["turn"]}',
            expected_turn=current['turn'], human_action='wait')
        record = value['current']['transition']['records'][0]
        assert record['human_action'] == 'wait'
        assert record['decision'] == before
        assert record['decision']['ai_action_overrides'] == 0
        assert record['decision']['execution_source'] == 'bound_frozen_NN_with_physical_mask'
        assert value['current']['binding'] == private['current']['binding']
        return value


def basic_case(check, domain, group):
    start = perf_counter()
    store = check.store
    enrolled = check.call(store.enroll, domain=domain, group=group, admin=True,
        participant_reference=f'actual-software-only-{domain}-{group}',
        scenario_seed=SEEDS[domain][0], consent=True, consent_version=CONSENT_VERSION,
        question_transmission_consent=False)
    token, iid = enrolled['session_token'], enrolled['instance']['id']
    initial = check.call(store.view, token, iid)
    assert initial['study']['scientific_sample'] is False
    assert initial['study']['recruitment_opened'] is False
    assert initial['study']['question_transmission_consent'] is False
    assert initial['current'] is None
    check.call(store.complete_preparation, token, iid, kind='controls', observed_codes=CODES[domain])
    demo = check.call(store.demonstration, token, iid)
    assert demo['two_frozen_NN'] is True
    assert demo['binding'] == enrolled['instance']['binding']
    info = check.call(store.view, token, iid)['study']['preparation']
    assert info['demonstration_sha256'] == digest(demo)
    assert info['demonstration_frame_count'] == len(demo['records'])
    check.call(store.complete_preparation, token, iid, kind='demonstration',
        bundle_sha256=info['demonstration_sha256'], final_frame_count=info['demonstration_frame_count'])
    check.call(store.acknowledge_demonstration, token, iid)
    view = check.call(store.start_task, token, iid)
    public = check.no_next_action(token, iid, view)
    assert public['current']['last_executed_action'] is None
    view = check.action(token, iid, view, label='basic')
    public = check.no_next_action(token, iid, view)
    assert public['current']['last_executed_action']['result_turn'] == 1
    restored = check.call(store.view, token, iid)
    assert restored['current']['state'] == view['current']['state']
    replay = check.call(store.replay, token, iid, task=1)
    assert replay['all_AI_actions_recomputed'] is True or replay.get('immutable_history_validation_cache') is True
    assert replay['frames'][-1] == {k:v for k,v in restored['current'].items() if k != 'run_id'}
    binding = enrolled['instance']['binding']
    result = dict(domain=domain, group=group, public_scene_seed=SEEDS[domain][0], passed=True,
        actor_sha256=binding['actor_sha256'], ai_actor_sha256=binding['ai_actor_sha256'],
        binding_sha256=digest(binding), demonstration_frames=len(demo['records']),
        human_AI_committed_steps=1, controls_receipt=True, playback_receipt=True,
        bound_two_NN_demonstration=True, executed_action_equals_bound_NN=True,
        no_next_action_exposed=True, restore_exact=True, replay_exact=True,
        elapsed_seconds=round(perf_counter()-start, 3))
    return result, (token, iid, view)


def complete_flow(check, domain, session, seconds, *, group='A'):
    start = perf_counter()
    token, iid, view = session
    result = dict(domain=domain, group=group, status='running', completed_tasks=0,
        wait_only=True, counts_by_task={}, historical_question_checked=False,
        answers_revoked_in_Task3=False, survey_saved=False, understanding_saved=False,
        scientific_task_quality_evaluated=False, maximum_seconds=seconds)
    store = check.store
    try:
        with deadline(seconds):
            for task in (1, 2, 3):
                if task > 1:
                    view = check.call(store.start_task, token, iid)
                check.no_next_action(token, iid, view)
                if task == 2 and group == 'A':
                    before = digest(view['current']['state'])
                    answer = check.call(store.ask, token, iid,
                        question='请说明 Task 1 已执行第一步的队友动作；不能推断的动机请明确说明。',
                        target_task=1, decision_turn=0, human_actions=['wait'])
                    assert answer['target_task'] == 1 and answer['decision_turn'] == 0
                    assert answer['result_turn'] == 1 and answer['unique_causal_motive_established'] is False
                    assert answer['language_rendering']['provider_status'] == 'not_configured_template_only'
                    after = check.call(store.view, token, iid)
                    assert digest(after['current']['state']) == before
                    assert check.call(store.answers, token, iid)['questions'][0]['question_id'] == answer['question_id']
                    result['historical_question_checked'] = True
                    result['question_did_not_advance_real_state'] = True
                elif task == 2:
                    try:
                        check.call(store.ask,token,iid,question='Why?',target_task=1,decision_turn=0)
                    except StudyError as exc:
                        assert exc.status==403
                    else:
                        raise AssertionError('raw control group exposed explanations')
                    result['explanations_unavailable_in_control_group']=True
                if task == 3:
                    try:
                        check.call(store.answers, token, iid)
                    except StudyError as exc:
                        assert exc.status == 403
                    else:
                        raise AssertionError('Task3 exposed past explanations')
                    result['answers_revoked_in_Task3'] = True
                while not view['current']['state']['terminal']:
                    view = check.action(token, iid, view, label='full')
                    result['counts_by_task'][str(task)] = view['current']['turn']
                check.no_next_action(token, iid, view)
                replay = check.call(store.replay, token, iid, task=task)
                assert replay['all_AI_actions_recomputed'] is True or replay.get('immutable_history_validation_cache') is True
                assert replay['frames'][-1] == {k:v for k,v in view['current'].items() if k != 'run_id'}
                result['completed_tasks'] += 1
                print(json.dumps(dict(event='task_complete',domain=domain,task=task,
                    committed_steps=view['current']['turn'],elapsed_seconds=round(perf_counter()-start,3))),flush=True)
            bank = check.call(store.understanding, token, iid)
            # Arbitrary first-option software responses; no efficacy/scale score
            # is extracted or reported by this checker.
            answers = {q['id']: q['options'][0]['id'] for q in bank['questions']}
            check.call(store.submit_understanding, token, iid, answers=answers,
                version=bank['version'], sha256=bank['sha256'])
            result['understanding_saved'] = True
            final = check.call(store.submit_survey, token, iid,
                answers={field: 4 for field in RATINGS}, feedback='Automated software flow; not a participant response.')
            result['survey_saved'] = True
            assert final['study']['phase'] == 'finished'
            assert final['study']['scientific_sample'] is False
            exported = check.call(store.export_instance, token, iid, admin=True)
            assert len(exported['records']['study_understanding']) == 1
            assert len(exported['records']['study_surveys']) == 1
            assert len(exported['records']['runs']) == 3
            result.update(status='completed', exact_action_binding_restore_replay=True,
                          questionnaires_saved_and_export_verified=True)
    except FlowDeadline:
        result['status'] = 'bounded_stop'
    except Exception as exc:
        # Never serialize exception arguments, participant tokens or state data.
        result.update(status='failed', error_type=type(exc).__name__)
    result['elapsed_seconds'] = round(perf_counter()-start, 3)
    return result


def run(output, flow_seconds=180, domains=('warehouse','pong','kitchen')):
    if output.exists():
        raise ValueError('refusing to replace an existing software receipt')
    assert 1 <= flow_seconds <= 180
    assert domains and len(set(domains))==len(domains) and set(domains)<={'warehouse','pong','kitchen'}
    began = perf_counter()
    result = dict(version=VERSION, checked_at=datetime.now(timezone.utc).isoformat(),
        scope='actual frozen-model software flow on existing public manual-preview scenes',
        scientific_model_acceptance=False, scientific_task_quality_evaluated=False,
        heldout_evaluated=False, human_study_sample=False, recruitment_opened=False,
        external_question_transmission=False, persistence='isolated temporary SQLite',
        cases=[], complete_wait_only_flows=[])
    result['requested_domains']=list(domains)
    result['runtime_source_sha256']={name:sha256((ROOT/name).read_bytes()).hexdigest() for name in (
        'runtime_v642/store.py','study_vnext/store.py','study_vnext/history.py','study_vnext/qa.py',
        'study_vnext/server.py','ui/model_preview_fast_runtime.py','scripts/check_study_vnext_models.py')}
    with TemporaryDirectory(prefix='policylens-study-software-') as temporary:
        store, preview = compose_preview(Path(temporary)/'software-only.sqlite')
        check = Check(store)
        sessions, identities = {}, {}
        try:
            for domain in domains:
                identities[domain] = {}
                for group in ('A', 'B', 'C'):
                    case, session = basic_case(check, domain, group)
                    result['cases'].append(case)
                    identities[domain][group] = case['binding_sha256']
                    if group in ('A','C'):
                        sessions[(domain,group)] = session
                    print(json.dumps(dict(event='basic_case',domain=domain,group=group,
                        passed=True,elapsed_seconds=case['elapsed_seconds'])), flush=True)
                assert identities[domain]['A'] == identities[domain]['B']
                assert identities[domain]['A'] != identities[domain]['C']
            result['AB_same_regularized_C_distinct_raw'] = True
            for domain in domains:
                for group in ('A','C'):
                    flow = complete_flow(check, domain, sessions[(domain,group)], flow_seconds,group=group)
                    result['complete_wait_only_flows'].append(flow)
                    print(json.dumps(dict(event='complete_flow',**flow)), flush=True)
            result['six_distinct_frozen_actor_bundles_checked'] = len({c['actor_sha256'] for c in result['cases']}) == 6
            result['all_requested_frozen_actor_bundles_distinct'] = len({c['actor_sha256'] for c in result['cases']}) == 2*len(domains)
            result['request_context_per_call'] = True
            result['call_count'] = check.calls
            result['call_counts'] = check.counts
            result['call_elapsed_seconds'] = {name:round(value,3) for name,value in check.elapsed.items()}
        finally:
            store.close()
            preview.close()
    result['passed'] = (len(result['cases']) == 3*len(domains) and result['all_requested_frozen_actor_bundles_distinct']
        and len(result['complete_wait_only_flows']) == 2*len(domains)
        and all(f['status'] == 'completed' for f in result['complete_wait_only_flows']))
    result['full_three_task_flow_count'] = sum(f['status'] == 'completed' for f in result['complete_wait_only_flows'])
    result['elapsed_seconds'] = round(perf_counter()-began, 3)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],cases=len(result['cases']),
        full_three_task_flows=result['full_three_task_flow_count'],elapsed_seconds=result['elapsed_seconds'],
        receipt=str(output))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/study_vnext/software-20261029/actual-model-final-check.json')
    parser.add_argument('--flow-seconds', type=int, default=180)
    parser.add_argument('--domains',nargs='+',choices=('warehouse','pong','kitchen'),default=('warehouse','pong','kitchen'))
    args = parser.parse_args()
    receipt = run(args.output, args.flow_seconds,tuple(args.domains))
    raise SystemExit(0 if receipt['passed'] else 1)
