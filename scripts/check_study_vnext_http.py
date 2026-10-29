"""Remote researcher-preview smoke checks against the exact local frozen NNs.

Creates synthetic preview records only. No training, quality evaluation, sealed
seeds, external question service, real participant or pilot admission is used.
The private operator credential is read from a private file and never reported.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from datetime import datetime, timezone
import gzip
from hashlib import sha256
from http.client import HTTPConnection, HTTPSConnection
import json
from pathlib import Path
import stat
import sys
from time import perf_counter
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VERSION = 'study-vnext-live-http-software-smoke.v1'
DOMAINS = ('warehouse', 'pong', 'kitchen')
CODES = dict(warehouse=['KeyW','KeyA','KeyS','KeyD','Space'],
    pong=['KeyA','KeyD','Space'], kitchen=['KeyW','KeyA','KeyS','KeyD','Space','KeyE'])
STATIC = {
    '/study/': 'study_vnext/web/index.html',
    '/study-assets/app.js': 'study_vnext/web/app.js',
    '/study-assets/styles.css': 'study_vnext/web/styles.css',
    '/study-assets/board.js': 'study_v3/web/board.js',
    '/study-assets/preview-board.js': 'ui/model_preview/board.js',
    '/study-assets/input-queue.js': 'ui/model_preview/input_queue.js',
}


class SmokeFailure(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise SmokeFailure(code)


def read_admin(path):
    require(not path.is_symlink(), 'private_admin_file_required')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_mode & 0o077 == 0
        and info.st_uid == os.getuid(), 'private_admin_file_permissions_required')
    credential = path.read_text().rstrip('\r\n')
    require(len(credential) >= 32 and '\n' not in credential and '\r' not in credential,
        'private_admin_file_invalid')
    return credential


class Client:
    def __init__(self, base, admin):
        parsed = urlsplit(base)
        require(parsed.scheme in ('https','http') and parsed.hostname
            and not parsed.username and not parsed.password and not parsed.query
            and not parsed.fragment and parsed.path in ('','/'), 'origin_base_url_required')
        require(parsed.scheme == 'https' or parsed.hostname in ('127.0.0.1','localhost','::1'),
            'https_required_except_loopback')
        self.base, self.admin = base.rstrip('/'), admin
        self.request_prefix = ''
        self.host_header = parsed.netloc
        proxy = getproxies().get(parsed.scheme) if not proxy_bypass(parsed.netloc) else None
        if proxy:
            endpoint = urlsplit(proxy if '://' in proxy else 'http://' + proxy)
            require(endpoint.scheme == 'http' and endpoint.hostname
                and not endpoint.username and not endpoint.password
                and endpoint.path in ('','/') and not endpoint.query and not endpoint.fragment,
                'unauthenticated_http_proxy_required')
            if parsed.scheme == 'https':
                self.connection = HTTPSConnection(endpoint.hostname, endpoint.port or 80, timeout=180)
                # CONNECT contains only the destination. Authorization and the
                # researcher key are sent inside verified target-host TLS.
                self.connection.set_tunnel(parsed.hostname, parsed.port or 443)
            else:
                self.connection = HTTPConnection(endpoint.hostname, endpoint.port or 80, timeout=180)
                self.request_prefix = self.base
        else:
            transport = HTTPSConnection if parsed.scheme == 'https' else HTTPConnection
            self.connection = transport(parsed.hostname, parsed.port, timeout=180)
        self.calls = []

    def close(self):
        self.connection.close()

    def call(self, path, payload=None, *, token=None, admin=False, status=200, raw=False):
        headers = {'Accept-Encoding':'gzip', 'Origin':self.base, 'Host':self.host_header}
        if payload is not None:
            headers['Content-Type'] = 'application/json'
        if token:
            headers['Authorization'] = 'Bearer ' + token
        if admin:
            headers['X-Admin-Token'] = self.admin
        began = perf_counter()
        # Reuse TLS for sequential probes. Never retry an uncertain mutation,
        # and never follow a redirect carrying authorization to another origin.
        self.connection.request('GET' if payload is None else 'POST',self.request_prefix + path,
            body=None if payload is None else json.dumps(payload).encode(),headers=headers)
        response = self.connection.getresponse()
        with response:
            body = response.read(32*1024*1024 + 1)
            require(len(body) <= 32*1024*1024, 'response_size_limit')
            received_status = response.status
            fields = {key.lower(): value for key,value in response.headers.items()}
            if response.headers.get('Content-Encoding') == 'gzip':
                body = gzip.decompress(body)
            require(len(body) <= 32*1024*1024, 'uncompressed_response_size_limit')
            # Never retain a session token, identifier, response body, operator
            # credential or enrollment reference in this deployment receipt.
            safe_path = path
            if path.startswith('/study-api/') and len(path.split('/')) > 3:
                parts = path.split('/'); parts[2] = '<synthetic-instance>'; safe_path = '/'.join(parts)
            self.calls.append(dict(path=safe_path, status=received_status,
                elapsed_ms=round((perf_counter()-began)*1000,3), bytes=len(body),
                gzip=response.headers.get('Content-Encoding') == 'gzip',
                server_timing=response.headers.get('Server-Timing','')))
            require(received_status == status, 'unexpected_http_status')
            return (body if raw else json.loads(body)), fields


def check_public(view, binding, local_state, adapter, *, turn):
    current = view['current']
    require(current['binding'] == binding.to_dict(), 'current_binding_mismatch')
    require(current['decision'] == {} and current['decision_committed'] is False,
        'unexecuted_action_exposed')
    require(current['turn'] == turn and current['public'] == adapter.public_state(local_state,binding),
        'local_frozen_nn_physics_public_state_mismatch')
    require(view['explanation_questions_allowed'] is False, 'task1_explanations_enabled')
    require(view['study']['scientific_sample'] is False and view['study']['recruitment_opened'] is False
        and view['study']['question_transmission_consent'] is False, 'preview_scope_mismatch')


def run(base, output, admin_file, expected_commit=None):
    require(not output.exists(), 'refusing_to_replace_existing_receipt')
    from runtime_v642.contracts import digest
    from study_vnext.store import CONSENT_VERSION, PROTOCOL_ID
    from ui.model_preview_server import Preview, RELEASE, SEEDS
    began = perf_counter()
    result = dict(version=VERSION, checked_at=datetime.now(timezone.utc).isoformat(),
        base_url=base.rstrip('/'), scope='synthetic researcher-preview HTTP software checks',
        passed=False, scientific_model_acceptance=False, scientific_task_quality_evaluated=False,
        heldout_evaluated=False, human_study_sample=False, pilot_opened=False,
        recruitment_opened=False, external_question_transmission=False,
        cases=[], static_assets=[], requests=[])
    client = Client(base, read_admin(admin_file))
    local = None
    try:
        health, _ = client.call('/health')
        require(health['pilot_opened'] is False and health['researcher_preview_available'] is True,
            'deployment_not_closed_researcher_preview')
        deployed = health['deployment_commit']
        if expected_commit:
            require(7 <= len(expected_commit) <= 40 and deployed.startswith(expected_commit),
                'deployment_commit_mismatch')
        result['deployed_commit'] = deployed
        metadata, _ = client.call('/api/release')
        require(metadata['release_id'] == RELEASE and metadata['deployment_commit'] == deployed
            and metadata['pilot_opened'] is False and metadata['study_ready'] is False
            and metadata['database_is_participant_database'] is False, 'release_scope_mismatch')
        status, _ = client.call('/study-api/status')
        require(status['protocol_id'] == PROTOCOL_ID and status['mode'] == 'preview'
            and status['recruitment_opened'] is False and status['scientific_sample'] is False
            and status['question_service_mode'] == 'template_only', 'study_status_mismatch')
        for path, source in STATIC.items():
            body, headers = client.call(path, raw=True)
            expected_sha = sha256((ROOT/source).read_bytes()).hexdigest()
            require(sha256(body).hexdigest() == expected_sha, 'static_asset_mismatch')
            require(headers.get('content-encoding') == 'gzip' and headers.get('server-timing'),
                'compression_or_server_timing_missing')
            result['static_assets'].append(dict(path=path, sha256=expected_sha, exact=True))
        for path in ('/prolific/','/api/start'):
            client.call(path,status=404)
        unauthorized = dict(domain='warehouse',participant_reference='synthetic-unauthorized',
            consent=True,consent_version=CONSENT_VERSION)
        client.call('/study-api/enroll',unauthorized,status=403)
        local = Preview(':memory:')
        local_metadata = local.metadata()
        require(metadata['models'] == local_metadata['models'], 'six_model_catalog_mismatch')
        identities, demos = {}, {}
        prefix = 'synthetic-http-only-' + uuid.uuid4().hex
        for domain in DOMAINS:
            identities[domain] = {}
            for group in ('A','C','B'):
                case_start = perf_counter()
                variant = 'raw' if group == 'C' else 'regularized'
                binding = local.catalog.binding(domain,variant)
                adapter = local.adapters[domain]
                enrollment, _ = client.call('/study-api/enroll',dict(domain=domain,group=group,
                    participant_reference=f'{prefix}-{domain}-{group}',consent=True,
                    consent_version=CONSENT_VERSION,question_transmission_consent=False),admin=True)
                token, iid = enrollment['session_token'], enrollment['instance']['id']
                route = lambda tail: '/study-api/' + iid + '/' + tail
                instance = enrollment['instance']
                require(instance['binding'] == binding.to_dict() and instance['group'] == group
                    and instance['mode'] == 'preview' and instance['scenario_seed'] in SEEDS[domain][:6],
                    'enrollment_binding_or_assignment_mismatch')
                identities[domain][group] = instance['binding_sha256']
                view, _ = client.call(route('view'),token=token)
                require(view['current'] is None, 'task_started_before_preparation')
                client.call(route('preparation'),dict(kind='controls',observed_codes=CODES[domain]),token=token)
                demo, _ = client.call(route('demo'),{},token=token)
                require(demo['two_frozen_NN'] is True and demo['binding'] == binding.to_dict()
                    and demo['replay_verified_by_server'] is True
                    and demo['contains_restorable_private_snapshot'] is False, 'demonstration_scope_mismatch')
                with local.runtime.request():
                    if (domain,variant) not in demos:
                        expected = adapter.demonstration(binding,seed=SEEDS[domain][-1],task=1,
                            steps=360 if domain == 'kitchen' else 120)
                        demos[(domain,variant)] = expected
                    expected = demos[(domain,variant)]
                    require(len(demo['records']) == len(expected['records']), 'demonstration_length_mismatch')
                    require(demo['initial_public'] == adapter.public_state(expected['initial_state'],binding)
                        and demo['final_public'] == adapter.public_state(expected['final_state'],binding),
                        'demonstration_physics_mismatch')
                    for remote, record in zip(demo['records'],expected['records']):
                        require(remote['human_action'] is None and remote['NN_decision']['action'] == record['decision']['action']
                            and remote['NN_decision']['NN_pair_actions'] == record['decision']['policy_actions']
                            and remote['public_after'] == adapter.public_state(record['state'],binding),
                            'demonstration_nn_action_or_physics_mismatch')
                view, _ = client.call(route('view'),token=token)
                prep = view['study']['preparation']
                client.call(route('preparation'),dict(kind='demonstration',
                    bundle_sha256=prep['demonstration_sha256'],
                    final_frame_count=prep['demonstration_frame_count']),token=token)
                client.call(route('demo-complete'),{},token=token)
                view, _ = client.call(route('next-task'),{},token=token)
                with local.runtime.request():
                    state = adapter.initial_state(instance['scenario_seed'],1,binding)
                    check_public(view,binding,state,adapter,turn=0)
                    require(view['current']['last_executed_action'] is None, 'initial_last_action_mismatch')
                for index in range(3):
                    with local.runtime.request():
                        decision = adapter.decide(state,binding)
                        state = adapter.step(state,'wait',binding,decision=decision)
                    payload = dict(command_id=f'software-smoke-step-{index}',expected_turn=index,human_action='wait')
                    view, fields = client.call(route('action'),payload,token=token)
                    require(bool(fields.get('server-timing')), 'action_server_timing_missing')
                    with local.runtime.request():
                        check_public(view,binding,state,adapter,turn=index+1)
                    last = view['current']['last_executed_action']
                    require(last == dict(decision_turn=index,result_turn=index+1,
                        human_action='wait',ai_action=decision['action']), 'executed_ai_action_not_bound_nn')
                    repeated, _ = client.call(route('action'),payload,token=token)
                    require(repeated == view, 'command_retry_not_idempotent')
                restored, _ = client.call(route('view'),token=token)
                require(restored['current'] == view['current'], 'restored_state_mismatch')
                replay, _ = client.call(route('replay')+'?task=1',token=token)
                require(replay['binding'] == binding.to_dict() and len(replay['frames']) == 4
                    and replay['frames'][-1] == {k:v for k,v in view['current'].items() if k != 'run_id'},
                    'replay_head_mismatch')
                for frame in replay['frames']:
                    require(frame['turn'] <= 3, 'replay_uncommitted_future_frame')
                client.call(route('answers'),token=token,status=403)
                client.call(route('question'),dict(question='Only an automated software test.',
                    target_task=1,decision_turn=0),token=token,status=403)
                result['cases'].append(dict(domain=domain,group=group,variant=variant,passed=True,
                    actor_sha256=binding.actor_sha256,human_actor_sha256=binding.human_actor_sha256,
                    ai_actor_sha256=binding.ai_actor_sha256,binding_sha256=digest(binding.to_dict()),
                    public_scene_seed=instance['scenario_seed'],demonstration_frames=len(demo['records']),
                    demonstration_all_frame_NN_and_physics_exact=True,committed_human_wait_steps=3,
                    human_takeover_keeps_frozen_ai=True,retry_idempotent=True,restore_exact=True,
                    replay_exact=True,no_unexecuted_action_exposed=True,task1_questions_refused=True,
                    elapsed_seconds=round(perf_counter()-case_start,3)))
                print(json.dumps(dict(event='case',domain=domain,group=group,passed=True)),flush=True)
            require(identities[domain]['A'] == identities[domain]['B']
                and identities[domain]['A'] != identities[domain]['C'], 'AB_C_binding_relation_failed')
        result['AB_same_regularized_C_distinct_raw'] = True
        result['six_distinct_actor_bundles_checked'] = len({c['actor_sha256'] for c in result['cases']}) == 6
        require(result['six_distinct_actor_bundles_checked'], 'six_distinct_actor_bundles_required')
        result['passed'] = True
    except Exception as error:
        # Exception arguments may contain URLs/headers/credentials; never emit.
        result['error_type'] = type(error).__name__
        if type(error) is SmokeFailure:
            result['failure_code'] = str(error)
    finally:
        client.close()
        if local is not None:
            local.close()
        result['requests'] = client.calls
        result['request_count'] = len(client.calls)
        result['elapsed_seconds'] = round(perf_counter()-began,3)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2); stream.write('\n')
    print(json.dumps(dict(passed=result['passed'],cases=len(result['cases']),
        requests=result['request_count'],receipt=str(output))),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--admin-file',required=True,type=Path)
    parser.add_argument('--expected-commit')
    args = parser.parse_args()
    try:
        receipt = run(args.base,args.output,args.admin_file,args.expected_commit)
    except Exception as error:
        print(json.dumps(dict(passed=False,error_type=type(error).__name__)),flush=True)
        raise SystemExit(1)
    raise SystemExit(0 if receipt['passed'] else 1)
