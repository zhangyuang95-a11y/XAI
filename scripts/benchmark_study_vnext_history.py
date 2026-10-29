#!/usr/bin/env python3
"""Matched local server-path benchmark; never model/study/Render acceptance.

Creates synthetic researcher-only temporary databases, copies the same saved
Task1 start for each branch and uses only the first existing public manual
scene. Every branch performs 10 warmup + 30 measured wait actions. The reference
is the pinned historical prefix algorithm with an explicit current-Store
compatibility wrapper, not an old live website. All six frozen raw/regularized
models are checked without training.
"""
from pathlib import Path
import argparse
from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
import gc
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import types
from tempfile import TemporaryDirectory
from time import perf_counter, get_clock_info

for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

VERSION = 'study-vnext-six-model-matched-warm-benchmark.v1'
WARMUP_ACTIONS = 10
SAMPLE_ACTIONS = 30
MODELS = tuple((domain,variant) for domain in ('warehouse','pong','kitchen')
               for variant in ('raw','regularized'))
ARCHIVED_HISTORY = ROOT/'tests/fixtures/study_vnext_history_prefix_reference.py'
ARCHIVED_HISTORY_SHA256 = '4e54f65b86450ad9efe183fe95efb11cf06f3054ff9023c9f44ea8cf82337c68'
SOURCE_NAMES = (
    'scripts/benchmark_study_vnext_history.py','study_vnext/history.py',
    'study_vnext/store.py','study_vnext/server.py','study_vnext/protocol.py',
    'runtime_v642/store.py','runtime_v642/server.py','runtime_v642/contracts.py',
    'runtime_v642/catalog.py','runtime_v642/registry.py','runtime_v642/legacy_domain.py',
    'ui/model_preview_fast_runtime.py','ui/model_preview_server.py',
    'ui/model_preview_warehouse.py',
    'study_vnext/request_physics.py',
    'tests/fixtures/study_vnext_history_prefix_reference.py',
)


def require(ok,message):
    if not ok:raise RuntimeError(message)


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    return {name:file_sha(ROOT/name) for name in SOURCE_NAMES}


def percentile(values,quantile):
    require(values and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values),
        'finite nonnegative measurements required')
    require(0<=quantile<=1,'invalid quantile')
    ordered=sorted(values);position=(len(ordered)-1)*quantile;low=math.floor(position);high=math.ceil(position)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def measurements(values):
    require(len(values)==SAMPLE_ACTIONS,'benchmark needs exactly 30 measured requests')
    return dict(N=len(values),P50=percentile(values,.5),P95=percentile(values,.95),mean=statistics.mean(values),
                minimum=min(values),maximum=max(values))


def compare_traces(baseline,accelerated):
    require(len(baseline)==len(accelerated)==WARMUP_ACTIONS+SAMPLE_ACTIONS,
        'both branches must finish all 40 actions')
    for index,(first,second) in enumerate(zip(baseline,accelerated)):
        require(first==second,'bound action/state/score/response mismatch at action '+str(index))
    return dict(compared_actions=len(baseline),exact_actions_states_scores_bindings=True,
        physical_state_hashes_identical=True,public_responses_identical=True)


def _write(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)


def _prepare_template(path,domain,variant,seed):
    from study_vnext.server import compose_preview
    from study_vnext.store import CONSENT_VERSION, _CONTROL_CODES
    from runtime_v642.contracts import digest
    store,preview=compose_preview(path)
    try:
        with store.request_context():
            enrolled=store.enroll(domain=domain,participant_reference='synthetic-engineering-benchmark-only',
                group='B' if variant=='regularized' else 'C',scenario_seed=seed,admin=True,
                consent=True,consent_version=CONSENT_VERSION)
            token,instance=enrolled['session_token'],enrolled['instance']['id']
            require(enrolled['instance']['scenario_seed']==seed,'requested public seed not bound')
            store.complete_preparation(token,instance,kind='controls',observed_codes=list(_CONTROL_CODES[domain]))
            demo=store.demonstration(token,instance)
            store.complete_preparation(token,instance,kind='demonstration',
                bundle_sha256=digest(demo),final_frame_count=len(demo['records']))
            store.acknowledge_demonstration(token,instance);origin=store.start_task(token,instance)
        return token,instance,deepcopy(origin['instance']['binding']),digest(origin['current']['state'])
    finally:
        store.close();preview.close()


def _run_branch(path,token,instance,*,mode,starting_sha):
    from study_vnext.server import compose_preview
    from runtime_v642.contracts import digest
    from runtime_v642.server import _plain
    from runtime_v642.store import encode
    require(file_sha(path)==starting_sha,'branch database is not an exact copy of the saved start')
    store,preview=compose_preview(path)
    require(mode in ('archived_prefix_normal','accelerated_normal','accelerated_normal_original_physics'),
        'unknown benchmark branch')
    if mode=='accelerated_normal_original_physics':
        from ui.model_preview_fast_runtime import FastWarehouseAdapter
        from runtime_v642.registry import RuntimeRegistry
        original=FastWarehouseAdapter(preview.catalog,preview.runtime)
        original._actors=dict(preview.adapters['warehouse']._actors)
        preview.adapters['warehouse']=original
        store.registry=RuntimeRegistry(preview.catalog,adapters=preview.adapters)
    if mode=='archived_prefix_normal':
        # There is no captured old Store source. This normal-mode historical
        # algorithm reference uses the pinned old prefix implementation plus
        # an explicitly disclosed summary compatibility adapter in this file.
        # It is not presented as the old website or a literal old deployment.
        require(file_sha(ARCHIVED_HISTORY)==ARCHIVED_HISTORY_SHA256,'archived prefix reference source differs')
        sys.dont_write_bytecode=True
        spec=importlib.util.spec_from_file_location('matched_archived_prefix_reference',ARCHIVED_HISTORY)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        from study_vnext.history import DemonstrationSummary,full_verification_enabled
        store._history=legacy.HistoryCertificates()
        def historical_run(self,db,bound,run,*,through=None):
            context=legacy.full_history_verification() if full_verification_enabled() else nullcontext()
            with context:return legacy.verified_run(self,self._history,db,bound,run,through=through)
        def summary(self,row,binding):
            bundle=self.parsed_demonstration(row,binding)
            return DemonstrationSummary(len(bundle['records']),row['bundle_sha256'],row['binding_sha256'],True)
        store._verified_run=types.MethodType(historical_run,store)
        store._history.demonstration_summary=types.MethodType(summary,store._history)
    counter={'value':0};original_memo=preview.runtime.memo
    def counted_memo(key,compute,*,category='inference'):
        def observed():
            if key and key[0] in ('warehouse','pong','kitchen'):counter['value']+=1
            return compute()
        return original_memo(key,observed,category=category)
    preview.runtime.memo=counted_memo
    trace=[];samples=[]
    try:
        require(file_sha(path)==starting_sha,'opening the branch changed its persisted start')
        gc.collect()
        for turn in range(WARMUP_ACTIONS+SAMPLE_ACTIONS):
            before_count=counter['value'];began=perf_counter()
            with store.request_context() as request_scope:
                result=store.action(token,instance,command_id='matched-'+str(turn),expected_turn=turn,human_action='wait')
                public=store.participant_view(token,instance,result)
                raw=encode(_plain(public)).encode('utf-8')
            elapsed_ms=(perf_counter()-began)*1000
            current=result['current'];state=current['state']
            require(current['turn']==turn+1 and state['task']==1 and not state['terminal'],
                'public fixture terminated before completing all 40 wait actions')
            incoming=current['transition']['records'][0]
            # Only already executed decisions enter the engineering receipt.
            # The newly calculated live-head decision is deliberately omitted.
            trace.append(dict(decision_turn=turn,result_turn=current['turn'],
                executed_human_action=incoming['human_action'],executed_AI_action=incoming['decision']['action'],
                executed_NN_pair_actions=deepcopy(incoming['decision']['policy_actions']),
                complete_state_sha256=digest(state),public_state_sha256=digest(current['public']),
                frame_sha256=current['frame_sha256'],binding_sha256=digest(current['binding']),
                score=deepcopy(current['public'].get('score')),public_response_sha256=sha256(raw).hexdigest()))
            if turn>=WARMUP_ACTIONS:
                samples.append(dict(before_turn=turn,after_turn=turn+1,milliseconds=elapsed_ms,
                    asset_bytes_read=sum(len(value) for value in request_scope['bytes'].values()),
                    unique_assets_read=len(request_scope['bytes']),
                    uncached_frozen_pair_decision_computations=counter['value']-before_count,
                    serialized_response_bytes=len(raw),
                    unique_physical_transition_computations=request_scope.get('study_warehouse_physics',{}).get('computed'),
                    repeated_physical_validation_witnesses=request_scope.get('study_warehouse_physics',{}).get('reused')))
        require(len(samples)==SAMPLE_ACTIONS,'incomplete benchmark cannot be reported')
        return dict(baseline_kind=mode,
            starting_database_sha256=starting_sha,trace=trace,samples=samples,
            latency_ms=measurements([x['milliseconds'] for x in samples]),
            asset_bytes=measurements([x['asset_bytes_read'] for x in samples]),
            uncached_pair_decision_computations=measurements([x['uncached_frozen_pair_decision_computations'] for x in samples]),
            no_NN_forward_output_cache_across_requests=True,
            inference_count_definition='actual compute callbacks for frozen pair decisions in existing request-local runtime memo; not individual dense-layer calls')
    finally:
        store.close();preview.close()


def run(output):
    from ui.model_preview_server import SEEDS,CATALOG_SHA256
    output=Path(output).resolve()
    require(output.is_relative_to((ROOT/'output/study_vnext/engineering').resolve()),'use an isolated engineering output directory')
    output.mkdir(parents=True,exist_ok=False)
    sources=source_files()
    require(file_sha(ARCHIVED_HISTORY)==ARCHIVED_HISTORY_SHA256,'pinned historical prefix source is missing or changed')
    for name,expected in sources.items():
        raw=(ROOT/name).read_bytes();require(sha256(raw).hexdigest()==expected,'source changed preparing benchmark')
        path=output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(0o444)
    value=dict(version=VERSION,scope='local_software_engineering_not_model_or_human_acceptance',
        requested_models=[dict(domain=d,variant=v) for d,v in MODELS],
        warmup_actions=WARMUP_ACTIONS,measured_actions=SAMPLE_ACTIONS,
        timing_scope='local action+participant projection+JSON serialization+complete request asset verification; excludes HTTP/network/browser',
        baseline_annotation='archived prefix algorithm in normal mode with current Store and explicit summary compatibility wrapper; no old Store snapshot; not old live website',
        archived_prefix_source=dict(path=str(ARCHIVED_HISTORY),sha256=ARCHIVED_HISTORY_SHA256),
        sample_actions='wait',task=1,source_files=sources,catalog_sha256=CATALOG_SHA256,
        environment=dict(python=sys.version,platform=platform.platform(),GC_enabled=gc.isenabled(),
            BLAS_threads={key:os.environ[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            clock_resolution_seconds=get_clock_info('perf_counter').resolution,
            load_control='other heavy project checkers held by operator; OS scheduling not isolated'),
        pairs=[],all_six_exact_pairs_completed=False,network_or_Render_measured=False,
        engineering_release_gate_passed=False,scientific_acceptance_claimed=False,pilot_opened=False)
    _write(output/'started.json',value)
    try:
        with TemporaryDirectory(prefix='policylens-six-model-history-benchmark-') as temporary:
            temporary=Path(temporary)
            for index,(domain,variant) in enumerate(MODELS):
                require(source_files()==sources,'benchmark source changed between model pairs')
                seed=SEEDS[domain][0];folder=temporary/f'{domain}-{variant}';folder.mkdir()
                template=folder/'template.sqlite'
                token,instance,binding,initial_state_sha=_prepare_template(template,domain,variant,seed)
                start_sha=file_sha(template);branches={}
                order=('archived_prefix_normal','accelerated_normal') if index%2==0 else \
                    ('accelerated_normal','archived_prefix_normal')
                for branch in order:
                    path=folder/(branch+'.sqlite');shutil.copyfile(template,path)
                    worker=dict(path=str(path),token=token,instance=instance,mode=branch,starting_sha=start_sha,
                        expected_source_files=sources)
                    process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker'],
                        input=json.dumps(worker),text=True,capture_output=True,check=False,cwd=ROOT)
                    require(process.returncode==0,'isolated worker failed for '+domain+'/'+variant+'/'+branch+': '+process.stderr[-1200:])
                    branches[branch]=json.loads(process.stdout)
                equality=compare_traces(branches['archived_prefix_normal']['trace'],branches['accelerated_normal']['trace'])
                pair=dict(domain=domain,variant=variant,public_scenario_seed=seed,binding=binding,
                    initial_complete_state_sha256=initial_state_sha,starting_database_sha256=start_sha,
                    execution_order=list(order),equality=equality,branches=branches)
                _write(output/(domain+'-'+variant+'.json'),pair);value['pairs'].append(pair)
                print(json.dumps(dict(domain=domain,variant=variant,
                    archived_method_ms=branches['archived_prefix_normal']['latency_ms'],
                    accelerated_ms=branches['accelerated_normal']['latency_ms'],exact_equality=True)),flush=True)
        require(source_files()==sources,'benchmark source changed during execution')
        value['all_six_exact_pairs_completed']=len(value['pairs'])==len(MODELS)
        require(value['all_six_exact_pairs_completed'],'all six models must complete')
        _write(output/'report.json',value)
        return value
    except BaseException as error:
        value['failure']=dict(type=type(error).__name__,reason=str(error))
        _write(output/'failed.json',value)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        worker=json.load(sys.stdin)
        require(worker.pop('expected_source_files')==source_files(),'worker source differs from the controlled run')
        value=_run_branch(**worker)
        print(json.dumps(value,ensure_ascii=False,allow_nan=False));return value
    require(args.output is not None,'explicit engineering output required')
    return run(args.output)


if __name__=='__main__':main()
