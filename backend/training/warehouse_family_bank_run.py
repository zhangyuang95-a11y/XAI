"""Finite private question-bank work through genuine registered 197 runtimes.

The family bank supplies operation digests, not raw callbacks. This driver uses
temporary per-instance observation wrappers around the genuine bound methods:
no type, Actor, action, return value or runtime signature is replaced. Every
operation is reserved first; each nested physical call saves its full result
before acknowledgment. Failed/pending work cannot be automatically repeated.
Completed caches verify saved evidence without constructing another runtime.
Content checks and independent replay never grant research/release eligibility.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import re

from backend import warehouse_runtime_family as registry
from backend.warehouse_public_history_runtime import PublicHistoryRuntime
from backend.warehouse_shutdown_runtime import ShutdownRuntime
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.training import warehouse_public_history_bank_run as old_io
from backend.training.warehouse_native_revision_provenance import _absolute, _open_dir
from ui import warehouse_family_bank as bank_api

VERSION='warehouse-runtime-family-bank-workflow.v1'
LEDGER_VERSION='warehouse-runtime-family-bank-finite-ledger.v1'
INPUT_NAMES=('actor.npz','protocol.json','actor_bindings.json','selection.json','pool_scenes.json','exclusion_pools.json')
PHASES=('generation','verification')
KINDS={'next_action':0,'selfplay_decision':0,'trajectory':1,'wait_three':3}
CLASSES={'observed197':PublicHistoryRuntime,'retained_beta197':ShutdownRuntime}
HEX=re.compile(r'[a-f0-9]{64}\Z')


def _binding(raw):return {'sha256':sha256(raw).hexdigest(),'size':len(raw)}
def _read(path):
    path=_absolute(path);fd=_open_dir(path.parent);os.close(fd)
    return old_io._read(path)
def _json(path):return json.loads(_read(path))
def _mkdir(path):
    path=Path(path)
    if not path.parent.exists():_mkdir(path.parent)
    fd=_open_dir(path.parent)
    try:
        os.mkdir(path.name,mode=0o700,dir_fd=fd);os.fsync(fd)
    finally:os.close(fd)
def put(path,value,*,replace=False):
    path=_absolute(path)
    if not path.parent.exists():_mkdir(path.parent)
    fd=_open_dir(path.parent);os.close(fd)
    if path.is_symlink():raise ValueError('Symlink output is forbidden')
    old_io.put(path,value,replace=replace)


def execution_sources(runtime):
    result=bank_api.bank_sources(runtime)
    for name in (__file__,old_io.__file__,str(ROOT/'backend/training/warehouse_native_revision_provenance.py')):
        result[str(Path(name).relative_to(ROOT))]=file_hash(name)
    return result


def prepare(output,*,runtime,frozen_selection,pool_scenes,exclusion_pools,trajectory_steps,
            generation_step_cap,verification_step_cap,minimum_frame=1,allow_test_fixture=False):
    """Freeze a real runtime's input bytes and finite caps; zero inference/steps."""
    output=_absolute(output)
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    identity=registry.verify(runtime,allow_test_fixture=allow_test_fixture)
    if output.is_relative_to(runtime._actor_path.parent) or runtime._actor_path.is_relative_to(output):
        raise ValueError('New workflow must be separate from original Actor storage')
    old_io._selection(frozen_selection,runtime)
    for name in ('runtime_family','runtime_signature','shutdown_arm','own_shutdown_beta'):
        expected=identity['family'] if name=='runtime_family' else runtime.signature if name=='runtime_signature' else runtime.actor.metadata.get(name)
        if name in frozen_selection and frozen_selection[name]!=expected:raise ValueError('Selection runtime family differs')
    bank_api._pools(runtime,pool_scenes,exclusion_pools,allow_test_fixture=allow_test_fixture)
    if (type(trajectory_steps) is not int or not 1<=trajectory_steps<=runtime.config.horizon
            or type(minimum_frame) is not int or not 0<=minimum_frame<trajectory_steps):
        raise ValueError('Finite source trajectory interval is required')
    maximum=len(pool_scenes)*(trajectory_steps+3*(trajectory_steps-minimum_frame))
    zero=len(pool_scenes)*(2*trajectory_steps-minimum_frame)
    if any(type(n) is not int or not 0<=n<=maximum for n in (generation_step_cap,verification_step_cap)):
        raise ValueError('Finite caps may not exceed the declared trajectory/branch bound')
    actor_raw=_read(runtime._actor_path)
    if sha256(actor_raw).hexdigest()!=runtime.actor_sha256:raise ValueError('Actual Actor bytes differ')
    raws=dict(zip(INPUT_NAMES,(actor_raw,*[canonical(v).encode() for v in (runtime.protocol,
        {**runtime.actor.metadata,'actor_sha256':runtime.actor_sha256},frozen_selection,pool_scenes,exclusion_pools)])))
    sources=execution_sources(runtime)
    spec={'version':VERSION,'test_fixture':allow_test_fixture,'runtime':identity,
        'inputs':{name:_binding(raw) for name,raw in raws.items()},'sources':sources,'sources_sha256':digest(sources),
        'trajectory_steps':trajectory_steps,'minimum_frame':minimum_frame,
        'caps':{phase:{'environment_steps':n,'zero_step_nn_queries':zero,'nn_queries':n+zero}
            for phase,n in zip(PHASES,(generation_step_cap,verification_step_cap))}}
    if registry.verify(runtime,allow_test_fixture=allow_test_fixture)!=identity or execution_sources(runtime)!=sources:
        raise ValueError('Runtime source changed during preparation')
    _mkdir(output)
    for name,raw in raws.items():put(output/'inputs'/name,raw)
    prepared={'identity':spec,'identity_sha256':digest(spec),'status':'prepared_candidate','eligible':False,
        'release_ready':False,'formal_ready':False,'qualification_evaluated':False,
        'prepare_environment_steps':0,'prepare_nn_queries':0}
    put(output/'prepared.json',prepared)
    ledger={'version':LEDGER_VERSION,'identity_sha256':digest(spec),'revision':0,'automatic_resume':False,
        'reservations_refunded':False,'operations':[],
        'phases':{phase:{'status':'not_started','caps':deepcopy(spec['caps'][phase]),
            'reserved':{k:0 for k in spec['caps'][phase]},'confirmed':{k:0 for k in spec['caps'][phase]},
            'report':None,'bank':None,'error':None} for phase in PHASES}}
    put(output/'ledger.json',ledger);put(output/'run.lock',b'')
    return {**deepcopy(prepared),'prepared_sha256':file_hash(output/'prepared.json')}


def _static(output,expected,fixture):
    """Byte/source preflight, including a cache's records, before Actor creation."""
    output=_absolute(output);raw=_read(output/'prepared.json')
    if type(expected) is not str or not HEX.fullmatch(expected) or sha256(raw).hexdigest()!=expected:
        raise ValueError('External prepared byte anchor differs')
    p=json.loads(raw);s=p['identity']
    if s['version']!=VERSION or type(fixture) is not bool or s['test_fixture'] is not fixture:
        raise ValueError('Workflow or explicit fixture identity differs')
    if p['identity_sha256']!=digest(s) or set(s['inputs'])!=set(INPUT_NAMES) or set(s['caps'])!=set(PHASES):
        raise ValueError('Prepared identity differs')
    if any(p[k] is not False for k in ('eligible','release_ready','formal_ready','qualification_evaluated')):
        raise ValueError('Candidate workflow cannot grant qualification')
    for name,binding in s['inputs'].items():
        if _binding(_read(output/'inputs'/name))!=binding:raise ValueError('Frozen input bytes differ')
    if digest(s['sources'])!=s['sources_sha256']:raise ValueError('Source digest differs')
    for name,expected_sha in s['sources'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(ROOT/name)!=expected_sha:
            raise ValueError('Frozen execution source changed')
    ledger=_json(output/'ledger.json')
    _validate_records(output,p,ledger)
    if ledger['phases']['verification']['status']=='completed':
        if _json(output/'completion_receipt.json')!=_receipt(output,p,ledger):
            raise ValueError('Fixed completed replay receipt differs; it cannot be reconstructed on cache read')
    return output,p,ledger


def _runtime(output,p):
    identity=p['identity']['runtime'];kind=identity['family'];folder=output/'inputs'
    if kind not in CLASSES:raise ValueError('Unregistered runtime family')
    runtime=CLASSES[kind](folder/'actor.npz',protocol=_json(folder/'protocol.json'),
        expected_actor_sha256=identity['actor_sha256'],expected_protocol_sha256=identity['protocol_sha256'],
        expected_bindings=_json(folder/'actor_bindings.json'),allow_test_fixture=p['identity']['test_fixture'],
        config=old_io._configuration(identity['configuration']))
    if registry.verify(runtime,allow_test_fixture=p['identity']['test_fixture'])!=identity:
        raise ValueError('Reconstructed genuine runtime identity differs')
    if execution_sources(runtime)!=p['identity']['sources']:raise ValueError('Runtime source closure differs')
    old_io._selection(_json(folder/'selection.json'),runtime)
    return runtime


def _validate_records(output,p,ledger):
    if (ledger['version']!=LEDGER_VERSION or ledger['identity_sha256']!=p['identity_sha256']
            or ledger['automatic_resume'] is not False or ledger['reservations_refunded'] is not False
            or set(ledger['phases'])!=set(PHASES)):
        raise ValueError('Finite ledger identity or no-retry policy differs')
    seen=set();totals={phase:{field:{k:0 for k in p['identity']['caps'][phase]} for field in ('reserved','confirmed')} for phase in PHASES}
    roots={phase:[] for phase in PHASES};raw_counts={phase:{name:0 for name in ('decision','step','counterfactual')} for phase in PHASES}
    for index,op in enumerate(ledger['operations']):
        c=op['context'];phase=op['phase'];kind=c['kind'];maximum=KINDS.get(kind)
        if (phase not in PHASES or c['operation_id'] in seen or op['index']!=index or digest(c)!=op['context_sha256']
                or type(maximum) is not int or c['maximum_steps']!=maximum
                or c['phase']!=('generation' if phase=='generation' else 'load_reverification')):
            raise ValueError('Operation reservation identity differs')
        seen.add(c['operation_id']);reserved={'environment_steps':maximum,'zero_step_nn_queries':int(maximum==0),'nn_queries':max(1,maximum)}
        if op['reserved']!=reserved or op['status'] not in ('reserved','confirmed'):raise ValueError('Operation reservation changed')
        calls=op['calls'];actual=[];root_results=[]
        for j,call in enumerate(calls):
            if call['index']!=j or call['kind'] not in ('decision','step','counterfactual') or call['status'] not in ('entered','raw_saved'):
                raise ValueError('Raw call journal differs')
            if call['parent'] is not None and (type(call['parent']) is not int or not 0<=call['parent']<j):raise ValueError('Nested call parent differs')
            if call['status']=='raw_saved':
                name=f'operations/{index:06d}/{j:03d}.json'
                if call['raw']['path']!=name:raise ValueError('Raw evidence path differs')
                raw=_read(output/name)
                if _binding(raw)!={k:call['raw'][k] for k in ('sha256','size')}:raise ValueError('Raw operation bytes differ')
                value=json.loads(raw)
                if value['operation_id']!=c['operation_id'] or value['call_index']!=j or value['kind']!=call['kind']:
                    raise ValueError('Raw call belongs to another operation')
                result=value['result'];actual.append((call['kind'],result))
                if call['parent'] is None:root_results.append(result)
                if call['kind']=='step':
                    if (result['runtime_signature']!=p['identity']['runtime']['runtime_signature']
                            or result['submitted_actions']['robot_2']!=result['policy_actions']['robot_2']
                            or result['after']['state']['frame']!=result['before']['state']['frame']+1):
                        raise ValueError('Actual transition identity or unchanged NN submission differs')
        counts=CounterLike(actual)
        for name,n in counts.items():raw_counts[phase][name]+=n
        if counts['step']>maximum or counts['decision']>max(1,maximum):raise ValueError('Recorded work exceeds the operation cap')
        confirmed={k:0 for k in reserved}
        if op['status']=='confirmed':
            if len(root_results)!=1 or any(x['status']!='raw_saved' for x in calls):raise ValueError('Confirmed operation lacks complete raw evidence')
            result=root_results[0];completion=op['completion']
            if any(completion.get(k)!=v for k,v in c.items()) or completion['result_sha256']!=digest(result):
                raise ValueError('Completion differs from the original raw result')
            if completion['actual_steps']!=counts['step'] or counts['decision']!=counts['step']+int(maximum==0):
                raise ValueError('Completion count differs from actual nested calls')
            if (kind=='wait_three' and (counts['counterfactual']!=1 or result['steps_executed']!=counts['step']
                    or digest(result['transitions'])!=digest([r for k,r in actual if k=='step']))) or (kind!='wait_three' and counts['counterfactual']):
                raise ValueError('Counterfactual raw steps differ')
            confirmed={'environment_steps':counts['step'],'zero_step_nn_queries':int(maximum==0),'nn_queries':counts['decision']}
            roots[phase].append({'kind':kind,'scenario_id':c['scenario_id'],'frame':c['frame'],'result_sha256':digest(result)})
        if op['confirmed']!=confirmed:raise ValueError('Confirmed totals differ from original records')
        for field,value in (('reserved',reserved),('confirmed',confirmed)):
            for k,v in value.items():totals[phase][field][k]+=v
    for phase,value in ledger['phases'].items():
        if value['caps']!=p['identity']['caps'][phase] or any(value[field]!=totals[phase][field] for field in ('reserved','confirmed')):
            raise ValueError('Phase totals differ from immutable reservations')
        if any(value['reserved'][k]>cap for k,cap in value['caps'].items()):raise ValueError('Finite phase budget exceeded')
        if value['status']=='completed':
            if any(op['status']!='confirmed' for op in ledger['operations'] if op['phase']==phase):raise ValueError('Completed phase has pending operations')
            raw=_read(output/(phase+'_report.json'))
            if _binding(raw)!=value['report'] or _binding(_read(output/'bank.private.json'))!=value['bank']:
                raise ValueError('Completed cache artifact bytes differ')
            report=json.loads(raw)
            if (report['identity_sha256']!=p['identity_sha256'] or report['confirmed_work']!=value['confirmed']
                    or report['actual_driver_calls']!=raw_counts[phase]
                    or report['actor_sha256']!=p['identity']['runtime']['actor_sha256']
                    or any(report[k] is not False for k in ('eligible','release_ready','formal_ready','explanation_qualified','qualification_evaluated'))):
                raise ValueError('Cached report identity/count/qualification differs')
            if report['original_operation_results_sha256']!=digest(roots[phase]):raise ValueError('Cache operation evidence changed')
            _validate_content_report(output,p,phase,report,value['bank'])
    if ledger['phases']['verification']['status']=='completed' and roots['generation']!=roots['verification']:
        raise ValueError('Independent same-Actor replay differs from generation raw results')


def CounterLike(actual):return {name:sum(k==name for k,_ in actual) for name in ('decision','step','counterfactual')}


def _validate_content_report(output,p,phase,report,binding):
    """Pure saved-content/check/public-projection validation, with no loader replay."""
    bank=_json(output/'bank.private.json');runtime=p['identity']['runtime'];spec=p['identity']
    if (set(bank)!=bank_api._FIELDS or bank['version']!=bank_api.VERSION or bank['status']!='candidate'
            or bank['test_fixture'] is not spec['test_fixture']
            or bank['runtime_family']!=runtime['family'] or bank['runtime_version']!=runtime['runtime_version']
            or bank['runtime_signature']!=runtime['runtime_signature'] or bank['actor_sha256']!=runtime['actor_sha256']
            or bank['protocol_sha256']!=runtime['protocol_sha256'] or bank['registry_sources_sha256']!=runtime['registry_sources_sha256']
            or bank['formal_ready'] is not False or bank['release_ready'] is not False
            or bank['sources_sha256']!=digest(bank['sources'])
            or any(spec['sources'].get(k)!=v for k,v in bank['sources'].items())
            or bank['excluded_pools_sha256']!=digest(_json(output/'inputs/exclusion_pools.json'))
            or digest(bank['pool_scenes'])!=digest(_json(output/'inputs/pool_scenes.json'))
            or bank['trajectory_steps']!=spec['trajectory_steps'] or bank['minimum_frame']!=spec['minimum_frame']
            or bank['pool_namespace']!=bank_api.POOL_NAMESPACE or bank['counterfactual_filter']!=bank_api.FILTER):
        raise ValueError('Cached bank schema, input or runtime family differs')
    checks=bank_api._checks(bank['items'])
    if digest(bank['checks'])!=digest(checks):raise ValueError('Cached bank content checks differ from its actual items')
    public=None
    if phase=='verification':
        public=[] if not checks['passed'] else [{'id':item['id'],'type':'choice','prediction_kind':item['kind'],'required':True,
            'prompt':deepcopy(item['prompt']),'options':[{'value':x['value'],'label':deepcopy(x['label'])} for x in item['options']],
            'preview':deepcopy(item['preview']),'source_frame':item['frame'],'source_scenario':item['scenario_id']} for item in bank['items']]
    expected={'version':VERSION,'phase':phase,'status':'candidate_generated' if phase=='generation' else 'candidate_content_verified',
        'test_fixture':spec['test_fixture'],'runtime_family':runtime['family'],'runtime_signature':runtime['runtime_signature'],
        'bank_binding':binding,'independent_replay_completed':phase=='verification','content_checks':checks,
        'content_eligible':checks['passed'],'public_items':public}
    if digest({k:report.get(k) for k in expected})!=digest(expected):raise ValueError('Cached report content or public projection differs')


def _receipt(output,p,ledger):
    return {'version':VERSION,'scope':'fixed_completed_candidate_bank_and_independent_replay_records',
        'prepared_sha256':file_hash(output/'prepared.json'),'identity_sha256':p['identity_sha256'],
        'ledger_binding':_binding(_read(output/'ledger.json')),'bank_binding':_binding(_read(output/'bank.private.json')),
        'reports':{phase:_binding(_read(output/(phase+'_report.json'))) for phase in PHASES},
        'original_operation_results_sha256':{phase:_json(output/(phase+'_report.json'))['original_operation_results_sha256'] for phase in PHASES},
        'sources_sha256':p['identity']['sources_sha256'],'runtime':p['identity']['runtime'],
        'exclusion_pools_binding':p['identity']['inputs']['exclusion_pools.json'],
        'eligible':False,'release_ready':False,'formal_ready':False,'explanation_qualified':False,
        'physics_replay_on_cache_read':False}


class _Budget:
    def __init__(self,output,p,ledger,phase):
        self.output,self.p,self.ledger,self.phase=output,p,ledger,phase
        self.active=None;self.stack=[];self.actual_returns={'decision':0,'step':0,'counterfactual':0}
    def commit(self,candidate):
        candidate['revision']=self.ledger['revision']+1;put(self.output/'ledger.json',candidate,replace=True);self.ledger=candidate
    def start(self):
        value=deepcopy(self.ledger);value['phases'][self.phase]['status']='running';self.commit(value)
    def before(self,c):
        if self.active is not None or c.get('kind') not in KINDS or c.get('maximum_steps')!=KINDS[c['kind']]:raise ValueError('Nested/unregistered bank operation')
        if c['phase']!=('generation' if self.phase=='generation' else 'load_reverification'):raise ValueError('Wrong bank phase')
        value=deepcopy(self.ledger);phase=value['phases'][self.phase];maximum=c['maximum_steps']
        reserve={'environment_steps':maximum,'zero_step_nn_queries':int(maximum==0),'nn_queries':max(1,maximum)}
        if any(phase['reserved'][k]+n>phase['caps'][k] for k,n in reserve.items()):raise ValueError('Finite bank budget exhausted')
        for k,n in reserve.items():phase['reserved'][k]+=n
        value['operations'].append({'index':len(value['operations']),'phase':self.phase,'context':deepcopy(c),'context_sha256':digest(c),
            'status':'reserved','reserved':reserve,'confirmed':{k:0 for k in reserve},'completion':None,'calls':[]})
        self.commit(value);self.active=len(value['operations'])-1
        return {'execution_permitted':True}
    def observe(self,kind,function,*args,**kwargs):
        if self.active is None:raise ValueError('Runtime call has no durable operation reservation')
        op=self.ledger['operations'][self.active];counts={k:sum(c['kind']==k for c in op['calls']) for k in ('decision','step','counterfactual')}
        limit=op['reserved']['nn_queries'] if kind=='decision' else op['reserved']['environment_steps'] if kind=='step' else int(op['context']['kind']=='wait_three')
        if counts[kind]>=limit:raise ValueError('Actual runtime calls exceed prior reservation')
        value=deepcopy(self.ledger);calls=value['operations'][self.active]['calls'];index=len(calls)
        calls.append({'index':index,'kind':kind,'parent':self.stack[-1] if self.stack else None,'status':'entered','raw':None})
        self.commit(value);self.stack.append(index)
        try:
            result=function(*args,**kwargs);self.actual_returns[kind]+=1
            name=f'operations/{self.active:06d}/{index:03d}.json'
            raw={'operation_id':op['context']['operation_id'],'call_index':index,'kind':kind,'result':result}
            put(self.output/name,raw)
            value=deepcopy(self.ledger);value['operations'][self.active]['calls'][index].update(status='raw_saved',
                raw={'path':name,**_binding(_read(self.output/name))})
            self.commit(value)
            return result
        finally:self.stack.pop()
    def after(self,result):
        if self.active is None or self.stack:raise ValueError('Missing original runtime completion')
        value=deepcopy(self.ledger);op=value['operations'][self.active]
        if result['operation_id']!=op['context']['operation_id']:raise ValueError('Completion operation differs')
        op.update(status='confirmed',completion=deepcopy(result));maximum=op['reserved']['environment_steps']
        counts={kind:sum(c['kind']==kind and c['status']=='raw_saved' for c in op['calls']) for kind in ('decision','step')}
        op['confirmed']={'environment_steps':counts['step'],'zero_step_nn_queries':int(maximum==0),'nn_queries':counts['decision']}
        for k,n in op['confirmed'].items():value['phases'][self.phase]['confirmed'][k]+=n
        _validate_records(self.output,self.p,value)
        self.commit(value);self.active=None
    def finish(self,report,bank_binding):
        value=deepcopy(self.ledger);value['phases'][self.phase].update(status='completed',bank=bank_binding,
            report=_binding(_read(self.output/(self.phase+'_report.json'))))
        _validate_records(self.output,self.p,value);self.commit(value)


@contextmanager
def _observe(runtime,budget):
    """Private genuine-instance instrumentation, reverted even on BaseException."""
    names=('decision','step','counterfactual');saved={name:runtime.__dict__.get(name) for name in names}
    present={name:name in runtime.__dict__ for name in names}
    try:
        for name in names:
            original=getattr(runtime,name)
            def observed(*args,_name=name,_original=original,**kwargs):return budget.observe(_name,_original,*args,**kwargs)
            setattr(runtime,name,observed)
        yield
    finally:
        for name in names:
            if present[name]:setattr(runtime,name,saved[name])
            else:runtime.__dict__.pop(name,None)


def _work(output,phase,anchor,fixture):
    output,p,ledger=_static(output,anchor,fixture)
    fd=os.open(output/'run.lock',os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output,p,ledger=_static(output,anchor,fixture)
        status=ledger['phases'][phase]['status']
        if status=='completed':return deepcopy(_json(output/(phase+'_report.json')))
        if status!='not_started' or any(op['status']!='confirmed' for op in ledger['operations']):
            raise ValueError('Failed/pending work requires diagnosis; automatic replay and refunds are forbidden')
        if phase=='verification' and ledger['phases']['generation']['status']!='completed':raise ValueError('Complete generation is required before independent replay')
        runtime=_runtime(output,p);pool=_json(output/'inputs/pool_scenes.json');excluded=_json(output/'inputs/exclusion_pools.json')
        budget=_Budget(output,p,ledger,phase);budget.start()
        try:
            with _observe(runtime,budget):
                if phase=='generation':
                    bank=bank_api.generate_bank(runtime,pool,excluded,trajectory_steps=p['identity']['trajectory_steps'],minimum_frame=p['identity']['minimum_frame'],
                        before_operation=budget.before,after_operation=budget.after,allow_test_fixture=fixture)
                    put(output/'bank.private.json',bank);checks=bank['checks'];audit=bank['generation_audit'];public=None
                else:
                    bank=bank_api.FamilyQuestionBank(output/'bank.private.json',runtime,excluded,
                        expected_bank_sha256=ledger['phases']['generation']['bank']['sha256'],before_operation=budget.before,
                        after_operation=budget.after,allow_test_fixture=fixture)
                    checks=bank.checks;audit=bank.audit;public=bank.public_items()
            if registry.verify(runtime,allow_test_fixture=fixture)!=p['identity']['runtime'] or execution_sources(runtime)!=p['identity']['sources']:
                raise ValueError('Runtime/driver sources changed during actual execution')
            roots=[]
            for op in budget.ledger['operations']:
                if op['phase']==phase:
                    c=op['context'];roots.append({'kind':c['kind'],'scenario_id':c['scenario_id'],'frame':c['frame'],'result_sha256':op['completion']['result_sha256']})
            confirmed=budget.ledger['phases'][phase]['confirmed']
            if audit['actual_environment_steps']!=confirmed['environment_steps'] or audit['completed_operations']!=len(roots):raise ValueError('Bank audit differs from measured work')
            report={'version':VERSION,'status':'candidate_generated' if phase=='generation' else 'candidate_content_verified','phase':phase,
                'test_fixture':fixture,'identity_sha256':p['identity_sha256'],'runtime_family':p['identity']['runtime']['family'],
                'actor_sha256':runtime.actor_sha256,'runtime_signature':runtime.signature,'content_checks':checks,
                'content_eligible':checks['passed'],'eligible':False,'release_ready':False,'formal_ready':False,
                'explanation_qualified':False,'qualification_evaluated':False,'bank_binding':_binding(_read(output/'bank.private.json')),
                'confirmed_work':deepcopy(confirmed),'bank_operation_audit':audit,'original_operation_results_sha256':digest(roots),
                'actual_driver_calls':deepcopy(budget.actual_returns),'independent_replay_completed':phase=='verification',
                'raw_collection_mechanism':'temporary_private_genuine_runtime_instance_observation_wrappers',
                'cache_scope':'Saved input/source/raw/count verification, never an additional physics replay',
                'public_items':public,'training_steps':0,'optimizer_updates':0,'checkpoint_decodes':0}
            put(output/(phase+'_report.json'),report);budget.finish(report,report['bank_binding'])
            if phase=='verification':put(output/'completion_receipt.json',_receipt(output,p,budget.ledger))
            return deepcopy(report)
        except BaseException as exc:
            # Durable state is authoritative after uncertain I/O; never report
            # in-memory progress as a committed observation or permit a rerun.
            try:
                disk=_json(output/'ledger.json')
                calls=[c for op in disk['operations'] if op['phase']==phase for c in op['calls']]
                failure={'version':VERSION,'phase':phase,'error':repr(exc),'actual_driver_returns':budget.actual_returns,
                    'durable_phase':disk['phases'][phase],
                    'durable_entered_driver_calls':{name:sum(c['kind']==name for c in calls) for name in budget.actual_returns},
                    'durable_saved_driver_calls':{name:sum(c['kind']==name and c['status']=='raw_saved' for c in calls) for name in budget.actual_returns},
                    'unconfirmed_partial_call_work_may_be_unknown':any(c['status']=='entered' for c in calls),
                    'retry_allowed':False,'eligible':False}
                put(output/(phase+'_failure.json'),failure)
                if disk==budget.ledger:
                    value=deepcopy(disk);value['phases'][phase].update(status='failed',error=repr(exc));budget.commit(value)
            except BaseException:pass
            raise


def generate(output,*,expected_prepared_sha256,allow_test_fixture=False):return _work(output,'generation',expected_prepared_sha256,allow_test_fixture)
def verify(output,*,expected_prepared_sha256,allow_test_fixture=False):return _work(output,'verification',expected_prepared_sha256,allow_test_fixture)
def read_completed(output,*,expected_prepared_sha256,allow_test_fixture=False):
    output,p,ledger=_static(output,expected_prepared_sha256,allow_test_fixture)
    if ledger['phases']['verification']['status']!='completed':raise ValueError('Independent bank replay is incomplete')
    return {'report':deepcopy(_json(output/'verification_report.json')),
        'bank_private_data':deepcopy(_json(output/'bank.private.json')),
        'bank_binding':deepcopy(ledger['phases']['verification']['bank']),
        'replay_receipt':deepcopy(_json(output/'completion_receipt.json')),
        'replay_receipt_sha256':file_hash(output/'completion_receipt.json')}
