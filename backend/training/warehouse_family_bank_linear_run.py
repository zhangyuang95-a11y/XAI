"""Linear append-only bank accounting with genuine independent physical replay.

Per-operation checks inspect only new files. Full input/source/raw/chain checks
run at phase entry, completion and cache reads. This is not an assertion that
arbitrary external edits are detected immediately between those boundaries.
All outputs remain component evidence, never research/release qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path

from backend.training import warehouse_family_bank_run as original
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.training.warehouse_native_revision_provenance import _absolute

VERSION='warehouse-runtime-family-bank-linear-workflow.v1'
LEDGER_VERSION='warehouse-runtime-family-bank-linear-ledger.v1'
OP_VERSION='warehouse-runtime-family-bank-linear-operation.v1'
INPUT_NAMES=original.INPUT_NAMES
PHASES=original.PHASES
KINDS=original.KINDS
CLASSES=original.CLASSES
HEX=original.HEX
registry=original.registry
bank_api=original.bank_api
old_io=original.old_io
_read,_json,_mkdir,put,_binding=original._read,original._json,original._mkdir,original.put,original._binding
_observe=original._observe


def execution_sources(runtime):
    values=original.execution_sources(runtime)
    values[str(Path(__file__).relative_to(ROOT))]=file_hash(__file__)
    return values


def _same(a,b,reason):
    if canonical(a)!=canonical(b):raise ValueError(reason)


def _zeros():return {'environment_steps':0,'zero_step_nn_queries':0,'nn_queries':0}


def _initial_head(p):
    return {'version':LEDGER_VERSION,'identity_sha256':p['identity_sha256'],'revision':0,
        'automatic_resume':False,'reservations_refunded':False,'next_sequence':0,
        'last_ack_sha256':digest({'version':LEDGER_VERSION,'identity_sha256':p['identity_sha256']}),
        'pending':None,'phases':{phase:{'status':'not_started','caps':deepcopy(p['identity']['caps'][phase]),
            'reserved':_zeros(),'confirmed':_zeros(),'report':None,'bank':None,'error':None} for phase in PHASES}}


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
    put(output/'head.json',_initial_head(prepared));put(output/'run.lock',b'')
    return {**deepcopy(prepared),'prepared_sha256':file_hash(output/'prepared.json')}

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


def _bound(output,item):
    path=Path(item['path'])
    if path.is_absolute() or '..' in path.parts or str(path)!=item['path']:raise ValueError('Unsafe original evidence path')
    raw=_read(output/path)
    _same(_binding(raw),{k:item[k] for k in ('sha256','size')},'Original operation bytes differ')
    return json.loads(raw)


def _reservation(p,sequence,phase,context,previous):
    maximum=KINDS.get(context.get('kind'))
    if (type(maximum) is not int or context.get('maximum_steps')!=maximum
            or context.get('phase')!=('generation' if phase=='generation' else 'load_reverification')
            or type(context.get('operation_id')) is not str or not context['operation_id']):
        raise ValueError('Unknown operation/phase/finite limit')
    return {'version':OP_VERSION,'identity_sha256':p['identity_sha256'],'sequence':sequence,'phase':phase,
        'context':deepcopy(context),'context_sha256':digest(context),'previous_ack_sha256':previous,
        'reserved':{'environment_steps':maximum,'zero_step_nn_queries':int(maximum==0),'nn_queries':max(1,maximum)},
        'permanent':True,'refund_allowed':False}


def _operation(output,p,reservation,reservation_binding,ack=None):
    """Bounded new-operation check, also reused by the linear full audit."""
    _same(_bound(output,reservation_binding),reservation,'Durable reservation changed before confirmation')
    index=reservation['sequence'];folder=output/f'operations/{index:06d}';context=reservation['context']
    calls=[];actual=[];roots=[];files=set()
    call_dir=folder/'calls'
    entered=sorted(call_dir.glob('*.enter.json')) if call_dir.exists() else []
    for j,path in enumerate(entered):
        name=f'operations/{index:06d}/calls/{j:03d}.enter.json'
        if path!=output/name:raise ValueError('Call sequence gap')
        raw=_read(path);value=json.loads(raw);parent=value['parent'];kind=value['kind']
        if (value.get('version')!=OP_VERSION or value.get('operation_id')!=context['operation_id']
                or value.get('index')!=j or kind not in ('decision','step','counterfactual')
                or (parent is not None and (type(parent) is not int or not 0<=parent<j))):
            raise ValueError('Nested original call identity differs')
        expected_root='counterfactual' if context['kind']=='wait_three' else 'step' if reservation['reserved']['environment_steps'] else 'decision'
        if ((parent is None and (j!=0 or kind!=expected_root))
                or (parent is not None and (calls[parent]['kind'],kind) not in (('counterfactual','step'),('step','decision')))):
            raise ValueError('Original runtime nested call graph differs')
        item={'index':j,'kind':kind,'parent':parent,'entered':{'path':name,**_binding(raw)},'raw':None}
        files.add(path.name);result_path=path.with_name(f'{j:03d}.raw.json')
        if result_path.exists():
            result_raw=_read(result_path);stored=json.loads(result_raw)
            if (stored.get('operation_id')!=context['operation_id'] or stored.get('call_index')!=j or stored.get('kind')!=kind):
                raise ValueError('Raw result belongs to another call')
            item['raw']={'path':str(result_path.relative_to(output)),**_binding(result_raw)}
            result=stored['result'];actual.append((kind,result));files.add(result_path.name)
            if parent is None:roots.append((kind,result))
            if kind=='step' and (result['runtime_signature']!=p['identity']['runtime']['runtime_signature']
                    or result['submitted_actions']['robot_2']!=result['policy_actions']['robot_2']
                    or result['after']['state']['frame']!=result['before']['state']['frame']+1):
                raise ValueError('Physical transition or raw NN submission differs')
        calls.append(item)
    if call_dir.exists() and (call_dir.is_symlink() or {x.name for x in call_dir.iterdir()}!=files):
        raise ValueError('Unregistered nested raw files')
    counts={kind:sum(k==kind for k,_ in actual) for kind in ('decision','step','counterfactual')}
    maximum=reservation['reserved']['environment_steps'];kind=context['kind']
    entered_counts={k:sum(c['kind']==k for c in calls) for k in counts}
    if (entered_counts['step']>maximum or entered_counts['decision']>max(1,maximum)
            or entered_counts['counterfactual']>int(kind=='wait_three')):
        raise ValueError('Entered calls exceed permanent reservation')
    confirmed=_zeros();root=None
    if ack is not None:
        if (any(c['raw'] is None for c in calls) or len(roots)!=1
                or roots[0][0]!=('counterfactual' if kind=='wait_three' else 'step' if maximum else 'decision')):
            raise ValueError('Acknowledgment lacks one complete original root call')
        if any(sum(c['parent']==step['index'] and c['kind']=='decision' for c in calls)!=1
                for step in calls if step['kind']=='step'):
            raise ValueError('Original step call graph lacks its single NN decision')
        _same(ack['calls'],calls,'Immutable call bindings differ')
        _same(ack['reservation'],reservation_binding,'ACK reservation bytes differ')
        if (ack.get('version')!=OP_VERSION or ack.get('sequence')!=index or ack.get('phase')!=reservation['phase']
                or ack.get('previous_ack_sha256')!=reservation['previous_ack_sha256']):
            raise ValueError('ACK chain or producer differs')
        result=roots[0][1];completion=ack['completion']
        for k,v in context.items():_same(completion.get(k),v,'Original operation completion differs')
        if (completion['result_sha256']!=digest(result) or completion['actual_steps']!=counts['step']
                or counts['decision']!=counts['step']+int(maximum==0)):
            raise ValueError('Completion differs from actual nested NN/step calls')
        if kind=='wait_three':
            if (counts['counterfactual']!=1 or result['steps_executed']!=counts['step']
                    or digest(result['transitions'])!=digest([r for k,r in actual if k=='step'])):
                raise ValueError('Counterfactual original transitions differ')
        elif counts['counterfactual']:raise ValueError('Unexpected counterfactual execution')
        confirmed={'environment_steps':counts['step'],'zero_step_nn_queries':int(maximum==0),'nn_queries':counts['decision']}
        _same(ack['confirmed'],confirmed,'ACK work counters differ')
        root={'kind':kind,'scenario_id':context['scenario_id'],'frame':context['frame'],'result_sha256':digest(result)}
    return {'calls':calls,'raw_counts':counts,'entered_counts':entered_counts,'confirmed':confirmed,'root':root}


def _full(output,p,head):
    """One pass over all original operations, with constant reads per raw file."""
    expected=_initial_head(p)
    for key in ('version','identity_sha256','automatic_resume','reservations_refunded'):
        _same(head[key],expected[key],'Head identity or permanence differs')
    if type(head['next_sequence']) is not int or head['next_sequence']<0:raise ValueError('Invalid operation count')
    directory=output/'operations';names={f'{i:06d}' for i in range(head['next_sequence'])}
    if directory.exists() and (directory.is_symlink() or {x.name for x in directory.iterdir()}!=names):
        raise ValueError('Uncommitted, missing or extra operation directory')
    roots={phase:[] for phase in PHASES};raw_counts={phase:{k:0 for k in ('decision','step','counterfactual')} for phase in PHASES}
    seen=set();previous=expected['last_ack_sha256'];pending=None
    for index in range(head['next_sequence']):
        folder=output/f'operations/{index:06d}';raw=_read(folder/'reservation.json');reservation=json.loads(raw)
        phase=reservation['phase'];context=reservation['context']
        if phase not in PHASES or context['operation_id'] in seen or pending is not None:
            raise ValueError('Duplicate operation or sampling after pending')
        if phase=='generation' and roots['verification']:raise ValueError('Generation resumed after independent verification')
        seen.add(context['operation_id'])
        _same(reservation,_reservation(p,index,phase,context,previous),'Original reservation differs')
        binding={'path':str((folder/'reservation.json').relative_to(output)),**_binding(raw)}
        ack_raw=_read(folder/'ack.json') if (folder/'ack.json').exists() else None
        ack=json.loads(ack_raw) if ack_raw is not None else None
        allowed={'reservation.json','calls'}|({'ack.json'} if ack is not None else set())
        if {x.name for x in folder.iterdir()}-allowed:raise ValueError('Unregistered operation artifact')
        value=_operation(output,p,reservation,binding,ack)
        for field,values in (('reserved',reservation['reserved']),('confirmed',value['confirmed'])):
            for k,v in values.items():expected['phases'][phase][field][k]+=v
        for k,n in value['raw_counts'].items():raw_counts[phase][k]+=n
        if ack is None:pending={'sequence':index,'reservation':binding}
        else:previous=sha256(ack_raw).hexdigest();roots[phase].append(value['root'])
    _same(head['pending'],pending,'Pending reservation differs')
    _same(head['last_ack_sha256'],previous,'Last confirmed chain SHA differs')
    if set(head['phases'])!=set(PHASES):raise ValueError('Unknown workflow phase')
    for phase,value in head['phases'].items():
        for key in ('caps','reserved','confirmed'):_same(value[key],expected['phases'][phase][key],'Phase counters or caps differ')
        if any(value['reserved'][k]>cap for k,cap in value['caps'].items()):raise ValueError('Finite phase exhausted')
        if value['status'] not in ('not_started','running','completed','failed'):raise ValueError('Unknown phase status')
        if value['status']=='not_started' and any(value['reserved'].values()):raise ValueError('Unstarted phase already consumed budget')
        if value['status']=='completed':
            if pending is not None:raise ValueError('Completed phase has pending work')
            report_raw=_read(output/(phase+'_report.json'));report=json.loads(report_raw)
            _same(_binding(report_raw),value['report'],'Completed report bytes differ')
            _same(_binding(_read(output/'bank.private.json')),value['bank'],'Completed bank bytes differ')
            if (report['identity_sha256']!=p['identity_sha256'] or report['actor_sha256']!=p['identity']['runtime']['actor_sha256']
                    or report['confirmed_work']!=value['confirmed'] or report['actual_driver_calls']!=raw_counts[phase]
                    or report['original_operation_results_sha256']!=digest(roots[phase])
                    or any(report[k] is not False for k in ('eligible','release_ready','formal_ready','explanation_qualified','qualification_evaluated'))):
                raise ValueError('Completed original report/count/scope differs')
            _validate_content_report(output,p,phase,report,value['bank'])
    if head['phases']['verification']['status']=='completed':
        if head['phases']['generation']['status']!='completed' or roots['generation']!=roots['verification']:
            raise ValueError('Independent replay original results differ')
    return {'roots':roots,'raw_counts':raw_counts,'seen_ids':seen}


def _inputs(output,expected,fixture):
    if type(expected) is not str or not HEX.fullmatch(expected):raise ValueError('Explicit prepared SHA required')
    output=_absolute(output);raw=_read(output/'prepared.json')
    if sha256(raw).hexdigest()!=expected:raise ValueError('Prepared byte anchor differs')
    p=json.loads(raw);s=p['identity']
    if (s['version']!=VERSION or type(fixture) is not bool or s['test_fixture'] is not fixture
            or p['identity_sha256']!=digest(s) or set(s['inputs'])!=set(INPUT_NAMES) or set(s['caps'])!=set(PHASES)
            or any(p[k] is not False for k in ('eligible','release_ready','formal_ready','qualification_evaluated'))):
        raise ValueError('Explicit workflow identity or candidate scope differs')
    for name,item in s['inputs'].items():_same(_binding(_read(output/'inputs'/name)),item,'Frozen input bytes differ')
    _same(digest(s['sources']),s['sources_sha256'],'Execution source digest differs')
    for name,value in s['sources'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(ROOT/name)!=value:
            raise ValueError('Original execution source changed')
    return output,p


def _receipt(output,p,head):
    return {'version':VERSION,'ledger_version':LEDGER_VERSION,'prepared_sha256':file_hash(output/'prepared.json'),
        'identity_sha256':p['identity_sha256'],'head_binding':_binding(_read(output/'head.json')),
        'last_ack_sha256':head['last_ack_sha256'],'operation_count':head['next_sequence'],
        'bank_binding':_binding(_read(output/'bank.private.json')),
        'reports':{phase:_binding(_read(output/(phase+'_report.json'))) for phase in PHASES},
        'runtime':p['identity']['runtime'],'sources_sha256':p['identity']['sources_sha256'],
        'exclusion_pools_binding':p['identity']['inputs']['exclusion_pools.json'],
        'eligible':False,'release_ready':False,'formal_ready':False,'explanation_qualified':False,
        'physics_replay_on_cache_read':False,'scope':'same-Actor bank plus actual independently repeated generation trajectory'}


def _static(output,expected,fixture):
    output,p=_inputs(output,expected,fixture);head=_json(output/'head.json');audit=_full(output,p,head)
    if head['phases']['verification']['status']=='completed':
        _same(_json(output/'completion_receipt.json'),_receipt(output,p,head),'Original completed replay receipt differs')
    return output,p,head,audit


class _Budget:
    def __init__(self,output,p,head,phase,audit):
        self.output,self.p,self.head,self.phase=output,p,head,phase
        self.active=None;self.calls=[];self.stack=[];self.seen=set(audit['seen_ids'])
        self.roots=list(audit['roots'][phase]);self.actual_returns={'decision':0,'step':0,'counterfactual':0}

    def commit(self,value):
        value['revision']=self.head['revision']+1
        put(self.output/'head.json',value,replace=True);self.head=value

    def start(self):
        value=deepcopy(self.head);value['phases'][self.phase]['status']='running';self.commit(value)

    def before(self,context):
        if self.active is not None or self.head['pending'] is not None or context.get('operation_id') in self.seen:
            raise ValueError('Pending or duplicate bank operation cannot be executed')
        _same(_json(self.output/'head.json'),self.head,'Head changed between operations')
        index=self.head['next_sequence'];reservation=_reservation(self.p,index,self.phase,context,self.head['last_ack_sha256'])
        value=deepcopy(self.head);phase=value['phases'][self.phase]
        if any(phase['reserved'][k]+n>phase['caps'][k] for k,n in reservation['reserved'].items()):
            raise ValueError('Finite bank budget exhausted')
        path=self.output/f'operations/{index:06d}/reservation.json';put(path,reservation)
        binding={'path':str(path.relative_to(self.output)),**_binding(_read(path))}
        for k,n in reservation['reserved'].items():phase['reserved'][k]+=n
        value['next_sequence']+=1;value['pending']={'sequence':index,'reservation':binding};self.commit(value)
        self.active=reservation;self.calls=[];self.seen.add(context['operation_id'])
        return {'execution_permitted':True}

    def observe(self,kind,function,*args,**kwargs):
        if self.active is None:raise ValueError('No durable operation reservation')
        reserved=self.active['reserved'];maximum=reserved['nn_queries'] if kind=='decision' else reserved['environment_steps'] if kind=='step' else int(self.active['context']['kind']=='wait_three')
        if sum(c['kind']==kind for c in self.calls)>=maximum:raise ValueError('Nested calls exceed finite reservation')
        index=len(self.calls);seq=self.active['sequence'];parent=self.stack[-1] if self.stack else None
        base=f'operations/{seq:06d}/calls/{index:03d}';value={'version':OP_VERSION,
            'operation_id':self.active['context']['operation_id'],'index':index,'kind':kind,'parent':parent}
        put(self.output/(base+'.enter.json'),value)
        item={'index':index,'kind':kind,'parent':parent,'entered':{'path':base+'.enter.json',**_binding(_read(self.output/(base+'.enter.json')))},'raw':None}
        self.calls.append(item);self.stack.append(index)
        try:
            result=function(*args,**kwargs);self.actual_returns[kind]+=1
            raw={'operation_id':self.active['context']['operation_id'],'call_index':index,'kind':kind,'result':result}
            put(self.output/(base+'.raw.json'),raw)
            item['raw']={'path':base+'.raw.json',**_binding(_read(self.output/(base+'.raw.json')))}
            return result
        finally:self.stack.pop()

    def after(self,result):
        if self.active is None or self.stack:raise ValueError('Missing complete original root return')
        reservation=self.active;maximum=reservation['reserved']['environment_steps'];sequence=reservation['sequence']
        counts={k:sum(c['kind']==k and c['raw'] is not None for c in self.calls) for k in ('decision','step')}
        confirmed={'environment_steps':counts['step'],'zero_step_nn_queries':int(maximum==0),'nn_queries':counts['decision']}
        ack={'version':OP_VERSION,'sequence':sequence,'phase':self.phase,'previous_ack_sha256':reservation['previous_ack_sha256'],
            'reservation':deepcopy(self.head['pending']['reservation']),'calls':deepcopy(self.calls),
            'confirmed':confirmed,'completion':deepcopy(result)}
        actual=_operation(self.output,self.p,reservation,ack['reservation'],ack)
        path=self.output/f'operations/{sequence:06d}/ack.json';put(path,ack)
        value=deepcopy(self.head);value['last_ack_sha256']=sha256(_read(path)).hexdigest();value['pending']=None
        for k,n in confirmed.items():value['phases'][self.phase]['confirmed'][k]+=n
        self.commit(value);self.roots.append(actual['root']);self.active=None;self.calls=[]

    def finish(self,report):
        value=deepcopy(self.head);value['phases'][self.phase].update(status='completed',bank=report['bank_binding'],
            report=_binding(_read(self.output/(self.phase+'_report.json'))))
        _full(self.output,self.p,value);self.commit(value)


def _work(output,phase,anchor,fixture):
    output=_absolute(output)
    fd=os.open(output/'run.lock',os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output,p,head,audit=_static(output,anchor,fixture)
        status=head['phases'][phase]['status']
        if status=='completed':return deepcopy(_json(output/(phase+'_report.json')))
        if status!='not_started' or head['pending'] is not None:
            raise ValueError('Failed/pending stages cannot be automatically resampled')
        if phase=='verification' and head['phases']['generation']['status']!='completed':
            raise ValueError('Complete generation required before independent replay')
        runtime=_runtime(output,p);pool=_json(output/'inputs/pool_scenes.json');excluded=_json(output/'inputs/exclusion_pools.json')
        budget=_Budget(output,p,head,phase,audit);budget.start()
        try:
            with _observe(runtime,budget):
                if phase=='generation':
                    bank=bank_api.generate_bank(runtime,pool,excluded,trajectory_steps=p['identity']['trajectory_steps'],
                        minimum_frame=p['identity']['minimum_frame'],before_operation=budget.before,after_operation=budget.after,allow_test_fixture=fixture)
                    put(output/'bank.private.json',bank);checks=bank['checks'];measured=bank['generation_audit'];public=None
                else:
                    bank=bank_api.FamilyQuestionBank(output/'bank.private.json',runtime,excluded,
                        expected_bank_sha256=head['phases']['generation']['bank']['sha256'],
                        before_operation=budget.before,after_operation=budget.after,allow_test_fixture=fixture)
                    checks=bank.checks;measured=bank.audit;public=bank.public_items()
            if registry.verify(runtime,allow_test_fixture=fixture)!=p['identity']['runtime'] or execution_sources(runtime)!=p['identity']['sources']:
                raise ValueError('Genuine runtime or driver source changed')
            confirmed=budget.head['phases'][phase]['confirmed']
            if measured['actual_environment_steps']!=confirmed['environment_steps'] or measured['completed_operations']!=len(budget.roots):
                raise ValueError('Original bank audit differs from actual acknowledged work')
            report={'version':VERSION,'status':'candidate_generated' if phase=='generation' else 'candidate_content_verified',
                'phase':phase,'test_fixture':fixture,'identity_sha256':p['identity_sha256'],'runtime_family':p['identity']['runtime']['family'],
                'actor_sha256':runtime.actor_sha256,'runtime_signature':runtime.signature,'content_checks':checks,'content_eligible':checks['passed'],
                'eligible':False,'release_ready':False,'formal_ready':False,'explanation_qualified':False,'qualification_evaluated':False,
                'bank_binding':_binding(_read(output/'bank.private.json')),'confirmed_work':deepcopy(confirmed),
                'bank_operation_audit':measured,'original_operation_results_sha256':digest(budget.roots),
                'actual_driver_calls':deepcopy(budget.actual_returns),'independent_replay_completed':phase=='verification',
                'raw_collection_mechanism':'temporary_private_genuine_runtime_instance_observation_wrappers',
                'history_integrity_scope':'full-chain checks at entry/completion/cache; incremental checks inspect only new operation',
                'public_items':public,'training_steps':0,'optimizer_updates':0,'checkpoint_decodes':0}
            put(output/(phase+'_report.json'),report);budget.finish(report)
            if phase=='verification':put(output/'completion_receipt.json',_receipt(output,p,budget.head))
            return deepcopy(report)
        except BaseException as error:
            # Never replace uncertain durable ACK/head state with in-memory counts.
            try:
                disk=_json(output/'head.json');value=deepcopy(disk)
                value['phases'][phase].update(status='failed',error=repr(error))
                put(output/(phase+'_failure.json'),{'version':VERSION,'phase':phase,'error':repr(error),
                    'durable_head':disk,'actual_driver_returns':budget.actual_returns,
                    'partial_execution_may_be_unknown':disk['pending'] is not None,
                    'orphan_reservation_or_ack_may_require_diagnosis':True,'retry_allowed':False,'eligible':False})
                value['revision']=disk['revision']+1;put(output/'head.json',value,replace=True)
            except BaseException:pass
            raise


def generate(output,*,expected_prepared_sha256,allow_test_fixture=False):
    return _work(output,'generation',expected_prepared_sha256,allow_test_fixture)


def verify(output,*,expected_prepared_sha256,allow_test_fixture=False):
    return _work(output,'verification',expected_prepared_sha256,allow_test_fixture)


def read_completed(output,*,expected_prepared_sha256,allow_test_fixture=False):
    output=_absolute(output);fd=os.open(output/'run.lock',os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output,p,head,_=_static(output,expected_prepared_sha256,allow_test_fixture)
        if head['phases']['verification']['status']!='completed':raise ValueError('Independent replay is incomplete')
        return {'version':VERSION,'report':deepcopy(_json(output/'verification_report.json')),
            'bank_private_data':deepcopy(_json(output/'bank.private.json')),
            'bank_binding':deepcopy(head['phases']['verification']['bank']),
            'replay_receipt':deepcopy(_json(output/'completion_receipt.json')),
            'replay_receipt_sha256':file_hash(output/'completion_receipt.json')}
