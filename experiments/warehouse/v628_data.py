"""Single-use frozen-NN development data producer and scene-level partitions.

No ordinary API can obtain authority from this module. The source-pinned plan,
exact historical seed audit and a single-use collection directory are required
before any environment constructor consumes RNG.
"""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import gzip,json,os
from pathlib import Path
import numpy as np
from . import v628_protocol as v
from scripts.run_warehouse_v625 import write_new

ACTIVE=ContextVar('warehouse_v628_finite_collection',default=None)
HISTORY_NAMES=('XAI','XAI-study-v3','XAI-v6-integration','XAI-v6-recovery-20261015T165508Z')


def cases():
    from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
    out=[]
    for mode,count,offset in (('argmax',300,0),('sampled',1350,300)):
        for i in range(count):
            task=i%3+1;index=offset+i;base=86280000+index;action=5628000000+index
            out.append(dict(case_id=f'dev-{index:04}',base=base,scenario_seed=10*base+task,task=task,
                action_seed=action,sampling_mode=mode,draw=0))
    for key in ('base','scenario_seed','action_seed'):
        v.require(len({c[key] for c in out})==1650 and not any(is_sealed_scenario_seed(c[key]) for c in out),'seed overlaps or sealed argument')
    sets=[{c[key] for c in out} for key in ('base','scenario_seed','action_seed')]
    v.require(all(not a&b for i,a in enumerate(sets) for b in sets[:i]),'RNG namespaces overlap')
    return out


def folds():
    c=cases();out=[]
    for fold in range(3):
        outer={s:[] for s in ('fit','calibration','eligibility','validation')}
        inner={s:[] for s in outer}
        for task in (1,2,3):
            arg=[r['case_id'] for r in c if r['task']==task and r['sampling_mode']=='argmax']
            sampled=[r['case_id'] for r in c if r['task']==task and r['sampling_mode']=='sampled']
            shift=100*fold;sampled=sampled[shift:]+sampled[:shift]
            outer['fit']+=arg+sampled[:200];outer['calibration']+=sampled[200:250]
            outer['eligibility']+=sampled[250:350];outer['validation']+=sampled[350:450]
            inner['fit']+=arg+sampled[:50];inner['calibration']+=sampled[50:80]
            inner['eligibility']+=sampled[80:140];inner['validation']+=sampled[140:200]
        v.require({k:len(a) for k,a in outer.items()}==dict(fit=900,calibration=150,eligibility=300,validation=300),'outer counts differ')
        v.require({k:len(a) for k,a in inner.items()}==dict(fit=450,calibration=90,eligibility=180,validation=180),'inner counts differ')
        for groups in (outer,inner):
            v.require(all(not set(a)&set(b) for i,a in enumerate(groups.values()) for b in list(groups.values())[:i]),'scene leakage')
        out.append(dict(fold=fold,outer=outer,inner=inner))
    return out


def exact_plan():
    uses=[]
    for c in cases():
        for kind,arg,key in (('scenario','task_expansion_base','base'),('env','random.Random(config.seed)','scenario_seed'),
            ('env','reset(seed)','scenario_seed'),('action_rng','FrozenActor.act(base_seed)','action_seed')):
            row=dict(kind=kind,purpose='v628_development',argument=arg,value=c[key],group=c['case_id'],reuse_tag=key+':'+str(c[key]))
            if kind=='action_rng':row['consumed_as_random_draw']=c['sampling_mode']=='sampled'
            uses.append(row)
    return dict(schema='warehouse-exact-seed-plan.v1',candidate_id=v.CANDIDATE,uses=uses,authorization_granted=False)


def audit_seeds():
    """Complete existing bounded/triaged tools; no model weight content reads."""
    from . import exact_seed_audit as exact,seed_metadata_audit as meta,seed_metadata_triage as triage
    folder=v.OUTPUT/'seed-ledger';folder.mkdir(exist_ok=False)
    write_new(folder/'exact-plan.json',exact_plan());families=exact.compile_families(exact_plan())
    inventory=v.bound(dict(path=str(v.ROOT/'output/warehouse_v621/seed-ledger-03/rebound-01/inventory.json'),
        sha256='98ea8b4d0aed59978d8537e45b8677fa84af848416a64ccc6aa1b0d9edf62b19'))
    traces=inventory['registered_trace_paths']
    proposal=dict(schema='warehouse-seed-proposal.v1',candidate_id=v.CANDIDATE,
        families=[{k:r[k] for k in ('name','kind','values','query_only')} for r in families],
        exclude_paths=[str(v.OUTPUT)],registered_trace_paths=traces)
    write_new(folder/'proposal.json',proposal)
    roots=[v.ROOT.parent/name for name in HISTORY_NAMES]
    meta.audit(folder/'proposal.json',roots,folder/'bounded')
    result=triage.triage(folder/'bounded/inventory.json',folder/'proposal.json',folder/'triage')
    evidence=exact.assess(folder/'exact-plan.json',folder/'triage/result.json',folder/'exact',context_source_root=v.ROOT)
    v.require(not evidence['summary']['unresolved_paths'],'unresolved historical metadata')
    for row in evidence['summary']['families'].values():
        v.require(not any(row.get(k) for k in ('references_or_reservations','reservation_overlaps',
            'audit_references_not_original_execution','excluded_weight_path_seed_hints')),'historical seed conflict')
    # Record the complete stat selection for a pre-constructor delta check.
    stats={};selection=[]
    for p in meta.candidates(roots,[v.OUTPUT],registered_trace_paths=traces,selection_stats=stats):
        s=p.stat();selection.append(dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in result['inventory'] if r['classification']!='source_template_environment_reservation'}
    v.require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in selection},'history changed during seed audit')
    receipt=dict(schema='warehouse-v628-seed-clearance.v1',cases_sha256=v.digest(cases()),exact=v.pointer(folder/'exact/result.json'),
        selection=selection,weight_paths=stats['excluded_weights_name_stat_only'],registered_traces=traces,
        approved_finite_development_resets=1650,sealed_authority=False,training_authority=False)
    write_new(folder/'clearance.json',receipt);return receipt


def checked_clearance():
    from . import seed_metadata_audit as meta
    c=v.bound(v.pointer(v.OUTPUT/'seed-ledger/clearance.json'));v.bound(c['exact'])
    v.require(c['cases_sha256']==v.digest(cases()) and c['approved_finite_development_resets']==1650,'wrong seed clearance')
    stats={};current=[]
    for p in meta.candidates([v.ROOT.parent/name for name in HISTORY_NAMES],[v.OUTPUT],registered_trace_paths=c['registered_traces'],selection_stats=stats):
        s=p.stat();current.append(dict(path=str(p.resolve()),bytes=s.st_size,mtime_ns=s.st_mtime_ns))
    v.require(current==c['selection'] and stats['excluded_weights_name_stat_only']==c['weight_paths'],'historical metadata changed after clearance')
    return c


@contextmanager
def capability():
    path,plan=v.checked_plan();checked_clearance()
    v.require(ACTIVE.get() is None,'finite collection cannot nest')
    folder=v.OUTPUT/'collection';folder.mkdir(exist_ok=False)
    write_new(folder/'started.json',dict(plan=v.pointer(path),scheduled_resets=1650,single_use_even_if_interrupted=True))
    token=ACTIVE.set(dict(cursor=0,plan=plan,folder=folder))
    try:
        yield ACTIVE.get()
        v.require(ACTIVE.get()['cursor']==1650,'incomplete finite schedule')
        v.checked_plan()
    except BaseException as e:
        write_new(folder/'failed.json',dict(error_type=type(e).__name__,error=str(e),preauthorized_resets=ACTIVE.get()['cursor'],automatic_retry=False));raise
    finally:ACTIVE.reset(token)


def collect():
    from backend.training.warehouse_v6_runner import _set_dual_nn_mode
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,encode_actor_tree_features
    from backend.training.warehouse_v622_program import masked_probabilities
    from env.warehouse.domain import participant_study_config
    from env.warehouse.environment import WarehouseMultiAgentEnv
    from env.warehouse.public_action_features_v617 import validate_study_action_physics
    from env.warehouse.public_observation_v6 import encode_public_observation,public_pre_action_snapshot
    from env.warehouse.pure_nn import AGENTS,action_masks
    from env.warehouse.navigation import ACTIONS
    from env.warehouse.decision_protocol import independent_agent_seed
    from env.warehouse.reset_audit_v614 import capture_stage1_resets,stage1_reset_phase
    with capability() as cap:
        ref=cap['plan']['actor'];actor=FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],expected_role_sha256=v.old.ROLE_SHA256)
        refs={};outcomes=[];checks=0
        journal=cap['folder']/'reset-preauthorization.jsonl'
        with journal.open('x') as auth,capture_stage1_resets(cap['folder']/'actual-resets.jsonl') as actual:
            for c in cap['plan']['cases']:
                v.require(ACTIVE.get() is cap and c==cap['plan']['cases'][cap['cursor']],'reset outside finite schedule')
                # Source/config identity is checked before each RNG constructor.
                v.require(v.source_files()==cap['plan']['source_files'],'collection source changed')
                auth.write(json.dumps(dict(index=cap['cursor'],case=c,plan_sha256=v.file_sha(v.OUTPUT/'plan.json')))+'\n');auth.flush();os.fsync(auth.fileno());cap['cursor']+=1
                config=participant_study_config(seed=c['scenario_seed']);validate_study_action_physics(config)
                with stage1_reset_phase('program'):env=WarehouseMultiAgentEnv(config);env.reset(seed=c['scenario_seed'])
                _set_dual_nn_mode(env);data={k:[] for k in ('x','m','p','sampled')};previous=None
                trace=cap['folder']/(c['case_id']+'.jsonl.gz')
                with gzip.open(trace,'xt') as h:
                    while not (env.state.terminated or env.state.truncated):
                        obs,m=encode_public_observation(env.state),action_masks(env.state)
                        before=public_pre_action_snapshot(env.state)
                        v.require(previous is None or previous==v.digest(before),'disconnected public trace')
                        actions,dist=actor.act(env.state,base_seed=c['action_seed'],deterministic=c['sampling_mode']=='argmax')
                        p=masked_probabilities(actor.logits(obs,m),m);sampled=[]
                        for i,role in enumerate(AGENTS):
                            seed=independent_agent_seed(base_seed=c['action_seed'],episode_id=int(env.state.episode_id),frame=int(env.state.frame),agent_id=role)
                            a=int(p[i].argmax()) if c['sampling_mode']=='argmax' else int(np.random.default_rng(seed).choice(5,p=p[i]))
                            v.require(actions[role]==ACTIONS[a] and m[i,a]==1 and np.array_equal(np.array(dist[role].probabilities,np.float32),p[i]),'action differs from canonical frozen NN')
                            sampled.append(a);checks+=1
                        frame=int(env.state.frame);_,_,_,_,info=env.step(actions)
                        after=public_pre_action_snapshot(env.state);previous=v.digest(after)
                        v.require(info['proposed_actions']==actions and not info['action_override_agents']
                            and {r['agent_id']:r['last_action'] for r in after['agents']}==actions,'post-NN action override')
                        h.write(json.dumps(dict(case=c,frame=frame,pre_state_public=before,post_state_public=after,
                            proposed_actions=actions,actor_sha256=ref['sha256']),separators=(',',':'))+'\n')
                        for k,value in (('x',encode_actor_tree_features(obs,m)),('m',m),('p',p),('sampled',sampled)):data[k].append(value)
                path=cap['folder']/(c['case_id']+'.npz')
                with path.open('xb') as h:np.savez_compressed(h,**{k:np.asarray(a,dtype=np.int64 if k=='sampled' else np.float32) for k,a in data.items()});h.flush();os.fsync(h.fileno())
                path.chmod(0o444);trace.chmod(0o444)
                refs[c['case_id']]=dict(arrays=v.pointer(path),trace=v.pointer(trace))
                outcomes.append(dict(case=c,deliveries=int(env.state.total_deliveries),robot_collisions=int(env.state.robot_collision_events),
                    shutdown=bool(env.state.shutdown_count),physical_frames=len(data['x'])))
                if cap['cursor']%25==0:print(json.dumps(dict(collected=cap['cursor'],total=1650)),flush=True)
        v.require(len(actual)==1650 and [r['seed'] for r in actual]==[c['scenario_seed'] for c in cap['plan']['cases']],'reset journal mismatch')
        result=dict(schema='warehouse-v628-collection.v1',plan_sha256=v.file_sha(v.OUTPUT/'plan.json'),
            actor=ref,cases=refs,outcomes=outcomes,NN_action_checks=checks,actual_resets=v.pointer(cap['folder']/'actual-resets.jsonl'),
            preauthorization=v.pointer(journal),actual_reset_count=1650,post_policy_interventions=0,actor_updates=0,PPO_joint_steps=0,pilot_opened=False)
        write_new(cap['folder']/'result.json',result)
    write_new(v.OUTPUT/'collection/completed.json',dict(result_sha256=v.file_sha(v.OUTPUT/'collection/result.json')))
    return result


def load_parts(plan,fold,role,*,inner=False):
    receipt=v.bound(v.pointer(v.OUTPUT/'collection/result.json'))
    v.require(json.loads((v.OUTPUT/'collection/completed.json').read_bytes())==dict(result_sha256=v.file_sha(v.OUTPUT/'collection/result.json')),'incomplete common collection')
    v.require(receipt['plan_sha256']==v.file_sha(v.OUTPUT/'plan.json') and receipt['actual_reset_count']==1650,'wrong collection binding')
    rindex=list(v.old.ROLES).index(role);groups=plan['folds'][fold]['inner' if inner else 'outer'];byid={r['case_id']:r for r in plan['cases']};parts={}
    for split,ids in groups.items():
        arrays={k:[] for k in ('x','m','p','sampled')};rows=[]
        for case_id in ids:
            ref=receipt['cases'][case_id]['arrays'];v.require(v.pointer(ref['path'])==ref,'common target arrays changed')
            with np.load(ref['path'],allow_pickle=False) as z:
                for k in arrays:arrays[k].append(z[k][:,rindex].copy())
                count=len(z['x'])
            c=byid[case_id];rows.extend(dict(episode=case_id,base_seed=str(c['base']),task=c['task']) for _ in range(count))
        parts[split]={k:np.concatenate(a) for k,a in arrays.items()};parts[split]['rows']=rows
    return parts
