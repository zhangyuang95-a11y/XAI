"""Actual HTTP play checked against local frozen NN; never a heldout evaluation."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ui.model_preview_server import Preview, RELEASE


def compare_replay(remote, local):
    """Physics, identities and chosen actions exact; only NN float reductions tolerate CPU variation."""
    errors=[]
    def compare(a,b,path=()):
        if isinstance(a,dict) and isinstance(b,dict):
            assert set(a)==set(b), path
            for k in a: compare(a[k],b[k],path+(k,))
        elif isinstance(a,list) and isinstance(b,list):
            assert len(a)==len(b), path
            for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+(i,))
        elif a!=b:
            numeric_path=(len(path)>=6 and path[:2]==('replay','records') and path[3]=='decision'
                          and path[4] in ('distributions','scores'))
            assert numeric_path and type(a) in (float,int) and type(b) in (float,int), path
            probability=('probabilities' in path)
            tolerance=2e-6 if probability else 1e-4
            assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance, (path,a,b)
            errors.append(dict(path='/'.join(map(str,path)),absolute_error=abs(a-b),allowed_absolute_error=tolerance))
    compare(remote,local)
    return dict(exact_actions_bindings_and_physics=True,NN_float_differences=len(errors),
        maximum_absolute_error=max((e['absolute_error'] for e in errors),default=0),differences=errors)


def verify(base, output):
    base=base.rstrip('/')
    requests=[]

    def call(path, payload=None, token=None):
        headers={}
        if payload is not None: headers['Content-Type']='application/json'
        if token: headers['Authorization']='Bearer '+token
        request=Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers)
        with urlopen(request,timeout=90) as response:
            raw=response.read()
            requests.append(dict(path=path,status=response.status,bytes=len(raw),sha256=sha256(raw).hexdigest()))
            return json.loads(raw)

    meta=call('/api/release')
    assert meta['release_id']==RELEASE and meta['pilot_opened'] is meta['study_ready'] is False
    local=Preview(':memory:')
    cases=[]
    try:
        for domain,seed in (('warehouse',40786),('pong',880101),('kitchen',1000)):
            for variant in ('raw','regularized'):
                choice=dict(domain=domain,variant=variant,task=1,seed=seed)
                remote=call('/manual-api/start',choice)
                expected=local.start(choice)
                token,local_token=remote.pop('session_token'),expected.pop('session_token')
                assert remote==expected
                for index,action in enumerate((None,'wait','left')):
                    payload=dict(command_id='production-smoke-'+str(index),expected_turn=index,human_action=action)
                    remote=call('/manual-api/action',payload,token)
                    expected=local.step(local_token,payload)
                    assert remote==expected, (domain,variant,index)
                    assert remote['last_action']['ai_action_overrides']==0
                    assert call('/manual-api/action',payload,token)==remote
                    assert call('/manual-api/session',token=token)['state']==remote['state']
                replay=call('/manual-api/export',token=token)
                expected_replay=local.export(local_token)
                comparison=compare_replay(replay,expected_replay)
                cases.append(dict(domain=domain,variant=variant,seed=seed,task=1,joint_steps=3,
                    ai_action_overrides=0,actor_sha256=remote['binding']['actor_sha256'],
                    binding_identity_sha256=remote['state']['binding_identity_sha256'],
                    independent_replay_passed=True,local_frozen_NN_physics_match=True,cross_platform_numerics=comparison))
        for path in ('/','/warehouse/','/pong/','/kitchen/'):
            with urlopen(base+path,timeout=60) as response:
                assert response.status==200 and '模型试玩' in response.read().decode()
        for path in ('/prolific/','/api/start'):
            try: urlopen(base+path,timeout=60)
            except HTTPError as error: assert error.code==404
            else: raise AssertionError('old participant entry is exposed')
    finally:
        local.close()
    result=dict(format='manual-model-preview-live-HTTP-verification.v1',base_url=base,
        checked_at=datetime.now(timezone.utc).isoformat(),passed=True,release_id=RELEASE,
        deployed_commit=meta['commit'],pilot_opened=False,scientific_model_acceptance=False,
        heldout_evaluated=False,human_study_sample=False,cases=cases,requests=requests)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise ValueError('refusing to replace prior deployment verification')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(cases),requests=len(requests),deployed_commit=meta['commit'],receipt=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    verify(args.base,args.output)
