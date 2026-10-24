"""Same persistent HTTPS route: paired warm health/action latency, manual seeds only."""
import argparse
from datetime import datetime, timezone
import http.client
import gzip
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',required=True);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--per-model',type=int,default=5)
    args=parser.parse_args(); url=urlsplit(args.base)
    cls=http.client.HTTPSConnection if url.scheme=='https' else http.client.HTTPConnection
    connection=cls(url.netloc,timeout=90)
    def call(path,payload=None,token=None):
        headers={'Accept':'application/json','Accept-Encoding':'gzip'}
        if payload is not None:headers['Content-Type']='application/json'
        if token:headers['Authorization']='Bearer '+token
        began=time.perf_counter()
        connection.request('POST' if payload is not None else 'GET',path,
            None if payload is None else json.dumps(payload),headers)
        response=connection.getresponse();raw=response.read();elapsed=(time.perf_counter()-began)*1000
        if response.getheader('Content-Encoding')=='gzip':raw=gzip.decompress(raw)
        assert response.status==200,(path,response.status,raw[:200])
        return json.loads(raw),elapsed,response.getheader('Server-Timing','')
    wake=call('/health')[1];meta=call('/api/release')[0];samples=[]
    try:
        for domain,seed in [('warehouse',40786),('pong',880101),('kitchen',1000)]:
            for variant in ('raw','regularized'):
                state=call('/manual-api/start',dict(domain=domain,variant=variant,task=1,seed=seed))[0]
                token=state['session_token']
                call('/manual-api/action',dict(command_id='http-latency-warmup',expected_turn=0,human_action=None),token)
                for index in range(args.per_model):
                    health=call('/health')[1]
                    result,action,timing=call('/manual-api/action',dict(command_id=f'http-latency-{index}',expected_turn=index+1,human_action=None if index%2 else 'wait'),token)
                    assert result['state']['turn']==index+2 and result['last_action']['ai_action_overrides']==0
                    samples.append(dict(domain=domain,variant=variant,health_ms=health,action_ms=action,server_timing=timing))
                print(domain,variant,'done',flush=True)
    finally:connection.close()
    def p95(key):return sorted(s[key] for s in samples)[min(len(samples)-1,int(len(samples)*.95))]
    report=dict(mode='manual_preview_not_scientific_evaluation',checked_at=datetime.now(timezone.utc).isoformat(),
        deployed_commit=meta['commit'],runtime=meta.get('preview_runtime_version','v1'),
        initial_connection_and_wake_ms=wake,sample_count=len(samples),health_p95_ms=p95('health_ms'),
        action_p95_ms=p95('action_ms'),route_gate_passed=p95('action_ms')<=p95('health_ms')+150,samples=samples)
    assert not args.output.exists(),'refusing to overwrite performance evidence'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}),flush=True)

if __name__=='__main__':main()
