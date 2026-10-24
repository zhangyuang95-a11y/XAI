"""Warm manual-preview benchmark; never consumes scientific evaluation seeds."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    if args.baseline:
        spec = importlib.util.spec_from_file_location('ui._baseline_latency', args.baseline)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        module.ROOT = ROOT
        module.PACKAGE = ROOT / 'releases' / module.RELEASE
        module.WEB = ROOT / 'ui' / 'model_preview'
    else:
        from ui import model_preview_server as module
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Role
    from domains.pong import nn_policy
    from domains.kitchen.neural_policy import FrozenActors
    from runtime_v642.contracts import digest
    counters = Counter()
    read_bytes = Path.read_bytes
    def read(path):
        raw = read_bytes(path)
        if path.is_relative_to(module.PACKAGE):
            counters['reads'] += 1; counters['bytes'] += len(raw)
            counters['file:' + str(path.relative_to(module.PACKAGE))] += 1
        return raw
    def counted(original):
        def call(*a, **kw):
            counters['role_forwards'] += 1
            return original(*a, **kw)
        return call
    preview = module.Preview(':memory:')
    rows = []
    try:
        with patch.object(Path, 'read_bytes', read), \
             patch.object(FrozenWarehouseV6Role, 'logits_encoded', counted(FrozenWarehouseV6Role.logits_encoded)), \
             patch.object(nn_policy, 'logits_from_payload', counted(nn_policy.logits_from_payload)), \
             patch.object(FrozenActors, 'choose', counted(FrozenActors.choose)):
            for domain in ('warehouse', 'pong', 'kitchen'):
                for variant in ('raw', 'regularized'):
                    for task in (1, 2, 3):
                        start = preview.start(dict(domain=domain, variant=variant, task=task, seed=module.SEEDS[domain][0]))
                        token = start.pop('session_token'); samples = []; hashes = []
                        for i in range(args.steps + 1):
                            counters.clear()
                            payload = dict(command_id=f'warm-benchmark-{i}', expected_turn=i, human_action=(None if i % 3 == 0 else 'wait'))
                            began = time.perf_counter(); result = preview.step(token, payload)
                            elapsed = (time.perf_counter() - began) * 1000
                            hashes.append(digest(result))
                            if i:  # first action warms allocator/code paths
                                samples.append(dict(ms=elapsed, reads=counters['reads'], bytes=counters['bytes'],
                                    role_forwards=counters['role_forwards'], max_reads_per_file=max(
                                        (v for k,v in counters.items() if k.startswith('file:')), default=0)))
                        ordered = sorted(s['ms'] for s in samples)
                        row = dict(domain=domain,variant=variant,task=task,samples=samples,
                            p50_ms=ordered[len(ordered)//2],p95_ms=ordered[min(len(ordered)-1,int(.95*len(ordered)))],
                            exact_frame_digest=digest(hashes), binding_sha256=digest(result['binding']))
                        rows.append(row)
                        print(domain, variant, task, round(row['p50_ms'],2), round(row['p95_ms'],2), flush=True)
        report = dict(mode='manual_preview_performance_not_scientific_evaluation', steps_per_case=args.steps,
            baseline_source_sha256=sha256(Path(args.baseline).read_bytes()).hexdigest() if args.baseline else None,
            implementation='baseline' if args.baseline else 'request_scoped_v2', rows=rows)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    finally:
        preview.close()

if __name__ == '__main__':
    main()
