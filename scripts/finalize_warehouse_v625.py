#!/usr/bin/env python3
"""Append-only complete or interrupted v6.25 evidence seal. Never publishes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.warehouse import v625_protocol as v


def stream_sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def finalize(plan_path=None, *, interruption=False):
    from scripts.run_warehouse_v625 import checked_plan, collect_results
    path, plan = checked_plan(plan_path)
    folder = path.parent
    destination = folder / 'finalizations' / ('engineering-interruption-01' if interruption else 'completed-development-01')
    v.require(not (folder/'finalizations').exists(), 'candidate already sealed; no repeated finalization')
    disposition = []
    if interruption:
        failed = json.loads((folder/'execution/failed.json').read_bytes())
        v.require(failed.get('all_owned_workers_waited') is True, 'wait all owned workers before seal')
        selected = None
        status = 'closed_engineering_interruption'
        for spec in plan['jobs']:
            jobdir = folder/'jobs'/spec['job_id']
            kind = ('completed_before_interruption' if (jobdir/'completed.json').exists() else
                    'engineering_failed' if (jobdir/'failed.json').exists() else
                    'terminated_without_completion' if (jobdir/'started.json').exists() else 'not_started')
            disposition.append(dict(job=spec,status=kind))
    else:
        results = collect_results(path, plan)
        summary = json.loads((folder/'summary.json').read_bytes())
        audit = json.loads((folder/'independent-audit.json').read_bytes())
        v.require(audit.get('passed') is True and audit.get('summary_sha256') == v.file_sha(folder/'summary.json'),
                  'full independent audit must pass before scientific sealing')
        from scripts.audit_warehouse_v625 import audit as re_audit
        current_audit=re_audit(path)
        for key in ('passed','source_sha256','plan_sha256','summary_sha256','jobs','evidence_sha256'):
            v.require(current_audit.get(key)==audit.get(key),'audited evidence changed before seal: '+key)
        selection = v.comparison(results)
        for arm in v.ARMS:
            for row in selection['arms'][arm]['role_folds']:
                row['result']=v.pointer(folder/'jobs'/row['job_id']/'result.json')
        v.require(all(summary.get(k)==value for k,value in selection.items()), 'summary selection changed')
        selected = selection['selected_arm']
        status = 'stage_A_passed' if selected else 'closed_failed_program_precheck'
        disposition = [dict(job=r['job'],status=r['status'],passed=r['gate']['passed'],
                            result=v.pointer(folder/'jobs'/r['job']['job_id']/'result.json')) for r in results]
    files = []
    for asset in sorted(folder.rglob('*')):
        if asset.is_dir(): continue
        v.require(not asset.is_symlink(), 'evidence symlinks refused')
        if asset.name == 'worker.lock': continue
        files.append(dict(path=str(asset.resolve()), size=asset.stat().st_size, sha256=stream_sha(asset)))
    manifest = dict(schema='warehouse-v625-evidence-manifest.v1',candidate_id=v.CANDIDATE,
                    source_sha256=plan['source_sha256'],plan=v.pointer(path),files=files,
                    file_count=len(files),total_bytes=sum(r['size'] for r in files))
    destination.mkdir(parents=True, exist_ok=False)
    write_new(destination/'evidence-manifest.json', manifest)
    receipt = dict(v.ZERO,schema='warehouse-v625-stage-A-seal.v1',candidate_id=v.CANDIDATE,
                   stage='A',status=status,candidate_closed=selected is None,selected_arm=selected,
                   complete_scientific_comparison=not interruption,
                   source_sha256=plan['source_sha256'],plan=v.pointer(path),
                   at_utc=datetime.now(timezone.utc).isoformat(),jobs=disposition,
                   evidence_manifest=v.pointer(destination/'evidence-manifest.json'),
                   training_authorized=False,fresh_data_authorized=False,
                   interruption_is_not_scientific_failure=interruption)
    write_new(destination/'receipt.json', receipt)
    # The compact tracked pointer preserves evidence identity without adding the
    # ignored experiment directory or model files to Git.
    tracked = ROOT/'experiments/warehouse/evidence/warehouse-v625-stage-a-01.json'
    write_new(tracked, dict(receipt=v.pointer(destination/'receipt.json'),
                           status=status, candidate_id=v.CANDIDATE, **v.ZERO))
    return dict(receipt=v.pointer(destination/'receipt.json'),status=status)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interruption',action='store_true')
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    print(json.dumps(finalize(args.plan,interruption=args.interruption),sort_keys=True))
