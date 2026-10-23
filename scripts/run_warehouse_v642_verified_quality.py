#!/usr/bin/env python3
"""Explicit entry points for the separately source-bound cached quality profile."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_quality_verified_cache as c

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','audit-seeds','collect','audit'))
    p.add_argument('--plan',type=Path);p.add_argument('--completed-assets',type=Path)
    p.add_argument('--final-programs',type=Path);p.add_argument('--run-id')
    p.add_argument('--base-start',type=int);p.add_argument('--action-start',type=int)
    p.add_argument('--quality-result',type=Path);p.add_argument('--supervisor-exit',type=Path)
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.stage=='prepare':
        if any(v is None for v in (a.completed_assets,a.final_programs,a.run_id,a.base_start,a.action_start)):p.error('missing prepare arguments')
        result=c.prepare(c.pointer(a.completed_assets),final_programs_ref=c.pointer(a.final_programs),run_id=a.run_id,base_start=a.base_start,action_start=a.action_start)
    elif a.stage=='audit':
        if any(v is None for v in (a.quality_result,a.supervisor_exit,a.output)):p.error('missing audit arguments')
        value=c.audit(c.pointer(a.quality_result),c.pointer(a.supervisor_exit));result=c._save(a.output,value)
    else:
        if a.plan is None:p.error('missing plan')
        result=(c.audit_seeds if a.stage=='audit-seeds' else c.collect)(c.pointer(a.plan))
    print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':main()
