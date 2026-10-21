#!/usr/bin/env python3
"""Conditional one-attempt fixed-program confirmation, never a training run."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v630_confirmation as c
from scripts import audit_warehouse_v630_confirmation as audit
from scripts.run_warehouse_v625 import write_new


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for stage in ('allocate','prepare','audit-seeds','collect','audit-collection','score','finalize','pipeline'):
        group.add_argument('--'+stage,action='store_true')
    parser.add_argument('--run-id');parser.add_argument('--base-start',type=int);parser.add_argument('--action-start',type=int)
    parser.add_argument('--allocation',type=Path);parser.add_argument('--plan',type=Path)
    args=parser.parse_args(argv)
    if args.allocate:
        if args.run_id is None or args.base_start is None or args.action_start is None:
            parser.error('allocation requires run ID and explicit history-audited candidate seed starts')
        result=dict(allocation=str(c.allocate(args.run_id,args.base_start,args.action_start)),collection_authorized=False)
    elif args.prepare:
        if args.allocation is None:parser.error('prepare requires --allocation')
        result=dict(plan=str(c.prepare(args.allocation)),collection_authorized=False)
    else:
        if args.plan is None:parser.error('stage requires --plan')
        stages={'audit_seeds':c.audit_seeds,'collect':c.collect,'audit_collection':audit.audit_collection,
            'score':audit.score,'finalize':audit.finalize}
        if args.pipeline:
            # A failed/partial attempt is retained; there is no automatic resume.
            c.v.require(not c.claim_path().exists() and not any((c.v.OUTPUT/name).exists()
                for name in ('confirmation.passed.json','confirmation.failed.json')),
                'confirmation pipeline already consumed or terminal; repeated invocation cannot change verdict')
            try:
                for key in ('audit_seeds','collect','audit_collection','score','finalize'):
                    result=stages[key](args.plan)
            except BaseException as error:
                claim=c.v.bound(c.v.pointer(c.claim_path())) if c.claim_path().exists() else None
                if claim is not None and claim.get('owner_pid')==os.getpid() and claim.get('plan')==c.v.pointer(args.plan) \
                        and not any((c.v.OUTPUT/name).exists() for name in ('confirmation.passed.json','confirmation.failed.json')):
                    write_new(c.v.OUTPUT/'confirmation.failed.json',dict(
                        confirmation_passed=False,plan=c.v.pointer(args.plan),claim=c.v.pointer(c.claim_path()),
                        error_type=type(error).__name__,error=str(error),automatic_retry=False,
                        candidate_closed=True,pilot_opened=False))
                raise
        else:
            key=next(key for key in stages if getattr(args,key))
            result=stages[key](args.plan)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return result


if __name__=='__main__':main()
