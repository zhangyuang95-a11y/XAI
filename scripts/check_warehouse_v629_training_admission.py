#!/usr/bin/env python3
"""Read-only prerequisite check; cannot start or authorize a training run."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse.v629_training_admission import load_readiness

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference-dir',type=Path,required=True);args=parser.parse_args()
    try:
        result=load_readiness(args.reference_dir)
        print(json.dumps(dict(status='prerequisites_passed',readiness_sha256=result.payload_sha256,**result.value)),flush=True)
    except Exception as error:
        print(json.dumps(dict(status='closed',reason=str(error),environment_authority=False,optimizer_authority=False,pilot_opened=False)),flush=True)
        sys.exit(2)
