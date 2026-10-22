#!/usr/bin/env python3
"""Read-only A/B and reference-buffer verifier; never a training command."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse.v642_training_admission import load_readiness


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-directory',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(load_readiness(args.reference_directory).value,ensure_ascii=False,allow_nan=False))
