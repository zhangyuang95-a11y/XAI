#!/usr/bin/env python3
"""Read-only shared initial reference states; grants no training authority."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.training.warehouse_v629_reference_buffer import prepare,audit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    value=audit(args.output_dir) if args.audit else prepare(args.output_dir)
    print(json.dumps({k:value[k] for k in ('complete_episodes','joint_frames','NN_action_checks','training_authority')},sort_keys=True),flush=True)
