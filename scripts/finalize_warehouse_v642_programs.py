#!/usr/bin/env python3
"""Once-only final program fitting after the real completed20k Actor pair."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_final_programs as f


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completed-assets',required=True,type=Path)
    parser.add_argument('--reference-directory',required=True,type=Path)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args(argv)
    result=f.run(completed_assets_ref=f.pointer(args.completed_assets),
        reference_directory=args.reference_directory,run_id=args.run_id)
    print(json.dumps(result,allow_nan=False),flush=True);return result


if __name__=='__main__':main()
