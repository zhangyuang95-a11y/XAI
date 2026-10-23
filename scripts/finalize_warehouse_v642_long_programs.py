#!/usr/bin/env python3
"""Once-only fixed-language final program fit for a completed long development pair."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_long_training_final_programs as f


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completed-assets',required=True,type=Path)
    p.add_argument('--reference-directory',required=True,type=Path)
    a=p.parse_args(argv)
    result=f.run(completed_assets_ref=f.pointer(a.completed_assets),reference_directory=a.reference_directory)
    print(json.dumps(result,allow_nan=False),flush=True)
    return result


if __name__=='__main__':main()
