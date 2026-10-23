#!/usr/bin/env python3
"""Independently replay one long-development stream; retain real failed gates."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_long_training_quality as q


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True,type=Path);parser.add_argument('--supervisor-exit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args(argv)
    report=q.audit(q.pointer(args.result),q.pointer(args.supervisor_exit))
    q.write_new(args.output,report);args.output.chmod(0o444)
    result=q.pointer(args.output);print(json.dumps(result,allow_nan=False),flush=True);return result


if __name__=='__main__':main()
