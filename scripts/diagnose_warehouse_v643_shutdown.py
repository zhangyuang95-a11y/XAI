"""Reproduce spent-development diagnosis; never run training or evaluation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.warehouse.v643_shutdown_diagnostic import diagnose,pointer,VERSION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-result',type=Path,required=True)
    parser.add_argument('--training-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=diagnose(pointer(args.quality_result),pointer(args.training_result))
    result['producer_source_refs']=[pointer(ROOT/name) for name in (
        'experiments/warehouse/v643_shutdown_diagnostic.py','scripts/diagnose_warehouse_v643_shutdown.py')]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,sort_keys=True,indent=2);handle.write('\n')
    args.output.chmod(0o444)
    print(json.dumps({'version':VERSION,'output':pointer(args.output),'quality':result['quality'],
        'training_cost_signal':result['training_cost_signal'],'training_authority':False},ensure_ascii=False))


if __name__=='__main__':main()
