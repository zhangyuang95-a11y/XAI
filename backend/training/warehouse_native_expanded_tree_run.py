"""Single finite expanded-evidence RCPD search from an externally bound corpus.

All arrays originate in completed neural collection receipts. This reader does
no model loading or simulation. A completed reliable fit is evidence for a
separate, versioned feedback preparation, never a participant release.
"""
from pathlib import Path
import argparse
import gzip
import io
import json

import numpy as np

from . import warehouse_native_expanded_rcpd as extraction
from . import warehouse_native_extraction_expansion as expansion
from . import warehouse_native_expansion_data as merge
from .warehouse_native_common import ROOT,digest,file_hash

initial=expansion.initial
VERSION='warehouse-native-expanded-tree-run.v1'


def producer_sources():
    sources=expansion.sources();sources[str(Path(merge.__file__).relative_to(ROOT))]=file_hash(Path(merge.__file__))
    return sources


def sources():
    value=producer_sources();value.update(extraction.execution_sources(extraction.own))
    value[str(Path(__file__).relative_to(ROOT))]=file_hash(Path(__file__))
    return value


def read_corpus(root,expected_manifest_sha256):
    root=Path(root).expanduser().resolve()
    if file_hash(root/'manifest.json')!=expected_manifest_sha256:raise ValueError('External corpus byte anchor differs')
    manifest=initial.read_json(root/'manifest.json')
    if (manifest.get('version')!='warehouse-native-expanded-extraction-corpus.v1'
            or manifest.get('status')!='verified_merged_corpus' or manifest.get('test_fixture') is not False
            or manifest.get('base_manifest_sha256')!=expansion.BASE_SHA
            or manifest.get('runtime_sources')!=producer_sources()
            or manifest.get('feedback_ready') is not False or manifest.get('release_ready') is not False
            or manifest.get('explanation_qualified') is not False):raise ValueError('Verified production corpus contract differs')
    for name,sha in manifest['runtime_sources'].items():
        if file_hash(root/'source_snapshot'/name)!=sha:raise ValueError('Corpus source snapshot changed')
    base,extra=Path(manifest['base']),Path(manifest['expansion'])
    if file_hash(base/'manifest.json')!=expansion.BASE_SHA:raise ValueError('Original source collection changed')
    for name,sha in manifest['expansion_anchors'].items():
        if file_hash(extra/name)!=sha:raise ValueError('Additional source collection changed')
    base_plan=initial.read_json(base/'plan.json');extra_plan=initial.read_json(extra/'plan.json')
    if (manifest['actor_bindings']!=base_plan['actor_bindings'] or extra_plan['actor_bindings']!=base_plan['actor_bindings']
            or extra_plan['base_plan_sha256']!=digest(base_plan)
            or file_hash(base/'actor.npz')!=manifest['actor_bindings']['actor_sha256']
            or file_hash(extra/'actor.npz')!=manifest['actor_bindings']['actor_sha256']):raise ValueError('The original same-Actor source differs')
    datasets={}
    for pool in ('train','selection'):
        files=manifest['files'][pool]
        data=json.loads(gzip.decompress(initial.bound_bytes(root,files['records'])))
        with np.load(io.BytesIO(initial.bound_bytes(root,files['arrays'])),allow_pickle=False) as values:
            if set(values.files)!={'observations','probabilities'}:raise ValueError('Unexpected corpus arrays')
            data.update({k:values[k].copy() for k in values.files})
        extraction.own._validate_data(data,False)
        if (data['pool']!=pool or data['data_sha256']!=files['data_sha256']
                or data['collector_receipt_sha256']!=files['collector_receipt_sha256']
                or data['actor_bindings']!=manifest['actor_bindings']):raise ValueError('Corpus dataset identity differs')
        datasets[pool]=data
    train,selection=datasets['train'],datasets['selection'];receipt=manifest['merge_receipt']
    if (receipt.get('version')!=merge.VERSION or receipt.get('test_fixture') is not False
            or receipt.get('merged_data_sha256')!=train['data_sha256']
            or receipt.get('merged_collector_receipt_sha256')!=train['collector_receipt_sha256']
            or receipt.get('rows_removed')!=0 or receipt.get('rows')!=len(train['observations'])
            or receipt.get('episodes')!=768 or len(train['episodes'])!=768 or len(train['scene_fingerprints'])!=192
            or len(selection['episodes'])!=128 or len(selection['scene_fingerprints'])!=32
            or manifest['selection_data_sha256']!=selection['data_sha256']
            or manifest['selection_rows']!=len(selection['observations'])):raise ValueError('Fixed expanded coverage or original selection differs')
    if file_hash(root/'manifest.json')!=expected_manifest_sha256:raise ValueError('Corpus changed while reading')
    return manifest,train,selection


def run(corpus,expected_manifest_sha256,output):
    corpus,output=Path(corpus).expanduser().resolve(),Path(output).expanduser().resolve()
    if output.exists():raise FileExistsError('Never retry or overwrite a reserved fit; preserve earlier attempts')
    if corpus==output or corpus in output.parents or output in corpus.parents:raise ValueError('Independent output is required')
    manifest,train,selection=read_corpus(corpus,expected_manifest_sha256)
    code=sources();step=train['actor_training_clock']
    plan={'version':VERSION,'corpus':str(corpus),'corpus_manifest_sha256':expected_manifest_sha256,
        'actor_bindings':manifest['actor_bindings'],'training_data_sha256':train['data_sha256'],
        'selection_data_sha256':selection['data_sha256'],'cumulative_fit_step':step,
        'extraction_contract':extraction.contract(),'runtime_sources':code,
        'maximum_rcpd_calls':len(extraction.CANDIDATES),'maximum_sklearn_fits':sum(d for d,_ in extraction.CANDIDATES),
        'new_environment_steps':0,'new_ppo_steps':0,'automatic_fit_retry':False,'test_fixture':False,
        'explanation_qualified':False,'release_ready':False}
    output.mkdir(parents=True,exist_ok=False)
    initial.write_json(output/'plan.json',plan)
    for name in code:initial.write_bytes(output/'source_snapshot'/name,(ROOT/name).read_bytes())
    initial.write_json(output/'manifest.json',{'version':VERSION,'status':'fit_reserved','plan_sha256':digest(plan),
        'feedback_training_started':False,'explanation_qualified':False,'release_ready':False})
    try:
        result=extraction.fit(train,selection,step=step,feature_names=train['feature_names'])
        if sources()!=code or file_hash(corpus/'manifest.json')!=expected_manifest_sha256:raise ValueError('Source inputs changed during fitting')
        initial.write_json(output/'fit_result.json',result)
        initial.write_json(output/'program.json',result['program'])
        completed={'version':VERSION,'status':'completed','plan_sha256':digest(plan),'reliable':result['reliable'],
            'fit_result':initial.binding(output,output/'fit_result.json'),'program':initial.binding(output,output/'program.json'),
            'feedback_training_started':False,'explanation_qualified':False,'release_ready':False}
        initial.write_json(output/'manifest.json',completed,replace=True)
        return {'status':'expanded_tree_completed','reliable':result['reliable'],
            'selected':result['fit_report']['selected'],'manifest_sha256':file_hash(output/'manifest.json'),
            'feedback_training_started':False,'release_ready':False}
    except BaseException as error:
        initial.write_json(output/'failure.json',{'error':repr(error),'automatic_retry':False});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',required=True);parser.add_argument('--manifest-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(args.corpus,args.manifest_sha256,args.output)),flush=True)


if __name__=='__main__':main()
