"""Merge two verified training collections without deleting or relabelling rows."""
from copy import deepcopy
import numpy as np

from . import warehouse_native_shutdown_rcpd as extraction
from .warehouse_native_common import digest

VERSION = 'warehouse-native-expanded-training-data.v1'


def merge_training(left, right, *, allow_test_fixture=False):
    for data in (left,right):
        extraction._validate_data(data,allow_test_fixture)
        if data['pool']!='train' or 'episode' in data:
            raise ValueError('Two already verified aggregate training collections are required')
        if (len(data.get('episodes',[]))!=len(data['collector_receipt'].get('episode_receipts',[]))
                or not data.get('episodes')):raise ValueError('Full individual episode receipts are required')
    fields=('version','pool','actor_bindings','feature_names','teacher_rows_in_fit',
        'neural_submitted_overrides','test_fixture','extraction_config_sha256','actor_training_clock')
    if any(digest(left[k])!=digest(right[k]) for k in fields):raise ValueError('Training collections belong to different Actors or contracts')
    a,b=left['collector_receipt'],right['collector_receipt']
    if any(digest(a[k])!=digest(b[k]) for k in ('actor_metadata','weights_sha256','configuration','actor_training_clock')):
        raise ValueError('Training collection source or physics differs')
    if (set(left['episode_ids']) & set(right['episode_ids'])
            or set(left['scene_fingerprints']) & set(right['scene_fingerprints'])):
        raise ValueError('Additional training must use distinct episodes and initial states')
    merged={k:deepcopy(left[k]) for k in fields}
    for k in ('observations','probabilities'):merged[k]=np.concatenate((left[k],right[k]))
    for k in ('episode_ids','groups','row_sources','episodes'):merged[k]=deepcopy(left[k]+right[k])
    merged['scene_fingerprints']=sorted(set(left['scene_fingerprints'])|set(right['scene_fingerprints']))
    merged['joint_transitions']=left['joint_transitions']+right['joint_transitions']
    merged['data_sha256']=extraction.data_api._data_digest(merged)
    contract={k:a[k] for k in ('weights_sha256','configuration','actor_training_clock')}
    contract['metadata']=a['actor_metadata']
    extraction._receipt(merged,contract=contract,sources=extraction.execution_sources(),
        episodes=deepcopy(a['episode_receipts']+b['episode_receipts']))
    extraction._validate_data(merged,allow_test_fixture)
    receipt={'version':VERSION,'source_data_sha256':[left['data_sha256'],right['data_sha256']],
        'source_collector_receipts':[left['collector_receipt_sha256'],right['collector_receipt_sha256']],
        'merged_data_sha256':merged['data_sha256'],'merged_collector_receipt_sha256':merged['collector_receipt_sha256'],
        'rows':len(merged['observations']),'scenarios':len(merged['scene_fingerprints']),
        'episodes':len(merged['episodes']),'rows_removed':0,'new_environment_steps':0,'neural_forwards':0,
        'qualification_granted':False,'test_fixture':allow_test_fixture}
    return merged,receipt
