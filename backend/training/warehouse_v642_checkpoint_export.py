"""Read-only frozen Actor export from exact checkpoint tensors, no RNG.

No training model, critic, optimizer or environment is constructed. The
frozen reader validates the resulting complete inference tensor contract.
"""
from copy import deepcopy
from pathlib import Path
import io,json
import numpy as np
import torch
from .warehouse_v6_actor import MODEL_VERSION,_tensor_names,role_sha256,FrozenWarehouseV6Actor
from .warehouse_v622_actor_training import CHECKPOINT_SCHEMA
from experiments.warehouse.v626_protocol import require,pointer,file_sha

VERSION='warehouse-v642-checkpoint-tensors-frozen-export.v1'
ROLES=('robot_1','robot_2')


def export_checkpoint_actor(checkpoint_ref,output,*,variant,candidate_id,expected_joint_steps):
    require(variant in ('raw','regularized') and isinstance(candidate_id,str) and candidate_id and
        type(expected_joint_steps) is int and expected_joint_steps>=0,'explicit checkpoint/variant identity required')
    require(pointer(checkpoint_ref['path'])==checkpoint_ref,'checkpoint hash differs')
    value=torch.load(io.BytesIO(Path(checkpoint_ref['path']).read_bytes()),map_location='cpu',weights_only=True)
    require(value['schema']==CHECKPOINT_SCHEMA and value['model_version']==MODEL_VERSION and value['joint_steps']==expected_joint_steps,
        'checkpoint schema, architecture or joint-step boundary differs')
    require(file_sha(value['reference_path'])==value['reference_sha256'],'original complete Actor changed')
    anchor=FrozenWarehouseV6Actor.load(value['reference_path'],expected_sha256=value['reference_sha256'],expected_role_sha256=value['reference_role_sha256'])
    with np.load(value['reference_path'],allow_pickle=False) as a:metadata=json.loads(str(a['metadata_json'].item()))
    weights={}
    for role in ROLES:
        prefix='actors.'+role+'.';state=value['model']
        fixed=state[prefix+'tree.support_fixed'].detach().cpu().numpy()
        mask=state[prefix+'tree.support_mask'].detach().cpu().numpy()
        require(fixed.shape==(1,) and fixed[0]==1 and np.isin(mask,(0,1)).all(),
            'checkpoint must retain the imported fixed support contract')
        weights[role]={name:state[prefix+name].detach().cpu().numpy().astype(np.float32,copy=True) for name in _tensor_names(MODEL_VERSION)}
        w=weights[role]['tree.split.weight'];require(mask.shape==w.shape,'fixed support shape differs')
        # v6.18's exact hard-forward grouping is w*mask+(s-s.detach()).
        # Adding positive zero preserves its signed-zero hash semantics.
        weights[role]['tree.split.weight']=np.add(np.multiply(w,mask.astype(np.float32),dtype=np.float32),np.float32(0.),dtype=np.float32)
    hashes={r:role_sha256(w,model_version=MODEL_VERSION) for r,w in weights.items()}
    metadata=deepcopy(metadata);metadata.update(pair_seed=value['initialization_seed'],variant=variant,candidate_id=candidate_id,
        role_sha256=hashes,training_recipe='warehouse-v642-paired-preserving',reference_sha256=value['reference_sha256'],
        checkpoint=checkpoint_ref,joint_steps=expected_joint_steps,frozen_export_method=VERSION,publishable=False)
    buffer=io.BytesIO();np.savez_compressed(buffer,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)),
        **{f'{r}.{k}':a for r,w in weights.items() for k,a in w.items()})
    require(pointer(checkpoint_ref['path'])==checkpoint_ref and file_sha(value['reference_path'])==value['reference_sha256'],
        'checkpoint/reference changed during read-only export')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as h:h.write(buffer.getvalue())
    output.chmod(0o444);ref=pointer(output)
    actor=FrozenWarehouseV6Actor.load(output,expected_sha256=ref['sha256'],expected_role_sha256=hashes)
    return dict(version=VERSION,actor=ref,role_sha256=hashes,checkpoint=checkpoint_ref,
        reference_actor=dict(path=value['reference_path'],sha256=value['reference_sha256']),joint_steps=expected_joint_steps,
        actor_updates=value['actor_updates'],critic_updates=value['critic_updates'],optimizer_steps=0,
        environment_resets=0,new_RNG_constructors=0,publishable=False)
