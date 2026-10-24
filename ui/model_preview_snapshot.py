"""Data-only snapshot codec copied from the frozen v6.29 reader.

Only the four named snapshot functions are carried; training imports are absent.
Equivalence to the original functions and numerical transitions is tested.
"""
from dataclasses import fields, is_dataclass
import math
import random
from env.warehouse.domain import WarehouseConfig, WarehouseState, AgentState, DeliveryTask
from env.warehouse.rewards import RewardConfig
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import get_map_layout
from env.warehouse.public_action_features_v617 import validate_study_action_physics
from runtime_v642.contracts import require, digest
SNAPSHOT_VERSION = 'warehouse-v629-full-environment-snapshot.v1'
CLASSES = {c.__name__:c for c in (WarehouseConfig, WarehouseState, AgentState, DeliveryTask, RewardConfig)}
ENV_FIELDS = {'config','layout','_rng','_episode_counter','state'}

def encode(value):
    """Explicit data-only types; no pickle, imports, callbacks or object hooks."""
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float:
        require(math.isfinite(value),'nonfinite private environment value');return value
    if type(value) in (list,tuple):return dict(kind=type(value).__name__,items=[encode(x) for x in value])
    if type(value) is dict:
        require(all(type(k) is str for k in value),'environment dictionaries need string keys')
        return dict(kind='dict',items=[[k,encode(value[k])] for k in sorted(value)])
    if is_dataclass(value) and type(value).__name__ in CLASSES and type(value) is CLASSES[type(value).__name__]:
        return dict(kind=type(value).__name__,fields={f.name:encode(getattr(value,f.name)) for f in fields(value)})
    raise ValueError('unsupported private environment type')

def decode(value):
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float:
        require(math.isfinite(value),'nonfinite snapshot value');return value
    require(type(value) is dict and type(value.get('kind')) is str,'unknown private snapshot type')
    kind=value['kind']
    if kind in ('list','tuple'):
        require(set(value)=={'kind','items'} and type(value['items']) is list,'malformed environment sequence')
        items=[decode(x) for x in value['items']];return items if kind=='list' else tuple(items)
    if kind=='dict':
        require(set(value)=={'kind','items'} and type(value['items']) is list,'malformed environment dictionary')
        pairs=value['items']
        require(all(type(p) is list and len(p)==2 and type(p[0]) is str for p in pairs)
            and [p[0] for p in pairs]==sorted({p[0] for p in pairs}),'duplicate or noncanonical environment dictionary')
        return {k:decode(v) for k,v in pairs}
    require(kind in CLASSES and set(value)=={'kind','fields'} and type(value['fields']) is dict
        and set(value['fields'])=={f.name for f in fields(CLASSES[kind])},'unknown or incomplete environment dataclass')
    return CLASSES[kind](**{k:decode(v) for k,v in value['fields'].items()})

def capture_environment(env):
    require(type(env) is WarehouseMultiAgentEnv and set(vars(env))==ENV_FIELDS
        and type(env._rng) is random.Random and type(env.state) is WarehouseState
        and type(env._episode_counter) is int and env._episode_counter==env.state.episode_id
        and env.layout==get_map_layout(env.config.map_layout_id),'unsupported or incomplete environment instance')
    validate_study_action_physics(env.config)
    require(not env.validate_state(env.state),'invalid private environment state')
    payload=dict(schema=SNAPSHOT_VERSION,config=encode(env.config),state=encode(env.state),
        rng=encode(env.get_rng_state()),episode_counter=env._episode_counter)
    return dict(payload=payload,payload_sha256=digest(payload))

def restore_environment(snapshot):
    require(type(snapshot) is dict and set(snapshot)=={'payload','payload_sha256'}
        and digest(snapshot['payload'])==snapshot['payload_sha256'],'environment snapshot integrity differs')
    p=snapshot['payload']
    require(set(p)=={'schema','config','state','rng','episode_counter'} and p['schema']==SNAPSHOT_VERSION,
        'unknown environment snapshot schema')
    config,state,rng=decode(p['config']),decode(p['state']),decode(p['rng'])
    require(type(config) is WarehouseConfig and type(state) is WarehouseState,'environment config/state type differs')
    # __new__ allocates empty objects without __init__/seed/reset. set_state
    # intentionally is not used: it would regenerate private goal bookkeeping.
    env=WarehouseMultiAgentEnv.__new__(WarehouseMultiAgentEnv)
    env.config=config;env.layout=get_map_layout(config.map_layout_id)
    env._rng=random.Random.__new__(random.Random);env._rng.setstate(rng)
    env._episode_counter=p['episode_counter'];env.state=state
    require(capture_environment(env)==snapshot,'environment snapshot does not roundtrip exactly')
    return env
