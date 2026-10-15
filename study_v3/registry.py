"""Domain registry: no controller receives experimental group information."""
from functools import lru_cache
from importlib import import_module
from .domain_contracts import LAYERS, validate_adapter

MODULES = {domain: layers.participant_adapter for domain, layers in LAYERS.items()}

@lru_cache(maxsize=3)
def engine(domain):
    if domain not in MODULES:
        raise ValueError('unknown_domain')
    module = import_module(MODULES[domain])
    validate_adapter(domain, module)
    return module


def engine_for_recorded_state(state):
    """Read historical Kitchen evidence with the controller that recorded it."""
    current = engine(state['domain'])
    if (state['domain'] == 'kitchen'
            and state.get('version') not in (current.VERSION, getattr(current, 'RELEASE_VERSION', current.VERSION))):
        return import_module('domains.kitchen.engine')
    return current

def demonstration(domain, policy_id):
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError('policy_id_required')
    # A cached rollout must never outlive its accepted Actor/program binding.
    # Check admission on every call, then cache only by the verified assets.
    from .policies import (_accepted, _catalog_entry, binding_for_group,
                           PolicyUnavailable)
    variant = 'raw' if policy_id == _catalog_entry(domain, 'raw')['policy_id'] else 'regularized'
    if policy_id != _catalog_entry(domain, variant)['policy_id']:
        raise ValueError('invalid_policy_id')
    binding = binding_for_group(domain, 'C' if variant == 'raw' else 'A')
    if not _accepted(_catalog_entry(domain, variant), binding):
        raise PolicyUnavailable('policy_not_accepted')
    import json
    from hashlib import sha256
    bound_hash = sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return _accepted_demonstration(domain, policy_id, bound_hash)


@lru_cache(maxsize=9)
def _accepted_demonstration(domain, policy_id, bound_hash):
    from . import RELEASE_ID
    return engine(domain).demonstration(policy_id, release_id=RELEASE_ID)

@lru_cache(maxsize=3)
def scenario_config(domain):
    import json
    from pathlib import Path
    if domain not in MODULES: raise ValueError('unknown_domain')
    if domain == 'kitchen':
        from .release_catalog import runtime_config
        return json.loads(runtime_config('kitchen').read_text())
    names = {'warehouse': 'study_v3_warehouse_neural_v6.json',
             'pong': 'study_v3_pong.json'}
    name = names[domain]
    return json.loads((Path(__file__).resolve().parents[1]/'configs'/name).read_text())
