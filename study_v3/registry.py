"""Domain registry: no controller receives experimental group information."""
from functools import lru_cache
from importlib import import_module
import re
from .domain_contracts import LAYERS, validate_adapter

MODULES = {domain: layers.participant_adapter for domain, layers in LAYERS.items()}
_WAREHOUSE_V6_POLICY = re.compile(
    r"warehouse-(raw|regularized)-v6(?:\.([1-9][0-9]*))?\Z")


def _demonstration_release(domain: str, policy_id: str) -> str:
    """Recover the canonical Warehouse candidate release from its policy ID."""
    from . import RELEASE_ID
    if domain != "warehouse":
        return RELEASE_ID
    matched = _WAREHOUSE_V6_POLICY.fullmatch(policy_id)
    if matched is None:
        return RELEASE_ID
    from .release_catalog import warehouse_release_spec
    variant, minor = matched.groups()
    prefix = RELEASE_ID.rsplit(".v6.", 1)[0]
    release_id = f"{prefix}.v6.{minor or '0'}"
    if warehouse_release_spec(release_id).policy_ids[variant] != policy_id:
        raise ValueError("invalid_policy_id")
    return release_id

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
    from . import RELEASE_ID
    release_id = _demonstration_release(domain, policy_id)
    def entry_for(variant):
        return (_catalog_entry(domain, variant) if release_id == RELEASE_ID else
                _catalog_entry(domain, variant, release_id=release_id))

    variant = 'raw' if policy_id == entry_for('raw')['policy_id'] else 'regularized'
    entry = entry_for(variant)
    if policy_id != entry['policy_id']:
        raise ValueError('invalid_policy_id')
    group = 'C' if variant == 'raw' else 'A'
    binding = (binding_for_group(domain, group) if release_id == RELEASE_ID else
               binding_for_group(domain, group, release_id=release_id))
    accepted = (_accepted(entry, binding) if release_id == RELEASE_ID else
                _accepted(entry, binding, release_id=release_id))
    if not accepted:
        raise PolicyUnavailable('policy_not_accepted')
    import json
    from hashlib import sha256
    bound_hash = sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return _accepted_demonstration(domain, policy_id, bound_hash, release_id)


@lru_cache(maxsize=18)
def _accepted_demonstration(domain, policy_id, bound_hash, release_id):
    return engine(domain).demonstration(policy_id, release_id=release_id)

@lru_cache(maxsize=3)
def scenario_config(domain):
    import json
    from pathlib import Path
    if domain not in MODULES: raise ValueError('unknown_domain')
    if domain in ('pong', 'kitchen'):
        from .release_catalog import runtime_config
        return json.loads(runtime_config(domain).read_text())
    names = {'warehouse': 'study_v3_warehouse_neural_v6.json'}
    name = names[domain]
    return json.loads((Path(__file__).resolve().parents[1]/'configs'/name).read_text())
