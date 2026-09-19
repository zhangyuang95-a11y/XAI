"""Domain registry: no controller receives experimental group information."""
from functools import lru_cache
from importlib import import_module

MODULES = {
    'warehouse': 'domains.warehouse.turnbased',
    'pong': 'domains.pong.turnbased',
    'kitchen': 'domains.kitchen.engine',
}

@lru_cache(maxsize=3)
def engine(domain):
    if domain not in MODULES:
        raise ValueError('unknown_domain')
    return import_module(MODULES[domain])

@lru_cache(maxsize=3)
def demonstration(domain):
    return engine(domain).demonstration()

@lru_cache(maxsize=3)
def scenario_config(domain):
    import json
    from pathlib import Path
    if domain not in MODULES: raise ValueError('unknown_domain')
    return json.loads((Path(__file__).resolve().parents[1]/'configs'/('study_v3_'+domain+'.json')).read_text())
