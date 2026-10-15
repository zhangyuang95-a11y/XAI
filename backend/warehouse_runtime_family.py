"""Closed runtime-family boundary for future warehouse component consumers.

Only the two actual frozen runtime classes are supported. This factory neither
changes Actor metadata nor supplies actions, and never grants model, explanation
or participant qualification. Existing exact-type consumers are not modified.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable

from backend import warehouse_public_history_runtime as observed
from backend import warehouse_shutdown_runtime as shutdown
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from env.warehouse_native.policy import NumPyNativeActor

VERSION = "warehouse-native-runtime-family-registry.v1"


@dataclass(frozen=True)
class RuntimeFamily:
    name: str
    runtime_version: str
    runtime_type: type
    source_provider: Callable


_FAMILIES = MappingProxyType({
    observed.PublicHistoryRuntime: RuntimeFamily("observed197", observed.RUNTIME_VERSION,
        observed.PublicHistoryRuntime, observed.runtime_sources),
    shutdown.ShutdownRuntime: RuntimeFamily("retained_beta197", shutdown.RUNTIME_VERSION,
        shutdown.ShutdownRuntime, shutdown.runtime_sources),
})


def execution_sources():
    result = {}
    for spec in _FAMILIES.values():
        for name, value in spec.source_provider().items():
            if name in result and result[name] != value: raise ValueError("Runtime-family source closures disagree")
            result[name] = value
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def family(runtime):
    """Identify only exact registered classes, without treating shape as trust."""
    spec = _FAMILIES.get(type(runtime))
    if spec is None: raise ValueError("Unregistered runtime class or subclass")
    return spec


def verify(runtime, *, allow_test_fixture=False, expected_family=None, expected_signature=None):
    """Zero-forward source verification; return a detached report, never a permit."""
    spec = family(runtime)
    if expected_family is not None and expected_family != spec.name: raise ValueError("Runtime family differs from its caller binding")
    if (type(allow_test_fixture) is not bool or type(runtime.test_fixture) is not bool
            or runtime.test_fixture is not allow_test_fixture): raise ValueError("Explicit matching runtime fixture scope required")
    if type(runtime.actor) is not NumPyNativeActor: raise ValueError("Only the real NumPy Actor may supply actions")
    if (runtime.actor.metadata.get("test_fixture", False) is not allow_test_fixture
            or runtime.protocol.get("test_fixture", False) is not allow_test_fixture):
        raise ValueError("Runtime, Actor and protocol fixture scopes differ")
    signature = runtime.verify_binding()
    # Older frozen runtimes do not themselves re-hash config/signature. Keep
    # their implementation intact while enforcing the full signed input here.
    actual_sources = spec.source_provider()
    rebuilt = digest({"version": spec.runtime_version, "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256, "actor_metadata_sha256": digest(runtime.actor.metadata),
        "configuration": asdict(runtime.config), "sources": actual_sources})
    if signature != rebuilt or runtime.sources != actual_sources:
        raise ValueError("Runtime signature, configuration or source contract changed")
    if expected_signature is not None and expected_signature != signature:
        raise ValueError("Runtime differs from the caller's external signature")
    sources = execution_sources()
    return {"version": VERSION, "family": spec.name, "runtime_version": spec.runtime_version,
        "runtime_signature": signature, "runtime_sources": deepcopy(actual_sources),
        "runtime_sources_sha256": digest(actual_sources), "registry_sources": sources,
        "registry_sources_sha256": digest(sources), "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256, "actor_metadata_sha256": digest(runtime.actor.metadata),
        "actor_experiment_version": runtime.actor.metadata["experiment_version"],
        "configuration": asdict(runtime.config), "test_fixture": allow_test_fixture,
        "scope": "runtime_identity_and_source_only", "qualification_evaluated": False,
        "release_ready": False, "explanation_qualified": False}


def fresh_instance(runtime, *, allow_test_fixture=False, expected_family=None, expected_signature=None):
    """Construct a private instance through that family's genuine constructor.

    The Actor bytes, complete protocol, metadata and runtime signature remain
    unchanged. NumPy loads private arrays; no existing Actor or live environment
    is copied/reused. A caller restores any selected frame through from_snapshot.
    """
    before = verify(runtime, allow_test_fixture=allow_test_fixture,
        expected_family=expected_family, expected_signature=expected_signature)
    spec = family(runtime)
    bindings = deepcopy(runtime.actor.metadata)
    bindings["actor_sha256"] = runtime.actor_sha256
    fresh = spec.runtime_type(runtime._actor_path, protocol=deepcopy(runtime.protocol),
        expected_actor_sha256=runtime.actor_sha256, expected_protocol_sha256=runtime.protocol_sha256,
        expected_bindings=bindings, allow_test_fixture=allow_test_fixture, config=deepcopy(runtime.config))
    after = verify(runtime, allow_test_fixture=allow_test_fixture,
        expected_family=spec.name, expected_signature=before["runtime_signature"])
    new = verify(fresh, allow_test_fixture=allow_test_fixture,
        expected_family=spec.name, expected_signature=before["runtime_signature"])
    if before != after or after != new: raise ValueError("Runtime inputs or sources changed during private construction")
    return fresh
