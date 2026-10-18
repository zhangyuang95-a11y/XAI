"""Source-bound, frozen-only authorization for independent Warehouse rollouts.

This module does not arm training, create sealed claims, or publish a model.
The runner uses the validated descriptor only for its evaluation capability.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.warehouse import protocol as p


SCHEMA = "warehouse-frozen-evaluation-authorization.v1"
DEVELOPMENT_PURPOSE = "unsealed_development_diagnostic"
SEALED_PURPOSES = ("final_program_audit", "heldout")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _sealed_base_seed(seed: int) -> bool:
    # Every release minor reserves the same offsets in a new 100k block.
    return any(seed >= block[0] and (seed-block[0]) % 100_000 < len(block)
               for block in (p.FINAL_PROGRAM_AUDIT_SEEDS, p.HELDOUT_BASE_SEEDS))


def is_sealed_scenario_seed(seed: int) -> bool:
    """Reject all versioned sealed bases and their task-expanded scenarios."""
    return (_sealed_base_seed(seed)
            or (seed % 10 in p.TASKS and _sealed_base_seed(seed // 10)))


def development_evaluation_authorization(actor) -> dict:
    from .warehouse_v6_runner import source_closure_sha256
    return {
        "schema": SCHEMA,
        "purpose": DEVELOPMENT_PURPOSE,
        "expected_source_closure_sha256": source_closure_sha256(),
        "authorization_module_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "actor_sha256": actor.artifact_sha256,
        "actor_role_sha256": dict(actor.role_sha256),
        "publishable": False,
    }


def sealed_evaluation_authorization(actor, *, claim_path: str | Path,
                                     expected_claim_sha256: str) -> dict:
    path = Path(claim_path).resolve()
    raw = path.read_bytes()
    if not _sha(expected_claim_sha256) or sha256(raw).hexdigest() != expected_claim_sha256:
        raise RuntimeError("Warehouse sealed evaluation claim hash mismatch")
    claim = json.loads(raw)
    result = development_evaluation_authorization(actor)
    result.update(purpose=claim.get("phase"), sealed_claim_path=str(path),
                  sealed_claim_sha256=expected_claim_sha256,
                  candidate_id=claim.get("candidate_id"),
                  release_id=claim.get("release_id"))
    return result


def validate_evaluation_authorization(*, actor, seeds: Sequence[int],
                                      authorization: Mapping | None = None,
                                      consume_sealed: bool = False) -> dict:
    from .warehouse_v6_actor import FrozenWarehouseV6Actor
    from .warehouse_v6_runner import source_closure_sha256
    if not isinstance(actor, FrozenWarehouseV6Actor):
        raise TypeError("Warehouse independent evaluation requires a frozen Actor")
    if (not _sha(actor.artifact_sha256)
            or set(actor.role_sha256) != set(p.ROLES)
            or any(not _sha(value) for value in actor.role_sha256.values())):
        raise RuntimeError("Warehouse evaluation requires artifact and dual-role hashes")
    ordered = list(seeds)
    if (not ordered or any(type(seed) is not int or seed < 0 for seed in ordered)
            or len(set(ordered)) != len(ordered)):
        raise ValueError("Warehouse evaluation requires distinct integer scenario seeds")
    spec = dict(development_evaluation_authorization(actor)
                if authorization is None else authorization)
    if (spec.get("schema") != SCHEMA
            or spec.get("expected_source_closure_sha256") != source_closure_sha256()
            or spec.get("authorization_module_sha256") !=
               sha256(Path(__file__).read_bytes()).hexdigest()
            or spec.get("actor_sha256") != actor.artifact_sha256
            or spec.get("actor_role_sha256") != dict(actor.role_sha256)
            or spec.get("publishable") is not False):
        raise RuntimeError("Warehouse frozen evaluation source or Actor binding mismatch")
    purpose = spec.get("purpose")
    if purpose == DEVELOPMENT_PURPOSE:
        if (any(is_sealed_scenario_seed(seed) for seed in ordered)
                or any(key.startswith("sealed_claim") for key in spec)):
            raise RuntimeError("Warehouse development evaluation cannot use sealed seeds")
    elif purpose in SEALED_PURPOSES:
        path = Path(spec.get("sealed_claim_path", ""))
        if not path.is_file():
            raise RuntimeError("Warehouse sealed evaluation requires an existing one-use claim")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != spec.get("sealed_claim_sha256"):
            raise RuntimeError("Warehouse sealed evaluation claim changed")
        claim = json.loads(raw)
        release = spec.get("release_id")
        expected = list(p.sealed_seed_blocks(release)[purpose])
        hashes = claim.get("actor_hashes")
        if (claim.get("phase") != purpose
                or not isinstance(spec.get("candidate_id"), str)
                or not spec["candidate_id"]
                or claim.get("candidate_id") != spec["candidate_id"]
                or claim.get("release_id") != release
                or claim.get("seeds") != expected
                or claim.get("seed_sha256") != _digest(expected)
                or not isinstance(hashes, Mapping)
                or set(hashes) != set(p.VARIANTS)
                or any(not _sha(value) for value in hashes.values())
                or actor.artifact_sha256 not in hashes.values()):
            raise RuntimeError("Warehouse sealed evaluation claim binding mismatch")
        permitted = ([expected] if purpose == "final_program_audit" else
                     [list(values) for values in p.heldout_schedule(release).values()])
        if ordered not in permitted:
            raise RuntimeError("Warehouse sealed evaluation requires the complete claimed batch")
        if consume_sealed:
            usage = (path.parent / (path.name + ".executions") /
                     actor.artifact_sha256 / (_digest(ordered) + ".started.json"))
            usage.parent.mkdir(parents=True, exist_ok=True)
            # A crash after this point spends the batch just like a failed audit.
            with usage.open("x", encoding="utf-8") as stream:
                json.dump({"actor_sha256": actor.artifact_sha256,
                           "source_closure_sha256": spec["expected_source_closure_sha256"],
                           "claim_sha256": spec["sealed_claim_sha256"],
                           "seeds_sha256": _digest(ordered)}, stream,
                          sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            directory_fd = os.open(usage.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            spec["sealed_execution_claim_path"] = str(usage)
            spec["sealed_execution_claim_sha256"] = sha256(usage.read_bytes()).hexdigest()
    else:
        raise ValueError("Warehouse evaluation purpose is not authorized")
    spec.update(seeds_sha256=_digest(ordered), scenario_count=len(ordered))
    return spec
