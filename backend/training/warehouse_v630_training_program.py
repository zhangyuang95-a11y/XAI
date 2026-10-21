"""Pure fixed-language surrogate refit for a future admitted training owner.

No seeds, environments, Actor updates or feedback authority are supplied.
The owner must establish A/B readiness and current-NN target provenance before
calling this numerical primitive. Every new optimizer call uses its budget.
"""
from copy import deepcopy

import numpy as np

from .warehouse_v630_program import Program
from .warehouse_v628_program import qualify, measure, payload
from .warehouse_v626_program import fit_leaf, calibrate
from .warehouse_v628_certificate import region
from .warehouse_v629_online_evidence import gate
from experiments.warehouse import v630_semantic_protection as semantic
from experiments.warehouse.v626_protocol import digest, require

VERSION = 'warehouse-v630-fixed-language-protected-descendant-refit.v1'
SPLITS = ('fit', 'calibration', 'eligibility', 'validation')


def protected_indices(program):
    """Include every child of each original protected route prefix."""
    program.validate()
    return frozenset(semantic.verify(program)['protected_descendants'])


def language_lock(program):
    program.validate()
    return dict(version=VERSION, program_version=program.to_dict()['version'],
        role=program.role, anchor_actor_role_sha256=program.actor_role_sha256,
        maximum_leaves=program.maximum_leaves, structure_sha256=digest(program.structure),
        supports=[list(leaf['support']) for leaf in program.leaves],
        witnesses=[leaf['witness'] for leaf in program.leaves],
        protection_sha256=digest(program.protection),
        protected_descendants=sorted(protected_indices(program)),
        protected_payload_sha256={str(i):digest(payload(program.leaves[i]))
                                  for i in protected_indices(program)})


def limits(program):
    n = len(program.leaves) - len(protected_indices(program))
    # Same fixed-support counterexample/calibration bound as v629; protected
    # descendants need no solves. No topology or support search is permitted.
    return dict(leaf=max(1, 8*n), node=0, proof=52*n)


def validate_parts(program, parts):
    require(set(parts) == set(SPLITS), 'four disjoint training-only parts required')
    episode_sets, scene_sets, routes = {}, {}, {}
    for split in SPLITS:
        a = parts[split]
        require(set(a) == {'x','m','p','sampled','rows'}, 'unknown fixed-language training part')
        n = len(a['x'])
        require(n > 0 and len(a['rows']) == n and a['m'].shape == a['p'].shape == (n,5)
                and a['sampled'].shape == (n,), 'incomplete physical-frame training part')
        p, m = a['p'], a['m']
        require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                and np.allclose(p.sum(1),1,rtol=0,atol=2e-6)
                and np.all(p[m == 0] == 0), 'targets must be full masked NN distributions')
        require(np.issubdtype(a['sampled'].dtype,np.integer)
                and np.all((a['sampled'] >= 0) & (a['sampled'] < 5))
                and np.all(m[np.arange(n),a['sampled']] == 1), 'invalid recorded physical action')
        # Uses the native feature-domain and explicit-mask checks as well.
        routes[split] = program.distributions(a['x'],m)[0]
        episode_sets[split] = {str(row['episode']) for row in a['rows']}
        scene_sets[split] = {str(row['base_seed']) for row in a['rows']}
    require(all(not episode_sets[a] & episode_sets[b] and not scene_sets[a] & scene_sets[b]
                for a in SPLITS for b in SPLITS if a < b),
            'fit/calibration/qualification/validation whole scenes overlap')
    return routes


def complete_program(program, parts, budget, *, refit=True):
    """Copy input, refit unprotected coefficients, recertify, requalify all.

    The anchor hash continues to describe immutable parent protection. This
    helper does not assert that those weights are the current target NN.
    Current target hashes belong in the owning evidence envelope.
    """
    require(isinstance(program,Program), 'fixed-language refit needs the native v630 reader')
    program.validate()
    routes = validate_parts(program,parts)
    out = deepcopy(program)
    before_lock = language_lock(program)
    protected = protected_indices(program)
    legal = program.certificates()
    before = []
    for i, leaf in enumerate(program.leaves):
        budget.check()
        if refit and i not in protected:
            a=parts['fit'];take=routes['fit']==i
            fitted=fit_leaf(a['x'][take],a['m'][take],a['p'][take],arm='C3',
                path=region(program.structure,i,program.maximum_leaves),legal=legal[i],
                budget=budget,initial=deepcopy(leaf),final=True)
            require(fitted['support']==leaf['support'] and fitted['witness']==leaf['witness'],
                    'fixed-language fit changed its support or witness')
            before.append(deepcopy(fitted))
            a=parts['calibration'];take=routes['calibration']==i
            out.leaves[i]=calibrate(fitted,a['x'][take],a['m'][take],a['p'][take],arm='C3',
                path=region(program.structure,i,program.maximum_leaves),legal=legal[i],budget=budget)
        else:
            before.append(deepcopy(leaf))
    require(language_lock(out)==before_lock,'refit changed the locked tree or protected descendant outputs')
    # qualify() recomputes all children. Correct its old exact-path-only audit
    # metadata; this does not change qualification thresholds or metrics.
    qualify(out,parts,before)
    out.report['protected_refit_and_calibration_skipped']=sorted(protected)
    out.report['qualification_only_no_leaf_fitting']=not (refit and len(protected)<len(out.leaves))
    out.report['fixed_language_refit_method']=VERSION
    out.report['current_target_NN_provenance_requires_owner_evidence']=True
    out.validate()
    measurements={split:measure(out,a) for split,a in parts.items()}
    for value in measurements.values():value['gate']=gate(value)
    require(language_lock(out)==before_lock,'requalification changed the locked program language')
    return out, measurements
