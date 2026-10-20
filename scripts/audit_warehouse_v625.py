#!/usr/bin/env python3
"""Independent arithmetic and certificate audit for v6.25 sparse logistic leaves.

No production program loader, leaf fitter, certificate or metric implementation
is invoked. Public feature encoding, frozen NN inference and immutable data
loading are the only shared scientific infrastructure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from fractions import Fraction
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import audit_warehouse_v621_diagnostics as math
from scripts.audit_warehouse_v622_capacity import (
    encoded_data, check_action_streams, section as old_section,
)
from scripts.audit_warehouse_v624_upgrade import (
    Evidence, require, finite, integer, valid_sha, zero, development_gate,
    reconstruct_summary as old_summary, validate_job_set,
)

CANDIDATE = "warehouse-v625-sparse-logistic-01"
ARMS = ("P0", "P1")
MAX_DEPTH, MAX_LEAVES, MAX_LEAF_COEFFICIENTS = 6, 32, 20
TEMPERATURES = (.8, .9, 1., 1.1, 1.2)
LEAF_VERSION = "warehouse-sparse-logistic-leaf.v1"
ARITHMETIC = "ordered-fp32-multiply-add-k3-raw-logit-argmax-first.v1"
CERTIFICATE_VERSION = "path-legal-fixed-witness-global-box-exact-corners.v1"
ZERO = {
    "environment_resets": 0, "actor_updates": 0, "PPO_joint_steps": 0,
    "pilot_opened": False, "publishable": False,
    "sealed_audit_used": False, "heldout_used": False,
}


def canonical(value):
    return json.dumps(math.plain(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def source_cost(indices, names):
    return sum(2 if names[i] == "public_hard_risk_context" else 1 for i in indices)


def independent_bounds(names):
    """Public bounds, never estimated from fit/calibration/validation extrema."""
    unit = {
        "self_row", "self_column", "partner_row", "partner_column",
        "self_battery_fraction", "public_hard_risk_context", "public_near_robots_context",
        *(f"{role}_recent_position_{i}_{axis}"
          for role in ("self", "partner") for i in range(3) for axis in ("row", "column")),
        *(f"order_{order}_{endpoint}_{axis}" for order in (0, 1)
          for endpoint in ("pickup", "delivery") for axis in ("row", "column")),
    }
    result = {}
    for i, name in enumerate(names):
        if name in unit or name in {f"legal_{a.lower()}" for a in math.ACTIONS}:
            result[i] = (0., 1.)
        elif name == "self_charger_return_margin" or name.startswith("charger_return_margin_if_"):
            result[i] = (-2., 1.)
    return result


def parse_leaf(leaf, names, mode):
    """Validate coefficient count and shared five-action feature support."""
    require(isinstance(leaf, dict), "leaf is not an object")
    require(leaf.get("version") == LEAF_VERSION and leaf.get("arithmetic") == ARITHMETIC
            and leaf.get("feature_names") == list(names), "leaf numeric/schema contract differs")
    support = leaf.get("support")
    require(isinstance(support, list) and len(support) <= 3
            and support == sorted(set(support))
            and all(integer(i, 0, len(names)-1) for i in support),
            "invalid shared leaf support")
    require(source_cost(support, names) <= 3,
            "leaf exceeds three public source features")
    require(mode != "P0" or not support, "P0 must contain only constant leaves")
    require(leaf.get("support_names") == [names[i] for i in support], "leaf support names differ")
    bounds = independent_bounds(names)
    require(set(support) <= set(bounds), "leaf has an unknown public domain")
    weights = np.asarray(leaf.get("weights"), dtype=np.float32)
    bias = np.asarray(leaf.get("bias"), dtype=np.float32)
    if not support and weights.size == 0:
        weights = np.empty((5, 0), np.float32)
    require(weights.shape == (5, len(support)) and bias.shape == (5,)
            and np.isfinite(weights).all() and np.isfinite(bias).all(),
            "invalid five-action logistic coefficients")
    require(np.max(np.abs(weights), initial=0) <= 32 and np.max(np.abs(bias), initial=0) <= 32
            and np.array_equal(np.asarray(leaf["bias"], np.float64), bias.astype(np.float64))
            and np.array_equal(np.asarray(leaf["weights"], np.float64), weights.astype(np.float64)),
            "coefficients must be bounded exact serialized float32")
    require(5 * (len(support)+1) <= MAX_LEAF_COEFFICIENTS,
            "too many stored leaf coefficients")
    witness = leaf.get("witness")
    require(integer(witness, 0, 4), "invalid fixed legality witness")
    return support, weights, bias, witness, bounds


def sequential_logits(encoded, support, weights, bias):
    """Explicit ordered float32 products and additions, no BLAS dispatch."""
    x = np.asarray(encoded, dtype=np.float32)
    require(x.ndim == 2 and np.isfinite(x).all(), "invalid leaf input")
    values = np.tile(bias, (len(x), 1)).astype(np.float32, copy=False)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        for offset, index in enumerate(support):
            for action in range(5):
                product = np.float32(x[:, index] * weights[action, offset])
                values[:, action] = np.float32(values[:, action] + product)
    require(np.isfinite(values).all(), "leaf arithmetic overflow")
    return values


def certificate(leaf, names, mode, legal_actions):
    """Independently prove that every raw argmax is path-legal.

    Corner checks establish the affine bound. A conservative IEEE float32
    forward-error bound then covers interior rounding, not just the corners.
    A legal witness strictly dominates every forbidden action over the whole
    public box; no environment mask is used to manufacture that conclusion.
    """
    support, weights, bias, witness, bounds = parse_leaf(leaf, names, mode)
    allowed = frozenset(legal_actions)
    require(4 in allowed and witness in allowed and all(integer(a, 0, 4) for a in allowed),
            "leaf witness is not proved legal by its path")
    axes = [bounds[i] for i in support]
    corners = list(itertools.product(*axes)) if axes else [()]
    require(len(corners) <= 8, "certificate corner budget")
    x = np.zeros((len(corners), len(names)), np.float32)
    for column, feature in enumerate(support):
        x[:, feature] = np.asarray([point[column] for point in corners], np.float32)
    actual = sequential_logits(x, support, weights, bias)
    operations = 2*len(support)+2
    eps = np.finfo(np.float32).eps
    gamma = operations*eps/(1-operations*eps)
    magnitudes = np.abs(bias.astype(np.float64))
    for column, feature in enumerate(support):
        magnitudes += np.abs(weights[:, column].astype(np.float64))*max(map(abs, bounds[feature]))
    error = (np.nextafter(gamma*magnitudes + operations*float(np.finfo(np.float32).tiny), np.inf)
             if support else np.zeros(5, np.float64))
    comparisons = []
    for action in sorted(set(range(5))-allowed):
        gaps = []
        for point in corners:
            exact = Fraction.from_float(float(bias[witness]))-Fraction.from_float(float(bias[action]))
            for column, coordinate in enumerate(point):
                exact += (Fraction.from_float(float(weights[witness, column]))
                    - Fraction.from_float(float(weights[action, column])))*Fraction.from_float(coordinate)
            gaps.append(exact)
        lower = min(gaps)
        required = Fraction.from_float(.001)+Fraction.from_float(float(error[witness]))+Fraction.from_float(float(error[action]))
        require(lower > required and bool(np.all(actual[:, witness] > actual[:, action])),
                "leaf fails global float32 witness proof")
        comparisons.append({"action": action, "passed": True,
            "minimum_exact_gap": [lower.numerator, lower.denominator],
            "minimum_real_gap": float(lower), "required_gap": float(required)})
    require(all(int(a) in allowed for a in actual.argmax(axis=1)), "illegal raw corner argmax")
    binding = {"version": LEAF_VERSION, "arithmetic": ARITHMETIC, "support": support,
        "support_names": [names[i] for i in support], "bias": bias.tolist(), "weights": weights.tolist(),
        "witness": witness, "legal_actions": sorted(allowed),
        "bounds": [[names[i], *bounds[i]] for i in support]}
    return {"version": CERTIFICATE_VERSION, "passed": True,
        "proof_scope": "all_public_box_values_raw_top_in_path_proven_actions",
        "path_legal_actions": sorted(allowed), "witness": witness, "bounds": binding["bounds"],
        "corner_count": len(corners), "coefficient_count": int(bias.size+weights.size),
        "source_feature_cost": source_cost(support, names), "float32_error_per_action": error.tolist(),
        "strict_margin": .001, "illegal_action_checks": comparisons, "binding_sha256": digest(binding),
        "tie_break": "numpy_argmax_first_index_strict_witness_gap_excludes_illegal_ties"}



def context(raw, data, names, *, mode=None):
    """Reconstruct full masked distributions without the production program."""
    mode = raw.get("mode") if mode is None else mode
    require(mode in ARMS, "unregistered leaf family")
    require(tuple(raw.get("feature_names", names)) == tuple(names), "feature schema drift")
    leaves = raw.get("leaves")
    require(isinstance(leaves, list), "missing logistic leaf models")
    tree, capacity = math.structure(dict(raw["structure"], leaf_count=len(leaves)), names,
                                    enforce_limits=False)
    require(capacity["depth"] <= 6 and capacity["leaf_count"] <= 32
            and capacity["maximum_source_features_per_node"] <= 3,
            "program exceeds charged topology limits")
    require(sum(5*(len(leaf["support"])+1) for leaf in leaves) <= 640,
            "tree exceeds shared leaf coefficient budget")
    x, masks, p, sampled, episodes, scenes = data
    proof = math.certificates(tree, names, explicit_crossing=True)
    certificates = [certificate(leaf, names, mode, proof[i]) for i, leaf in enumerate(leaves)]
    for i, leaf in enumerate(leaves):
        math.assert_equal(certificates[i], leaf.get("certificate"), "stored certificate."+str(i))
    for i, name in enumerate(names):
        if name in {"legal_"+a.lower() for a in math.ACTIONS}:
            require(np.array_equal(x[:, i], masks[:, math.ACTIONS.index(name[6:].upper())]),
                    "physical legal feature differs from mask")
    routes = math.route(tree, x)
    raw_logits = np.empty((len(x), 5), np.float32)
    for i, leaf in enumerate(leaves):
        support, weights, bias, _, bounds = parse_leaf(leaf, names, mode)
        selected = routes == i
        for feature in support:
            lo, hi = bounds[feature]
            require(np.all((x[selected, feature] >= lo) & (x[selected, feature] <= hi)),
                    "observed public feature exceeds certified domain")
        raw_logits[selected] = sequential_logits(x[selected], support, weights, bias)
    q = math.softmax(raw_logits, masks)
    require(np.isfinite(q).all() and np.all(q[masks == 0] == 0),
            "invalid physical masked program distribution")
    raw_action, top, nn_top = raw_logits.argmax(-1), q.argmax(-1), p.argmax(-1)
    legal = masks[np.arange(len(x)), raw_action] > .5
    proven = np.asarray([int(a) in proof[int(leaf)] for a, leaf in zip(raw_action, routes)])
    kl = np.sum(p*(np.log(np.maximum(p, 1.e-8))-np.log(np.maximum(q, 1.e-8))), axis=-1)
    return {
        "tree": tree, "capacity": capacity, "proof": proof,
        "independent_leaf_certificates": certificates, "x": x, "m": masks, "p": p,
        "sampled": sampled, "episode": episodes, "scenario": scenes,
        "leaves": routes, "q": q, "raw_action": raw_action, "top": top,
        "nn_top": nn_top, "raw_legal": legal, "proven": proven,
        "row_kl": kl, "names": tuple(names),
    }


def section(c, leaves, purpose):
    result = old_section(c, leaves, purpose)
    selected = np.isin(c['leaves'], leaves)
    valid = c['raw_legal'] & c['proven']
    matched = valid & (c['top'] == c['sampled'])
    expected = valid*c['p'][np.arange(len(c['p'])), c['top']]
    if purpose == 'feedback':
        matched &= c['sampled'] == c['nn_top']
        expected *= c['top'] == c['nn_top']
    take = c['x'][:, c['names'].index('self_battery_fraction')] <= np.float32(.2)
    rows = take & selected; count, denominator = int(rows.sum()), int(take.sum())
    result['public_strata']['low_battery_le_20_percent'] = {
        'rows': count, 'physical_frames': denominator,
        'qualified_coverage': count/denominator if denominator else None,
        'agreement': float((c['top'][rows] == c['nn_top'][rows]).mean()) if count else None,
        'kl': float(c['row_kl'][rows].mean()) if count else None,
        'illegal_count': int((rows & ~c['raw_legal']).sum()),
        'actual_actionable_rows': int((rows & matched).sum()),
        'expected_actionable_rows': float(expected[rows].sum()),
    }
    return result


def selection(jobs):
    """Only six-pass P1 is an upgrade; passing P0 is never a fallback."""
    require(len(jobs) == 12 and len({r["job"]["job_id"] for r in jobs}) == 12,
            "selection requires all twelve unique jobs")
    arms = {}
    for arm in ARMS:
        rows = [r for r in jobs if r["job"]["arm"] == arm]
        require(len(rows) == 6 and {(r["job"]["fold"], r["job"]["role"]) for r in rows}
                == {(f, role) for f in range(3) for role in ("robot_1", "robot_2")},
                "selection requires six distinct role/folds per arm")
        passed = [development_gate(r["validation"])["passed"] for r in rows]
        arms[arm] = {"all_six_pass": all(passed),
                     "role_folds": [{"job_id": r["job"]["job_id"], "passed": value}
                                    for r, value in zip(rows, passed)]}
    selected = "P1" if arms["P1"]["all_six_pass"] else None
    return {"arms": arms, "selected_arm": selected,
            "next_stage": "fresh_confirmation" if selected else "closed_failed_program_precheck"}


def audit_ledger(events, *, identity_sha=None, expected_calls=None, incomplete=False, names=()):
    """Independently reconcile the durable hash chain and every predebit."""
    require(isinstance(events, list) and events, 'missing durable search ledger')
    prior, calls, pending, failures, solved = '0'*64, 0, {}, 0, 0
    previous_elapsed = -1.
    for index, event in enumerate(events, 1):
        require(isinstance(event, dict), 'ledger record is not an object')
        body = {k: value for k, value in event.items() if k != 'record_sha256'}
        require(event.get('sequence') == index and event.get('previous_sha256') == prior
                and event.get('record_sha256') == digest(body), 'durable ledger chain mismatch')
        require(finite(event.get('elapsed_seconds')) and event['elapsed_seconds'] >= previous_elapsed,
                'durable ledger time decreased')
        previous_elapsed, prior = event['elapsed_seconds'], event['record_sha256']
        kind, data = event.get('event'), event.get('data', {})
        if index == 1:
            require(kind == 'created' and (identity_sha is None or data.get('identity_sha256') == identity_sha)
                    and data.get('budget') == {'maximum_calls': 19968, 'wall_seconds': 4800.},
                    'durable initial identity/budget changed')
        if kind == 'call_started':
            calls += 1
            require(data.get('call') == calls and not pending, 'missing/duplicate leaf predebit')
            meta = data.get('metadata', {})
            require(valid_sha(meta.get('leaf_key')) and integer(meta.get('rows'))
                    and meta.get('phase') in ('initial', 'proposal', 'candidate')
                    and isinstance(meta.get('path'), list) and len(meta['path']) <= 6
                    and all(type(v) is int and v in (0, 1) for v in meta['path']), 'leaf debit metadata')
            legal = meta.get('path_legal_actions')
            require(isinstance(legal, list) and legal == sorted(set(legal)) and 4 in legal
                    and all(integer(a, 0, 4) for a in legal), 'leaf debit path proof')
            pending[calls] = meta
        elif kind == 'call_completed':
            require(data.get('call') in pending and type(data.get('success')) is bool, 'settlement lacks debit')
            meta = pending.pop(data['call']); detail = data.get('detail', {})
            support = detail.get('support')
            if support is not None:
                require(isinstance(support, list) and support == sorted(set(support)) and len(support) <= 3
                        and all(integer(i, 0, len(names)-1) for i in support)
                        and source_cost(support, names) <= 3 and detail.get('witness') in meta['path_legal_actions'],
                        'solver support/witness exceeds debit domain')
            if detail.get('solver') == 'SLSQP':
                solved += 1
                require(support is not None and integer(detail.get('iterations'), 0, 200)
                        and detail.get('convex_fixed_support_witness_problem') is True
                        and detail.get('success') == data['success'], 'actual solver record differs')
                if data['success']:
                    require(detail.get('solver_success') is True and detail.get('certificate_passed') is True,
                            'accepted solver has no convergence/certificate')
            else:
                solved += 'exception' in detail
                require(data['success'] is False and bool(detail.get('exception') or detail.get('error_type')),
                        'successful debit lacks actual solver evidence')
            failures += not data['success']
        elif kind not in ('created', 'resumed', 'commit'):
            raise RuntimeError('unrecognized durable ledger event')
    require(calls <= 19968 and (expected_calls is None or calls == expected_calls), 'API total differs')
    require(incomplete or not pending, 'complete search has in-flight fit')
    return {'search_calls': calls, 'solver_completions': solved, 'failed_calls': failures,
            'pending_calls': len(pending), 'ledger_tail': prior, 'ledger_sequence': len(events)}


def checkpoint_graph(manifest, work, evidence):
    """Resolve packed logical objects without the production storage API.

    Every physical registry/index/data object and every indexed logical byte
    string is hash-checked, including obsolete objects no longer reachable
    from the final state. Nothing is unpickled.
    """
    import io
    objects={};logical={};root=work/'checkpoint.json.objects'
    def ref_valid(ref):
        require(isinstance(ref,dict) and set(ref)=={'sha256','kind','size'}
                and valid_sha(ref['sha256']) and ref['kind'] in ('json','npy')
                and integer(ref['size']), 'malformed checkpoint object pointer')
    def physical(ref):
        ref_valid(ref)
        raw=evidence.raw(root/(ref['sha256']+'.'+ref['kind']),ref['sha256'])
        require(len(raw)==ref['size'],'physical checkpoint size mismatch')
        return raw
    registry_ref=manifest.get('object_pack_registry')
    ref_valid(registry_ref);require(registry_ref['kind']=='json','checkpoint pack registry must be JSON')
    registry=json.loads(physical(registry_ref))
    require(set(registry)=={'schema','packs'} and registry['schema']=='warehouse-v625-CAS-pack-registry.v1'
            and isinstance(registry['packs'],list),'checkpoint pack registry schema')
    pack_keys=set()
    for index_ref in registry['packs']:
        ref_valid(index_ref);require(index_ref['kind']=='json' and index_ref['sha256'] not in pack_keys,
                                    'duplicate or non-JSON pack index')
        pack_keys.add(index_ref['sha256']);index=json.loads(physical(index_ref))
        require(set(index)=={'schema','data','entries'} and index['schema']=='warehouse-v625-CAS-packed-index.v1'
                and isinstance(index['entries'],list),'packed index schema')
        require(index['data']['kind']=='npy','packed data must be an unpickled byte array')
        array=np.load(io.BytesIO(physical(index['data'])),allow_pickle=False)
        require(array.dtype==np.uint8 and array.ndim==1,'packed data is not uint8 vector')
        raw=array.tobytes();cursor=0
        for entry in index['entries']:
            require(isinstance(entry,list) and len(entry)==4,'malformed packed logical entry')
            identity,kind,size,offset=entry
            require(valid_sha(identity) and kind in ('json','npy') and integer(size) and integer(offset)
                    and offset==cursor and offset+size<=len(raw),'invalid packed offset or overlapping/gapped records')
            key=(identity,kind);require(key not in logical,'logical object was stored more than once')
            value=raw[offset:offset+size]
            require(sha256(value).hexdigest()==identity,'packed logical SHA mismatch')
            logical[key]=(size,value);cursor+=size
        require(cursor==len(raw),'unindexed pack bytes')
    def walk(value):
        if isinstance(value,dict) and set(value)=={'sha256','kind','size'}:
            ref_valid(value);key=(value['sha256'],value['kind'])
            require(key in logical and logical[key][0]==value['size'],'logical checkpoint reference missing/size mismatch')
            if key in objects:return
            raw=logical[key][1]
            if value['kind']=='json':
                obj=json.loads(raw);objects[key]=obj;walk(obj)
            else:
                obj=np.load(io.BytesIO(raw),allow_pickle=False)
                require(not obj.dtype.hasobject,'object-valued checkpoint array');objects[key]=obj
        elif isinstance(value,dict):
            for item in value.values():walk(item)
        elif isinstance(value,list):
            for item in value:walk(item)
    walk(manifest['state'])
    def get(ref):return objects[(ref['sha256'],ref['kind'])]
    state=get(manifest['state'])
    require(state.get('schema')=='warehouse-v625-search-state.v1'
            and len(state.get('states',[]))<=128 and len(state.get('leaf_cache',[]))<=512
            and len(state.get('transitions',[]))<=8,'checkpoint bounded state schema')
    best=get(state['best']);require(best.get('kind')=='fitted_tree_state','missing best state')
    models=[get(ref)['model'] for ref in best['leaves']]
    return {'verified_objects':len(objects),'verified_packed_logical_objects':len(logical),
            'verified_physical_pack_count':len(pack_keys),'best_models':models,
            'best_score':best['score'],'best_tree_sha256':best['tree_sha256']}


def checkpoint_timing_checks(report):
    """Fresh wall fractions and resumed conservative bounds stay distinct."""
    timing=report.get('checkpoint_timing',{})
    io=report.get('checkpoint_io_seconds')
    numerator=report.get('checkpoint_fraction_numerator_seconds')
    denominator=report.get('checkpoint_fraction_denominator_seconds')
    scope=report.get('checkpoint_fraction_scope')
    segment=timing.get('checkpoint_segment_io_seconds')
    lower,upper=timing.get('checkpoint_total_io_lower_bound_seconds'),timing.get('checkpoint_total_io_upper_bound_seconds')
    actual_lower,actual_upper=timing.get('actual_elapsed_lower_bound_seconds'),timing.get('actual_elapsed_upper_bound_seconds')
    require(all(finite(v) and v>=0 for v in (io,numerator,segment,lower,upper,actual_lower,actual_upper))
            and finite(denominator) and denominator>0 and segment<=lower<=upper
            and actual_lower<=actual_upper and io==numerator==upper
            and timing.get('checkpoint_total_io_seconds')==upper, 'checkpoint actual/upper-bound quantities differ')
    if scope=='fresh_measured_segment':
        require(report.get('resume_count')==0 and denominator==report.get('runner_segment_elapsed_seconds')
                and lower==upper==segment, 'fresh checkpoint time is not an actual segment')
    elif scope=='resumed_cumulative_io_upper_over_actual_lower_bound':
        require(integer(report.get('resume_count'),1) and denominator==actual_lower,
                'resumed overhead must use cumulative I/O upper over actual lower bound')
    else:raise RuntimeError('checkpoint fraction scope missing or ambiguous')
    require(finite(report.get('checkpoint_fraction'))
            and np.isclose(report['checkpoint_fraction'],numerator/denominator,atol=1e-12,rtol=1e-10)
            and numerator/denominator<=.10 and report.get('checkpoint_overhead_at_most_ten_percent') is True,
            'actual/conservative checkpoint overhead exceeded gate')
    return {'checkpoint_fraction_scope':scope,'checkpoint_fraction':numerator/denominator}


def search_checks(report, plan, spec, initial_sha, data, names, work, evidence, incomplete=False):
    x, masks, p, sampled, episode_ids, scenes = data
    identity = {'mode': spec['arm'], 'sources': {name: evidence.ref(ROOT/'backend/training'/name)['sha256']
        for name in ('warehouse_v625_search.py', 'warehouse_v625_checkpoint.py', 'warehouse_v625_program.py',
                     'warehouse_v623_search.py', 'warehouse_v622_search.py')},
        'budget': {k: val for k, val in report['budget'].items() if k != 'pause_after_candidates'},
        'configuration': report['configuration'], 'actor_role_sha256': plan['actor_role_sha256'][spec['role']],
        'initial_structure_sha256': initial_sha, 'feature_names': list(names),
        'fit_episode_ids_sha256': digest(list(episode_ids)),
        **{name: sha256(np.ascontiguousarray(value, dtype='<f4').tobytes()).hexdigest()
           for name, value in (('encoded', x), ('masks', masks), ('targets', p))}}
    recipe = evidence.obj(ROOT/'configs/warehouse_v625_recipe.json', plan['recipe_sha256'])
    require(report.get('mode') == spec['arm'] and report.get('actor_role_sha256') == identity['actor_role_sha256']
            and report.get('initial_structure_sha256') == initial_sha
            and report.get('fit_input_sha256') == digest(identity)
            and report.get('fit_target_sha256') == identity['targets']
            and report.get('configuration') == recipe['search_configuration']
            and report.get('budget') == recipe['production_budget']
            and report.get('production_configuration') is True, 'search identity/recipe binding differs')
    require(incomplete or report.get('menu_configuration_sha256') == digest(report['configuration']), 'proposal menu SHA differs')
    for k, value in {'search_uses_fit_only': True, 'calibration_eligibility_validation_used': False,
                     'execution_action_override': False, 'zero_actor_updates': True, 'zero_environment_resets': True}.items():
        require(report.get(k) is value, 'search authority '+k)
    events = [json.loads(row) for row in evidence.raw(work/'checkpoint.json.ledger.jsonl').splitlines()]
    require(events == report.get('leaf_api_attempts'), 'reported ledger differs from durable bytes')
    ledger = audit_ledger(events, identity_sha=digest(identity), expected_calls=report.get('new_leaf_calls'),
                         incomplete=incomplete, names=names)
    require(finite(report.get('elapsed_seconds')) and 0 <= report['elapsed_seconds'] <= 4800,
            'search elapsed exceeds immutable bound')
    require(report.get('actual_convex_solver_calls') == ledger['solver_completions'], 'actual solver count differs')
    timing = report.get('checkpoint_timing', {})
    if not incomplete:
        require(report.get('schema') == 'warehouse-v625-fit-search.v1'
                and report.get('complete_incumbent') is True and report.get('stop_reason') in (
                'wall_budget', 'leaf_call_budget', 'three_complete_recipe_cycles_without_improvement', 'maximum_rounds'),
                'search cannot finalize paused/incomplete state')
        require(not (work/'checkpoint.json.dirty').exists(), 'completed checkpoint remains dirty')
        manifest = evidence.obj(work/'checkpoint.json')
        require(manifest.get('status') == 'completed' and manifest.get('identity_sha256') == digest(identity)
                and manifest.get('calls') == ledger['search_calls']
                and manifest.get('ledger_tail') == ledger['ledger_tail']
                and manifest.get('ledger_sequence') == ledger['ledger_sequence'], 'checkpoint completion binding')
        require(events[-1]['event']=='commit' and events[-1]['data'].get('status')=='completed'
                and events[-1]['data'].get('state')==manifest.get('state')
                and events[-1]['data'].get('object_pack_registry')==manifest.get('object_pack_registry'),
                'manifest is not bound to final durable state/pack commit')
        graph=checkpoint_graph(manifest, work, evidence)
        require(graph['best_score']==report.get('best_score') and graph['best_tree_sha256']==report.get('best_tree_sha256'),
                'checkpoint best-state score/tree differs')
        ledger.update(graph)
        ledger.update(checkpoint_timing_checks(report))
        require(timing.get('calls')==ledger['search_calls'] and timing.get('pending_calls')==0
                and timing.get('ledger_tail')==ledger['ledger_tail'], 'checkpoint timing/ledger mismatch')
        require(valid_sha(report.get('candidate_order_sha256')) and integer(report.get('evaluated_candidates')),
                'actual proposal sequence missing')
        require(sum(report.get('operation_counts', {}).values()) == report['evaluated_candidates'],
                'candidate operation counts differ')
        for number, row in enumerate(report.get('rounds', []), 1):
            require(row.get('round') == number and row.get('recipe') == (number-1)%8
                    and integer(row.get('new_leaf_calls'), 0, ledger['search_calls']), 'round ledger differs')
    return ledger


def check_calibration(leaf, report, c, leaf_id, mode):
    pre = report.get('pre_calibration_leaf')
    require(isinstance(pre, dict), 'calibration lacks immutable pre-temperature coefficients')
    names, proof = c['names'], c['proof'][leaf_id]
    certificate(pre, names, mode, proof)
    require(pre['support'] == leaf['support'] and pre['witness'] == leaf['witness'], 'calibration changed support/witness')
    take = c['leaves'] == leaf_id; x, m, p = c['x'][take], c['m'][take], c['p'][take].astype(np.float64)
    records, choices = [], []
    for index, temp in enumerate(TEMPERATURES):
        trial = dict(pre, bias=(np.asarray(pre['bias'], np.float32)/np.float32(temp)).tolist(),
                     weights=(np.asarray(pre['weights'], np.float32)/np.float32(temp)).tolist())
        try:
            proof_value = certificate(trial, names, mode, proof)
        except RuntimeError:
            records.append({'temperature': temp, 'full_distribution_forward_kl': None, 'certificate_passed': False})
            continue
        support, weights, bias, _, _ = parse_leaf(trial, names, mode)
        q = math.softmax(sequential_logits(x, support, weights, bias), m).astype(np.float64)
        loss = float(np.mean(np.sum(p*(np.log(np.maximum(p, 1e-300))-np.log(np.maximum(q, 1e-300))), -1))) if len(p) else 0.
        records.append({'temperature': temp, 'full_distribution_forward_kl': loss, 'certificate_passed': True})
        choices.append((loss, index if len(p) else abs(temp-1), temp, trial, proof_value))
    require(choices, 'no legal calibration candidate')
    winner = min(choices, key=lambda item: (item[0], item[1]))
    math.assert_equal(records, report.get('candidate_reports'), 'calibration temperature table')
    require(report.get('temperature') == winner[2] and report.get('row_count') == len(p)
            and report.get('support_reselected') is False and report.get('witness_reselected') is False
            and report.get('certificate_recomputed_after_float32_scaling') is True, 'calibration selection differs')
    for key in ('support', 'witness', 'bias', 'weights'):
        math.assert_equal(winner[3][key], leaf[key], 'selected calibration '+key)
    math.assert_equal(winner[4], leaf['certificate'], 'calibration certificate')


def final_fit_checks(raw, contexts, structure, search, result, final_ledger):
    report, count = raw['report'], len(raw['leaves'])
    require(report.get('leaf_optimization_calls') == report.get('leaf_optimization_calls_outside_search') == count
            and report.get('convex_solver_calls') == count and result.get('final_calls') == count
            and result.get('search_calls') == search['new_leaf_calls']
            and count+search['new_leaf_calls'] <= 20000, 'final solve/API totals differ')
    require(final_ledger.get('schema') == 'warehouse-v625-final-refit-predebit.v1'
            and final_ledger.get('calls') == count and final_ledger.get('inflight') is None
            and final_ledger.get('maximum') == 32
            and final_ledger.get('prior_leaf_optimization_calls') == search['new_leaf_calls'], 'final durable debit mismatch')
    require(len(final_ledger.get('records', [])) == count, 'missing final solve settlement')
    for i, leaf in enumerate(raw['leaves']):
        fit = report['leaf_fits'][str(i)]
        require(fit['row_count'] == int((contexts['fit']['leaves'] == i).sum())
                == report['fit_leaf_row_counts'][i] and fit['full_fit_calls'] == fit['convex_solver_calls'] == 1,
                'final fit rows/calls differ')
        require(fit.get('support_search_uses_fit_only') is True and fit.get('regularization') == 1e-4,
                'unregistered final fitting objective')
        initial = structure['initial_leaf_models'][i]
        require(leaf['support'] == initial['support'] and leaf['witness'] == initial['witness'],
                'final fit reselected support/witness')
        records = fit['candidate_reports']; require(len(records) == 1 and records[0]['success'] is True,
                                                   'final fixed solve failed or duplicated')
        debit = final_ledger['records'][i]
        require(debit.get('token') == i+1 and debit.get('success') is True and debit.get('detail') == records[0],
                'final durable solver settlement differs')
        check_calibration(leaf, report['leaf_calibrations'][str(i)], contexts['calibration'], i, raw['mode'])


def reconstruct_summary(plan, plan_sha, audited):
    value = old_summary(plan, plan_sha, audited)
    value.update(schema='warehouse-v625-sparse-logistic-summary.v1', candidate_id=CANDIDATE,
                 selected_arm='P1' if value['arms']['P1']['all_six_pass'] else None, P0_is_control_only=True)
    value['next_stage'] = 'fresh_confirmation' if value['selected_arm'] else 'closed_failed_program_precheck'
    return value


def audit(plan_path):
    """Recompute all twelve jobs; no fitting, environment or write operations."""
    from experiments.warehouse import v625_protocol as v
    from scripts.run_warehouse_v621_diagnostics import runtime_identity
    from backend.training.warehouse_v6_actor import FEATURE_SCHEMA_SHA256, ACTION_PHYSICS_CONFIG_SHA256
    begin = time.monotonic(); evidence = Evidence(); own = evidence.ref(__file__)
    path = Path(plan_path).resolve(strict=True); folder = path.parent; plan = evidence.obj(path)
    plan_sha = evidence.files[str(path)]; validate_job_set(plan.get('jobs')); zero(plan)
    require(plan.get('schema') == 'warehouse-v625-sparse-logistic-plan.v1' and plan.get('candidate_id') == CANDIDATE
            and plan.get('stage') == 'A' and plan['jobs'] == v.jobs()
            and evidence.obj(folder/'plan.claim.json') == {'sha256': plan_sha, 'single_use': True}, 'plan binding')
    require(plan['source_files'] == v.source_files() and plan['source_sha256'] == v.source_sha()
            and plan['recipe_sha256'] == v.recipe() and plan['runtime'] == runtime_identity()
            and plan['folds'] == v.old.grouped_folds() and plan.get('budget') == v.BUDGET
            and plan.get('gates') == v.GATES, 'source/recipe/runtime drift')
    for key in ('fresh_confirmation_authorized', 'automatic_retry', 'scientific_retry'):
        require(plan.get(key) is False, 'unregistered stage authority '+key)
    for relative, expected in plan['source_files'].items():
        evidence.raw(ROOT/relative, expected); evidence.raw(folder/'source'/plan['source_sha256']/relative, expected)
    engineering = evidence.bound(plan['engineering'])
    require(engineering.get('passed') is True and engineering.get('source_sha256') == plan['source_sha256'],
            'engineering preflight changed')
    actor, actor_path = v.old.checked_actor(ROOT)
    require(plan['actor'] == evidence.ref(actor_path) and plan['actor']['sha256'] == v.old.ACTOR_SHA256
            and plan['actor_role_sha256'] == v.old.ROLE_SHA256, 'frozen Actor identity')
    baseline, initial = v.checked_baseline(); require(plan['baseline'] == baseline and plan['initial_S1'] == initial,
                                                     'S1 starting topology binding')
    evidence.bind_graph(baseline); evidence.bind_graph(initial)
    episodes, manifest = v.old.checked_inputs(ROOT)
    require(plan['inputs'] == manifest and plan['targets'] == v.old.target_identity(actor, episodes), 'data/targets changed')
    for file, expected in manifest['files_sha256'].items(): evidence.raw(file, expected)
    action_checks = check_action_streams(episodes); cache = {}; audited = []
    for spec in plan['jobs']:
        work = folder/'jobs'/spec['job_id']; name, role = spec['job_id'], spec['role']
        require(not (work/'failed.json').exists(), 'engineering failure cannot enter scientific closure')
        result = evidence.obj(work/'result.json'); ref = evidence.ref(work/'result.json'); zero(result)
        require(evidence.obj(work/'started.json') == dict(ZERO, job=spec, stage='A', plan_sha256=plan_sha)
                and result.get('job') == spec and result.get('plan_sha256') == plan_sha
                and result.get('schema') == 'warehouse-v625-sparse-logistic-result.v1', 'job binding '+name)
        require(evidence.obj(work/'completed.json') == {'plan_sha256': plan_sha, 'result_sha256': ref['sha256'],
                'passed': result['gate']['passed']}, 'completion binding '+name)
        names = tuple(actor.roles[role].feature_names); key = (spec['fold'], role)
        if key not in cache:
            cache[key] = {split: encoded_data(actor, role, [r for case in plan['folds'][spec['fold']]['splits'][split]
                for r in episodes[case['base']][role]]) for split in v.SPLITS}
        data = cache[key]; initial_ref = plan['initial_S1'][f"fold-{spec['fold']}-{role}"]['structure']
        initial_tree, _ = math.structure(evidence.bound(initial_ref), names, enforce_limits=False)
        initial_sha = digest(initial_tree)
        search = evidence.bound(result['search_report'])
        ledger = search_checks(search, plan, spec, initial_sha, data['fit'], names, work, evidence,
                               incomplete=result.get('error_type') == 'SearchIncompleteError')
        if result.get('status') != 'completed':
            require(result.get('status') == 'scientific_budget_exhausted' and result.get('gate') == {'passed': False}
                    and result.get('error_type') == 'SearchIncompleteError' and result.get('failure') == search
                    and search.get('failure_kind') in ('initial_deadline_exhausted', 'initial_leaf_api_exhausted', 'initial_solver_infeasible')
                    and 'program' not in result and 'metrics' not in result, 'unproved scientific failure')
            if search['failure_kind'] == 'initial_leaf_api_exhausted':
                require(ledger['search_calls'] >= 19968, 'API budget not exhausted')
            elif search['failure_kind']=='initial_deadline_exhausted':
                require(search.get('error_type')=='DeadlineExceeded' and search['elapsed_seconds']>=4790, 'deadline not exhausted')
            else:
                require(search.get('error_type') == 'ConstrainedLeafFitError' and ledger['failed_calls'] > 0,
                        'missing typed solver failure')
            audited.append(dict(job=spec,status=result['status'],gate_passed=False,validation=None,
                                leaf_count=None,result=ref,ledger=ledger)); continue
        require(result.get('actor') == plan['actor'] and result.get('initial_structure') == initial_ref
                and result.get('initial_structure_sha256') == initial_sha, 'result initial/model binding')
        raw = evidence.bound(result['program'], work/'program.json'); structure = evidence.bound(result['structure'], work/'structure.json')
        require(raw.get('version') == 'warehouse-role-program.v6.25-sparse-logistic' and raw.get('mode') == spec['arm']
                and raw.get('role') == role and raw.get('actor_role_sha256') == v.old.ROLE_SHA256[role], 'program binding')
        math.assert_equal(ledger.pop('best_models'),structure.get('initial_leaf_models'),'search final leaf models differ')
        tree, capacity = math.structure(structure, names, enforce_limits=False)
        embedded, _ = math.structure(raw['structure'], names, enforce_limits=False); math.assert_equal(tree,embedded,'embedded structure')
        require(result.get('leaf_count') == search.get('leaf_count') == capacity['leaf_count']
                and search.get('depth') == capacity['depth'], 'capacity metadata differs')
        contexts = {split: context(raw, arrays, names) for split, arrays in data.items()}; report = raw['report']
        require(report.get('fit_target_sha256') == search['fit_target_sha256']
                and report.get('full_NN_distribution_targets') is True and report.get('external_program_selects_execution_action') is False
                and report.get('excluded_frames') == 0 and report.get('feature_schema_sha256') == FEATURE_SCHEMA_SHA256
                and report.get('action_physics_config_sha256') == ACTION_PHYSICS_CONFIG_SHA256, 'program target/physics binding')
        for label,index in (('split_episode_ids',4),('split_scenario_seed_ids',5)):
            ids={split:sorted(set(map(str,arrays[index]))) for split,arrays in data.items()}
            math.assert_equal(ids,report[label],label)
            require(all(not set(ids[a])&set(ids[b]) for a in v.SPLITS for b in v.SPLITS if a!=b),'partition leakage')
        math.assert_equal({s:len(a[0]) for s,a in data.items()},report['physical_frames_per_split'],'frame counts')
        for split,c in contexts.items():
            for purpose in ('feedback','explanation'):
                metric=section(c,raw[purpose+'_leaves'],purpose)
                math.assert_equal(metric,result['metrics'][split][purpose],name+'.'+split+'.'+purpose)
                if split=='validation': math.assert_equal(metric,report['validation'][purpose],name+'.report.validation')
                for i in range(capacity['leaf_count']):
                    one=section(c,[i],purpose)
                    math.assert_equal(one,result['metrics'][split]['per_leaf'][str(i)][purpose],name+'.'+split+'.leaf.'+str(i))
                    if split=='eligibility':math.assert_equal(one,report['leaf_eligibility'][str(i)][purpose],name+'.eligibility')
        for purpose,threshold in (('feedback',.93),('explanation',.95)):
            expected=[i for i in range(capacity['leaf_count']) if (contexts['fit']['leaves']==i).any()
                      and math.eligible(section(contexts['eligibility'],[i],purpose),threshold)]
            math.assert_equal(expected,raw[purpose+'_leaves'],'qualified leaves')
        final_fit_checks(raw,contexts,structure,search,result,evidence.obj(work/'final-checkpoint.json'))
        validation={p:section(contexts['validation'],raw[p+'_leaves'],p) for p in ('feedback','explanation')}
        gate=development_gate(validation); math.assert_equal(gate,result['gate'],'independent gate')
        require(finite(result.get('fitting_wall_seconds')) and 0<=result['fitting_wall_seconds']<=5400
                and result.get('budget_checks')==dict.fromkeys(('search_calls_with_reserved_final','final_calls_at_most_32',
                'total_new_calls_at_most_20000','all_final_leaves_refitted','search_wall_at_most_4800',
                'final_wall_at_most_600','checkpoint_overhead_at_most_10_percent','no_actor_change'),True),'final budget differs')
        audited.append(dict(job=spec,status='completed',gate_passed=gate['passed'],ledger=ledger,
                            validation=validation,leaf_count=capacity['leaf_count'],result=ref))
    expected=reconstruct_summary(plan,plan_sha,audited); summary_ref=evidence.ref(folder/'summary.json')
    math.assert_equal(expected,evidence.obj(folder/'summary.json'),'independent summary')
    require(v.source_sha()==plan['source_sha256'],'source changed during audit'); evidence.recheck()
    return dict(ZERO,schema='warehouse-v625-independent-audit.v1',candidate_id=CANDIDATE,stage='A',passed=True,
                summary_verified=True,audit_script=own,plan_sha256=plan_sha,source_sha256=plan['source_sha256'],
                summary_sha256=summary_ref['sha256'],jobs=audited,independently_reconstructed_summary=expected,
                independent_action_stream_checks=action_checks,evidence_sha256=dict(sorted(evidence.files.items())),
                at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-begin,
                scope='spent development arithmetic only; no fitting or new environment')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.plan), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
