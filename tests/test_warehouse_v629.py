"""Scientific contract tests; synthetic inputs, zero environment resets."""
from copy import deepcopy
from itertools import combinations, product
import json
import numpy as np
import pytest
from backend.training.warehouse_v6_actor import FEATURE_NAMES
from backend.training.warehouse_v622_program import MASK_INDICES, masked_probabilities
from backend.training.warehouse_v626_domain import domain_table, domain_sha
from backend.training.warehouse_v626_certificate import certify
from backend.training.warehouse_v626_search import Node, encode, decode, replace
from backend.training.warehouse_v626_program import Program as Parent
from backend.training.warehouse_v628_program import Program as Previous, payload
from backend.training.warehouse_v628_certificate import region
from backend.training.warehouse_v629_guards import predicate, branch_proof, verify_branch, leaf_actions
from backend.training.warehouse_v629_program import Program, from_parent
from backend.training.warehouse_v629_search import Context, Budget, optimize
from experiments.warehouse.v626_protocol import pointer


def axis(feature='self_row'):
    w = np.zeros(len(FEATURE_NAMES), np.float32); w[FEATURE_NAMES.index(feature)] = 1
    return tuple(map(float, w))


def constant_leaf(path, legal):
    leaf = dict(version='warehouse-role-program.v6.26-domain-complete', domain_sha256=domain_sha(),
        normalized=True, support=[], bias=[-3., -3., -3., -3., 3.], weights=[[] for _ in range(5)], witness=4)
    leaf['certificate'] = certify(leaf, path, legal, use_path=True)[0]
    return leaf


def fixture(tmp_path):
    t = encode(Node(axis(), -.5, None, Node(axis('self_column'), -.5)))[0]
    leaves = [constant_leaf(region(t, i), {4}) for i in range(3)]
    p = Parent('robot_1', 'a'*64, t, leaves, 'C3', 64, frozenset({0}), frozenset({0}))
    path = tmp_path/'parent.json'; path.write_text(json.dumps(p.to_dict()))
    model = from_parent(p, pointer(path))
    parts = {}
    for split in ('fit', 'calibration', 'eligibility', 'validation'):
        x = np.tile(np.array([f['center'] for f in domain_table()['features']], np.float32), (240, 1))
        x[:, MASK_INDICES] = 1
        x[:, FEATURE_NAMES.index('self_row')] = np.linspace(0, 1, 240, dtype=np.float32)
        x[:, FEATURE_NAMES.index('self_column')] = np.tile(np.linspace(0, 1, 60, dtype=np.float32), 4)
        m = np.ones((240, 5), np.float32)
        parts[split] = dict(x=x, m=m, p=masked_probabilities(np.tile(np.asarray(leaves[0]['bias'], np.float32), (240, 1)), m),
            sampled=np.full(240, 4), rows=[dict(episode=f'{split}-{i//10}', base_seed=f'{split}-{i//10}', task=i//10%3+1) for i in range(240)])
    return model, parts


@pytest.mark.parametrize('count', [2, 3])
@pytest.mark.parametrize('negative', [False, True])
def test_conjunction_exhaustively_proves_only_real_physical_bits(count, negative):
    from backend.training.warehouse_v6_tree_routing import split_scores, HARD_SPLIT_TIE_EPSILON
    for actions in combinations(range(4), count):
        w, b = predicate(actions, negative=negative)
        proof = branch_proof(w, b); assert verify_branch(w, b, proof)
        x = np.zeros((16, len(FEATURE_NAMES)), np.float32)
        x[:, MASK_INDICES] = np.array([(*bits, 1) for bits in product((0, 1), repeat=4)], np.float32)
        right = split_scores(x, np.array([w], np.float32), np.array([b], np.float32))[:, 0] > HARD_SPLIT_TIE_EPSILON
        for take, key in ((~right, 'left_proven'), (right, 'right_proven')):
            assert take.any()
            assert all(np.all(x[take, MASK_INDICES[a]] == 1) for a in proof[key])
        assert set(proof['left_proven'] if negative else proof['right_proven']) == set(actions)
        bad = deepcopy(proof); bad['right_proven'] = [0, 1, 2, 3]
        with pytest.raises(ValueError): verify_branch(w, b, bad)


def test_noncanonical_predicate_never_grants_new_facts_and_four_sources_rejected():
    w, b = predicate((0, 1, 2)); w = list(w); w[MASK_INDICES[0]] = .999
    proof = branch_proof(w, b)
    assert proof['left_proven'] == proof['right_proven'] == []
    with pytest.raises(ValueError): predicate((0, 1, 2, 3))


def test_two_levels_can_prove_all_four_moves_and_count_fallback_leaves():
    w, b = predicate((0, 1, 2)); w1, b1 = predicate((3,))
    t = encode(Node(w, b, None, Node(w1, b1)))[0]
    actions = leaf_actions(t)
    assert t['leaf_count'] == 3 and actions == (frozenset({4}), frozenset({0,1,2,4}), frozenset(range(5)))


def test_new_envelope_rejects_tamper_and_old_reader_does_not_reinterpret(tmp_path):
    model, parts = fixture(tmp_path)
    encoded = model.to_dict()
    assert Program.from_dict(encoded, model.actor_role_sha256).to_dict() == encoded
    with pytest.raises(ValueError): Previous.from_dict(encoded, model.actor_role_sha256)
    encoded['guard_evidence']['nodes'][0]['right_proven'] = [0]
    with pytest.raises(ValueError): Program.from_dict(encoded, model.actor_role_sha256)
    # Qualification starts empty; frozen parent bytes still enforced.
    assert not model.feedback_leaves and not model.explanation_leaves
    model.leaves[0]['bias'][4] = 2
    with pytest.raises(ValueError): model.validate()


def mock_fitting(monkeypatch):
    import backend.training.warehouse_v629_search as module
    calls = []
    def fit(x, m, p, **kw):
        calls.append(kw['path']['leaf']); return constant_leaf(kw['path'], kw['legal'])
    monkeypatch.setattr(module, 'fit_leaf', fit)
    monkeypatch.setattr(module, 'calibrate', lambda leaf, *args, **kwargs: leaf)
    return calls


def test_local_edit_refits_only_changed_leaves_preserving_other_repairs(tmp_path, monkeypatch):
    model, parts = fixture(tmp_path); calls = mock_fitting(monkeypatch)
    budget = Budget(tmp_path/'budget.json', dict(synthetic=True))
    ctx = Context(parts, model, budget)
    before, _, _ = ctx.evaluate(decode(model.structure)); assert calls == []
    tree = replace(decode(before.structure), (1, 0), Node(axis(), -.7))
    first, _, _ = ctx.evaluate(tree, anchor=before); assert len(calls) == 2
    tree = replace(decode(first.structure), (1, 1), Node(axis(), -.8))
    second, _, _ = ctx.evaluate(tree, anchor=first); assert len(calls) == 4
    assert payload(model.leaves[0]) == payload(second.leaves[0])
    assert len(second.leaves) == 5 and ctx.reused_leaves >= 8
    budget.session.close()


def test_same_sweep_composes_disjoint_repairs_and_safe_resume_is_identical(tmp_path, monkeypatch):
    mock_fitting(monkeypatch)
    class Synthetic(Context):
        def evaluate(self, tree, **kw):
            p, _, metrics = super().evaluate(tree, **kw)
            return p, (len(p.leaves),), metrics  # deterministic toy objective
        def proposals(self, program, path):
            if path not in ((1, 0), (1, 1)): return [], 'checked'
            t = replace(decode(program.structure), path, Node(axis(), -.7))
            return [(encode(t)[0], 'synthetic_local_split')], 'checked'
    model, parts = fixture(tmp_path)
    b1 = Budget(tmp_path/'whole.json', dict(synthetic='resume'))
    expected, report = optimize(Synthetic(parts, model, b1), rounds=1)
    accepted = [r for r in report['trace'] if r.get('accepted')]
    assert len(expected.leaves) == 5 and len(accepted) == 2
    assert accepted[1]['applied_to_sha256'] == accepted[0]['result_sha256']
    assert report['mandatory_complete']
    b2 = Budget(tmp_path/'pause.json', dict(synthetic='resume'))
    _, pause = optimize(Synthetic(parts, model, b2), rounds=1, pause_after=2)
    assert pause['paused']
    resumed = Budget(tmp_path/'pause.json', dict(synthetic='resume'), resume=True)
    actual, ar = optimize(Synthetic(parts, model, resumed), rounds=1, resume=True)
    assert actual.to_dict() == expected.to_dict() and ar == report and b1.counts == resumed.counts
    b1.session.close(); resumed.session.close()


def test_hard_timer_exit_recheck_preserves_returned_best_for_final_metrics(monkeypatch):
    import time
    import scripts.run_warehouse_v629 as runner
    from backend.training.warehouse_v625_checkpoint import DeadlineExceeded
    marker = object()
    def bounded_optimizer(ctx):
        try: time.sleep(.2)
        except DeadlineExceeded: return marker, dict(stopped='synthetic_search_deadline')
        raise AssertionError('expected real POSIX timer')
    monkeypatch.setattr(runner, 'optimize', bounded_optimizer)
    model, report = runner.bounded_search(None, time.monotonic(), .02)
    assert model is marker and report['hard_timer_exit_recheck']


def test_terminal_old_best_import_is_readonly_and_uses_no_resume(tmp_path, monkeypatch):
    import scripts.run_warehouse_v629 as runner
    from backend.training.warehouse_v628_search import Budget as OldBudget
    from backend.training.warehouse_v628_program import qualify
    model, parts = fixture(tmp_path)
    old = Previous(**deepcopy(model.__dict__)); qualify(old,parts)
    spec = dict(job_id='C64-fold-0-robot_1', capacity=64, fold=0, role='robot_1')
    groups = {s:sorted({r['episode'] for r in a['rows']}) for s,a in parts.items()}
    oldplan = dict(jobs=[spec],folds=[dict(inner=groups)],
        cases=[dict(case_id=i,base=i) for ids in groups.values() for i in ids])
    root = tmp_path/'v628'; folder = root/'jobs'/spec['job_id'];folder.mkdir(parents=True)
    planfile = root/'plan.json';planfile.write_text(json.dumps(oldplan))
    plan = dict(previous_plan=pointer(planfile),actor_role_sha256={'robot_1':model.actor_role_sha256})
    budget = OldBudget(folder/'checkpoint.json',dict(plan_sha256=pointer(planfile)['sha256'],job=spec))
    ref = budget.session.store.put_json(old.to_dict())
    budget.checkpoint(dict(algorithm='protected_region_subtree_refinement_v1', best=dict(program=ref),
        machine=dict(mandatory_complete=False)))
    result = dict(status='scientific_budget_exhausted',solver=budget.summary())
    budget.session.close()
    (folder/'result.json').write_text(json.dumps(result))
    (folder/'completed.json').write_text(json.dumps(dict(result_sha256=pointer(folder/'result.json')['sha256'])))
    monkeypatch.setattr(runner.previous,'OUTPUT',root)
    before={str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    imported, provenance, _ = runner.initial_program(plan,spec)
    assert provenance['kind'] == 'v628_terminal_committed_inner_CAS_best'
    assert provenance['no_old_search_resumed'] and not provenance['old_search_completed']
    assert imported.role == model.role
    assert before == {str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    manifest=json.loads((folder/'checkpoint.json').read_bytes());manifest['identity_sha256']='0'*64
    (folder/'checkpoint.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):runner.initial_program(plan,spec)


def test_independent_audit_rejects_wrong_reference_gain_or_size(tmp_path, monkeypatch):
    import scripts.audit_warehouse_v629 as auditor
    import scripts.run_warehouse_v629 as runner
    from backend.training.warehouse_v629_program import complete_program
    p,parts=fixture(tmp_path)
    job=dict(job_id='G64-fold-0-robot_1',name='G64',capacity=64,guard_bundles=True,fold=0,role='robot_1')
    output=tmp_path/'audit';folder=output/'jobs'/job['job_id'];folder.mkdir(parents=True)
    plan=dict(jobs=[job],actor_role_sha256={'robot_1':p.actor_role_sha256})
    oldplan=tmp_path/'old-plan.json';oldplan.write_text('{}');plan['previous_plan']=pointer(oldplan)
    planpath=output/'plan.json';planpath.write_text(json.dumps(plan))
    monkeypatch.setattr(auditor.v,'OUTPUT',output)
    monkeypatch.setattr(auditor.data,'load_parts',lambda *args,**kwargs:parts)
    b=Budget(folder/'checkpoint.json',dict(synthetic=True))
    p,measurements=complete_program(p,parts,b,refit=False)
    initial=deepcopy(p);initial_ref=dict(synthetic='initial')
    monkeypatch.setattr(runner,'initial_program',lambda plan,job:(initial,initial_ref,{}))
    for name in ('program','reference'):(folder/(name+'.json')).write_text(json.dumps(p.to_dict()))
    search=dict(mandatory_complete=True)
    (folder/'search-report.json').write_text(json.dumps(search))
    result=dict(job=job,status='completed',plan_sha256=pointer(planpath)['sha256'],
        program=pointer(folder/'program.json'),reference=pointer(folder/'reference.json'),initial=initial_ref,
        metrics=measurements,reference_metrics=deepcopy(measurements),leaf_count=3,depth=2,
        search_report=pointer(folder/'search-report.json'),gate=runner.gate(measurements['validation'],search),solver=b.summary())
    b.session.close()
    assert auditor.audit(plan,[result])['integrity_passed']
    result['reference_metrics']['validation']['explanation']['coverage']=.123
    with pytest.raises(RuntimeError):auditor.audit(plan,[result])
    result['reference_metrics']=deepcopy(measurements);result['leaf_count']=4
    with pytest.raises(RuntimeError):auditor.audit(plan,[result])
