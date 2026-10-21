#!/usr/bin/env python3
"""Single-use v629 batch; wait for v628, then measure all 24 program units."""
from pathlib import Path
import argparse, json, os, signal, subprocess, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from experiments.warehouse import v629_protocol as v, v628_protocol as previous, v628_data as data
from scripts.run_warehouse_v625 import write_new
from scripts.run_warehouse_v626 import stop_workers
from scripts.run_warehouse_v621_diagnostics import runtime_identity
from scripts.verify_warehouse_v625_native import real_deadline
from backend.training.warehouse_v628_program import Program as PreviousProgram
from backend.training.warehouse_v629_program import Program, complete_program, protected_indices, measure
from backend.training.warehouse_v629_search import Budget, Context, optimize
from backend.training.warehouse_v628_certificate import structure_info
from backend.training.warehouse_v625_checkpoint import BudgetExhausted, DeadlineExceeded, ContentStore


def process_identity(pid):
    row = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart=,command='], capture_output=True, text=True)
    return row.stdout.strip() if row.returncode == 0 else None


def prepare(wait_pid):
    path, old = previous.checked_plan(); source = v.source_files()
    identity = process_identity(wait_pid)
    v.require(identity is not None and 'continue_warehouse_v628_seed_audit_bounded.py' in identity,
        'must verify actual v628 driver handle before waiting')
    v.require(json.loads((ROOT/'configs/warehouse_v629_recipe.json').read_bytes()) == v.recipe(), 'recipe changed')
    folder = v.OUTPUT; folder.mkdir(parents=True, exist_ok=False)
    source_sha = v.digest(source)
    for name, sha in source.items():
        target = folder/'source'/source_sha/name; target.parent.mkdir(parents=True, exist_ok=True)
        raw = (ROOT/name).read_bytes(); v.require(v.file_sha(ROOT/name) == sha, 'source changed preparing')
        target.write_bytes(raw); target.chmod(0o444)
    plan = dict(schema='warehouse-v629-plan.v1', candidate_id=v.CANDIDATE,
        source_files=source, source_sha256=source_sha, recipe=v.recipe(), jobs=v.jobs(),
        previous_plan=v.pointer(path), collection=v.pointer(previous.OUTPUT/'collection/result.json'),
        dependency_process=dict(pid=wait_pid, identity=identity),
        actor=old['actor'], actor_role_sha256=old['actor_role_sha256'], runtime=runtime_identity(),
        environment_resets=0, training_authority=False, sealed_authority=False, pilot_opened=False)
    write_new(folder/'plan.json', plan)
    write_new(folder/'plan.claim.json', dict(sha256=v.file_sha(folder/'plan.json'), single_use=True))
    return plan


def wait_dependency(plan):
    identity = plan['dependency_process']; started = time.monotonic()
    write_new(v.OUTPUT/'dependency-wait-started.json', dict(process=identity, started_monotonic=started))
    while process_identity(identity['pid']) == identity['identity']:
        # A bounded poll verifies this exact live process, never a lock file.
        time.sleep(10)
    summary = v.bound(v.pointer(previous.OUTPUT/'summary.json'))
    execution = v.bound(v.pointer(previous.OUTPUT/'execution/completed.json'))
    v.require(summary['all_18_settled'] and execution['all_owned_workers_waited']
        and len(execution['exits']) == 18, 'v628 interrupted or not all workers reaped')
    for row in execution['exits']:
        current = process_identity(row['pid'])
        v.require(current is None or 'run_warehouse_v628.py --job' not in current,
            'v628 optimizing worker still live')
    write_new(v.OUTPUT/'dependency-cleared.json', dict(summary=v.pointer(previous.OUTPUT/'summary.json'),
        execution=v.pointer(previous.OUTPUT/'execution/completed.json'),
        elapsed_seconds=time.monotonic()-started, supervisor_returncode='not_our_child_unknown',
        no_old_job_interrupted=True))


def initial_program(plan, job):
    """Only inner-search output, never v628 outer-refitted program.json."""
    oldplan = v.bound(plan['previous_plan'])
    folder = previous.OUTPUT/'jobs'/f"C{job['capacity']}-fold-{job['fold']}-{job['role']}"
    result = v.bound(v.pointer(folder/'result.json'))
    v.require(json.loads((folder/'completed.json').read_bytes()) == dict(result_sha256=v.file_sha(folder/'result.json')),
        'old result seal differs')
    ref, encoded = read_initial_source(plan, job, folder, result, oldplan)
    old = PreviousProgram.from_dict(encoded, plan['actor_role_sha256'][job['role']])
    v.require(old.maximum_leaves == job['capacity'], 'old capacity differs')
    groups = oldplan['folds'][job['fold']]['inner']; cases = {c['case_id']:c for c in oldplan['cases']}
    v.require(old.report['split_episode_ids'] == {s:sorted(ids) for s,ids in groups.items()}
        and old.report['split_scenario_ids'] == {s:sorted(str(cases[i]['base']) for i in ids) for s,ids in groups.items()},
        'v628 initial did not use exactly the inner episode/scenario partitions')
    p = Program(**deepcopy_fields(old))
    v.require(p.certificates() == old.certificates(), 'new logical proof changes an inherited leaf legal set')
    p.validate()
    return p, ref, oldplan


def read_initial_source(plan, job, folder, result, oldplan):
    """Deterministic file-then-CAS fallback; never pick a source by its score.

    A final timer failure may leave a valid committed best without search-best.
    This reads that immutable snapshot without resuming a dirty/terminal job.
    Full terminal debits are audited separately from the committed prefix.
    """
    from scripts.audit_warehouse_v628 import ledger
    result_ref = v.pointer(folder/'result.json'); completion = v.pointer(folder/'completed.json')
    if (folder/'search-best.json').is_file():
        program = v.pointer(folder/'search-best.json')
        return dict(kind='v628_sealed_inner_search_file', program=program, result=result_ref, completion=completion), v.bound(program)
    v.require(result['status'] == 'scientific_budget_exhausted', 'missing v628 program without an admissible terminal budget outcome')
    counts = ledger(folder, result)
    cp = folder/'checkpoint.json'; raw = cp.read_bytes(); manifest = json.loads(raw)
    old_job = next(j for j in oldplan['jobs'] if j['job_id'] == folder.name)
    expected = v.digest(dict(plan_sha256=plan['previous_plan']['sha256'], job=old_job,
        solver_limits=previous.recipe()['limits'], budget_contract='v628-common-max256-reserve.v1'))
    v.require(manifest['identity_sha256'] == expected
        and manifest['budget'] == dict(maximum_calls=484096, wall_seconds=6000.), 'terminal v628 checkpoint identity differs')
    root = Path(str(cp)+'.objects'); v.require(root.is_dir(), 'terminal CAS store missing')
    store = ContentStore(root); store.load_registry(manifest['object_pack_registry'])
    state = store.get_json(manifest['state'])
    v.require(state.get('algorithm') == 'protected_region_subtree_refinement_v1', 'not a v628 search snapshot')
    ref = state['best']['program']; encoded = store.get_json(ref)
    # The prefix can precede settled rejected calls; it may never exceed the
    # independently verified complete terminal ledger.
    v.require(all(state['solver_counts'][k] <= counts[k] for k in counts), 'committed solver prefix exceeds terminal ledger')
    return dict(kind='v628_terminal_committed_inner_CAS_best', checkpoint=v.pointer(cp),
        state=manifest['state'], registry=manifest['object_pack_registry'], program=ref,
        result=result_ref, completion=completion, terminal_ledger=v.pointer(folder/'checkpoint.json.ledger.jsonl'),
        old_search_completed=False, old_mandatory_complete=state['machine']['mandatory_complete'],
        old_budget_counts=counts, no_old_search_resumed=True), encoded


def deepcopy_fields(program):
    from copy import deepcopy
    return deepcopy(program.__dict__)


def gate(metrics, search):
    from backend.training.warehouse_v622_program import gate_metrics
    result = gate_metrics(metrics, phase='development')
    result['checks']['mandatory_disjoint_roots_explored'] = search['mandatory_complete']
    result['passed'] = all(result['checks'].values())
    return result


def bounded_search(ctx, begin, seconds=4790):
    """Preserve a safe optimize return if the timer's exit check fires again.

    A signal during initial computation with no returned program still fails;
    this helper cannot synthesize or resurrect a missing candidate.
    """
    returned = None
    try:
        with real_deadline(begin, seconds): returned = optimize(ctx)
    except DeadlineExceeded:
        if returned is None: raise
        returned[1]['hard_timer_exit_recheck'] = True
    return returned


def run_job(job_id):
    path, plan = v.checked_plan(); spec = next(j for j in plan['jobs'] if j['job_id'] == job_id)
    v.require((v.OUTPUT/'dependency-cleared.json').is_file(), 'old batch still owns workers')
    folder = v.OUTPUT/'jobs'/job_id; folder.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic(); budget = None
    try:
        initial, ref, oldplan = initial_program(plan, spec)
        inner = data.load_parts(oldplan, spec['fold'], spec['role'], inner=True)
        begin = time.monotonic()
        write_new(folder/'started.json', dict(job=spec, initial=ref, plan_sha256=v.file_sha(path),
            started_monotonic=begin, search_deadline_monotonic=begin+4800, deadline_monotonic=begin+6000))
        budget = Budget(folder/'checkpoint.json', dict(plan_sha256=v.file_sha(path), job=spec, initial=ref), started_at=begin)
        ctx = Context(inner, initial, budget, guard_bundles=spec['guard_bundles'])
        program, search = bounded_search(ctx, begin)
        search['actual_search_seconds'] = time.monotonic()-begin
        write_new(folder/'search-best.json', program.to_dict()); write_new(folder/'search-report.json', search)
        del ctx, inner
        budget.phase = 'final'; final_start = time.monotonic()
        write_new(folder/'final-started.json', dict(deadline_monotonic=min(begin+6000, final_start+1200)))
        with real_deadline(final_start, min(1190, begin+5990-final_start)):
            parts = data.load_parts(oldplan, spec['fold'], spec['role'])
            program, measurements = complete_program(program, parts, budget)
            write_new(folder/'program.json', program.to_dict())
            reference, reference_metrics = complete_program(initial, parts, budget, refit=False)
            write_new(folder/'reference.json', reference.to_dict())
            budget.checkpoint(dict(stage='completed', program_sha256=v.file_sha(folder/'program.json')), status='completed')
        _, info = structure_info(program.structure, spec['capacity'])
        result = dict(schema='warehouse-v629-result.v1', status='completed', job=spec,
            plan_sha256=v.file_sha(path), initial=ref, program=v.pointer(folder/'program.json'),
            reference=v.pointer(folder/'reference.json'), search_report=v.pointer(folder/'search-report.json'),
            search_best=v.pointer(folder/'search-best.json'), leaf_count=len(program.leaves), depth=info['depth'],
            protected_leaves=len(protected_indices(program)), metrics=measurements, reference_metrics=reference_metrics,
            gate=gate(measurements['validation'], search), solver=budget.summary(),
            elapsed_seconds=time.monotonic()-begin, Actor_updates=0, PPO_joint_steps=0, environment_resets=0,
            pilot_opened=False, engineering_performance='unmeasured_independent_release_gate')
    except (BudgetExhausted, DeadlineExceeded) as error:
        result = dict(schema='warehouse-v629-result.v1', status='scientific_budget_exhausted', job=spec,
            error=str(error), gate=dict(passed=False), solver=budget.summary(), elapsed_seconds=time.monotonic()-begin)
        budget.session.close()
    except BaseException as error:
        result = dict(schema='warehouse-v629-result.v1', status='failed', job=spec, error_type=type(error).__name__,
            error=str(error), gate=dict(passed=False), elapsed_seconds=time.monotonic()-begin)
        if budget is not None:
            result['solver'] = budget.summary(); budget.session.close()
        write_new(folder/'failed.json', result)
    v.require(v.pointer(plan['actor']['path']) == plan['actor'], 'NN weights changed')
    write_new(folder/'result.json', result)
    write_new(folder/'completed.json', dict(result_sha256=v.file_sha(folder/'result.json')))
    print(json.dumps(dict(job=spec, status=result['status'], passed=result['gate']['passed'])), flush=True)
    return result


def execute():
    path, plan = v.checked_plan()
    v.require((v.OUTPUT/'dependency-cleared.json').exists(), 'dependency not cleared')
    control = v.OUTPUT/'execution'; control.mkdir(exist_ok=False)
    v.require(not (v.OUTPUT/'jobs').exists(), 'existing batch cannot be retried')
    pending, active, exits = list(plan['jobs']), [], []
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    def interrupt(signum, frame): raise RuntimeError('v629 supervisor interrupted')
    previous_handler = signal.signal(signal.SIGTERM, interrupt)
    events = (control/'events.jsonl').open('x', buffering=1)
    try:
        while pending or active:
            while pending and len(active) < 3:
                job = pending.pop(0); log = (control/(job['job_id']+'.log')).open('x')
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--job', job['job_id']],
                    cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                active.append((job, process, log)); events.write(json.dumps(dict(event='started', job=job, pid=process.pid))+'\n')
            for item in list(active):
                job, process, log = item; folder = v.OUTPUT/'jobs'/job['job_id']
                if process.poll() is None and (folder/'started.json').exists():
                    start = json.loads((folder/'started.json').read_bytes()); deadline = start['search_deadline_monotonic']
                    if (folder/'final-started.json').exists():
                        deadline = min(start['deadline_monotonic'], json.loads((folder/'final-started.json').read_bytes())['deadline_monotonic'])
                    if time.monotonic() >= deadline: os.killpg(process.pid, signal.SIGTERM)
                code = process.poll()
                if code is not None:
                    process.wait(); log.close(); active.remove(item)
                    row = dict(job=job, pid=process.pid, returncode=code, waited=True); exits.append(row)
                    events.write(json.dumps(dict(event='exited', **row))+'\n')
            if active: time.sleep(.25)
        write_new(control/'completed.json', dict(all_owned_workers_waited=True, jobs=24, exits=exits))
    except BaseException as error:
        stopped = stop_workers(active)
        write_new(control/'failed.json', dict(error_type=type(error).__name__, error=str(error), workers=stopped,
            all_owned_workers_waited=True, automatic_retry=False)); raise
    finally:
        events.close(); signal.signal(signal.SIGTERM, previous_handler)
    return finalize()


def finalize():
    from scripts.audit_warehouse_v629 import audit, report
    _, plan = v.checked_plan(); results = []
    for job in plan['jobs']:
        folder = v.OUTPUT/'jobs'/job['job_id']
        if (folder/'completed.json').exists():
            result = json.loads((folder/'result.json').read_bytes())
            v.require(json.loads((folder/'completed.json').read_bytes()) == dict(result_sha256=v.file_sha(folder/'result.json')),
                'result seal changed')
            v.require(result['job'] == job, 'wrong scientific unit')
        else: result = dict(job=job, status='failed_or_unfinished', gate=dict(passed=False))
        results.append(result)
    integrity = audit(plan, results); write_new(v.OUTPUT/'independent-audit.json', integrity)
    passing = [n for n in (64,128,256) if all(r['status'] == 'completed' and r['gate']['passed']
        for r in results if r['job']['name'] == f'G{n}')]
    settled = all((v.OUTPUT/'jobs'/j['job_id']/'completed.json').exists() for j in plan['jobs'])
    chosen = min(passing) if passing and settled and integrity['integrity_passed'] else None
    summary = dict(schema='warehouse-v629-summary.v1', candidate_id=v.CANDIDATE, all_24_settled=settled,
        selected_capacity=chosen, scientific_pass=chosen is not None,
        independent_audit=v.pointer(v.OUTPUT/'independent-audit.json'),
        next_stage='fixed_asset_fresh_confirmation' if chosen else 'closed_failed_program_precheck',
        engineering_performance='not_yet_independently_measured', actor_updates=0, PPO_joint_steps=0,
        environment_resets=0, sealed_audit_used=False, heldout_used=False, pilot_opened=False, publishable=False)
    write_new(v.OUTPUT/'summary.json', summary); report(results, summary)
    if chosen:
        write_new(v.OUTPUT/'method-lock.json', dict(method=v.recipe()['method'], capacity=chosen,
            programs=[r['program'] for r in results if r['job']['name'] == f'G{chosen}'], summary=v.pointer(v.OUTPUT/'summary.json')))
    else:
        write_new(v.OUTPUT/'candidate.closed.json', dict(status=summary['next_stage'], summary=v.pointer(v.OUTPUT/'summary.json'),
            scientific_pass=False, fresh_confirmation_used=False, pilot_opened=False))
    return summary


def pipeline(wait_pid):
    plan = prepare(wait_pid); print('v629 source pinned; waiting verified v628 PID '+str(wait_pid), flush=True)
    wait_dependency(plan); print('v628 dependency fully settled; starting measured v629 batch', flush=True)
    return execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pipeline', action='store_true'); group.add_argument('--job'); group.add_argument('--finalize', action='store_true')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    try:
        result = run_job(args.job) if args.job else finalize() if args.finalize else pipeline(args.wait_pid)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        if v.OUTPUT.exists() and not (v.OUTPUT/'driver.failed.json').exists():
            write_new(v.OUTPUT/'driver.failed.json', dict(error_type=type(error).__name__, error=str(error), automatic_retry=False))
        raise
