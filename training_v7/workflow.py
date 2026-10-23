"""Finite registration and engineering admission for the new shared method."""
from dataclasses import asdict
import json
from pathlib import Path
from .contracts import BC_STAGES, CANDIDATE, RELEASE, DOMAINS, MethodSpec, digest, file_sha, write_new
from .domains import domain
from .storage import source_files, verify_sources

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/three_domain_v7' / CANDIDATE
HISTORY_ROOTS = tuple(ROOT.parent / p for p in ('XAI', 'XAI-study-v3', 'XAI-v6-integration', 'XAI-v6-recovery-20261015T165508Z'))


def software_identity():
    import sys
    import numpy, scipy, sklearn, torch
    return dict(python=sys.version, executable=sys.executable, numpy=numpy.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, torch=str(torch.__version__), computation='CPU_float32_single_thread')


def registration_at(output):
    output = Path(output); pointer = output/'active-registration.json'
    path = output/'registration.json'
    if pointer.exists():
        ref = json.loads(pointer.read_text()); path = Path(ref['path'])
        if path.parent != output.resolve() or file_sha(path) != ref['sha256']:
            raise ValueError('preflight registration revision is not bound')
    return path, json.loads(path.read_text())


def amend_preflight(output=OUTPUT):
    """Append a seed-authorization-only revision before any scientific claim.

    Existing registrations, snapshots and audit outputs are never overwritten.
    Model, objectives, BC, PPO, program fitting and domain physics must remain
    byte-identical. This is not a way to continue a spent/failed model cohort.
    """
    output = Path(output)
    if any(output.rglob('claim.json')) or any(output.rglob('*.actor.npz')):
        raise ValueError('cannot amend preflight after any NN experiment claim')
    parent, old = registration_at(output)
    current = source_files(ROOT)
    allowed = {'training_v7/workflow.py', 'training_v7/storage.py', 'training_v7/seed_delta.py',
               'training_v7/seed_context.py', 'scripts/run_three_domain_unified.py',
               'training_v7/bridge.py', 'training_v7/collector.py', 'training_v7/reporting.py', 'training_v7/seed_resume.py'}
    changed = {p for p in set(old['source_files']) | set(current) if old['source_files'].get(p) != current.get(p)}
    if 'training_v7/program.py' in changed:
        # Reporting-only correction; search/qualification/gates remain exactly
        # the registered code. In particular, this cannot revise a fitted tree.
        snapshot = ('source' if old.get('preflight_revision', 1) == 1 else
                    f"source-preflight-{old['preflight_revision']:02d}")
        original = (output/snapshot/'training_v7/program.py').read_text()
        corrected = original.replace("selected & (top == data['sampled'])", "selected & agreement & (top == data['sampled'])")
        if corrected != (ROOT/'training_v7/program.py').read_text():
            raise ValueError('preflight cannot change program optimization or proof algorithms')
        allowed.add('training_v7/program.py')
    if 'training_v7/ppo.py' in changed:
        # Accept ONLY the named descriptive counters in the zero-claim v2
        # preflight. Removing these lines must reproduce the frozen engine.
        snapshot = ('source' if old.get('preflight_revision', 1) == 1 else
                    f"source-preflight-{old['preflight_revision']:02d}")
        original = (output/snapshot/'training_v7/ppo.py').read_text()
        expected = original.replace(
            "        self.program_norm = {r.name: 0. for r in model.domain.roles}\n",
            "        self.program_norm = {r.name: 0. for r in model.domain.roles}\n"
            "        self.program_feedback_frames = {r.name: 0 for r in model.domain.roles}\n"
            "        self.program_evaluated_frames = {r.name: 0 for r in model.domain.roles}\n")
        expected = expected.replace("                        if used.any() and ploss.requires_grad:\n",
            "                        self.program_evaluated_frames[n] += len(used)\n"
            "                        self.program_feedback_frames[n] += int(used.sum())\n"
            "                        if used.any() and ploss.requires_grad:\n")
        expected = expected.replace("            accepted_actor_change=self.accepted_actor_change, collector=collector_state, programs=programs))",
            "            program_feedback_frames=self.program_feedback_frames, program_evaluated_frames=self.program_evaluated_frames,\n"
            "            accepted_actor_change=self.accepted_actor_change, collector=collector_state, programs=programs))")
        expected = expected.replace("'program_updates', 'program_norm', 'accepted_actor_change'):",
            "'program_updates', 'program_norm',\n                      'program_feedback_frames', 'program_evaluated_frames', 'accepted_actor_change'):")
        if expected != (ROOT/'training_v7/ppo.py').read_text():
            raise ValueError('preflight cannot change PPO numerical objectives')
        allowed.add('training_v7/ppo.py')
    if changed - allowed or old['method_sha256'] != MethodSpec().sha256:
        raise ValueError('preflight amendment cannot change the scientific method')
    revision = old.get('preflight_revision', 1) + 1
    value = dict(old, source_files=current, preflight_revision=revision, seed_guard_version=2,
        parent_registration=dict(path=str(parent.resolve()), sha256=file_sha(parent)),
        preflight_amendment=dict(reason='final seed delta, post-20k evidence, current-only eligibility partitions and descriptive reporting; model/BC/PPO objectives unchanged',
                                changed_source_files=sorted(changed), NN_experiment_claims=0))
    path = output/f'registration-preflight-{revision:02d}.json'; write_new(path, value)
    for relative, h in current.items():
        target = output/f'source-preflight-{revision:02d}'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f: f.write((ROOT/relative).read_bytes())
        target.chmod(0o444)
        if file_sha(target) != h: raise ValueError('source changed during preflight amendment')
    pointer = output/'active-registration.json'
    if pointer.exists():
        write_new(output/f'active-registration-preflight-{revision-1:02d}.json', json.loads(pointer.read_text()))
        pointer.unlink()
    write_new(pointer, dict(path=str(path.resolve()), sha256=file_sha(path)))
    return value


def register(output=OUTPUT):
    output = Path(output)
    if output.resolve() != OUTPUT.resolve(): raise ValueError('scientific registration uses one canonical candidate directory')
    if output.exists(): raise FileExistsError('candidate folder already exists; preserve its original registration')
    domains = {name: asdict(domain(name).spec) for name in DOMAINS}
    method = MethodSpec()
    historical = ROOT/'output/warehouse_v621/seed-ledger-03/rebound-01/inventory.json'
    registry = json.loads(historical.read_text())
    proposal = dict(schema='warehouse-seed-proposal.v1', candidate_id=CANDIDATE, families=[],
                    registered_trace_paths=registry['registered_trace_paths'])
    plans = {}
    for domain_index, name in enumerate(DOMAINS):
        cases = []; base = 32_000_000_000 + domain_index * 10_000_000
        initialization_seed = 8_200_000_000 + domain_index * 100
        optimizer_seed = 8_200_000_050 + domain_index * 100
        bc_action_seed = 8_200_000_025 + domain_index * 100
        evidence_seed = 8_200_000_030 + domain_index * 100
        action_seed = 8_200_000_040 + domain_index * 100
        critic_seed = 8_200_000_060 + domain_index * 100
        for kind, offset, count in (('bc', 0, 1270), ('program', 100000, 1650), ('calibration_quality', 200000, 75), ('ppo', 300000, 20000)):
            proposal['families'].append(dict(name=f'{name}-{kind}', kind='scenario', start=base+offset,
                end_exclusive=base+offset+count, purpose=kind))
        proposal['families'].append(dict(name=f'{name}-initialization', kind='init', values=[initialization_seed]))
        proposal['families'].append(dict(name=f'{name}-critic', kind='critic', values=[critic_seed]))
        proposal['families'].append(dict(name=f'{name}-training-random-streams', kind='action_rng',
                                        values=[optimizer_seed, bc_action_seed, evidence_seed, action_seed]))
        proposal['families'].append(dict(name=f'{name}-evaluation-actions', kind='action_rng',
            start=base+1_000_000_000_000, end_exclusive=base+1_000_000_000_000+75))
        ordinal = 0
        for stage in BC_STAGES:
            for i in range(stage[1]):
                cases.append(dict(case_id=f'{name}:bc:{ordinal}', seed=base+ordinal, task=1+ordinal%3,
                                  purpose='bc', stage=stage[0]))
                ordinal += 1
        partitions = ('fit', 'calibration', 'eligibility', 'validation')
        ordinal = 0
        for partition, count in zip(partitions, method.evidence_counts, strict=True):
            for i in range(count):
                cases.append(dict(case_id=f'{name}:program:{ordinal}', seed=base+100000+ordinal,
                    task=1+ordinal%3, purpose='program', partition=partition))
                ordinal += 1
        for i in range(75):
            for policy in ('reference', 'raw', 'regularized'):
                for pairing in (('nn_nn', 'offline_human_proxy') if name == 'kitchen' else ('nn_nn',)):
                    cases.append(dict(case_id=f'{name}:calibration_quality:{i}:{policy}:{pairing}', seed=base+200000+i,
                        action_seed=base+1_000_000_000_000+i, task=1+i%3, purpose='calibration_quality',
                        policy=policy, pairing=pairing))
        horizon = domains[name]['horizon']
        for block in range((method.calibration_steps + 8*horizon-1)//(8*horizon)):
            for slot in range(8):
                ordinal = block*8+slot
                cases.append(dict(case_id=f'{name}:ppo:{ordinal}', seed=base+300000+ordinal, task=1+ordinal%3,
                    purpose='ppo', slot=slot, block=block, maximum_resets=horizon))
        plans[name] = dict(version='three-domain-unified-calibration-plan.v1', candidate_id=CANDIDATE,
            domain=name, domain_sha256=digest(domains[name]), method_sha256=method.sha256,
            initialization_seed=initialization_seed, optimizer_seed=optimizer_seed,
            bc_action_seed=bc_action_seed, evidence_seed=evidence_seed, action_seed=action_seed, critic_seed=critic_seed,
            pair_id=0, phase='calibration', cases=cases, sealed_authority=False, pilot_opened=False)
    registration = dict(version='three-domain-unified-registration.v1', candidate_id=CANDIDATE, release_id=RELEASE,
        method=asdict(method), method_sha256=method.sha256, domains=domains,
        calibration_plans=plans, source_files=source_files(ROOT), software=software_identity(), seed_guard_version=2,
        model_acceptance_passed=False, pilot_opened=False,
        old_acceptance_receipts_authorize_new_method=False)
    # Shared method identity is one hash, not three editable per-domain copies.
    if len({p['method_sha256'] for p in plans.values()}) != 1:
        raise ValueError('unregistered method differences')
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/'registration.json', registration)
    write_new(output/'seed-proposal.json', proposal)
    for relative, expected in registration['source_files'].items():
        target = output/'source'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle: handle.write((ROOT/relative).read_bytes())
        target.chmod(0o444)
        if file_sha(target) != expected: raise ValueError('source changed while creating immutable snapshot')
    return registration


def seed_audit(output=OUTPUT, *, previous_complete=None):
    """Use the existing parameterized inert metadata auditors on all four roots.

    No model, NN, environment, sample or reset is constructed by this command.
    A conflict, opaque metadata or missing root prevents clearance.
    """
    from .seed_audit import audit, SCOPE
    from experiments.warehouse.seed_triage_indexed_v642 import triage
    output = Path(output)
    proposal = output/'seed-proposal.json'
    inventory = audit(proposal, HISTORY_ROOTS, output/'seed-inventory')
    if previous_complete is None:
        result = triage(output/'seed-inventory/inventory.json', proposal, output/'seed-triage')
    else:
        from .seed_resume import resume
        result = resume(output,previous_complete)
    if (result['selection_scope'] != SCOPE or result['selection_adapter']['sha256'] != file_sha(ROOT/'training_v7/seed_audit.py')):
        raise ValueError('old Warehouse-only seed inventory cannot authorize the v7 method')
    summary = result['summary']
    passed = (not summary['unresolved_paths'] and not result.get('missing_roots') and
        all(not r['references_or_reservations'] and not r['reservation_overlaps']
            and not r['excluded_weight_path_seed_hints'] and not r['audit_references_not_original_execution']
            for r in summary['families'].values()))
    triage_path=output/('seed-triage/result.json' if previous_complete is None else 'seed-triage-resumed/result.json')
    evidence = [dict(path=str(p.resolve()), sha256=file_sha(p)) for p in
                (output/'seed-inventory/inventory.json', triage_path)]
    registration = json.loads((output/'registration.json').read_text())
    report = dict(version='three-domain-unified-seed-audit-summary.v1', passed=passed,
        unresolved_paths=summary['unresolved_paths'], families=summary['families'], evidence=evidence,
        environment_resets=0, Actor_updates=0, pilot_opened=False)
    write_new(output/'seed-audit-summary.json', report)
    if passed:
        return finalize_seeds(output)
    return report


def finalize_seeds(output=OUTPUT):
    from .seed_delta import verify
    from .seed_context import reconcile
    output = Path(output); registration_path, registration = registration_at(output)
    verify_sources(ROOT, registration['source_files'])
    if registration.get('seed_guard_version') != 2: raise ValueError('final-delta preflight registration required')
    context_path = output/'seed-context-result.json'
    if not context_path.exists():
        triage_path = output/'seed-triage-resumed/result.json'
        if not triage_path.exists(): triage_path=output/'seed-triage/result.json'
        reconcile(output/'seed-proposal.json', triage_path, context_path)
    delta = verify(output/'seed-proposal.json', context_path, output/'seed-final-delta.json')
    if not delta['passed']: return delta
    triage_path=output/'seed-triage-resumed/result.json'
    if not triage_path.exists():triage_path=output/'seed-triage/result.json'
    refs = [dict(path=str(p.resolve()),sha256=file_sha(p)) for p in (
        output/'seed-inventory/inventory.json',triage_path,output/'seed-context-result.json',output/'seed-final-delta.json')]
    for name, plan in registration['calibration_plans'].items():
        write_new(output/f'{name}-clearance-final.json', dict(version='three-domain-unified-seed-clearance.v2',
            plan_sha256=digest(plan), passed=True, evidence=refs, final_delta=refs[-1],
            registration=dict(path=str(registration_path.resolve()), sha256=file_sha(registration_path)),
            source_proposal_sha256=file_sha(output/'seed-proposal.json'),
            scope='declared_metadata_roots_not_global_freshness', sealed_authority=False, pilot_opened=False))
    return delta


def run_bc(name, output=OUTPUT):
    import torch
    from .bc import train_bc
    from .model import DualActorCritic
    from .storage import Run
    output = Path(output)
    _, registration = registration_at(output)
    verify_sources(ROOT, registration['source_files'])
    if registration['software'] != software_identity(): raise ValueError('numerical dependency versions changed')
    if registration['method_sha256'] != MethodSpec().sha256: raise ValueError('method registration differs')
    adapter = domain(name); plan = registration['calibration_plans'][name]
    if adapter.spec.sha256 != plan['domain_sha256']: raise ValueError('domain physical contract differs')
    run = Run(output/name/'calibration-0', plan, output/f'{name}-clearance-final.json')
    torch.set_num_threads(1)
    torch.manual_seed(plan['initialization_seed'])
    model = DualActorCritic(adapter.spec)
    result = train_bc(adapter, model, run, initialization_seed=plan['initialization_seed'], shuffle_seed=plan['optimizer_seed'], action_seed=plan['bc_action_seed'])
    verify_sources(ROOT, registration['source_files'])
    run.finish()
    return result


def run_calibration(name, output=OUTPUT):
    """BC, independent evidence, both actual 20k models, then fresh evaluation.

    Any precheck failure closes this shared method; no subsequent stage can
    interpret a partially completed domain as admission.
    """
    from copy import deepcopy
    import numpy as np
    import torch
    from .bc import train_bc
    from .bridge import Evidence, ProgramProvider
    from .collector import train_pair_variant
    from .model import DualActorCritic
    from .numerical import verify_export
    from .program import Program, metrics
    from .ppo import PPO
    from .quality_run import collect_quality, audit_quality
    from .runtime import Binding
    from .storage import Run, export_actor
    output = Path(output)
    if (output/'CLOSED.json').exists(): raise ValueError('shared candidate is closed')
    _, registration = registration_at(output)
    verify_sources(ROOT, registration['source_files'])
    if registration['software'] != software_identity(): raise ValueError('numerical dependency versions changed')
    adapter = domain(name); plan = registration['calibration_plans'][name]
    if plan['method_sha256'] != MethodSpec().sha256 or plan['domain_sha256'] != adapter.spec.sha256:
        raise ValueError('shared method or domain contract differs')
    run = Run(output/name/'calibration-0', plan, output/f'{name}-clearance-final.json')
    torch.set_num_threads(1); torch.manual_seed(plan['initialization_seed'])
    bc_model = DualActorCritic(adapter.spec)
    try:
        bc = train_bc(adapter, bc_model, run, initialization_seed=plan['initialization_seed'], shuffle_seed=plan['optimizer_seed'], action_seed=plan['bc_action_seed'])
        evidence = Evidence.collect(adapter, bc_model, run, seed=plan['evidence_seed'])
        verify_export(bc_model, evidence, bc['actor'], run.folder/'bc-export-numerical.json')
        provider = ProgramProvider.initial(evidence, bc_model, run.folder/'initial-programs', run.journal)
        # Exactly 300 distinct, previously collected NN-visitation episodes.
        files = sorted((run.folder/'bc').glob('*.npz'))
        selected = [p for p in files if not p.name.endswith('actor.npz') and 'public_teacher-' not in p.name][:300]
        if len(selected) != 300: raise ValueError('missing 300 training-only reference buffer episodes')
        reference_rows = {}
        for role in adapter.spec.roles:
            parts = {k: [] for k in ('x', 'mask', 'action_features')}
            for path in selected:
                with np.load(path, allow_pickle=False) as data:
                    for key, suffix in (('x', 'x'), ('mask', 'mask'), ('action_features', 'af')):
                        parts[key].append(data[role.name+'_'+suffix])
            reference_rows[role.name] = {k: np.concatenate(v) for k, v in parts.items()}
            for value in reference_rows[role.name].values(): value.flags.writeable = False
        write_new(run.folder/'reference-buffer.json', dict(episodes=300, partition='bc_train_only',
            files=[dict(path=str(p.resolve()), sha256=file_sha(p)) for p in selected]))
        torch.manual_seed(plan['critic_seed']); initial = DualActorCritic(adapter.spec)
        initial.actors.load_state_dict(bc_model.actors.state_dict())
        initial_hashes = initial.role_hashes()
        results, assets, programs = {}, {'reference': bc['actor']}, {}
        for variant in ('raw', 'regularized'):
            if (output/'CLOSED.json').exists(): raise ValueError('another domain closed the shared candidate')
            owner = Run(run.folder/variant, plan, output/f'{name}-clearance-final.json')
            engine = PPO(deepcopy(initial), optimizer_seed=plan['optimizer_seed'], reference_rows=reference_rows)
            cloned = {r.name: Program.from_dict(r, provider.programs[r.name].as_dict(), actor_sha256=initial_hashes[r.name]) for r in adapter.spec.roles}
            live_provider = ProgramProvider(evidence, cloned, owner.journal)
            results[variant] = train_pair_variant(adapter, engine, owner, live_provider, variant=variant,
                budget=20000, action_seed=plan['action_seed'], sources=registration['source_files'])
            assets[variant] = export_actor(engine.model, owner.folder/f'{variant}-final.actor.npz')
            verify_export(engine.model, evidence, assets[variant], owner.folder/'final-export-numerical.json')
            programs[variant] = live_provider.programs
        weights_distinct = all(assets['raw']['role_sha256'][r.name] != assets['regularized']['role_sha256'][r.name] for r in adapter.spec.roles)
        fresh = collect_quality(adapter, run, assets, programs, output=run.folder/'fresh-20k-quality')
        replay = audit_quality(adapter, run.folder/'fresh-20k-quality/result.json')
        # Recompute all four program results using independent stored arrays and
        # freshly verified certificates; do not trust serialized passed flags.
        actual_programs = {}
        for variant in ('raw', 'regularized'):
            actual_programs[variant] = {}
            for role in adapter.spec.roles:
                program = Program.from_dict(role, programs[variant][role.name].as_dict(), actor_sha256=assets[variant]['role_sha256'][role.name])
                with np.load(run.folder/'fresh-20k-quality'/f'{variant}-{role.name}-fresh-program.npz', allow_pickle=False) as data:
                    report = metrics(program, {k: data[k] for k in data.files})
                if report != fresh['program_metrics'][variant][role.name]: raise ValueError('fresh program metric replay differs')
                actual_programs[variant][role.name] = report
        evidence.verify_static()
        from .reporting import calibration_report
        calibration_report(adapter, run.folder, results, programs)
        verify_sources(ROOT, registration['source_files'])
        run.finish()
        training_passed = weights_distinct and all(all(r['training_gates'].values()) for r in results.values())
        passed = training_passed and replay['task_quality_passed'] and all(m['passed'] for p in actual_programs.values() for m in p.values())
        final = dict(version='three-domain-unified-20k-calibration-replay.v1', domain=name, passed=passed,
            training=results, training_gates_passed=training_passed, raw_regularized_role_weights_distinct=weights_distinct,
            initial_role_sha256=initial_hashes, quality=replay,
            program_metrics=actual_programs, method_sha256=MethodSpec().sha256,
            descriptive_report=dict(path=str((run.folder/'post-20k-metrics-and-intervals.json').resolve()),
                                    sha256=file_sha(run.folder/'post-20k-metrics-and-intervals.json')),
            model_acceptance_passed=False, formal_training_started=False, pilot_opened=False)
        write_new(run.folder/'calibration-result.json', final)
        if not passed: raise ValueError('fresh post-20k calibration failed')
        return final
    except BaseException as error:
        failure = dict(version='three-domain-unified-candidate-closed.v1', domain=name,
            exception=type(error).__name__, reason=str(error), method_sha256=MethodSpec().sha256,
            model_acceptance_passed=False, pilot_opened=False)
        try: write_new(output/'CLOSED.json', failure)
        except FileExistsError: pass  # Another domain owns the first immutable failure.
        write_new(run.folder/'failure.json', failure)
        raise
