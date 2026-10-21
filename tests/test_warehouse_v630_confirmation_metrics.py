"""Selection/immutability fixtures, not fresh Warehouse confirmation evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json

import numpy as np
import pytest

from experiments.warehouse import v630_confirmation_metrics as c
from scripts.run_warehouse_v630 import gate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return c.v.pointer(path)


def metric(passed=True):
    return {purpose: dict(coverage=.5 if passed else .4, agreement=.99, kl=.05,
        expected_actionable_coverage=.45, illegal_count=0, path_uncertified_count=0,
        whole_tree_raw_illegal_count=0, whole_tree_path_uncertified_count=0)
        for purpose in ('feedback', 'explanation')}


@pytest.fixture
def settled(tmp_path, monkeypatch):
    """Fake serialized programs isolate the final gate/asset plumbing."""
    output, old_output = tmp_path / 'candidate', tmp_path / 'previous'
    monkeypatch.setattr(c.v, 'OUTPUT', output)
    monkeypatch.setattr(c.v.previous, 'OUTPUT', old_output)
    parent = write(tmp_path / 'parent.json', dict(synthetic=True))
    actor = write(tmp_path / 'actor.json', dict(synthetic_not_NN=True))
    roles = {r: r + '-test-hash' for r in c.ROLES}
    old_jobs = [dict(job_id=f'G128-fold-{f}-{r}', fold=f, role=r, name='G128')
                for f in range(3) for r in c.ROLES]
    old_plan = write(tmp_path / 'old-plan.json', dict(jobs=old_jobs))
    quality = dict(sampled=dict(deliveries=15., collisions=3., shutdown_rate=.02))
    quality_ref = write(tmp_path / 'quality.json', quality)
    jobs = c.v.jobs()
    plan = dict(jobs=jobs, actor=actor, actor_role_sha256=roles,
                previous_plan=old_plan, common_quality=quality_ref)
    plan_path = output / 'plan.json'
    plan_ref = write(plan_path, plan)
    old_audit_rows = []
    for j in old_jobs:
        ref = write(old_output / 'jobs' / j['job_id'] / 'reference.json',
                    dict(role=j['role'], actor_role_sha256=roles[j['role']],
                         maximum_leaves=128, protection=dict(parent=parent)))
        result = dict(job=j, status='completed', plan_sha256=old_plan['sha256'], reference=ref)
        r = write(old_output / 'jobs' / j['job_id'] / 'result.json', result)
        write(old_output / 'jobs' / j['job_id'] / 'completed.json', dict(result_sha256=r['sha256']))
        old_audit_rows.append(dict(job=j, status='completed', independent_reference_metrics=True))
    old_audit = write(tmp_path / 'old-audit.json', dict(jobs=old_audit_rows))
    old_summary = write(tmp_path / 'old-summary.json', dict(scientific_pass=False))
    completion = dict(summary=old_summary, independent_audit=old_audit,
                      skip_new_candidate=False, terminal_assets=[])
    write(output / 'dependency-cleared.json', dict(completion=completion, sources={}))
    records, checked = [], []
    for j in jobs:
        folder = output / 'jobs' / j['job_id']
        p = write(folder / 'program.json', dict(role=j['role'], actor_role_sha256=roles[j['role']],
                    maximum_leaves=256, protection=dict(parent=parent)))
        reference = write(folder / 'reference.json', dict(mechanism_control_not_primary=True))
        search = write(folder / 'search-program.json', dict(synthetic=True))
        report = write(folder / 'search-report.json', dict(mandatory_complete=True, reliable_selection_passed=True))
        m = metric()
        result = dict(job=j, status='completed', plan=plan_ref, program=p, reference=reference,
                      search_report=report, search_program=search, metrics=dict(validation=m),
                      gate=gate(m, c.v.bound(report), j['scope']))
        ref = write(folder / 'result.json', result)
        write(folder / 'completed.json', dict(result_sha256=ref['sha256']))
        records.append(result)
        checked.append(dict(job=j, status='completed', program=p, all_old_distributions_preserved=True,
                            independent_numeric_and_qualification_replay=True))
    audit = write(output / 'independent-audit.json', dict(integrity_passed=True, jobs=checked, common_quality=quality))
    summary = dict(candidate_id=c.v.CANDIDATE, scientific_pass=True, all_18_settled=True, all_18_completed=True,
                   selected_method='P', independent_audit=audit, common_quality=quality_ref,
                   common_frozen_NN_development_quality_passed=True)
    summary_ref = write(output / 'summary.json', summary)
    write(output / 'method-lock.json', dict(method=c.v.recipe()['method'], arm='P',
          programs=[r['program'] for r in records if r['job']['name']=='P'], summary=summary_ref))
    execution = dict(all_owned_workers_waited=True, jobs=18,
                     exits=[dict(job=j, waited=True, returncode=0) for j in jobs])
    write(output / 'execution/completed.json', execution)
    monkeypatch.setattr(c.v, 'checked_plan', lambda: (plan_path, plan))
    monkeypatch.setattr(c.v, 'old_completion', lambda: completion)
    monkeypatch.setattr(c.v, 'common_quality', lambda p: quality)
    class FakeReference:
        @classmethod
        def from_dict(cls, a, sha):
            return cls(a)
        def __init__(self, a):
            self.a=deepcopy(a)
            for key, val in a.items():setattr(self, key, val)
        def to_dict(self):return deepcopy(self.a)
    class FakeProgram(FakeReference):pass
    monkeypatch.setattr(c, 'ReferenceProgram', FakeReference)
    monkeypatch.setattr(c, 'Program', FakeProgram)
    return SimpleNamespace(output=output, old_output=old_output, plan=plan, records=records,
                           summary=summary, audit=c.v.bound(audit), execution=execution)


def test_missing_real_winner_rejects_before_new_data_access(tmp_path, monkeypatch):
    monkeypatch.setattr(c.v, 'OUTPUT', tmp_path)
    with pytest.raises(RuntimeError, match='no complete locked audited winner'):
        c.load_locked_bundle()
    assert list(tmp_path.iterdir()) == []


def test_original_v628_baseline_is_kept_instead_of_unsplit_v630_control(settled):
    bundle = c.load_locked_bundle()
    assert len(bundle.units) == 6 and bundle.selected_method == 'P'
    assert all('/previous/jobs/G128-' in u.reference_ref['path'] for u in bundle.units)
    assert all(u.program.maximum_leaves == 256 and u.reference.maximum_leaves == 128 for u in bundle.units)
    assert all('/candidate/jobs/' in u.program_ref['path'] for u in bundle.units)
    assert bundle.verify() is bundle


@pytest.mark.parametrize('fault', ['incomplete', 'audit', 'reaped', 'gate', 'wrong_method', 'wrong_plan'])
def test_stored_pass_flag_cannot_bypass_full_selection_evidence(settled, fault):
    x=settled
    if fault=='incomplete':
        x.summary['all_18_completed']=False;write(x.output/'summary.json',x.summary)
    elif fault=='audit':
        x.audit['jobs'][0]['independent_numeric_and_qualification_replay']=False
        ref=write(x.output/'independent-audit.json',x.audit)
        x.summary['independent_audit']=ref;write(x.output/'summary.json',x.summary)
    elif fault=='reaped':
        x.execution['exits'][0]['waited']=False;write(x.output/'execution/completed.json',x.execution)
    elif fault in ('gate','wrong_plan'):
        r=deepcopy(x.records[0])
        if fault=='gate':r['metrics']['validation']=metric(False)
        else:r['plan']=dict(path='bad',sha256='0'*64)
        folder=x.output/'jobs'/r['job']['job_id'];ref=write(folder/'result.json',r)
        write(folder/'completed.json',dict(result_sha256=ref['sha256']))
    else:
        lock=c.v.bound(c.v.pointer(x.output/'method-lock.json'));lock['arm']='A'
        write(x.output/'method-lock.json',lock)
    with pytest.raises(RuntimeError):c.load_locked_bundle()


def test_post_lock_model_mutation_and_primary_baseline_replacement_are_rejected(settled):
    bundle=c.load_locked_bundle()
    bundle.units[0].program.a['extra']='silent qualifier change'
    with pytest.raises(RuntimeError,match='attempted to change'):bundle.verify()
    bundle=c.load_locked_bundle()
    write(Path(bundle.units[0].reference_ref['path']),dict(replaced=True))
    with pytest.raises(RuntimeError,match='changed'):bundle.verify()


def test_A_can_be_selected_only_after_P_has_a_real_failed_unit(settled):
    x=settled
    r=deepcopy(next(r for r in x.records if r['job']['name']=='P'))
    r['metrics']['validation']=metric(False)
    r['gate']=gate(r['metrics']['validation'],c.v.bound(r['search_report']),r['job']['scope'])
    folder=x.output/'jobs'/r['job']['job_id'];ref=write(folder/'result.json',r)
    write(folder/'completed.json',dict(result_sha256=ref['sha256']))
    x.summary['selected_method']='A';summary_ref=write(x.output/'summary.json',x.summary)
    write(x.output/'method-lock.json',dict(method=c.v.recipe()['method'],arm='A',
          programs=[r['program'] for r in x.records if r['job']['name']=='A'],summary=summary_ref))
    bundle=c.load_locked_bundle()
    assert bundle.selected_method=='A'
    assert all('/jobs/A-fold-' in u.program_ref['path'] for u in bundle.units)


def test_unselected_control_asset_is_still_bound_to_the_complete_audit(settled):
    bundle=c.load_locked_bundle()
    control=next(r for r in settled.records if r['job']['name']=='C0')
    write(Path(control['program']['path']),dict(tampered=True))
    with pytest.raises(RuntimeError,match='changed'):bundle.verify()


def test_wrong_model_anchor_and_old_bundle_cannot_be_used_as_new_confirmation(settled):
    bundle=c.load_locked_bundle();bundle.units[0].program.actor_role_sha256='other-model'
    with pytest.raises(RuntimeError,match='NN anchor'):bundle.verify()
    with pytest.raises(RuntimeError,match='own fixed asset binding'):
        c.evaluate_fixed(object(),{},[],[])


def test_native_new_asset_uses_unchanged_arithmetic_without_requalification(tmp_path):
    """Real program readers and forward; synthetic distributions, zero resets."""
    spec=importlib.util.spec_from_file_location('v630_confirmation_synthetic_helper',
            Path(__file__).with_name('test_warehouse_v629.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from backend.training.warehouse_v628_program import qualify
    from backend.training.warehouse_v630_program import from_previous
    from backend.training.warehouse_v630_partition_search import qualify_current
    source, parts=helper.fixture(tmp_path)
    source.maximum_leaves=128;qualify(source,parts)
    model=from_previous(source);model.maximum_leaves=256;qualify_current(model,parts)
    p=c.Program.from_dict(model.to_dict(),model.actor_role_sha256)
    x=parts['validation'];before=c.v.digest(p.to_dict())
    actual=c.arithmetic.measure(p,x)
    predicted=c.arithmetic.forward(p,x['x'],x['m'])
    for purpose,selected in [('feedback',p.feedback_leaves),('explanation',p.explanation_leaves)]:
        c.arithmetic.same_metrics(c.arithmetic.basic(p,x,predicted,selected,purpose),actual[purpose])
    ep=c.arithmetic.episode_endpoints(p,x,predicted)
    assert ep and c.v.digest(p.to_dict())==before
    assert all(np.array_equal(a,b) for a,b in zip(predicted,p.distributions(x['x'],x['m']),strict=True))
