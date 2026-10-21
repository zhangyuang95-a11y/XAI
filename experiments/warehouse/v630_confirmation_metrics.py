"""Fixed v630 asset selection and confirmation scoring, with no collection API.

The primary comparator remains the corresponding v628 search prefix used by
v629. v630's unsplit G128 source is a separate mechanism control, not a new
primary baseline. This module does not allocate seeds or authorize training.
"""
from dataclasses import dataclass
from copy import deepcopy

from . import v630_protocol as v
from . import v629_confirmation_metrics as arithmetic
from backend.training.warehouse_v630_program import Program
from backend.training.warehouse_v629_program import Program as ReferenceProgram

ROLES = arithmetic.ROLES
BASELINE = 'fixed_corresponding_v628_search_prefix_outer_qualified_in_v629_G128'


@dataclass(frozen=True)
class Bundle:
    units: tuple
    selected_method: str
    actor: dict
    actor_role_sha256: dict
    provenance: dict
    asset_refs: tuple

    def verify(self):
        expected = {(r, f) for r in ROLES for f in range(3)}
        v.require(self.selected_method in ('P', 'A') and len(self.units) == 6
                  and {(u.role, u.fold) for u in self.units} == expected,
                  'confirmation needs one locked method and all six role/fold units')
        v.require(set(self.actor_role_sha256) == set(ROLES), 'missing dual Actor hashes')
        refs = (self.actor, *self.provenance.values(), *self.asset_refs)
        v.require(all(v.pointer(ref['path']) == ref for ref in refs),
                  'fixed confirmation asset or provenance changed')
        for unit in self.units:
            v.require(isinstance(unit.program, Program)
                      and isinstance(unit.reference, ReferenceProgram)
                      and not isinstance(unit.reference, Program),
                      'primary baseline must retain the original v629 asset language')
            v.require(unit.program.role == unit.reference.role == unit.role
                      and unit.program.actor_role_sha256 == unit.reference.actor_role_sha256
                      == self.actor_role_sha256[unit.role]
                      and unit.program.maximum_leaves == 256
                      and unit.reference.maximum_leaves == 128,
                      'fixed role, NN anchor or registered capacities differ')
            v.require(v.pointer(unit.program_ref['path']) == unit.program_ref
                      and v.pointer(unit.reference_ref['path']) == unit.reference_ref,
                      'fixed program or primary comparator changed')
            v.require(v.digest(unit.program.to_dict()) == unit.program_digest
                      and v.digest(unit.reference.to_dict()) == unit.reference_digest,
                      'confirmation attempted to change program or qualification')
        return self


def load_locked_bundle():
    """Reject partial/failed development before any new data or RNG exists."""
    for name in ('summary.json', 'method-lock.json', 'independent-audit.json',
                 'execution/completed.json', 'dependency-cleared.json'):
        v.require((v.OUTPUT / name).is_file(),
                  'v630 has no complete locked audited winner; confirmation remains closed')
    path, plan = v.checked_plan()
    refs = {key: v.pointer(v.OUTPUT / name) for key, name in (
        ('plan', 'plan.json'), ('summary', 'summary.json'), ('lock', 'method-lock.json'),
        ('independent_audit', 'independent-audit.json'),
        ('execution', 'execution/completed.json'), ('dependency', 'dependency-cleared.json'))}
    summary, lock = v.bound(refs['summary']), v.bound(refs['lock'])
    audit, execution = v.bound(refs['independent_audit']), v.bound(refs['execution'])
    dependency = v.bound(refs['dependency'])
    v.require(summary['candidate_id'] == v.CANDIDATE
              and summary['scientific_pass'] is True and summary['all_18_settled'] is True
              and summary['all_18_completed'] is True
              and summary['selected_method'] in ('P', 'A')
              and summary['independent_audit'] == refs['independent_audit']
              and audit['integrity_passed'] is True
              and [row['job'] for row in audit['jobs']] == plan['jobs'],
              'partial, failed or incompletely audited candidate cannot confirm')
    v.require(execution['all_owned_workers_waited'] is True and execution['jobs'] == 18
              and len(execution['exits']) == 18
              and {row['job']['job_id'] for row in execution['exits']}
              == {job['job_id'] for job in plan['jobs']}
              and all(row['waited'] is True and row['returncode'] == 0
                      for row in execution['exits']),
              'all 18 original workers must be settled and actually reaped')
    v.require(dependency['completion'] == v.old_completion()
              and dependency['completion']['skip_new_candidate'] is False,
              'original v629 closure or upstream integrity changed')
    quality = v.common_quality(plan)
    v.require(summary['common_quality'] == plan['common_quality']
              and audit['common_quality'] == quality
              and summary['common_frozen_NN_development_quality_passed'] is True
              and all(m['deliveries'] >= 5.5 and m['collisions'] <= 6
                      and m['shutdown_rate'] <= .05 for m in quality.values()),
              'frozen NN task quality did not pass from its actual outcomes')
    from scripts.run_warehouse_v630 import gate
    results, assets = [], list(dependency['completion']['terminal_assets'])
    for binding in dependency['sources'].values():
        assets.extend(binding[key] for key in ('program', 'terminal', 'completed'))
    for job, checked in zip(plan['jobs'], audit['jobs'], strict=True):
        folder = v.OUTPUT / 'jobs' / job['job_id']
        result_ref, seal = v.pointer(folder / 'result.json'), v.pointer(folder / 'completed.json')
        result = v.bound(result_ref)
        v.require(v.bound(seal) == dict(result_sha256=result_ref['sha256'])
                  and result['job'] == job and result['plan'] == refs['plan']
                  and result['status'] == checked['status'] == 'completed',
                  'all 18 unit result identities and completion seals are required')
        v.require(checked['program'] == result['program']
                  and checked['all_old_distributions_preserved'] is True
                  and checked['independent_numeric_and_qualification_replay'] is True,
                  'unit audit lacks exact distribution or independent qualification replay')
        report = v.bound(result['search_report'])
        v.require(result['gate'] == gate(result['metrics']['validation'], report, job['scope']),
                  'stored strict gate differs from measured metrics')
        assets.extend((result_ref, seal, result['program'], result['reference'],
                       result['search_report'], result['search_program']))
        if 'option_tables' in report:
            assets.append(report['option_tables'])
        results.append(result)
    passing = [arm for arm in ('P', 'A') if all(r['gate']['passed']
               for r in results if r['job']['name'] == arm)]
    v.require(passing and summary['selected_method'] == passing[0]
              and lock == dict(method=v.recipe()['method'], arm=passing[0],
                  programs=[r['program'] for r in results if r['job']['name'] == passing[0]],
                  summary=refs['summary']),
              'selection must use first all-six passing P then A, without role mixing')
    old_plan_ref = plan['previous_plan']
    refs['v629_plan'] = old_plan_ref
    old_plan = v.bound(old_plan_ref)
    old_summary_ref = dependency['completion']['summary']
    old_audit_ref = dependency['completion']['independent_audit']
    refs.update(v629_summary=old_summary_ref, v629_independent_audit=old_audit_ref)
    old_audit = v.bound(old_audit_ref)
    units = []
    for result in results:
        job = result['job']
        if job['name'] != passing[0]:
            continue
        role, fold = job['role'], job['fold']
        old_folder = v.previous.OUTPUT / 'jobs' / f'G128-fold-{fold}-{role}'
        old_ref = v.pointer(old_folder / 'result.json')
        old = v.bound(old_ref)
        old_job = next(j for j in old_plan['jobs'] if j['job_id'] == old_folder.name)
        old_checked = next(row for row in old_audit['jobs'] if row['job'] == old_job)
        v.require(v.bound(v.pointer(old_folder / 'completed.json'))
                  == dict(result_sha256=old_ref['sha256'])
                  and old['job'] == old_job and old['status'] == old_checked['status'] == 'completed'
                  and old['plan_sha256'] == old_plan_ref['sha256']
                  and old_checked['independent_reference_metrics'] is True,
                  'primary v628 comparator lacks the original audited v629 reference')
        # This is the original v628 starting program, already outer-qualified
        # by v629. Do not replace it with v630's stronger unsplit G128 control.
        program = Program.from_dict(v.bound(result['program']), plan['actor_role_sha256'][role])
        reference = ReferenceProgram.from_dict(v.bound(old['reference']), plan['actor_role_sha256'][role])
        units.append(arithmetic.Unit(role, fold, program, reference,
            result['program'], old['reference'], v.digest(program.to_dict()), v.digest(reference.to_dict())))
        assets.extend((old_ref, v.pointer(old_folder / 'completed.json'), old['reference']))
        assets.extend((program.protection['parent'], reference.protection['parent']))
    return Bundle(tuple(units), passing[0], plan['actor'], deepcopy(plan['actor_role_sha256']),
                  refs, tuple(assets)).verify()


def evaluate_fixed(bundle, datasets, cases, outcomes):
    """Use unchanged gates and scene-paired endpoints; never refit or requalify."""
    v.require(isinstance(bundle, Bundle), 'v630 confirmation requires its own fixed asset binding')
    result = arithmetic.evaluate_fixed(bundle, datasets, cases, outcomes)
    result.update(schema='warehouse-v630-fixed-confirmation-metrics.v1',
                  selected_method=bundle.selected_method, primary_comparator=BASELINE,
                  mechanism_control='v630_C0_separate_development_ablation_not_primary_baseline',
                  program_distribution_changing_operations_in_confirmation=0,
                  NN_task_improvement_claimed=False)
    return result
