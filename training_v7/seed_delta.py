"""Final full-content metadata delta. A preliminary inventory grants no reset."""
from pathlib import Path
import json
from .contracts import digest, file_sha, write_new
from .seed_audit import candidates, SCOPE


def verify(proposal_path, triage_path, output):
    proposal_path, triage_path = Path(proposal_path).resolve(), Path(triage_path).resolve()
    result = json.loads(triage_path.read_text())
    if result['selection_scope'] != SCOPE or result['proposal']['sha256'] != file_sha(proposal_path):
        raise ValueError('final delta proposal or three-domain selection differs')
    from . import seed_audit
    if result['selection_adapter']['sha256'] != file_sha(seed_audit.__file__):
        raise ValueError('final delta metadata selector differs')
    roots = [Path(p) for p in result['scan_roots']]
    if any(not p.is_dir() for p in roots): raise ValueError('historical root missing at final delta')
    context = result.get('first_use_owner_exclusion')
    if context:
        owner = Path(context['path'])
        if (owner != proposal_path.parent or context['NN_claims'] != 0 or context['actor_artifacts'] != 0
                or any(owner.rglob('claim.json')) or any(owner.rglob('*.actor.npz'))):
            raise ValueError('first-use owner scope no longer holds')
        for ref in [result['contextual_decoder'], result['context_parent'], *result['contextual_producer_sources'],
                    *result['own_registration_metadata']]:
            if file_sha(ref['path']) != ref['sha256']: raise ValueError('context evidence changed')
        excludes = [owner]
    else:
        excludes = [proposal_path, Path(result.get('bounded_inventory', {}).get('path', triage_path)).parent]
        excludes += [Path(p).resolve() for p in json.loads(proposal_path.read_text()).get('exclude_paths', [])]
    resume = result.get('resume_adapter')
    if resume:
        for ref in [resume, resume['previous_complete'],resume['resumed_full_parse'],*resume['partial_evidence']]:
            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('seed parsing cache evidence changed')
    stats = {}
    live = sorted(candidates(roots, excludes,
        registered_trace_paths=result.get('registered_trace_paths', ()), selection_stats=stats))
    expected = {r['path']: r for r in result['inventory'] if r['classification'] != 'source_template_environment_reservation'}
    actual_paths = {str(p.resolve()) for p in live}
    selection_same = actual_paths == set(expected)
    changed = []
    checked = []
    # Rehash the complete selected input population, not just timestamps or a
    # sample. Unchanged bytes allow reuse of the previous parsing conclusion.
    for path in live:
        name = str(path.resolve()); h = file_sha(path)
        if name not in expected or h != expected[name].get('sha256'): changed.append(name)
        checked.append(dict(path=name, sha256=h))
    for row in result['inventory']:
        if row['classification'] == 'source_template_environment_reservation' and file_sha(row['path']) != row['sha256']:
            changed.append(row['path'])
    previous_weights = result['selection_stats'].get('excluded_weights_name_stat_only', [])
    current_weights = stats.get('excluded_weights_name_stat_only', [])
    opaque_same = sorted(previous_weights, key=lambda r: r['path']) == sorted(current_weights, key=lambda r: r['path'])
    s = result['summary']
    no_conflicts = (not s['unresolved_paths'] and not result.get('missing_roots') and all(
        not r['references_or_reservations'] and not r['reservation_overlaps']
        and not r['excluded_weight_path_seed_hints'] and not r['audit_references_not_original_execution']
        for r in s['families'].values()))
    value = dict(version='three-domain-unified-final-seed-delta.v1',
        proposal=dict(path=str(proposal_path), sha256=file_sha(proposal_path)),
        triage=dict(path=str(triage_path), sha256=file_sha(triage_path)),
        selection_scope=SCOPE, selection_unchanged=selection_same, opaque_exclusions_unchanged=opaque_same,
        checked=checked, selection_sha256=digest(checked), changed=changed,
        missing=sorted(set(expected)-actual_paths), no_known_conflicts=no_conflicts,
        passed=selection_same and opaque_same and not changed and no_conflicts,
        environment_resets=0, NN_updates=0, pilot_opened=False)
    write_new(output, value)
    return value
