"""New source-closed v6.43 original seed arguments and exact history audit.

Preparation and history clearance are read-only scientific planning operations,
not initialization/reset/optimizer capabilities. No old claim or 20k pass is
accepted. The owner must recheck the history stat delta immediately at claim.
"""
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os

from . import v643_training_partitions as partitions
from . import exact_seed_audit as exact
from . import v642_confirmation as history
from .v643_risk_recipe import CANDIDATE
from .v626_protocol import require, pointer, bound, digest, file_sha
from scripts.run_warehouse_v625 import write_new

ROOT = partitions.ROOT
OUTPUT_ROOT = ROOT/'output/warehouse_v643'/CANDIDATE/'seed-plans'
VERSION = 'warehouse-v643-finite-calibration-seed-plan.v1'
CLEARANCE_VERSION = 'warehouse-v643-finite-calibration-history-clearance.v1'
VARIANTS = ('raw', 'regularized')


def source_files():
    # No optional imports or caller-provided list: all implementation pieces
    # must exist before a scientific plan can snapshot its source closure.
    from .v643_program_provider import source_files as program_sources
    files = partitions.source_files()
    for name, sha in program_sources().items():
        require(name not in files or files[name] == sha, 'conflicting numerical/program source closure')
        files[name] = sha
    for name in ('experiments/warehouse/v643_training_seed_plan.py',
                 'experiments/warehouse/v643_calibration_owner.py',
                 'experiments/warehouse/v643_calibration_recovery.py',
                 'experiments/warehouse/exact_seed_audit.py',
                 'tests/test_warehouse_v643_training_seed_plan.py',
                 'tests/test_warehouse_v643_calibration_owner.py'):
        files[name] = file_sha(Path(__file__).resolve().parents[2]/name)
    return dict(sorted(files.items()))


def declared_calls(partition):
    partition.verify()
    p = partition.value
    training, program, quality = [bound(p[key]) for key in
        ('train_case_inventory','program_evidence_inventory','quality_case_inventory')]
    checked = partitions.validate_inventories(training, program, quality,
        initialization_seed=p['initialization_seed'], optimizer_seed=p['optimizer_seed'])
    require(checked == p['checked'], 'partition inventory semantics differ')
    uses, cases = [], []

    def use(kind, purpose, argument, value, group, tag, draw=None):
        row = dict(kind=kind, purpose=purpose, argument=argument, value=value, group=group, reuse_tag=tag)
        if draw is not None:
            row['consumed_as_random_draw'] = draw
        uses.append(row)

    def episode(policy, item):
        group = policy + ':' + item['episode']
        purpose = 'v643_calibration_' + item['scope']
        for kind, argument, key in (
            ('scenario','task_expansion_base','base'),
            ('env','random.Random(config.seed)','scenario_seed'),
            ('env','reset(seed)','scenario_seed'),
            ('action_rng','independent_agent_seed(base_seed,episode_id=1,frame,agent_id)','action_seed')):
            use(kind, purpose, argument, item[key], group, key+':'+str(item[key]),
                True if kind == 'action_rng' else None)
        cases.append(dict(variant=policy, **item))

    for variant in VARIANTS:
        use('init','v643_original_BC_import_and_new_critic','torch.default_generator.manual_seed',
            p['initialization_seed'],variant+':initialization','paired-initialization:'+str(p['initialization_seed']))
        use('action_rng','v643_PPO_reference_and_cost_replay','np.random.default_rng',
            p['optimizer_seed'],variant+':optimizer','paired-optimizer:'+str(p['optimizer_seed']),True)
        for c in training['cases']:
            episode(variant, dict(episode=c['episode_id'],base=c['scenario_seed']//10, scenario_seed=c['scenario_seed'],
                task=c['task'],action_seed=c['action_seed'],scope='train',slot=c['slot']))
        for c in program['cases']:
            episode(variant, dict(episode=c['episode'],base=c['base_seed'],scenario_seed=c['scenario_seed'],
                task=c['task'],action_seed=c['action_seed'],scope='program_'+c['split'],slot=None))
    for policy in ('reference', *VARIANTS):
        for c in quality['cases']:
            episode(policy, dict(episode=c['case_id'],base=c['base'],scenario_seed=c['scenario_seed'],
                task=c['task'],action_seed=c['action_seed'],scope='quality',slot=None))
    return dict(schema='warehouse-exact-seed-plan.v1',candidate_id=CANDIDATE,
        uses=uses,partition=deepcopy(partition.ref),case_reservations_sha256=digest(cases),
        planned_episode_construction='fresh_constructor_then_explicit_reset_for_each_used_case',
        planned_physical_episode_id=1,paired_case_reuse_not_independent_evidence=True,
        inventory_reservation_is_not_actual_reset=True,authorization_granted=False), cases


@dataclass(frozen=True)
class SeedPlan:
    ref: dict
    value: dict
    partition: object

    def verify(self):
        self.partition.verify()
        p = self.value
        require(partitions._immutable(self.ref) == p and p['version'] == VERSION
            and p['candidate_id'] == CANDIDATE and p['partition'] == self.partition.ref
            and p['source_files'] == source_files() and p['source_sha256'] == digest(source_files())
            and p['training_authority'] is False, 'v643 seed plan identity/source differs')
        declaration, cases = declared_calls(self.partition)
        require(partitions._immutable(p['exact_plan']) == declaration
            and partitions._immutable(p['case_reservations']) == cases
            and p['maximum_environment_constructors'] == p['maximum_explicit_resets'] == len(cases)
            and partitions._immutable(pointer(Path(self.ref['path']).with_name('completed.json')))
                == dict(plan_sha256=self.ref['sha256']), 'v643 finite original calls or completion seal differs')
        for name, sha in p['source_files'].items():
            path = Path(self.ref['path']).parent/'source'/p['source_sha256']/name
            require(file_sha(path) == sha and not path.is_symlink() and not path.stat().st_mode & 0o222,
                    'v643 source snapshot differs')
        return self


def prepare(*, reference_directory, partition_ref, run_id):
    partition = partitions.load(reference_directory, partition_ref)
    require(type(run_id) is str and partitions._NAME.fullmatch(run_id), 'safe explicit v643 run ID required')
    declaration, cases = declared_calls(partition)
    sources = source_files(); source_sha = digest(sources)
    output = OUTPUT_ROOT/run_id
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in sources.items():
        path = ROOT/name; raw = path.read_bytes()
        require(file_sha(path) == expected, 'source changed before snapshot')
        target = output/'source'/source_sha/name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(raw);handle.flush();os.fsync(handle.fileno())
        target.chmod(0o444)
    for filename, value in (('exact-plan.json',declaration),('case-reservations.json',cases)):
        write_new(output/filename,value); (output/filename).chmod(0o444)
    exact.read_exact_plan(output/'exact-plan.json')
    partition.verify(); require(source_files() == sources, 'source changed during v643 snapshot')
    value = dict(version=VERSION,candidate_id=CANDIDATE,run_id=run_id,partition=deepcopy(partition.ref),
        source_files=sources,source_sha256=source_sha,exact_plan=pointer(output/'exact-plan.json'),
        case_reservations=pointer(output/'case-reservations.json'),config=deepcopy(partition.value['config']),
        maximum_environment_constructors=len(cases),maximum_explicit_resets=len(cases),
        initialization_RNG_constructors=2,optimizer_RNG_constructors=2,quality_policy_reservations=225,
        history_freshness_verified=False,training_authority=False,new_environment_resets=0,
        new_Actor_updates=0,single_use_complete_owner_required=True,pilot_opened=False)
    write_new(output/'plan.json',value); (output/'plan.json').chmod(0o444)
    ref=pointer(output/'plan.json')
    write_new(output/'completed.json',dict(plan_sha256=ref['sha256']));(output/'completed.json').chmod(0o444)
    return SeedPlan(ref,value,partition).verify()


def load(reference_directory, ref):
    value=partitions._immutable(ref)
    require(Path(ref['path']).resolve() == OUTPUT_ROOT/value['run_id']/'plan.json',
            'v643 seed plan relocated outside its registered namespace')
    return SeedPlan(ref,value,partitions.load(reference_directory,value['partition'])).verify()


def exclusions(plan):
    return [Path(plan.ref['path']).parent, Path(plan.partition.ref['path']).parent,
            *(Path(plan.partition.value[key]['path']) for key in
              ('train_case_inventory','program_evidence_inventory','quality_case_inventory'))]


def history_selection(plan, traces):
    stats={}; selected=[]
    for path in history.metadata.candidates([ROOT.parent/name for name in history.HISTORY_NAMES],
            exclusions(plan),registered_trace_paths=traces,selection_stats=stats):
        stat=path.stat()
        selected.append(dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return selected,stats['excluded_weights_name_stat_only']


def _conflict_free(evidence):
    require(not evidence['summary']['unresolved_paths'] and not any(any(row.get(key) for key in
        ('references_or_reservations','reservation_overlaps','audit_references_not_original_execution',
         'excluded_weight_path_seed_hints')) for row in evidence['summary']['families'].values()),
        'v643 historical seed conflict or unresolved metadata')


def _template_bindings(inventory, *, check_current):
    rows=[row for row in inventory if row['classification']=='source_template_environment_reservation']
    recorded={row['path']:row['sha256'] for row in rows}
    require(len(rows)==len(recorded),'duplicate historical source template')
    if check_current:
        current={}
        for name in history.HISTORY_NAMES:
            path=ROOT.parent/name/'backend/training/seed_ledger.py'
            if path.is_file():
                raw=path.read_bytes()
                if b'500_004' in raw and b'10_500_204' in raw:
                    current[str(path.resolve())]=file_sha(path)
        require(current==recorded,'historical source reservation changed')


def audit_seeds(plan):
    """Real metadata→triage→exact/context reproduction; no science execution."""
    plan.verify()
    folder=Path(plan.ref['path']).parent/'seed-ledger'; folder.mkdir(exist_ok=False)
    declaration=bound(plan.value['exact_plan']); families=exact.compile_families(declaration)
    traces=bound(history.INVENTORY_REF)['registered_trace_paths']
    proposal=dict(schema='warehouse-seed-proposal.v1',candidate_id=CANDIDATE,
        families=[{key:row[key] for key in ('name','kind','values','query_only')} for row in families],
        exclude_paths=list(map(str,exclusions(plan))),registered_trace_paths=traces)
    write_new(folder/'proposal.json',proposal)
    history.metadata.audit(folder/'proposal.json',[ROOT.parent/name for name in history.HISTORY_NAMES],folder/'bounded')
    history.triage.triage(folder/'bounded/inventory.json',folder/'proposal.json',folder/'triaged')
    evidence=exact.assess(plan.value['exact_plan']['path'],folder/'triaged/result.json',folder/'exact',context_source_root=ROOT)
    _conflict_free(evidence)
    selected, weights=history_selection(plan,traces)
    audited={row['path']:(row['bytes'],row.get('mtime_ns')) for row in evidence['inventory']
             if row['classification']!='source_template_environment_reservation'}
    require(audited=={row['path']:(row['bytes'],row['mtime_ns']) for row in selected},
            'history changed during v643 exact audit')
    _template_bindings(evidence['inventory'],check_current=True); plan.verify()
    now=datetime.now(timezone.utc)
    value=dict(schema=CLEARANCE_VERSION,candidate_id=CANDIDATE,plan=deepcopy(plan.ref),
        registry=deepcopy(history.INVENTORY_REF),exact=pointer(folder/'exact/result.json'),
        registered_traces=traces,selection=selected,weight_paths=weights,
        issued_utc=now.isoformat(),expires_utc=(now+timedelta(hours=24)).isoformat(),
        approved_case_reservations_sha256=plan.value['case_reservations']['sha256'],
        maximum_finite_reset_reservations=plan.value['maximum_explicit_resets'],
        actual_resets=0,training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(folder/'clearance.json',value);(folder/'clearance.json').chmod(0o444)
    return checked_clearance(plan)


def checked_clearance(plan, *, execution=True):
    plan.verify();folder=Path(plan.ref['path']).parent/'seed-ledger'
    ref=pointer(folder/'clearance.json');value=partitions._immutable(ref)
    evidence=bound(value['exact'])
    issued=datetime.fromisoformat(value['issued_utc']);expires=datetime.fromisoformat(value['expires_utc'])
    require(value['schema']==CLEARANCE_VERSION and value['candidate_id']==CANDIDATE and value['plan']==plan.ref
        and value['registry']==history.INVENTORY_REF and pointer(history.INVENTORY_REF['path'])==history.INVENTORY_REF
        and issued.tzinfo is not None and expires.tzinfo is not None and expires-issued==timedelta(hours=24)
        and (not execution or issued<=datetime.now(timezone.utc)<expires), 'new candidate history binding/lifetime differs')
    require(value['approved_case_reservations_sha256']==plan.value['case_reservations']['sha256']
        and value['maximum_finite_reset_reservations']==plan.value['maximum_explicit_resets']
        and value['actual_resets']==0 and value['training_authority'] is value['sealed_authority'] is value['pilot_opened'] is False,
        'history clearance cannot grant scientific execution')
    require(value['exact']==pointer(folder/'exact/result.json') and evidence['exact_plan']==plan.value['exact_plan']
        and evidence['exact_auditor_sha256']==file_sha(exact.__file__), 'exact evidence/auditor binding differs')
    inventory=bound(evidence['bounded_or_triaged_inventory'])
    projection=exact.project_inventory(inventory,exact.compile_families(bound(plan.value['exact_plan'])))
    if evidence.get('context_annotations'):
        projection=exact.apply_context(projection,bound(evidence['context_annotations']))
    require(digest(evidence['inventory'])==digest(projection['inventory']) and evidence['summary']==projection['summary'],
        'exact/context evidence does not independently reproduce')
    roots=sorted(str((ROOT.parent/name).resolve()) for name in history.HISTORY_NAMES)
    require(sorted(evidence['scan_roots'])==roots and evidence['scan_roots']==inventory['scan_roots']
        and not inventory['missing_roots'] and all(Path(root).is_dir() for root in roots),
        'all four historical worktrees are required')
    traces=bound(history.INVENTORY_REF)['registered_trace_paths']
    excluded=list(map(str,[folder/'proposal.json',folder/'bounded',*exclusions(plan)]))
    require(value['registered_traces']==evidence['registered_trace_paths']==inventory['registered_trace_paths']==traces
        and evidence['selection_scope']==inventory['selection_scope']==history.metadata.SELECTION_SCOPE
        and evidence['read_model_weight_files'] is inventory['read_model_weight_files'] is False
        and sorted(evidence['scope_exclusions'])==sorted(inventory['scope_exclusions'])==sorted(excluded),
        'historical scan scope, registry or explicit exclusions differ')
    audited={row['path']:(row['bytes'],row.get('mtime_ns')) for row in evidence['inventory']
             if row['classification']!='source_template_environment_reservation'}
    require(audited=={row['path']:(row['bytes'],row['mtime_ns']) for row in value['selection']}
        and len(audited)==len(value['selection']), 'history selection differs from exact audit')
    _conflict_free(evidence);_template_bindings(evidence['inventory'],check_current=execution)
    if execution:
        selected,weights=history_selection(plan,traces)
        require(selected==value['selection'] and weights==value['weight_paths'], 'history changed after v643 clearance')
    return dict(clearance=ref,value=value)
