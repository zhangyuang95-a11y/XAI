"""Explicit Stage C seed call plan; no RNG, reset or training authority.

The plan fixes one fresh constructor + explicit reset for each used episode.
Raw and regularized share matching declared streams, not independent scenes.
Unused finite inventory entries remain reservations, never claimed executions.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime,timedelta,timezone
from . import v630_training_partitions as partitions
from . import exact_seed_audit as exact
from . import v630_confirmation as confirmation
from .v626_protocol import require,pointer,bound,digest,file_sha
from scripts.run_warehouse_v625 import write_new

ROOT=partitions.admission.v.ROOT
OUTPUT_ROOT=ROOT/'output/warehouse_v630/training-seed-plans'
VERSION='warehouse-v630-paired-calibration-seed-plan.v1'
VARIANTS=('raw','regularized')


def source_files():
    files=partitions.source_files()
    files['experiments/warehouse/v630_training_seed_plan.py']=file_sha(__file__)
    # History roots can be redirected for an isolated audit; source bindings
    # always refer to the module's actual checkout.
    files['tests/test_warehouse_v630_training_seed_plan.py']=file_sha(
        Path(__file__).resolve().parents[2]/'tests/test_warehouse_v630_training_seed_plan.py')
    return dict(sorted(files.items()))


def load_partition(reference_directory,proposal_ref):
    ready=partitions.admission.load_readiness(reference_directory)  # Before any proposal reads.
    ready.verify();value=bound(proposal_ref)
    path=Path(proposal_ref['path']).resolve()
    require(path.is_relative_to(partitions.OUTPUT_ROOT.resolve()) and path.name=='proposal.json',
        'partition proposal is outside its prepared namespace')
    refs=tuple(deepcopy([*ready.refs,value['train_case_inventory'],value['program_evidence_inventory']]))
    return partitions.PartitionProposal(proposal_ref,value,ready,refs).verify()


def declared_calls(proposal):
    """Exact original arguments, with explicit matched-pair reuse tags.

    Per-frame uint64 hash-derived action seeds are deterministic children of
    the declared action base, episode ID=1, frame and role. They are not fresh
    independent scenario/initialization arguments or a new RNG search.
    """
    proposal.verify();p=proposal.value
    training=bound(p['train_case_inventory']);evidence=bound(p['program_evidence_inventory'])
    checked=partitions.validate_inventories(training,evidence,initialization_seed=p['initialization_seed'],optimizer_seed=p['optimizer_seed'])
    require(checked==p['checked'],'partition semantics differ from prepared proposal')
    uses=[];cases=[]
    def item(kind,purpose,argument,value,group,tag,draw=None):
        row=dict(kind=kind,purpose=purpose,argument=argument,value=value,group=group,reuse_tag=tag)
        if draw is not None:row['consumed_as_random_draw']=draw
        uses.append(row)
    for variant in VARIANTS:
        item('init','v630_imported_actors_new_critic_initialization','torch.default_generator.manual_seed',
            p['initialization_seed'],variant+':initialization','paired-initialization:'+str(p['initialization_seed']))
        item('action_rng','v630_PPO_shuffle_and_reference_mix','np.random.default_rng',
            p['optimizer_seed'],variant+':optimizer','paired-optimizer:'+str(p['optimizer_seed']),True)
        streams=[]
        for case in training['cases']:
            streams.append(dict(episode=case['episode_id'],base=case['scenario_seed']//10,scenario_seed=case['scenario_seed'],
                task=case['task'],action_seed=case['action_seed'],scope='train',slot=case['slot']))
        for case in evidence['cases']:
            streams.append(dict(episode=case['episode'],base=case['base_seed'],scenario_seed=case['scenario_seed'],
                task=case['task'],action_seed=case['action_seed'],scope='program_'+case['split'],slot=None))
        for case in streams:
            group=variant+':'+case['episode'];purpose='v630_calibration_'+case['scope']
            for kind,argument,key in (('scenario','task_expansion_base','base'),
                ('env','random.Random(config.seed)','scenario_seed'),('env','reset(seed)','scenario_seed'),
                ('action_rng','independent_agent_seed(base_seed,episode_id=1,frame,agent_id)','action_seed')):
                item(kind,purpose,argument,case[key],group,key+':'+str(case[key]),True if kind=='action_rng' else None)
            cases.append(dict(variant=variant,**case))
    return dict(schema='warehouse-exact-seed-plan.v1',candidate_id=partitions.admission.v.CANDIDATE+'-paired-20k',
        uses=uses,partition_proposal=deepcopy(proposal.ref),case_reservations_sha256=digest(cases),
        planned_episode_construction='fresh_constructor_then_explicit_reset_for_each_used_case',
        planned_physical_episode_id=1,paired_case_reuse_not_independent_evidence=True,
        inventory_reservation_is_not_actual_reset=True,authorization_granted=False),cases


@dataclass(frozen=True)
class CalibrationSeedPlan:
    ref:dict
    value:dict
    proposal:object

    def verify(self):
        self.proposal.verify();p=self.value;path=Path(self.ref['path']);folder=path.parent
        require(pointer(path)==self.ref and bound(self.ref)==p and p['version']==VERSION
            and p['source_files']==source_files() and p['source_sha256']==digest(source_files())
            and p['partition_proposal']==self.proposal.ref and p['training_authority'] is False,
            'calibration seed plan or source binding changed')
        require(bound(pointer(folder/'completed.json'))==dict(plan_sha256=self.ref['sha256']),
            'calibration seed plan completion seal changed')
        declaration,cases=declared_calls(self.proposal)
        require(bound(p['exact_plan'])==declaration and bound(p['case_reservations'])==cases
            and p['maximum_environment_constructors']==p['maximum_explicit_resets']==len(cases),
            'calibration original arguments, pair reuse or finite ceilings changed')
        for name,sha in p['source_files'].items():
            snapshot=folder/'source'/p['source_sha256']/name
            require(file_sha(snapshot)==sha and not snapshot.stat().st_mode&0o222,'calibration source snapshot changed')
        return self


def prepare(*,reference_directory,proposal_ref,run_id):
    proposal=load_partition(reference_directory,proposal_ref)  # Must precede run/output handling.
    run=confirmation.run_name(run_id);sources=source_files();source_sha=digest(sources)
    declaration,cases=declared_calls(proposal)
    output=OUTPUT_ROOT/run;output.mkdir(parents=True,exist_ok=False)
    for name,sha in sources.items():
        source=ROOT/name;raw=source.read_bytes();require(file_sha(source)==sha,'source changed preparing calibration plan')
        target=output/'source'/source_sha/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o444)
    write_new(output/'exact-plan.json',declaration);(output/'exact-plan.json').chmod(0o444)
    exact.read_exact_plan(output/'exact-plan.json')
    write_new(output/'case-reservations.json',cases);(output/'case-reservations.json').chmod(0o444)
    proposal.verify();require(source_files()==sources,'source changed while freezing calibration seed plan')
    value=dict(version=VERSION,run_id=run,partition_proposal=deepcopy(proposal.ref),source_files=sources,source_sha256=source_sha,
        exact_plan=pointer(output/'exact-plan.json'),case_reservations=pointer(output/'case-reservations.json'),
        config=deepcopy(proposal.value['config']),maximum_environment_constructors=len(cases),maximum_explicit_resets=len(cases),
        variants=list(VARIANTS),new_critic_initializations=2,optimizer_RNG_constructors=2,
        history_freshness_verified=False,history_exact_audit_and_final_stat_delta_required=True,
        single_use_training_owner_required=True,training_authority=False,new_environment_resets=0,
        Actor_updates=0,PPO_joint_steps=0,pilot_opened=False,publishable=False)
    write_new(output/'plan.json',value);(output/'plan.json').chmod(0o444)
    ref=pointer(output/'plan.json');write_new(output/'completed.json',dict(plan_sha256=ref['sha256']));(output/'completed.json').chmod(0o444)
    return CalibrationSeedPlan(ref,value,proposal).verify()


def exclusions(plan):
    return [Path(plan.ref['path']).parent,Path(plan.proposal.ref['path']).parent,
        *(Path(plan.proposal.value[k]['path']) for k in ('train_case_inventory','program_evidence_inventory'))]


def history_selection(plan,traces):
    stats={};selected=[]
    roots=[ROOT.parent/name for name in confirmation.HISTORY_NAMES]
    for path in confirmation.metadata.candidates(roots,exclusions(plan),registered_trace_paths=traces,selection_stats=stats):
        stat=path.stat();selected.append(dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return selected,stats['excluded_weights_name_stat_only']


def conflict_free(evidence):
    require(not evidence['summary']['unresolved_paths'] and not any(any(row.get(k) for k in
        ('references_or_reservations','reservation_overlaps','audit_references_not_original_execution','excluded_weight_path_seed_hints'))
        for row in evidence['summary']['families'].values()),'historical seed conflict or unresolved metadata')


def template_bindings(inventory,*,check_current):
    rows=[r for r in inventory if r['classification']=='source_template_environment_reservation']
    recorded={r['path']:r['sha256'] for r in rows}
    require(len(recorded)==len(rows),'duplicate historical source template')
    if check_current:
        current={}
        for name in confirmation.HISTORY_NAMES:
            path=ROOT.parent/name/'backend/training/seed_ledger.py'
            if path.is_file():
                raw=path.read_bytes()
                if b'500_004' in raw and b'10_500_204' in raw:current[str(path.resolve())]=file_sha(path)
        require(current==recorded,'historical source reservation changed after calibration audit')
    return recorded


def audit_seeds(plan):
    """Bind existing exact/context auditors; still no RNG or reset authority."""
    plan.verify();folder=Path(plan.ref['path']).parent/'seed-ledger';folder.mkdir(exist_ok=False)
    declaration=bound(plan.value['exact_plan']);families=exact.compile_families(declaration)
    registry=confirmation.INVENTORY_REF;traces=bound(registry)['registered_trace_paths']
    proposal=dict(schema='warehouse-seed-proposal.v1',candidate_id=declaration['candidate_id'],
        families=[{k:r[k] for k in ('name','kind','values','query_only')} for r in families],
        exclude_paths=list(map(str,exclusions(plan))),registered_trace_paths=traces)
    write_new(folder/'proposal.json',proposal)
    confirmation.metadata.audit(folder/'proposal.json',[ROOT.parent/name for name in confirmation.HISTORY_NAMES],folder/'bounded')
    confirmation.triage.triage(folder/'bounded/inventory.json',folder/'proposal.json',folder/'triaged')
    result=exact.assess(plan.value['exact_plan']['path'],folder/'triaged/result.json',folder/'exact',context_source_root=ROOT)
    conflict_free(result);selected,weights=history_selection(plan,traces)
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in result['inventory'] if r['classification']!='source_template_environment_reservation'}
    require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in selected},'history changed during calibration seed audit')
    template_bindings(result['inventory'],check_current=True);plan.verify()
    now=datetime.now(timezone.utc)
    value=dict(schema='warehouse-v630-calibration-history-clearance.v1',plan=deepcopy(plan.ref),registry=deepcopy(registry),
        exact=pointer(folder/'exact/result.json'),registered_traces=traces,selection=selected,weight_paths=weights,
        issued_utc=now.isoformat(),expires_utc=(now+timedelta(hours=24)).isoformat(),
        approved_case_reservations_sha256=plan.value['case_reservations']['sha256'],
        maximum_finite_reset_reservations=plan.value['maximum_explicit_resets'],
        actual_resets=0,training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(folder/'clearance.json',value);(folder/'clearance.json').chmod(0o444)
    return checked_clearance(plan)


def checked_clearance(plan,*,execution=True):
    plan.verify();folder=Path(plan.ref['path']).parent/'seed-ledger'
    ref=pointer(folder/'clearance.json');clearance=bound(ref);evidence=bound(clearance['exact'])
    issued=datetime.fromisoformat(clearance['issued_utc']);expires=datetime.fromisoformat(clearance['expires_utc'])
    require(clearance['schema']=='warehouse-v630-calibration-history-clearance.v1' and clearance['plan']==plan.ref
        and clearance['registry']==confirmation.INVENTORY_REF and pointer(confirmation.INVENTORY_REF['path'])==confirmation.INVENTORY_REF
        and issued.tzinfo is not None and expires.tzinfo is not None and expires-issued==timedelta(hours=24)
        and (not execution or issued<=datetime.now(timezone.utc)<expires),'calibration clearance binding or lifetime differs')
    require(clearance['approved_case_reservations_sha256']==plan.value['case_reservations']['sha256']
        and clearance['maximum_finite_reset_reservations']==plan.value['maximum_explicit_resets']
        and clearance['actual_resets']==0 and clearance['training_authority'] is clearance['sealed_authority'] is clearance['pilot_opened'] is False,
        'history clearance cannot authorize resets, training, sealed data or pilot')
    require(clearance['exact']==pointer(folder/'exact/result.json') and evidence['exact_plan']==plan.value['exact_plan']
        and evidence['exact_auditor_sha256']==file_sha(exact.__file__),'calibration exact evidence or auditor differs')
    declaration=bound(plan.value['exact_plan']);inventory=bound(evidence['bounded_or_triaged_inventory'])
    projection=exact.project_inventory(inventory,exact.compile_families(declaration))
    if evidence.get('context_annotations'):projection=exact.apply_context(projection,bound(evidence['context_annotations']))
    require(digest(evidence['inventory'])==digest(projection['inventory']) and evidence['summary']==projection['summary'],
        'calibration exact/context evidence does not reproduce from its input inventory')
    roots=sorted(str((ROOT.parent/name).resolve()) for name in confirmation.HISTORY_NAMES)
    require(sorted(evidence['scan_roots'])==roots and evidence['scan_roots']==inventory['scan_roots'] and not inventory['missing_roots']
        and all(Path(root).is_dir() for root in roots),'all four historical worktrees are required')
    traces=bound(confirmation.INVENTORY_REF)['registered_trace_paths']
    excluded=list(map(str,[folder/'proposal.json',folder/'bounded',*exclusions(plan)]))
    require(clearance['registered_traces']==evidence['registered_trace_paths']==inventory['registered_trace_paths']==traces
        and evidence['selection_scope']==inventory['selection_scope']==confirmation.metadata.SELECTION_SCOPE
        and evidence['read_model_weight_files'] is inventory['read_model_weight_files'] is False
        and sorted(evidence['scope_exclusions'])==sorted(inventory['scope_exclusions'])==sorted(excluded),
        'calibration history scope, trace registry or exclusions differ')
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in evidence['inventory'] if r['classification']!='source_template_environment_reservation'}
    require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in clearance['selection']}
        and len(audited)==len(clearance['selection']),'calibration clearance history selection differs')
    conflict_free(evidence);template_bindings(evidence['inventory'],check_current=execution)
    if execution:
        selected,weights=history_selection(plan,traces)
        require(selected==clearance['selection'] and weights==clearance['weight_paths'],'history changed after calibration clearance')
    return dict(clearance=ref,value=clearance)
