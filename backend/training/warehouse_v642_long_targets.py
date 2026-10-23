"""Source-bound, lossless storage producer for NEW finite long owners only.

The numerical loop and native target semantics are unchanged. Fresh canonical
files are produced before binding, then represented by real rows hardlinks and
APFS byte-identical NPZ clones. No already sealed package is rewritten. Deleted
scratch inputs have explicit temporal lifetimes, never missing persistent refs.
"""
from pathlib import Path
import json,os,re,stat
import numpy as np
from . import warehouse_v642_observed_pool as native
from experiments.warehouse import v642_rows_cas as rows
from experiments.warehouse import v642_apfs_npz_clone as clones
from experiments.warehouse.v626_protocol import require,pointer,bound,digest,file_sha
from scripts.run_warehouse_v625 import write_new

VERSION='warehouse-v642-new-long-target-materialization.v1'
ROOT=Path(__file__).resolve().parents[2]
FILES=('backend/training/warehouse_v642_long_targets.py','tests/test_warehouse_v642_long_targets.py',
       'docs/warehouse_v642_long_targets.md','experiments/warehouse/v642_apfs_npz_clone.py',
       'tests/test_warehouse_v642_apfs_npz_clone.py','docs/warehouse_v642_apfs_npz_clone.md')


def source_files():
    files=rows.source_files();files.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(files.items()))


def _immutable(ref):
    require(type(ref) is dict and set(ref)=={'path','sha256'},'explicit persistent immutable pointer required')
    path=Path(ref['path']);require(path.is_absolute() and path.resolve()==path,'persistent path must be canonical')
    for part in (path,*path.parents):require(not part.is_symlink(),'persistent path contains a symlink')
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode) and not path.stat().st_mode&0o222
        and pointer(path)==ref,'persistent materialization input missing, writable or changed')
    return path


def _put(path,value):
    write_new(path,value);path.chmod(0o444);return pointer(path)


def _active_owner(active,variant,pool):
    # Local import avoids a source-contract import cycle. This is not a
    # generic provider retrofit or permission inferred from a passed JSON.
    from experiments.warehouse import v642_long_training_owner as owner
    require(type(active) is owner.LongTrainingOwner and owner.ACTIVE.get() is active,
        'only an active newly claimed finite long owner can produce these targets')
    # The owning update/claim already performs the full immutable seed/input
    # check. Do not multiply that multi-GB replay by every PPO epoch here:
    # this local boundary rechecks live claim/descriptor/finite lease and the
    # complete executor source closure, plus exact in-memory pool ownership.
    active.live()
    require(variant in owner.VARIANTS and active.pools.get(variant) is pool
        and pool.owner_ref==active.owners[variant] and active.sources==owner.source_files(),
        'long target source is not owned or source-bound by this variant')
    require(all(active.sources.get(name)==sha for name,sha in source_files().items()),
        'long source contract omitted the explicit storage producer/algorithms')


def _write_canonical(stage,pool,actor_ref):
    """Preserve native split/role order, scalar NN and recorded draws exactly."""
    actor=native.FrozenWarehouseV6Actor.load(actor_ref['path'],expected_sha256=actor_ref['sha256'])
    require(actor.model_version==native.MODEL_VERSION,'target NN architecture changed')
    differences={}
    for split,part in pool.parts.items():
        file=stage.payload/(split+'-source-actual-actions.npz')
        with file.open('xb') as handle:np.savez_compressed(handle,executed=part['executed'])
        file.chmod(0o444)
    for i,role in enumerate(native.AGENTS):
        differences[role]={}
        for split,part in pool.parts.items():
            x,m=part['x'][:,i],part['m'][:,i]
            p=native.online.runtime_probabilities(actor,role,x,m)
            sampled=np.array([np.random.default_rng(native.independent_agent_seed(
                base_seed=row['action_seed'],episode_id=row['episode_id'],frame=row['frame'],agent_id=role)).choice(5,p=prob)
                for row,prob in zip(part['rows'],p,strict=True)],np.int64)
            file=stage.payload/(role+'-'+split+'.npz')
            with file.open('xb') as handle:np.savez_compressed(handle,x=x,m=m,p=p,sampled=sampled)
            file.chmod(0o444)
            _put(stage.payload/(role+'-'+split+'-rows.json'),part['rows'])
            differences[role][split]=int(np.count_nonzero(sampled!=part['executed'][:,i]))
    return differences


def _baselines(stage,sources):
    folder=stage.owner/'npz-baselines';ledger=stage.owner/'npz-baselines.json'
    if ledger.exists():
        ref=pointer(ledger);value=bound(ref);_immutable(ref)
        require(value['version']==VERSION and value['source_files']==sources
            and value['first_stage']['path'].startswith(str(stage.owner/'stages')+'/')
            and set(value['baselines'])==rows.NPZ_NAMES,'permanent first-call NPZ baseline inventory/source differs')
        _immutable(value['first_stage'])
    else:
        require(not folder.exists(),'interrupted NPZ baseline publication cannot be repaired or retried')
        folder.mkdir()
        bases={}
        for name in sorted(rows.NPZ_NAMES):
            # Hardlink our fresh canonical inode once: no ordinary copy
            # fallback, and its first future-call bytes remain forever.
            path=folder/name;os.link(stage.payload/name,path,follow_symlinks=False);bases[name]=pointer(path)
        value=dict(version=VERSION,source_files=sources,first_stage=stage.header_ref,baselines=bases,
            immutable_permanent_first_future_call=True,historical_assets_adopted=False)
        ref=_put(ledger,value)
    for name,item in value['baselines'].items():
        require(Path(item['path'])==folder/name,'NPZ baseline escaped its exact owner namespace');_immutable(item)
    return ref,value['baselines']


def relabel_current_NN(active,variant,pool,target_actor_ref,*,stage_id):
    """Numerically generate and bind a new long package, never retrofit old data."""
    _active_owner(active,variant,pool)
    require(isinstance(pool,native.ObservedPool),'audited observed states required');pool.verify()
    require(type(stage_id) is str and re.fullmatch(r'(refit-[0-9]{1,6}|call-[0-9]{6})',stage_id),
        'long target stage must be the exact finite refit/call identity')
    _immutable(target_actor_ref);sources=source_files();owner_key=digest(pool.owner_ref)
    with rows.new_stage(owner_sha256=owner_key,stage_id=stage_id) as stage:
        differences=_write_canonical(stage,pool,target_actor_ref)
        names=sorted(rows.NPZ_NAMES);rows.register_temporal_npz(stage,names=names)
        rows_ref=rows.deduplicate_rows(stage,names=sorted(rows.ROW_NAMES))
        baseline_ref,bases=_baselines(stage,sources)
        output=stage.folder/'published-target';output.mkdir()
        npz={};witness=[]
        for name in names:
            scratch=pointer(stage.payload/name)
            report=clones.clone_patch_npz(bases[name],scratch,output/name)
            require(report['output']==pointer(output/name) and report['output']['sha256']==scratch['sha256']
                and report['kernel_clone_succeeded'] and report['ordinary_copy_fallback'] is False,
                'APFS output differs or clone silently copied')
            npz[name]=report['output']
            # The helper's temporary target pointer is deliberately not
            # serialized as a persistent dependency after scratch cleanup.
            temporal=dict(stage_relative_path='unbound-target/'+name,sha256=scratch['sha256'],
                bytes=(stage.payload/name).stat().st_size,lifecycle='created_and_consumed_before_final_binding',
                persisted_equivalent=npz[name])
            report={k:v for k,v in report.items() if k!='target'}
            report.update(consumed_temporal_input=temporal,source_bound_long_integration=True)
            witness.append(report)
        final_rows={}
        for name in sorted(rows.ROW_NAMES):
            os.link(stage.payload/name,output/name,follow_symlinks=False)
            final_rows[name]=pointer(output/name)
            a,b=(stage.payload/name).stat(),(output/name).stat()
            require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and b.st_nlink>=3,'published rows are not real CAS hardlinks')
        cleanup=rows.consume_temporal_npz(stage,substitutions=npz)
        witness_ref=_put(stage.folder/'npz-materialization.json',dict(version=VERSION,stage=stage.header_ref,
            baseline_ledger=baseline_ref,NPZ=witness,temporal_cleanup=cleanup,
            persistent_dependencies=[baseline_ref,*bases.values(),*npz.values()],
            canonical_scalar_NN_unchanged=True,recorded_draw_order_unchanged=True,
            ordinary_copy_fallback=False,NN_updates=0,environment_resets=0))
        pool.verify();_immutable(target_actor_ref)
        require(source_files()==sources,'long materialization source changed')
    closed=pointer(stage.folder/'storage-stage-closed.json')
    first_stage=bound(baseline_ref)['first_stage']
    cas_refs=[dict(path=r['CAS'],sha256=r['sha256']) for r in bound(rows_ref)['rows']]
    persistent=[stage.header_ref,closed,rows_ref,baseline_ref,witness_ref,cleanup,first_stage,
        *cas_refs,*bases.values(),*npz.values(),*final_rows.values()]
    provenance=_put(stage.folder/'materialization.json',dict(version=VERSION,owner=pool.owner_ref,variant=variant,
        stage_id=stage_id,stage=stage.header_ref,stage_closed=closed,source_files=sources,source_sha256=digest(sources),
        native_semantic_source_files=native.source_files(),observed_source=pool.source_ref,target_actor=target_actor_ref,
        rows_deduplication=rows_ref,npz_materialization=witness_ref,temporal_cleanup=cleanup,
        npz_baselines=baseline_ref,final_arrays=npz,final_rows=final_rows,persistent_dependencies=persistent,
        materialization_before_final_binding=True,old_packages_modified=False,
        numeric_method='unchanged_native_scalar_NN_and_recorded_draw_order',storage_only=True,
        NN_updates=0,environment_resets=0,PPO_joint_steps=0,scientific_acceptance_claimed=False,pilot_opened=False))
    pools={};actual={};refs=[]
    for split in pool.parts:
        actual[split]=npz[split+'-source-actual-actions.npz'];refs.append(actual[split])
    for role in native.AGENTS:
        parts={}
        for split in pool.parts:
            parts[split]=dict(arrays=npz[role+'-'+split+'.npz'],rows=final_rows[role+'-'+split+'-rows.json'])
            refs.extend(parts[split].values())
        pools[role]=_put(output/(role+'-pool.json'),dict(schema=native.online.MANIFEST_VERSION,
            scope='online_program_training_evidence',target_actor=target_actor_ref,role=role,parts=parts))
        refs.append(pools[role])
    # This is the existing target DATA schema with explicitly different
    # materialization provenance. It never claims the old producer executed.
    value=dict(schema=native.TARGET_VERSION,observed_source=pool.source_ref,observed_audit_sha256=pool.audit_sha256,
        observed_data_sha256=pool.observed_data_sha256,owner=pool.owner_ref,target_actor=target_actor_ref,
        source_files=native.source_files(),pools=pools,actual_source_actions=actual,
        counterfactual_difference_counts=differences,asset_refs=refs,long_materialization_provenance=provenance,
        current_policy_trajectory=False,counterfactual_draws_not_executed=True,actual_actions_replaced=False,
        actual_PPO_feedback_counts_from_real_rollouts_only=True,current_policy_task_quality_estimate=None,
        uses_existing_recorded_seeds_for_counterfactual_numerical_replay=True,fresh_randomness_claimed=False,
        environment_resets=0,NN_updates=0,PPO_joint_steps=0,training_authority=False,pilot_opened=False)
    ref=_put(output/'result.json',value);_put(output/'completed.json',dict(result_sha256=ref['sha256']))
    read_materialization(ref,expected_owner=pool.owner_ref)
    return ref


def read_materialization(target_ref,*,expected_owner):
    """Strict independent storage reader; caller still replays native NN targets.

    Only the specifically enumerated stage scratch NPZ may be absent. Every
    persistent pointer is rehashed, including algorithms, CAS and ledgers.
    No broad missing-ref exception is provided.
    """
    _immutable(target_ref);target=bound(target_ref)
    _immutable(target['long_materialization_provenance']);p=bound(target['long_materialization_provenance'])
    require(p['version']==VERSION and p['owner']==target['owner']==expected_owner
        and p['source_files']==source_files() and p['source_sha256']==digest(p['source_files'])
        and p['native_semantic_source_files']==native.source_files()==target['source_files']
        and p['target_actor']==target['target_actor'] and p['observed_source']==target['observed_source']
        and p['materialization_before_final_binding'] is True and p['old_packages_modified'] is False
        and p['storage_only'] is True and p['numeric_method']=='unchanged_native_scalar_NN_and_recorded_draw_order'
        and p['scientific_acceptance_claimed'] is p['pilot_opened'] is False
        and all(p[k]==0 and type(p[k]) is int for k in ('NN_updates','environment_resets','PPO_joint_steps')),
        'long target provenance/source/owner differs')
    folder=Path(p['stage']['path']).parent
    require(folder==rows.STORAGE_ROOT.resolve()/digest(expected_owner)/'stages'/p['stage_id']
        and Path(target_ref['path'])==folder/'published-target/result.json','long target escaped its new exact stage')
    names=dict(stage='stage.json',stage_closed='storage-stage-closed.json',rows_deduplication='rows-deduplication.json',
        npz_materialization='npz-materialization.json',temporal_cleanup='temporal-inputs-consumed.json')
    require(all(Path(p[key]['path'])==folder/name for key,name in names.items())
        and Path(target['long_materialization_provenance']['path'])==folder/'materialization.json'
        and Path(p['npz_baselines']['path'])==folder.parents[1]/'npz-baselines.json'
        and p['variant'] in ('raw','regularized'),'storage receipts escaped their exact registered stage')
    required=[p[k] for k in ('stage','stage_closed','rows_deduplication','npz_materialization','temporal_cleanup','npz_baselines')]
    for ref in [*p['persistent_dependencies'],*required,*target['asset_refs'],target['target_actor'],target['observed_source'],expected_owner]:_immutable(ref)
    header=bound(p['stage']);closed=bound(p['stage_closed']);clean=bound(p['temporal_cleanup'])
    row_receipt=bound(p['rows_deduplication']);npz_receipt=bound(p['npz_materialization']);base=bound(p['npz_baselines'])
    require(header['version']==rows.VERSION and header['owner_sha256']==digest(expected_owner)
        and header['stage_id']==p['stage_id'] and header['probe_only'] is False
        and header['source_files']==rows.source_files() and closed['version']==rows.VERSION and closed['stage']==p['stage']
        and closed['temporal_npz_consumed'] is True and row_receipt['stage']==clean['stage']==npz_receipt['stage']==p['stage']
        and row_receipt['version']==clean['version']==rows.VERSION and npz_receipt['version']==VERSION
        and clean['only_new_stage_private_paths_removed'] is True and clean['historical_source_bytes_deleted'] is False
        and clean['before_final_target_binding'] is True and base['version']==VERSION and base['source_files']==source_files()
        and npz_receipt['baseline_ledger']==p['npz_baselines'] and npz_receipt['temporal_cleanup']==p['temporal_cleanup'],
        'stage/cleanup/baseline lifecycle differs')
    _immutable(base['first_stage'])
    first=Path(base['first_stage']['path']);first_value=bound(base['first_stage'])
    require(first.parent.parent==folder.parent and first.name=='stage.json'
        and first_value['owner_sha256']==digest(expected_owner) and first_value['probe_only'] is False
        and first_value['source_files']==rows.source_files(),'first-call baseline stage differs')
    require(base['immutable_permanent_first_future_call'] is True and base['historical_assets_adopted'] is False,
        'baseline is not a permanent newly generated future-call input')
    require(set(p['final_arrays'])==set(base['baselines'])==rows.NPZ_NAMES
        and set(p['final_rows'])==rows.ROW_NAMES and len(npz_receipt['NPZ'])==12
        and len(clean['consumed_temporal_inputs'])==12 and len(row_receipt['rows'])==8,
        'finite materialization inventory differs')
    temporal={r['stage_relative_path']:r for r in clean['consumed_temporal_inputs']}
    require(len(temporal)==12,'duplicate temporal input')
    for report in npz_receipt['NPZ']:
        require('target' not in report,'temporary helper input cannot be a persistent pointer')
        item=report['consumed_temporal_input'];name=Path(item['stage_relative_path']).name
        require(name in rows.NPZ_NAMES and item['stage_relative_path']=='unbound-target/'+name
            and item['lifecycle']=='created_and_consumed_before_final_binding' and not (folder/item['stage_relative_path']).exists()
            and item['persisted_equivalent']==report['output']==p['final_arrays'][name]
            and report['base']==base['baselines'][name] and item['sha256']==report['output']['sha256']
            and item['bytes']==Path(report['output']['path']).stat().st_size
            and temporal[item['stage_relative_path']]==dict(item,mode=0o444)
            and report['kernel_clone_succeeded'] is report['source_bytes_and_modes_unchanged'] is True
            and report['ordinary_copy_fallback'] is False and report['kernel']=='APFS clonefile'
            and report['version']==clones.VERSION and report['source_bound_long_integration'] is True,
            'NPZ consumed-input/final-output/clone facts differ')
        require(Path(report['output']['path'])==folder/'published-target'/name
            and Path(report['base']['path'])==folder.parents[1]/'npz-baselines'/name,'NPZ output or baseline escaped owner')
        _immutable(report['base']);_immutable(report['output'])
        with open(report['base']['path'],'rb') as left,open(report['output']['path'],'rb') as right:
            shared=clones.shared_extent_bytes(clones.extent_map(left.fileno()),clones.extent_map(right.fileno()))
        require(type(report['shared_physical_extent_bytes']) is int and 0<=report['shared_physical_extent_bytes']<=shared,
            'current APFS extents do not support the recorded physical sharing')
        with np.load(report['output']['path'],allow_pickle=False) as package:
            arrays=[dict(name=k,dtype=package[k].dtype.str,shape=list(package[k].shape),bitwise_equal=True) for k in package.files]
        require(arrays==report['npz_arrays'],'final NPZ array inventory/dtype/shape differs')
    for record in row_receipt['rows']:
        name=Path(record['path']).name;final=p['final_rows'][name]
        require(name in rows.ROW_NAMES and Path(record['path'])==folder/'unbound-target'/name
            and Path(final['path'])==folder/'published-target'/name,'rows final/source path differs')
        cas=Path(record['CAS']);_immutable(dict(path=str(cas),sha256=record['sha256']))
        _immutable(dict(path=record['path'],sha256=record['sha256']))
        require(cas==folder.parents[1]/'rows'/(record['sha256']+'.rows.json') and final['sha256']==record['sha256'],
            'rows CAS location/hash differs')
        a,b,c=Path(record['path']).stat(),Path(final['path']).stat(),cas.stat()
        require((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)==(c.st_dev,c.st_ino) and c.st_nlink>=3
            and stat.S_IMODE(a.st_mode)==stat.S_IMODE(b.st_mode)==stat.S_IMODE(c.st_mode)==0o444,
            'rows physical hardlink sharing differs')
    require(closed['operations']==row_receipt['rows'] and row_receipt['physical_hardlinks_verified'] is True,
        'rows stage close or physical-sharing ledger differs')
    semantic=[]
    for split in native.online.SPLITS:
        require(target['actual_source_actions'][split]==p['final_arrays'][split+'-source-actual-actions.npz'],
            'source actions differ from storage materialization')
        semantic.append(target['actual_source_actions'][split])
    for role in native.AGENTS:
        manifest=bound(target['pools'][role]);semantic.append(target['pools'][role])
        require(manifest['target_actor']==target['target_actor'] and manifest['role']==role
            and Path(target['pools'][role]['path'])==folder/'published-target'/(role+'-pool.json'),
            'final semantic pool escaped materialized package')
        for split in native.online.SPLITS:
            expected_part=dict(arrays=p['final_arrays'][role+'-'+split+'.npz'],rows=p['final_rows'][role+'-'+split+'-rows.json'])
            require(manifest['parts'][split]==expected_part,'native semantic pool refers to other numerical files')
            semantic.extend(expected_part.values())
    require({digest(r) for r in semantic}=={digest(r) for r in target['asset_refs']},
        'semantic asset inventory differs from materialized target files')
    cas_refs=[dict(path=r['CAS'],sha256=r['sha256']) for r in row_receipt['rows']]
    expected=[*required,base['first_stage'],*cas_refs,*base['baselines'].values(),*p['final_arrays'].values(),*p['final_rows'].values()]
    require({digest(r) for r in p['persistent_dependencies']}=={digest(r) for r in expected}
        and {digest(r) for r in clean['persistent_dependencies']}=={digest(r) for r in p['final_arrays'].values()},
        'persistent dependencies missing or extra')
    seal=pointer(Path(target_ref['path']).with_name('completed.json'));_immutable(seal)
    require(bound(seal)==dict(result_sha256=target_ref['sha256']),
        'target completion seal differs')
    return dict(provenance=p,provenance_ref=target['long_materialization_provenance'],
        inputs=[target_ref,target['long_materialization_provenance'],*p['persistent_dependencies']],
        canonical_NN_replay_still_required=True)
