"""Independent registered fresh-pool audit, never an Actor-metadata rewrite.

expected_pool_manifest_sha256 anchors the exact bytes of pool_root/pool.json
(there is NO separate pool manifest file). expected_exclusions_sha256 anchors
pool_root/exclusions.json, a catalog of previously registered fingerprint
indices. Each index is separately SHA/size bound and retains its origin file
bindings; this reader verifies complete catalog sets, not an asserted complete
boolean. Producing/trusting those original indices is a separate preregistered
source-verification responsibility, not a way to infer missing final content.
expected_source_index_acceptance_sha256 independently anchors the prerequisite
pool_root/source_index_acceptance.json produced by actual source verification.

run returns plan_sha256 and manifest_sha256. read_completed requires both plus
the original expected_bindings; it imports no model or environment instance and
recalculates saved NN evidence/tree metrics, not another physical/NN replay.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

from backend import warehouse_runtime_family as registry
from backend import warehouse_family_explanation as renderer
from backend.training import warehouse_family_explanation_audit as original
from backend.training import warehouse_family_explanation_run as old_run
from backend.training import warehouse_family_answer_run as io
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from core.program import ExecutableProgram
from env.warehouse.layouts import get_map_layout
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS
from env.warehouse_native.scenarios import SPLIT_NAMES, scenario_fingerprint

VERSION='warehouse-family-fresh-explanation-run.v1'
POOL_VERSION='warehouse-family-fresh-explanation-pool.v1'
EXCLUSIONS_VERSION='warehouse-family-explanation-exclusions.v1'
INDEX_VERSION='warehouse-family-registered-initial-fingerprint-index.v1'
INDEX_ACCEPTANCE_VERSION='warehouse-family-explanation-source-index-acceptance.v1'
REGISTERED_COUNTS={'train':512,'calibration':100,'validation':50,'extraction':100,'explanation_test':100,'final_test':100,'play':12}
INDEX_CONTRACT={'version':INDEX_ACCEPTANCE_VERSION,'external_source_acceptance_required':True,
    'source_indices_require_prior_actual_origin_verification':True,'original_registered_counts':REGISTERED_COUNTS,
    'final_snapshot_content_rebuilt_by_this_component':False}
REGISTRY_ROOT=ROOT/'output/warehouse_native/.warehouse_family_fresh_explanation_consumption'
PHASES=('base','counterfactual')
PARTNERS=original.PARTNERS
_equal=original._equal
_put=original._put
_read=original._read
_groups=original._groups
_check_decision=original._check_decision
_summarize=original._summarize
physical_projection=original.physical_projection


def _sha(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('Explicit lowercase SHA256 required')


def sources():
    result=old_run.sources()
    for path in (Path(__file__),ROOT/'env/warehouse_native/scenarios.py'):
        result[str(path.relative_to(ROOT))]=file_hash(path)
    return result


def contract():
    return {'version':VERSION,'metrics_contract':original.contract(),
        'pool_version':POOL_VERSION,'exclusions_version':EXCLUSIONS_VERSION,
        'source_index_contract':deepcopy(INDEX_CONTRACT),
        'source_manifest_identity':'original Actor metadata; separate fresh evaluation pool',
        'pool_generation':'not implemented here; preregister physical rules only',
        'qualification_evaluated':False}


def _bytes(root,relative,anchor=None,size=None):
    part=Path(relative)
    if part.is_absolute() or '..' in part.parts or str(part)!=relative:raise ValueError('Unsafe registered input path')
    path=root/part
    if path.resolve()!=path or not path.is_file():raise ValueError('Missing, linked or nonregular input')
    raw=path.read_bytes()
    if anchor is not None:
        _sha(anchor)
        if sha256(raw).hexdigest()!=anchor:raise ValueError('Registered input SHA differs')
    if size is not None and (type(size) is not int or len(raw)!=size):raise ValueError('Registered input size differs')
    return raw


def _fingerprint(scene,configuration):
    snapshot=scene['snapshot'];state=snapshot['state']
    if (snapshot['configuration']!=configuration or state['frame']!=0
            or state['terminated'] is not False or state['truncated'] is not False
            or any(k.startswith('public_feedback') for k in snapshot)):
        raise ValueError('Fresh pool requires original raw frame-zero physical starts')
    layout=get_map_layout(configuration['map_layout_id'])
    agents=[SimpleNamespace(**a) for a in state['agents']]
    tasks=[SimpleNamespace(**t) for t in state['tasks']]
    if (len(agents)!=2 or [a.agent_id for a in agents]!=['robot_1','robot_2']
            or len({tuple(a.position) for a in agents})!=2
            or any(not layout.is_passable(tuple(a.position)) or not 0<=a.battery<=100 for a in agents)):
        raise ValueError('Invalid fresh physical initial robots')
    proxy=SimpleNamespace(state=SimpleNamespace(agents=agents,tasks=tasks),layout=layout,config=SimpleNamespace(**configuration))
    actual=scenario_fingerprint(proxy)
    if actual!=scene['fingerprint']:raise ValueError('Fresh physical fingerprint differs')
    return actual


def _fingerprints(pool):
    if not isinstance(pool,list) or not pool or any(type(x) is not str for x in pool):raise ValueError('Nonempty registered fingerprint set required')
    for value in pool:_sha(value)
    if len(set(pool))!=len(pool):raise ValueError('Duplicate registered fingerprints')
    return set(pool)


def _pool(root,pool_anchor,exclusions_anchor,acceptance_anchor,fixture):
    """Pure index and initial-state checks; no final-state content is opened."""
    _sha(pool_anchor);_sha(exclusions_anchor);_sha(acceptance_anchor)
    root=Path(root).expanduser().absolute()
    if root.resolve()!=root:raise ValueError('Pool path must not traverse symlinks')
    blobs={'pool.json':_bytes(root,'pool.json',pool_anchor),'exclusions.json':_bytes(root,'exclusions.json',exclusions_anchor)}
    pool,excluded=(json.loads(blobs[x]) for x in ('pool.json','exclusions.json'))
    if (pool.get('version')!=POOL_VERSION or excluded.get('version')!=EXCLUSIONS_VERSION
            or pool.get('test_fixture') is not fixture or excluded.get('test_fixture') is not fixture
            or pool.get('purpose')!='independent_explanation_test'
            or pool.get('exclusions_sha256')!=exclusions_anchor):raise ValueError('Fresh pool/exclusion producer differs')
    _sha(pool['source_scenario_manifest_sha256'])
    _equal(excluded['source_scenario_manifest_sha256'],pool['source_scenario_manifest_sha256'],'Original source manifest differs')
    records=excluded['registered_indices']
    if not isinstance(records,list) or len(records)<3:raise ValueError('Original, exposed-heldout and development indices are mandatory')
    acceptance_binding=excluded['source_index_acceptance']
    if acceptance_binding.get('path')!='source_index_acceptance.json' or acceptance_binding.get('sha256')!=acceptance_anchor:
        raise ValueError('Independent prior source-index acceptance anchor required')
    acceptance_raw=_bytes(root,'source_index_acceptance.json',acceptance_anchor,acceptance_binding['size']);blobs['source_index_acceptance.json']=acceptance_raw
    acceptance=json.loads(acceptance_raw)
    if (acceptance.get('version')!=INDEX_ACCEPTANCE_VERSION or acceptance.get('test_fixture') is not fixture
            or acceptance.get('status')!='source_indices_verified'
            or acceptance.get('source_scenario_manifest_sha256')!=pool['source_scenario_manifest_sha256']):raise ValueError('Prior source-index acceptance identity differs')
    _equal(acceptance['index_contract'],INDEX_CONTRACT,'Source-index provenance contract differs')
    _equal(acceptance['registered_indices'],records,'Previously verified complete source-index catalog differs')
    required_checks={'original_source_bytes_and_index_sets','all_original_seven_pool_counts','all_previously_exposed_explanation_pools','all_actual_training_tree_intervention_sources'}
    if (set(acceptance.get('checks',{}))!=required_checks or any(acceptance['checks'][k] is not True for k in required_checks)
            or not isinstance(acceptance.get('verifier_sources'),dict) or not acceptance['verifier_sources']):raise ValueError('Actual origin-index verification is missing or failed')
    for value in acceptance['verifier_sources'].values():_sha(value)
    scopes={'source_scenarios':[],'used_explanation_test':[],'development':[]};all_used=set();catalog=[]
    for entry in records:
        if entry['scope'] not in scopes or entry['path'] in blobs:raise ValueError('Duplicate/unknown registered exclusion index')
        raw=_bytes(root,entry['path'],entry['sha256'],entry['size']);blobs[entry['path']]=raw;index=json.loads(raw)
        if (index.get('version')!=INDEX_VERSION or index.get('test_fixture') is not fixture or index.get('scope')!=entry['scope']
                or index.get('source_scenario_manifest_sha256')!=pool['source_scenario_manifest_sha256']):raise ValueError('Registered index origin differs')
        origins=index['origin_bindings']
        if not isinstance(origins,dict) or not origins:raise ValueError('Registered index lacks immutable source anchors')
        for name,binding in origins.items():
            if not isinstance(name,str) or not name:raise ValueError('Index origin name required')
            _sha(binding['sha256'])
            if type(binding['size']) is not int or binding['size']<1:raise ValueError('Index origin size required')
        if entry['scope']=='source_scenarios' and origins.get('scenarios',{}).get('semantic_sha256')!=pool['source_scenario_manifest_sha256']:
            raise ValueError('Original index must bind the actual Actor scenario semantic SHA')
        required_origins={'source_scenarios':{'scenarios'},'used_explanation_test':{'audit_inputs','audit_manifest'},'development':{'plan','manifest'}}[entry['scope']]
        if not required_origins<=set(origins):raise ValueError('Required source artifacts omitted from index')
        sets={name:_fingerprints(values) for name,values in index['pools'].items()}
        if not sets or index['counts']!={k:len(v) for k,v in sets.items()}:raise ValueError('Index complete pool counts differ')
        _equal(entry['pools_sha256'],digest(index['pools']),'Preregistered index pool sets differ')
        _equal(entry['origin_bindings'],origins,'Preregistered source anchors omitted or changed')
        scopes[entry['scope']].append(sets);all_used.update(set().union(*sets.values()));catalog.append({k:deepcopy(entry[k]) for k in ('path','sha256','size','scope','pools_sha256','origin_bindings')})
    if len(scopes['source_scenarios'])!=1 or not scopes['used_explanation_test'] or not scopes['development']:
        raise ValueError('Exclusion catalog omitted an entire used-data family')
    registered=scopes['source_scenarios'][0]
    if set(registered)!=set(SPLIT_NAMES) or sum(map(len,registered.values()))!=len(set().union(*registered.values())):
        raise ValueError('Original seven registered pools are incomplete or overlap')
    if not fixture and {k:len(v) for k,v in registered.items()}!=REGISTERED_COUNTS:raise ValueError('All original registered split counts must be exact')
    exposed=set().union(*(set().union(*x.values()) for x in scopes['used_explanation_test']))
    if not registered['explanation_test']<=exposed:raise ValueError('Old exposed/failed heldout was omitted')
    if not fixture and any(len(x)!=100 for pools in scopes['used_explanation_test'] for x in pools.values()):raise ValueError('Used heldout must preserve every original hundred scenes')
    scenes=pool['scenes'];configuration=pool['configuration'];horizon=configuration['horizon']
    if (type(horizon) is not int or (not fixture and (len(scenes)!=100 or horizon!=120))
            or (fixture and (not 1<=len(scenes)<=2 or not 1<=horizon<=3))):raise ValueError('Fixed fresh pool size/horizon differs')
    if type(pool['pool_id']) is not str or not re.fullmatch('[A-Za-z0-9_-]{3,80}',pool['pool_id']):raise ValueError('Independent pool ID required')
    expected_ids=[f"explanation_test_{pool['pool_id']}_{i:04d}" for i in range(len(scenes))]
    if [s['id'] for s in scenes]!=expected_ids:raise ValueError('Fresh pool order/names differ')
    fingerprints=[_fingerprint(s,configuration) for s in scenes]
    if len(set(fingerprints))!=len(scenes) or set(fingerprints)&all_used:raise ValueError('Fresh initial content overlaps registered training/development/exposed/final/play pools')
    _equal(pool['initial_fingerprints_sha256'],digest(fingerprints),'Fresh physical matrix binding differs')
    if not isinstance(pool['generation'],dict) or pool['generation'].get('selection_rule')!='physical_uniqueness_and_registered_exclusions_only':raise ValueError('Physical-only preregistered generation rule required')
    _sha(pool['generation']['sources_sha256'])
    if type(pool['generation'].get('seed_start')) is not int or pool['generation']['seed_start']<0:raise ValueError('Fixed nonnegative generation seed required')
    if not fixture and any(a['battery'] not in (60,70,80,90,100) for s in scenes for a in s['snapshot']['state']['agents']):raise ValueError('Original initial battery support changed')
    return {'pool':pool,'exclusions':excluded,'blobs':blobs,'catalog':catalog,'fingerprints':fingerprints}


def input_bindings(runtime,program_path,pool_root,*,expected_pool_manifest_sha256,expected_exclusions_sha256,expected_source_index_acceptance_sha256,allow_test_fixture=False):
    identity=registry.verify(runtime,allow_test_fixture=allow_test_fixture)
    material=_pool(pool_root,expected_pool_manifest_sha256,expected_exclusions_sha256,expected_source_index_acceptance_sha256,allow_test_fixture);pool=material['pool']
    _equal(pool['configuration'],asdict(runtime.config),'Pool/runtime original configuration differs')
    _equal(pool['source_scenario_manifest_sha256'],runtime.actor.metadata['scenario_manifest_sha256'],'Original Actor scenario provenance differs')
    return {'runtime_signature':runtime.signature,'runtime_family':identity['family'],'runtime_version':identity['runtime_version'],
        'actor_sha256':runtime.actor_sha256,'actor_metadata_sha256':digest(runtime.actor.metadata),
        'actor_weights_sha256':runtime._weight_digest(),'feature_names_sha256':digest(runtime.actor.metadata['feature_names']),
        'program_sha256':file_hash(program_path),'protocol_sha256':runtime.protocol_sha256,
        'source_scenario_manifest_sha256':pool['source_scenario_manifest_sha256'],
        'evaluation_pool_sha256':expected_pool_manifest_sha256,'exclusions_sha256':expected_exclusions_sha256,
        'source_index_acceptance_sha256':expected_source_index_acceptance_sha256,
        'initial_fingerprints_sha256':pool['initial_fingerprints_sha256'],'contract_sha256':digest(contract()),'sources_sha256':digest(sources())}


def _caps(pool,fixture):
    return {'base':len(pool['scenes'])*3*pool['configuration']['horizon'],'counterfactual':len(pool['scenes'])*3*5} if fixture else {'base':36000,'counterfactual':18000}


def _key(bindings,configuration):
    # No output, execution ID, pool labels, NPZ packaging or tree bytes can
    # reopen the same Actor/physical-matrix test consumption.
    return digest({k:bindings[k] for k in ('actor_weights_sha256','feature_names_sha256','initial_fingerprints_sha256','contract_sha256')}|{'configuration':configuration})


class Journal(old_run.Journal):
    """Same durable step protocol, explicit producer version in new records."""
    def before(self,context):
        _equal(io._read(self.path),self.state,'Journal changed before reservation')
        state=self.state;phase=context['phase'];index=sum(state['reserved'].values())
        if (state['status']!='running' or state['pending'] is not None or state['reserved']!=state['acknowledged']
                or phase not in PHASES or state['reserved'][phase]>=self.plan['caps'][phase]):raise ValueError('Unconfirmed or exhausted fresh audit')
        if context['operation_id']!=f"{self.plan['execution_id']}:{phase}:{index:06d}" or context['reserved_steps']!=1:raise ValueError('Unregistered fresh operation')
        path=self.root/'reservations'/f'{index:06d}.json'
        io._write(path,{'version':VERSION,'context':deepcopy(context),'permanent':True,'refund_allowed':False})
        changed=deepcopy(state);changed['reserved'][phase]+=1;changed['pending']={'index':index,'reservation':io._binding(self.root,path)};self.commit(changed)
        return True
    def after(self,completed):
        _equal(io._read(self.path),self.state,'Journal changed before ACK')
        if self.state['status']!='running' or self.state['pending'] is None:raise ValueError('No pending fresh reservation')
        pending=self.state['pending'];index=pending['index'];reserved=io._bound(self.root,pending['reservation'],f'reservations/{index:06d}.json');context=reserved['context']
        for k,v in context.items():_equal(completed.get(k),v,'Completion context differs')
        path=self.root/'audit/steps'/f'{index:06d}.json.gz'
        if (completed['status']!='executed_unacknowledged' or type(completed['actual_steps']) is not int or completed['actual_steps']!=1
                or Path(completed['record_path'])!=path or file_hash(path)!=completed['record_sha256']):raise ValueError('Complete durable raw step required')
        old_run._validate_raw(io._read(path),context)
        io._write(self.root/'confirmations'/f'{index:06d}.json',{'version':VERSION,'reservation':pending['reservation'],
            'record':io._binding(self.root,path),'completion':deepcopy(completed)})
        changed=deepcopy(self.state);changed['acknowledged'][context['phase']]+=1;changed['pending']=None;self.commit(changed)
        return True


# Saved matrix mechanics retained; explicit fresh producer/header/source checks.
def _recompute(output, header, manifest):
    """Hash-bound saved facts only; neither restore/step nor NN load/forward."""
    _equal(header["contract"], contract(), "contract_changed")
    _equal(header["sources"], sources(), "sources_changed")
    if digest(header["sources"]) != header["bindings"]["sources_sha256"]: raise ValueError("source_binding_changed")
    if header["version"] != VERSION or manifest["version"] != VERSION: raise ValueError("audit_version_differs")
    if any(header.get(k) is not False for k in ("qualification_evaluated", "explanation_eligible", "participant_enabled")):
        raise ValueError("component_cannot_grant_qualification")
    if sha256(header["program_file_text"].encode()).hexdigest() != header["bindings"]["program_sha256"]:
        raise ValueError("original_program_bytes_differ")
    program = ExecutableProgram.from_dict(header["program"])
    raw = json.loads(header["program_file_text"])
    _equal(program.to_dict(), raw["program"] if raw.get("version") == "warehouse_native_rcpd_feedback_v1" else raw, "program_wrapper_differs")
    entries = manifest["entries"]; base = {}; anchors = {}; branches = {}; rows = []; physical_steps = {"base": 0, "counterfactual": 0}
    for index, entry in enumerate(entries):
        relative = f"steps/{index:06d}.json.gz"
        if entry["path"] != relative or entry["actual_steps"] != 1: raise ValueError("noncanonical_step_entry")
        path = output / relative
        if path.is_symlink() or file_hash(path) != entry["sha256"]: raise ValueError("step_hash_differs")
        _equal(_read(output / "acks" / f"{index:06d}.json"), entry, "durable_ack_differs")
        record = _read(path); phase = record["phase"]
        if phase not in physical_steps or record["operation_id"] != entry["operation_id"] or phase != entry["phase"]:
            raise ValueError("operation_binding_differs")
        physical_steps[phase] += 1
        if digest(record["before"]) != record["before_sha256"]: raise ValueError("before_hash_differs")
        before, after = record["before"]["state"], record["after"]["state"]
        if before["terminated"] or before["truncated"] or after["frame"] != before["frame"] + 1:
            raise ValueError("invalid_transition_frames")
        if record["frame"] != before["frame"] or record["decision"]["frame"] != before["frame"]:
            raise ValueError("decision_frame_differs")
        _equal(record["groups"], _groups(record["before"]), "saved_groups_differ")
        _check_decision(record["decision"], program, header["bindings"])
        _check_decision(record["next_decision"], program, header["bindings"])
        if (record["submitted_actions"]["robot_2"] != record["decision"]["neural_action"]
                or record["info"]["requested_actions"] != record["submitted_actions"]): raise ValueError("NN_override_in_saved_trace")
        _equal(record["executed_actions"], record["info"]["executed_actions"], "physical_actions_differ")
        _equal(record["events"], record["info"]["events"], "events_differ")
        for name, state in (("before_view", before), ("after_view", after)):
            _equal(physical_projection(record[name]), physical_projection(state), "physical_projection_differs_from_snapshot")
        _equal([record["terminated"], record["truncated"]], [after["terminated"], after["truncated"]], "terminal_flags_differ")
        key = (record["partner"], record["scenario_index"])
        scene = header["scenes"][key[1]]
        if key[0] not in PARTNERS or (record["scenario_id"], record["initial_fingerprint"], record["seed"]) != (scene["id"], scene["fingerprint"], 17000 + key[1]):
            raise ValueError("scene_binding_differs")
        if phase == "base":
            sequence = base.setdefault(key, {"steps": 0, "last_after_sha256": None, "done": False})
            if before["frame"] != sequence["steps"]: raise ValueError("incomplete_base_prefix")
            if sequence["steps"]:
                _equal(digest(record["before"]), sequence["last_after_sha256"], "base_snapshot_chain_broken")
            else:
                # Initial observed wrapper is extra; its physical state/RNG must retain source bytes.
                for field in ("state", "rng", "episode_counter"):
                    _equal(record["before"][field], scene["snapshot"][field], "initial_state_differs")
            if record["next_decision"] is not None: raise ValueError("unexpected_base_next_decision")
            sequence.update(steps=sequence["steps"] + 1, last_after_sha256=digest(record["after"]),
                done=bool(after["terminated"] or after["truncated"]))
            if not record["frame"] % 10 and record["groups"]:
                anchors[(*key, record["frame"])] = {"before_sha256": digest(record["before"]),
                    "decision_sha256": digest(record["decision"]), "groups": record["groups"],
                    "fingerprint": record["initial_fingerprint"]}
            rows.append({"fingerprint": scene["fingerprint"], "partner": key[0], "groups": record["groups"],
                "action": record["decision"]["neural_action"], "correct": record["decision"]["neural_action"] == record["decision"]["tree_action"]})
        else:
            action = record["intervention_action"]
            if record["submitted_actions"]["robot_1"] != action or action not in ACTIONS: raise ValueError("intervention_action_differs")
            anchor = (*key, before["frame"])
            pool = branches.setdefault(anchor, {})
            if action in pool: raise ValueError("duplicate_branch")
            if bool(record["next_decision"] is None) != bool(after["terminated"] or after["truncated"]): raise ValueError("missing_next_decision")
            if record["next_decision"] is not None and record["next_decision"]["frame"] != after["frame"]: raise ValueError("next_decision_frame_differs")
            if anchor not in anchors: raise ValueError("branch_precedes_unregistered_base_anchor")
            _equal(digest(record["before"]), anchors[anchor]["before_sha256"], "branch_did_not_start_from_anchor")
            _equal(digest(record["decision"]), anchors[anchor]["decision_sha256"], "branch_initial_NN_differs")
            pool[action] = {"physical_sha256": digest(physical_projection(record["after_view"])),
                "next": None if record["next_decision"] is None else {
                    "neural_action": record["next_decision"]["neural_action"], "tree_action": record["next_decision"]["tree_action"]}}
    expected_base = {(p, i) for p in PARTNERS for i in range(len(header["scenes"]))}
    if set(base) != expected_base: raise ValueError("incomplete_partner_scene_matrix")
    pairs = []
    for key, sequence in base.items():
        if not sequence["done"]: raise ValueError("base_episode_not_complete")
        if sequence["steps"] > header["configuration"]["horizon"]: raise ValueError("base_horizon_exceeded")
    for anchor, source in anchors.items():
        pool = branches.get(anchor, {})
        if set(pool) != set(ACTIONS): raise ValueError("five_branch_matrix_incomplete")
        baseline = pool["WAIT"]
        for action in ACTIONS:
            if action == "WAIT": continue
            changed = pool[action]
            if baseline["next"] is None or changed["next"] is None: continue
            b, c = baseline["next"], changed["next"]
            pairs.append({"fingerprint": source["fingerprint"], "partner": anchor[0], "groups": source["groups"],
                "action": action, "physical_effect": baseline["physical_sha256"] != changed["physical_sha256"],
                "nn_changed": b["neural_action"] != c["neural_action"],
                "correct": b["tree_action"] == b["neural_action"] and c["tree_action"] == c["neural_action"]})
    if set(branches) != set(anchors): raise ValueError("unexpected_branch_anchors")
    counts = manifest["execution"]["counts"]; caps = manifest["execution"]["caps"]
    expected_caps = {"base": 36000, "counterfactual": 18000} if not header["test_fixture"] else {
        "base": len(header["scenes"]) * 3 * header["configuration"]["horizon"], "counterfactual": len(header["scenes"]) * 3 * 5}
    _equal(caps, expected_caps, "fixed_caps_differ")
    for phase, actual in physical_steps.items():
        if actual != counts[phase + "_steps"] or actual != counts[phase + "_attempts"] or actual > caps[phase]: raise ValueError("actual_phase_accounting_differs")
    if counts["acknowledged_steps"] != len(entries) or manifest["execution"]["pending_operation"] is not None: raise ValueError("unconfirmed_records")
    if any(counts[k] != 0 for k in ("neural_updates", "tree_fits", "torch_loads")): raise ValueError("unexpected_training_or_fit")
    full = len(header["scenes"]) == 100 and header["configuration"]["horizon"] == 120 and not header["test_fixture"]
    return {"version": VERSION, "test_fixture": header["test_fixture"], "bindings": header["bindings"],
        "pool_id":header["pool_id"], "metrics_contract_version":original.VERSION,
        "statistics": _summarize(rows, pairs, full), "base_rows": len(rows), "branch_anchor_count": len(anchors),
        "episodes": len(base), "zero_NN_overrides": True, "execution": manifest["execution"],
        "saved_recalculation_scope": "hash-bound recorded logits/observations, scalar tree outputs, groups, matrices and metrics; not independent NN or physical replay",
        "qualification_evaluated": False, "explanation_eligible": False, "participant_enabled": False}


def _journal_read(root,plan,state):
    if (state.get('version')!=VERSION or state.get('plan_sha256')!=digest(plan)
            or state.get('status')!='completed' or state.get('pending') is not None
            or state.get('reserved')!=state.get('acknowledged')):raise ValueError('Pending/failed fresh audit cannot be used or rerun')
    counts={phase:0 for phase in PHASES};names=set()
    for index in range(sum(state['acknowledged'].values())):
        name=f'{index:06d}.json';names.add(name);ack=io._read(root/'confirmations'/name)
        reservation=io._bound(root,ack['reservation'],'reservations/'+name);ctx=reservation['context'];phase=ctx['phase']
        if (ack['version']!=VERSION or reservation['version']!=VERSION or reservation['permanent'] is not True
                or reservation['refund_allowed'] is not False or phase not in PHASES
                or ctx['operation_id']!=f"{plan['execution_id']}:{phase}:{index:06d}"
                or type(ctx['reserved_steps']) is not int or ctx['reserved_steps']!=1):raise ValueError('Saved fresh reservation differs')
        for key in ('evaluation_pool_sha256','source_scenario_manifest_sha256','exclusions_sha256'):
            _equal(ctx[key],plan['bindings'][key],'Step belongs to another source/pool')
        row=io._bound(root,ack['record'],f'audit/steps/{index:06d}.json.gz');old_run._validate_raw(row,ctx)
        for k,v in ctx.items():_equal(ack['completion'].get(k),v,'Saved ACK context differs')
        if (ack['completion']['record_sha256']!=ack['record']['sha256'] or ack['completion']['actual_steps']!=1
                or ack['completion']['status']!='executed_unacknowledged'):raise ValueError('Saved actual completion differs')
        # Original absolute producer path is provenance, not a new read target.
        if Path(ack['completion']['record_path'])!=Path(plan['original_output'])/ack['record']['path']:raise ValueError('Original producer output identity differs')
        counts[phase]+=1
    for directory in ('reservations','confirmations'):
        if {x.name for x in (root/directory).iterdir()}!=names:raise ValueError('Missing/extra original journal entries')
    if counts!=state['acknowledged'] or any(counts[p]>plan['caps'][p] for p in PHASES):raise ValueError('Finite journal totals differ')
    return counts


def read_completed(output,*,expected_plan_sha256,expected_manifest_sha256,expected_bindings):
    """Strict original records only: no runtime construction, NN, PT or step."""
    _sha(expected_plan_sha256);_sha(expected_manifest_sha256)
    root=Path(output).expanduser().absolute()
    if root.resolve()!=root:raise ValueError('Completed path is linked')
    for name in ('reservations','confirmations','audit','audit/steps','audit/acks'):
        path=root/name
        if path.resolve()!=path or not path.is_dir():raise ValueError('Linked or missing evidence directory')
    plan=json.loads(_bytes(root,'plan.json',expected_plan_sha256))
    if plan.get('version')!=VERSION:raise ValueError('Fresh producer required')
    _equal(plan['bindings'],expected_bindings,'External fresh audit bindings differ')
    _equal(plan['sources'],sources(),'Audit execution sources changed')
    _equal(plan['contract'],contract(),'Fresh metrics/pool contract changed')
    fixture=plan['test_fixture']
    if type(fixture) is not bool:raise ValueError('Explicit saved fixture scope required')
    inputs=_pool(root/'inputs',expected_bindings['evaluation_pool_sha256'],expected_bindings['exclusions_sha256'],expected_bindings['source_index_acceptance_sha256'],fixture);pool=inputs['pool']
    _equal(plan['input_files'],{name:{'sha256':sha256(raw).hexdigest(),'size':len(raw)} for name,raw in inputs['blobs'].items()},'Original index inventory differs')
    _equal(plan['caps'],_caps(pool,fixture),'Permanent full phase caps differ')
    for key in ('source_scenario_manifest_sha256','initial_fingerprints_sha256'):_equal(pool[key],expected_bindings[key],'Pool/source semantic identity differs')
    _equal(expected_bindings['contract_sha256'],digest(contract()),'Metrics contract anchor differs')
    _equal(expected_bindings['sources_sha256'],digest(sources()),'Source closure anchor differs')
    claim=io._read(root/'consumption.json')
    if (claim.get('version')!=VERSION or claim['consumption_key']!=_key(expected_bindings,pool['configuration'])
            or claim['plan_sha256']!=expected_plan_sha256 or claim['reserved_caps']!=plan['caps']
            or claim['automatic_retry'] is not False or claim['refund_allowed'] is not False):raise ValueError('Permanent sampling claim differs')
    if any((root/p).exists() for p in ('failure.json','audit/failed.json','audit/pending.json')):raise ValueError('Failed/incomplete sampling cannot be accepted')
    state=io._read(root/'state.json');counts=_journal_read(root,plan,state)
    _equal(state['manifest_sha256'],expected_manifest_sha256,'Completed manifest anchor differs')
    manifest=json.loads(_bytes(root,'audit/manifest.json',expected_manifest_sha256));header=json.loads(_bytes(root,'audit/inputs.json',manifest['header_sha256']))
    _equal(manifest['consumption_sha256'],file_hash(root/'consumption.json'),'Original permanent claim bytes differ')
    for name,suffix in (('steps','.json.gz'),('acks','.json')):
        if {p.name for p in (root/'audit'/name).iterdir()}!={f'{i:06d}'+suffix for i in range(len(manifest['entries']))}:raise ValueError('Extra/missing raw audit entries')
    _equal(header['bindings'],expected_bindings,'Saved NN evidence bindings differ')
    _equal(header['scenes'],pool['scenes'],'Full fresh scene order differs')
    _equal(header['configuration'],pool['configuration'],'Saved physical configuration differs')
    if header['pool_id']!=pool['pool_id'] or header['test_fixture'] is not fixture:raise ValueError('Saved pool/fixture producer differs')
    report=_recompute(root/'audit',header,manifest)
    _equal(json.loads(_bytes(root,'audit/report.json',manifest['report_sha256'])),report,'Saved summary differs from raw matrix')
    for phase in PHASES:
        if report['execution']['counts'][phase+'_steps']!=counts[phase]:raise ValueError('Raw/journal step counts differ')
    return {'version':VERSION,'status':'completed','plan_sha256':expected_plan_sha256,'manifest_sha256':expected_manifest_sha256,
        'audit_report':report,'reserved_steps':state['reserved'],'acknowledged_steps':counts,'caps':plan['caps'],
        'pool_sha256':expected_bindings['evaluation_pool_sha256'],'exclusions_sha256':expected_bindings['exclusions_sha256'],
        'consumption_sha256':file_hash(root/'consumption.json'),'original_registry_path':claim['registry_path'],
        'registry_reread_on_saved_verification':False,
        'source_index_contract':deepcopy(INDEX_CONTRACT),'source_index_acceptance_sha256':expected_bindings['source_index_acceptance_sha256'],
        'exclusion_verification_scope':'complete sets/counts and origin anchors from externally preregistered indices; no final-content or index-derivation replay',
        'qualification_evaluated':False,'explanation_eligible':False,'participant_enabled':False}


def run(runtime,program_path,pool_root,*,expected_pool_manifest_sha256,expected_exclusions_sha256,
        expected_source_index_acceptance_sha256,expected_bindings,output,execution_id,execution_permitted=False,allow_test_fixture=False):
    if execution_permitted is not True:raise ValueError('Explicit finite execution permission required')
    if type(execution_id) is not str or not re.fullmatch('[A-Za-z0-9_-]{3,80}',execution_id):raise ValueError('Execution ID required')
    bindings=input_bindings(runtime,program_path,pool_root,expected_pool_manifest_sha256=expected_pool_manifest_sha256,
        expected_exclusions_sha256=expected_exclusions_sha256,expected_source_index_acceptance_sha256=expected_source_index_acceptance_sha256,allow_test_fixture=allow_test_fixture)
    _equal(bindings,expected_bindings,'External fresh bindings differ')
    material=_pool(pool_root,expected_pool_manifest_sha256,expected_exclusions_sha256,expected_source_index_acceptance_sha256,allow_test_fixture);pool=material['pool']
    root=Path(output).expanduser().absolute()
    if root.resolve()!=root or root==Path(pool_root).resolve() or Path(pool_root).resolve() in root.parents:raise ValueError('Output must be separate and not linked')
    if root.exists():
        plan=io._read(root/'plan.json');state=io._read(root/'state.json')
        return read_completed(root,expected_plan_sha256=file_hash(root/'plan.json'),expected_manifest_sha256=state['manifest_sha256'],expected_bindings=bindings)
    program=renderer.FamilyExplainer(program_path,expected_program_sha256=bindings['program_sha256'],runtime=runtime,allow_test_fixture=allow_test_fixture).program
    caps=_caps(pool,allow_test_fixture)
    plan={'version':VERSION,'test_fixture':allow_test_fixture,'execution_id':execution_id,'original_output':str(root),
        'bindings':bindings,'sources':sources(),'contract':contract(),'caps':caps,
        'input_files':{name:{'sha256':sha256(raw).hexdigest(),'size':len(raw)} for name,raw in material['blobs'].items()},
        'automatic_retry':False,'refund_allowed':False,'qualification_evaluated':False}
    # A single atomic exclusive file permanently consumes this NN/matrix audit,
    # including failures before the first physical step. Output changes do not
    # grant a second reservation. No production registry is opened by readers.
    registry_root=Path(REGISTRY_ROOT).absolute()
    if registry_root.resolve()!=registry_root:raise ValueError('Registry path is linked')
    registry_root.mkdir(parents=True,exist_ok=True)
    key=_key(bindings,pool['configuration']);claim_path=registry_root/(key+'.json')
    plan_raw=original.canonical(plan).encode()
    # io._write uses canonical UTF-8 bytes without a newline.
    plan_anchor=sha256(plan_raw).hexdigest()
    claim={'version':VERSION,'consumption_key':key,'plan_sha256':plan_anchor,'reserved_caps':caps,
        'registry_path':str(claim_path),'automatic_retry':False,'refund_allowed':False,'test_fixture':allow_test_fixture}
    io._write(claim_path,claim)  # exclusive; no automatic retry even in another output
    root.mkdir(parents=True,exist_ok=False);io._sync(root.parent)
    for name,raw in material['blobs'].items():
        destination=root/'inputs'/name;destination.parent.mkdir(parents=True,exist_ok=True);io._write(destination,raw,raw=True)
    io._write(root/'plan.json',plan);_equal(file_hash(root/'plan.json'),plan_anchor,'Plan original bytes differ')
    io._write(root/'consumption.json',claim)
    for folder in ('reservations','confirmations','audit/steps','audit/acks'):(root/folder).mkdir(parents=True)
    state={'version':VERSION,'plan_sha256':digest(plan),'status':'running','reserved':dict.fromkeys(PHASES,0),
        'acknowledged':dict.fromkeys(PHASES,0),'pending':None,'manifest_sha256':None}
    io._write(root/'state.json',state);journal=Journal(root,plan)
    meter=original._Meter(root/'audit',execution_id,journal.before,journal.after,caps)
    header={'version':VERSION,'test_fixture':allow_test_fixture,'bindings':bindings,'contract':contract(),'sources':sources(),
        'pool_id':pool['pool_id'],'configuration':pool['configuration'],'scenes':pool['scenes'],
        'program':program.to_dict(),'program_file_text':Path(program_path).read_bytes().decode(),
        'qualification_evaluated':False,'explanation_eligible':False,'participant_enabled':False}
    io._write(root/'audit/inputs.json',header)
    try:
        private=registry.fresh_instance(runtime,allow_test_fixture=allow_test_fixture,expected_family=bindings['runtime_family'],expected_signature=runtime.signature)
        # Validate every original initial state before the first neural query.
        for scene in pool['scenes']:private.environment(scene)
        for partner in PARTNERS:
            for index,scene in enumerate(pool['scenes']):
                env=private.environment(scene);rng=np.random.default_rng(17000+index)
                while not env.done:
                    before=env.snapshot();decision=original._decision(env,private,program,meter);groups=_groups(before)
                    action=partner_action(env,'robot_1',partner,rng);_equal(env.snapshot(),before,'Program partner changed pre-state')
                    context={'phase':'base','partner':partner,'scenario_index':index,'scenario_id':scene['id'],
                        'initial_fingerprint':scene['fingerprint'],'seed':17000+index,'frame':before['state']['frame'],
                        **{k:bindings[k] for k in ('evaluation_pool_sha256','source_scenario_manifest_sha256','exclusions_sha256')}}
                    meter.step(env,private,program,decision,action,context)
                    if before['state']['frame']%10==0 and groups:
                        for action in ACTIONS:meter.step(private.from_snapshot(before),private,program,decision,action,{**context,'phase':'counterfactual','intervention_action':action})
                    if env.state.frame>private.config.horizon:raise ValueError('Horizon exceeded')
        _equal(sources(),plan['sources'],'Execution source changed')
        _equal(input_bindings(runtime,program_path,pool_root,expected_pool_manifest_sha256=expected_pool_manifest_sha256,
            expected_exclusions_sha256=expected_exclusions_sha256,expected_source_index_acceptance_sha256=expected_source_index_acceptance_sha256,allow_test_fixture=allow_test_fixture),bindings,'Input changed during audit')
        manifest={'version':VERSION,'header_sha256':file_hash(root/'audit/inputs.json'),'consumption_sha256':file_hash(root/'consumption.json'),'entries':meter.entries,'execution':meter.report(),'qualified':False}
        report=_recompute(root/'audit',header,manifest);io._write(root/'audit/report.json',report)
        manifest['report_sha256']=file_hash(root/'audit/report.json');io._write(root/'audit/manifest.json',manifest)
        done=deepcopy(journal.state);done.update(status='completed',manifest_sha256=file_hash(root/'audit/manifest.json'));journal.commit(done)
        return read_completed(root,expected_plan_sha256=plan_anchor,expected_manifest_sha256=done['manifest_sha256'],expected_bindings=bindings)
    except BaseException as error:
        failed=deepcopy(journal.state);failed['status']='failed'
        try:
            journal.commit(failed);io._write(root/'failure.json',{'version':VERSION,'reason':repr(error),'execution':meter.report(),
                'automatic_retry':False,'refund_allowed':False,'qualification_evaluated':False})
        except BaseException as persistence:error.journal_persistence_failure=str(persistence)
        raise
