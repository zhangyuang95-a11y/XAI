"""Source-bound contextual decoding of legacy CSV and inert trajectory metadata.

Never imports pickle globals or runs original constructors. Own first-use
registration is excluded only after proving this candidate has no execution
claim or Actor artifact; historical candidates remain in the scan.
"""
import ast
import csv
import gzip
import io
import json
from pathlib import Path
import numpy as np
from .contracts import file_sha, write_new
from .seed_audit import engine
from experiments.warehouse.seed_triage_indexed_v642 import InertUnpickler, InertRecord


class LabeledReader(InertUnpickler):
    def find_class(self, module, name):
        # This constructs a local inert record, NEVER the named original.
        return type('LiteralRecord', (InertRecord,), {'pickle_global': (module, name)})


def legacy_trajectory(path):
    raw = gzip.decompress(Path(path).read_bytes())
    if len(raw) > 256*1024*1024: raise ValueError('trajectory expansion bound')
    stream = io.BytesIO(raw); value = LabeledReader(stream).load()
    if stream.read() or value.get('format') != 'warehouse_collaborative_trajectory_v1':
        raise ValueError('unknown legacy trajectory format')
    if value.get('metadata', {}).get('source') != 'current_two_robot_training':
        raise ValueError('unknown trajectory producer')
    evidence = []; seen = set()
    def literal(v, path=()):
        if isinstance(v, (InertRecord, dict, list, tuple, set)):
            if id(v) in seen: return None
            seen.add(id(v))
        if isinstance(v, InertRecord):
            label = v.pickle_global
            args = getattr(v, 'constructor_args', ())
            if label == ('numpy._core.numeric', '_frombuffer'):
                if len(args) != 4 or not isinstance(args[0], (bytes, bytearray)) or args[3] != 'C':
                    raise ValueError('unknown numeric buffer encoding')
                dtype = args[1]
                if (not isinstance(dtype, InertRecord) or dtype.pickle_global != ('numpy', 'dtype')
                        or dtype.constructor_args != ('f4', False, True)
                        or tuple(dtype.literal_state) != (3, '<', None, None, None, -1, -1, 0)):
                    raise ValueError('unsupported array dtype')
                shape = args[2]
                if not isinstance(shape, tuple) or len(shape) != 1 or shape[0] not in (466, 940):
                    raise ValueError('unknown producer observation shape')
                a = np.frombuffer(args[0], dtype='<f4')
                if a.size != shape[0] or not np.isfinite(a).all(): raise ValueError('invalid public numeric buffer')
                evidence.append(dict(path=list(path), kind='complete_float32_observation', elements=a.size,
                                     bytes=len(args[0])))
                return [int(x) if float(x).is_integer() else float(x) for x in a]
            if label == ('torch._utils', '_rebuild_tensor_v2'):
                if (not path or not (path[-1] == 'policy_rng_state' or path[-2:] == ('rng_state','policy'))
                        or len(args) != 6):
                    raise ValueError('tensor outside producer policy RNG state')
                storage, offset, shape, stride, grad, hooks = args
                if (not isinstance(storage, InertRecord) or storage.pickle_global != ('torch.storage', '_load_from_bytes')
                        or len(storage.constructor_args) != 1 or not isinstance(storage.constructor_args[0], bytes)
                        or offset != 0 or shape != (5056,) or stride != (1,) or grad is not False
                        or not isinstance(hooks, InertRecord) or hooks.pickle_global != ('collections', 'OrderedDict')):
                    raise ValueError('unknown policy RNG tensor schema')
                from hashlib import sha256
                payload = storage.constructor_args[0]
                evidence.append(dict(path=list(path), kind='producer_policy_get_rng_state_not_seed_argument',
                    bytes=len(payload), sha256=sha256(payload).hexdigest(), state_bytes=5056))
                return {'producer_policy_rng_snapshot': True}
            return {k: literal(x, path+(k,)) for k,x in vars(v).items()}
        if isinstance(v, dict): return {str(k): literal(x, path+(str(k),)) for k,x in v.items()}
        if isinstance(v, (tuple, list, set)): return [literal(x, path+(i,)) for i,x in enumerate(v)]
        if isinstance(v, (bytes, bytearray)): raise ValueError('unrecognized opaque bytes')
        if v is None or isinstance(v, (str, bool, int, float)): return v
        raise ValueError('unknown literal object')
    return literal(value), evidence


def legacy_csv(path):
    rows = []
    with Path(path).open(newline='') as handle:
        reader = csv.reader(handle, strict=True); header = next(reader)
        if len(set(header)) != len(header) or header[0] != 'episode' or len(header) not in (52,54):
            raise ValueError('unknown legacy metrics CSV header')
        for record in reader:
            if len(record) != len(header): raise ValueError('ragged legacy metrics CSV')
            current = {}
            for key, text in zip(header, record, strict=True):
                try: value = ast.literal_eval(text)
                except (ValueError, SyntaxError): value = text
                def numbers(x):
                    if isinstance(x, float) and np.isfinite(x) and x.is_integer(): return int(x)
                    if isinstance(x, (tuple,list)): return [numbers(v) for v in x]
                    if isinstance(x,dict): return {str(k):numbers(v) for k,v in x.items()}
                    if isinstance(x,(str,int,float,bool)) or x is None: return x
                    raise ValueError('nonliteral CSV cell')
                current[key] = numbers(value)
            rows.append(current)
    return rows, dict(columns=header, complete_rows=len(rows))


def reconcile(proposal_path, triage_path, output):
    proposal_path = Path(proposal_path).resolve(); owner = proposal_path.parent
    if any(owner.rglob('claim.json')) or any(owner.rglob('*.actor.npz')):
        raise ValueError('own first-use context forbidden after any experiment claim')
    original = json.loads(Path(triage_path).read_text()); result = dict(original)
    proposal, families, pointer = engine.read_proposal(proposal_path)
    if original['proposal'] != pointer: raise ValueError('proposal differs')
    root = next(Path(p) for p in original['scan_roots'] if Path(p).name == 'XAI')
    producers = [root/p for p in ('backend/simulation/trajectory_store.py', 'backend/adapters/base.py',
        'backend/adapters/warehouse.py', 'backend/training/warehouse.py', 'core/policy_contracts.py')]
    # Bind exact producer text establishing that policy_rng_state is a snapshot,
    # not an initialization parameter. No original producer is imported.
    adapter = (root/'backend/adapters/warehouse.py').read_text()
    if 'deepcopy(policy.get_rng_state()) if policy is not None else None' not in adapter:
        raise ValueError('legacy RNG snapshot producer differs')
    result['contextual_producer_sources'] = [dict(path=str(p), sha256=file_sha(p)) for p in producers]
    result['contextual_decoder'] = dict(path=__file__, sha256=file_sha(__file__))
    result['context_parent'] = dict(path=str(Path(triage_path).resolve()), sha256=file_sha(triage_path))
    result['first_use_owner_exclusion'] = dict(path=str(owner), NN_claims=0, actor_artifacts=0,
        reason='this candidate registration, source snapshots and audit output only; not historical execution')
    result['inventory'] = []; resolved = []; own = []
    for old in original['inventory']:
        p = Path(old['path'])
        if p.is_relative_to(owner):
            own.append(dict(path=str(p), sha256=file_sha(p))); continue
        row = dict(old)
        if not row.get('resolved'):
            if file_sha(p) != row['sha256']: raise ValueError('legacy metadata changed')
            if p.read_bytes()[:2] == b'\x1f\x8b':
                value, evidence = legacy_trajectory(p); parser = 'v7_inert_labeled_float32_and_proven_rng_snapshot'
            else:
                value, evidence = legacy_csv(p); parser = 'v7_complete_legacy_CSV_52_or_54_columns'
            row.update(detail=engine.analyze_literal(value, families), parser=parser,
                       resolved=True, contextual_evidence=evidence)
            row.pop('triage_error', None); row.pop('opaque_pickle_fields', None)
            resolved.append(str(p))
        result['inventory'].append(row)
    result['own_registration_metadata'] = own
    result['contextually_resolved_paths'] = resolved
    result['summary'] = engine.summarize(result['inventory'], families,
        selection_stats=result['selection_stats'], missing_roots=result.get('missing_roots', ()))
    write_new(output, result)
    return result
