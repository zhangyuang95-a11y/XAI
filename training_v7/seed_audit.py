"""Explicit v7 metadata scope around the frozen, parameterized numeric auditor.

The native audit code object and its inert triager are reused unchanged. Its
candidate-selection dependency is supplied in an isolated globals namespace;
the original module is never patched. Every output records BOTH the original
engine hash and this selection adapter hash. This is a new audit scope, not a
claim that the old Warehouse-only inventory covered Pong and Kitchen.
"""
import os
from pathlib import Path
import re
from types import FunctionType
from experiments.warehouse import seed_metadata_audit_indexed_v642 as engine

SCOPE = 'all_four_repositories_whitelisted_three_domain_metadata_and_registered_traces.v1'
ADDITIONAL = re.compile(r'registration|events|status|failure|closed', re.I)


def candidates(roots, excludes, *, registered_trace_paths=(), selection_stats=None):
    registered = {str(Path(p).resolve()) for p in registered_trace_paths}
    stats = selection_stats if selection_stats is not None else {}
    stats.setdefault('excluded_counts', {}); stats.setdefault('excluded_weights_name_stat_only', [])
    def skip(reason):
        stats['excluded_counts'][reason] = stats['excluded_counts'].get(reason, 0) + 1
    for root in roots:
        for directory, dirs, names in os.walk(Path(root).resolve(), followlinks=False):
            directory = Path(directory)
            dirs[:] = sorted(n for n in dirs if n not in engine.PRUNE and not any(
                (directory/n).resolve().is_relative_to(p) for p in excludes))
            for name in sorted(names):
                path = directory/name
                # Reject uninteresting millions of frame basenames before
                # doing filesystem resolution, stat calls or content reads.
                is_registered = str(path) in registered
                start_guard = name == 'started.json' or name.endswith('.started.json')
                claim_context = any('claim' in p.lower() for p in directory.parts)
                config_context = 'configs' in directory.parts
                whitelisted = bool(engine.METADATA_NAME.search(name) or ADDITIONAL.search(name)
                                   or start_guard or claim_context or config_context or is_registered)
                weight_directory = any(p.lower() in engine.OPAQUE_WEIGHT_DIRECTORIES for p in path.parts)
                weight_file = (path.suffix.lower() in engine.OPAQUE_WEIGHT_SUFFIXES or
                    (path.suffix.lower() == '.json' and engine.WEIGHT_NAME.search(name)
                     and not engine.WEIGHT_METADATA.search(name) and not start_guard))
                if weight_directory or weight_file:
                    skip('opaque_or_possible_json_weight_not_read')
                    info = path.stat(); stats['excluded_weights_name_stat_only'].append(dict(
                        path=str(path.resolve()), bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                        reason='opaque_model_directory_or_weight_filename_not_read'))
                    continue
                if path.suffix.lower() not in engine.EXTENSIONS or not whitelisted:
                    skip('basename_not_metadata_whitelist_or_registered_trace'); continue
                if path.is_symlink(): skip('symlink_not_followed'); continue
                if any(path.resolve().is_relative_to(p) for p in excludes): skip('explicit_exclusion'); continue
                if is_registered and (path.suffix.lower() != '.jsonl' or path.name not in
                    {'fit.jsonl', 'calibration.jsonl', 'eligibility.jsonl', 'validation.jsonl'}):
                    raise ValueError('unknown registered physical program trace format')
                yield path


def audit(proposal, roots, output):
    adapter_sha = engine.file_sha(__file__)
    namespace = dict(vars(engine))
    namespace.update(candidates=candidates, SELECTION_SCOPE=SCOPE)
    def write(path, value):
        if Path(path).name == 'inventory.json':
            value['selection_adapter'] = dict(version='three-domain-v7-seed-selection.v1',
                path=str(Path(__file__).resolve()), sha256=adapter_sha,
                native_engine_sha256=engine.file_sha(engine.__file__),
                isolated_dependency_injection=True, original_module_monkeypatch=False)
            if engine.file_sha(__file__) != adapter_sha: raise ValueError('seed selector source changed')
        return engine.write_new(path, value)
    namespace['write_new'] = write
    native_body = FunctionType(engine.audit.__code__, namespace, 'audit_v7_scope', engine.audit.__defaults__)
    return native_body(proposal, roots, output)
