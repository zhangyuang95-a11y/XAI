"""An actual private local cluster, never an existing/production connection."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_study_vnext_postgres.py'


def test_real_local_postgres_three_domains_restore_and_concurrency(tmp_path):
    spec = importlib.util.spec_from_file_location('local_postgres_drill', SCRIPT)
    drill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drill)
    preflight = drill.prerequisites()
    if not preflight['available']:
        pytest.skip('real local PostgreSQL/driver unavailable: ' + ', '.join(preflight['missing']))
    output = tmp_path / 'synthetic-drill'
    # The command itself scrubs inherited PG configuration and uses no external
    # database address; stdout contains only the synthetic receipt location.
    result = subprocess.run([sys.executable, str(SCRIPT), '--output', str(output)],
                            capture_output=True, text=True, timeout=120,
                            env=dict(PATH=os.defpath, LC_ALL='C', LANG='C'))
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((output/'receipt.json').read_text())
    assert receipt['status'] == 'passed_synthetic_local_software_drill'
    assert receipt['synthetic_only'] is True and receipt['production_storage_receipt'] is False
    assert receipt['pilot_authorization'] is receipt['human_results'] is False
    assert receipt['owned_cluster_stopped'] and receipt['temporary_cluster_removed']
    assert receipt['existing_clusters_accessed'] is False and receipt['external_provider_calls'] == 0
    assert receipt['counts']['v642_instances'] == 9
    assert receipt['counts']['v642_runs'] == 27
    assert receipt['counts']['v642_commands'] == 54
    assert receipt['checks']['reopen_exact_exports'] is True
    assert receipt['checks']['pg_dump_restore_fresh_database_exact_exports'] is True
    assert receipt['checks']['read_while_provider_outside_database']['read_completed_while_provider_blocked'] is True
    assert receipt['checks']['concurrent_duplicate_command']['committed_commands'] == 1
    assert receipt['checks']['concurrent_duplicate_command']['automatic_backend_retries'] == 0
    assert len(receipt['source_sha256']) == 5
    assert len(receipt['synthetic_backup_sha256']) == 64
