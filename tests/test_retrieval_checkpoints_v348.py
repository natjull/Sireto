"""Tiny checkpoint/process tests only; no CRM labels or national metrics."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from src.xgb_matcher import retrieval_checkpoints_v348 as checkpoints


def test_atomic_and_exclusive_checkpoint_lock(tmp_path):
    with checkpoints.controller_lock(tmp_path):
        with pytest.raises(RuntimeError, match='ALREADY_RUNNING'):
            with checkpoints.controller_lock(tmp_path):
                pytest.fail('Duplicate controller')
        checkpoints.atomic_json(tmp_path/'checkpoint.json', {'complete': True})
    with checkpoints.controller_lock(tmp_path):
        assert json.loads((tmp_path/'checkpoint.json').read_text()) == {'complete': True}
    assert not (tmp_path/'checkpoint.json.pending').exists()


def test_real_killed_controller_releases_os_lock(tmp_path):
    code = ('import sys,time; from pathlib import Path; '
            'from src.xgb_matcher.retrieval_checkpoints_v348 import controller_lock; '
            'lock=controller_lock(Path(sys.argv[1])); lock.__enter__(); '
            'print("LOCKED",flush=True); time.sleep(30)')
    child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'LOCKED'
        with pytest.raises(RuntimeError):
            with checkpoints.controller_lock(tmp_path):
                pytest.fail('Held child lock')
        child.kill(); assert child.wait(timeout=5) == -signal.SIGKILL
        with checkpoints.controller_lock(tmp_path):
            pass
    finally:
        if child.poll() is None: child.kill()
        child.wait(timeout=5); child.stdout.close()


def test_retry_limit_and_contiguous_slots(tmp_path):
    for i in range(3):
        assert checkpoints.next_slot(tmp_path) == i
        (tmp_path/f'attempt{i:02d}').mkdir()
    with pytest.raises(RuntimeError, match='RETRIES_EXHAUSTED'):
        checkpoints.next_slot(tmp_path)


@pytest.mark.parametrize('name', ['attempt01', 'attempt000', 'attemptAA'])
def test_invalid_attempt_inventory_fatal(tmp_path, name):
    (tmp_path/name).mkdir()
    with pytest.raises(ValueError): checkpoints.slots(tmp_path)


@pytest.mark.parametrize('reason,retry', [
    ('STOP_CORROBORATED_QUERY_COST', True), ('STOP_CORROBORATED_BATCH_WORKER_COST', True),
    ('STOP_RSS_OR_WALL_RESOURCE_POLICY', True), ('STOP_ACTUAL_MEMORY_PRESSURE', True),
    ('SOURCE_HASH_DRIFT', False), ('UNKNOWN_EXCEPTION', False)])
def test_declared_only_failures(tmp_path, reason, retry):
    checkpoints.atomic_json(tmp_path/'worker_stop.json', {'reason': reason})
    assert checkpoints.failure_kind(tmp_path).startswith('RETRYABLE') is retry


def test_unreaped_child_blocks_resource_retry(tmp_path):
    checkpoints.atomic_json(tmp_path/'worker_stop.json', {'reason': 'STOP_CORROBORATED_QUERY_COST'})
    checkpoints.atomic_json(tmp_path/'supervisor_stop.json', {'own_child_exit_confirmed': False})
    assert checkpoints.failure_kind(tmp_path) == 'FATAL_UNREAPED_PRODUCER'


def test_pressure_and_unknown_preflight(tmp_path):
    path = tmp_path/'preflight_stop.json'
    checkpoints.atomic_json(path, {'reason': 'STOP_ACTUAL_MEMORY_PRESSURE', 'no_worker_started': True})
    assert checkpoints.failure_kind(tmp_path).startswith('RETRYABLE')
    checkpoints.atomic_json(path, {'reason': 'UNKNOWN', 'no_worker_started': True})
    assert checkpoints.failure_kind(tmp_path).startswith('FATAL')


def test_only_owned_interruption_retryable(tmp_path):
    path = tmp_path/'supervisor_stop.json'
    checkpoints.atomic_json(path, {'exit_code': -9, 'own_child_exit_confirmed': True})
    assert checkpoints.failure_kind(tmp_path).startswith('RETRYABLE')
    checkpoints.atomic_json(path, {'exit_code': 1, 'own_child_exit_confirmed': True})
    assert checkpoints.failure_kind(tmp_path).startswith('FATAL')


def test_no_receipt_never_admitted_and_invalid_json_fatal(tmp_path):
    assert checkpoints.failure_kind(tmp_path).startswith('UNCONFIRMED')
    (tmp_path/'worker_stop.json').write_text('invalid')
    with pytest.raises(ValueError): checkpoints.failure_kind(tmp_path)
