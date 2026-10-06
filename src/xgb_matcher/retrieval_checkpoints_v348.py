"""Atomic checkpoint primitives, exclusive controller and finite attempt slots."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

MAX_ATTEMPTS = 3
MAX_CONTROLLER_RUNS = 3


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


@contextmanager
def controller_lock(root):
    with (Path(root)/'controller.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('CONTROLLER_ALREADY_RUNNING') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def slots(parent):
    parent = Path(parent)
    values = []
    for path in parent.glob('attempt*'):
        suffix = path.name.removeprefix('attempt')
        if len(suffix) != 2 or not suffix.isdigit() or not path.is_dir():
            raise ValueError('Unexpected attempt inventory')
        values.append(int(suffix))
    if sorted(values) != list(range(len(values))) or len(values) > MAX_ATTEMPTS:
        raise ValueError('Contiguous bounded append-only attempts required')
    return sorted(values)


def failure_kind(directory):
    """Unknown faults/drift fatal; only declared cost/resource/crash retryable."""
    directory = Path(directory)
    worker = directory/'worker_stop.json'
    supervisor = directory/'supervisor_stop.json'
    preflight = directory/'preflight_stop.json'
    reason = None
    if preflight.exists():
        p = json.loads(preflight.read_text())
        if p.get('no_worker_started') is True and p.get('reason') == 'STOP_ACTUAL_MEMORY_PRESSURE':
            return 'RETRYABLE_DECLARED_PREFLIGHT_PRESSURE'
        return 'FATAL_PREFLIGHT_OR_INTEGRITY'
    if supervisor.exists():
        s = json.loads(supervisor.read_text())
        if s.get('own_child_exit_confirmed') is not True or s.get('cleanup_error'):
            return 'FATAL_UNREAPED_PRODUCER'
    if worker.exists():
        reason = json.loads(worker.read_text()).get('reason')
    if reason in ('STOP_CORROBORATED_QUERY_COST', 'STOP_CORROBORATED_BATCH_WORKER_COST',
                  'STOP_NATIVE_INDEX_OPEN_COST', 'STOP_RSS_OR_WALL_RESOURCE_POLICY',
                  'STOP_ACTUAL_MEMORY_PRESSURE'):
        return 'RETRYABLE_DECLARED_COST_OR_RESOURCE'
    if worker.exists():
        return 'FATAL_WORKER_OR_INTEGRITY'
    if supervisor.exists():
        if s.get('reason') in ('STOP_RSS_OR_WALL_RESOURCE_POLICY', 'STOP_ACTUAL_MEMORY_PRESSURE'):
            return 'RETRYABLE_DECLARED_COST_OR_RESOURCE'
        if s.get('own_child_exit_confirmed') is True and s.get('exit_code') in (-9, -15):
            return 'RETRYABLE_OWNED_PROCESS_INTERRUPTION'
        return 'FATAL_UNCLASSIFIED_SUPERVISION'
    return 'UNCONFIRMED_INTERRUPTED_ATTEMPT_NEVER_ADMIT'


def next_slot(parent):
    used = slots(parent)
    if len(used) == MAX_ATTEMPTS:
        raise RuntimeError('STOP_RETRIES_EXHAUSTED')
    return len(used)
