"""Tiny launcher + real dummy process restart; no retrieval/labels executed."""
import json
import signal
import subprocess
import sys
import time

import pytest

from scripts import launch_retrieval_v352_resumable as launch


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from scripts import retrieval_v349_checkpointed as driver
    repo = tmp_path/'repository'; repo.mkdir()
    runtime = repo/'.venv/bin/python'; runtime.parent.mkdir(parents=True); runtime.write_text('TEST')
    doc = repo/'docs/retrieval_v352_detached_launch_contract.md'; doc.parent.mkdir(); doc.write_text('TEST')
    control_base = tmp_path/'controls'; control_base.mkdir()
    generation_base = tmp_path/'generation'; generation_base.mkdir()
    panel = tmp_path/'panel.json'; panel.write_text('{}')
    monkeypatch.setattr(launch, 'ROOT', repo)
    monkeypatch.setattr(launch, 'CONTROL_BASE', control_base)
    monkeypatch.setattr(launch, 'GENERATION_BASE', generation_base)
    monkeypatch.setattr(launch, 'PANEL_MANIFEST', launch.reference(panel))
    monkeypatch.setattr(driver.population, 'sealed', lambda root: ({}, {}, {}))
    monkeypatch.setattr(driver, 'implementation', lambda: {'TEST': launch.reference(panel)})
    control, generation = control_base/'test_job', generation_base/'test_run'
    launch.prepare(control, generation)
    return control, generation


def test_code_hashes_and_exact_transport_protected(setup):
    control, _ = setup
    value = launch.load(control)
    assert value['automatic_retry'] is True and value['max_controller_runs'] == 3
    value['start_new_session'] = False; launch.write(control/'launch.json', value)
    with pytest.raises(ValueError): launch.load(control)


def test_supervisor_claim_still_single_and_progress_not_recall(setup):
    control, _ = setup
    launch.claim(control)
    with pytest.raises(FileExistsError): launch.claim(control)
    log = control/'workflow.log'
    log.write_text('{"generated":4}\n{"position":92}\n{"generated":true}\n{"generated":2001}')
    assert launch.progress(log, 0) == 4


@pytest.mark.parametrize('terminal', ['status.json', 'manifest.json'])
def test_real_controller_killed_then_restarted_but_terminal_not_restarted(setup, monkeypatch, terminal):
    control, generation = setup
    real_popen, real_sleep = subprocess.Popen, time.sleep
    children = []; controllers = []
    def popen(command, **kwargs):
        if command[0] == '/usr/bin/caffeinate':
            actual = [sys.executable, '-c', 'import time;time.sleep(30)']
        elif not controllers:
            actual = [sys.executable, '-c', 'import os,signal;os.kill(os.getpid(),signal.SIGKILL)']
        else:
            code = ('from pathlib import Path;import sys;'
                    'p=Path(sys.argv[1]);p.mkdir();(p/sys.argv[2]).write_text("{}");sys.exit(1)')
            actual = [sys.executable, '-c', code, str(generation), terminal]
        child = real_popen(actual, **kwargs); children.append(child)
        if command[0] != '/usr/bin/caffeinate':
            controllers.append(child)
            assert kwargs['start_new_session'] is True
            assert kwargs['stdin'] == subprocess.DEVNULL
        return child
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: real_sleep(.01))
    with pytest.raises(subprocess.CalledProcessError): launch.job(control)
    assert len(controllers) == 2 and [c.returncode for c in controllers] == [-signal.SIGKILL, 1]
    assert all(c.poll() is not None for c in children)
    receipts = [json.loads((control/f'workflow_supervision{i:02d}.json').read_text()) for i in range(2)]
    assert [r['exit_code'] for r in receipts] == [-signal.SIGKILL, 1]
    assert all(r['own_child_exit_confirmed'] for r in receipts)
    state = json.loads((control/'state.json').read_text())
    assert state['status'].startswith('STOP') and not state['recall_admitted']
    with pytest.raises(FileExistsError): launch.job(control)


def test_at_most_three_controller_attempts(setup, monkeypatch):
    control, _ = setup
    calls = []
    class Child:
        pid = 123456
        returncode = -9
        def poll(self): return self.returncode
        def wait(self, timeout): return self.returncode
        def terminate(self): pass
    def popen(command, **kwargs): calls.append(command); return Child()
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(launch, 'terminate_owned_group', lambda child: None)
    with pytest.raises(RuntimeError, match='RESTARTS_EXHAUSTED'): launch.job(control)
    assert len(calls) == 4  # One awake process + THREE controllers, not infinite.
    assert len(list(control.glob('controller_claim*.json'))) == 3


def test_unknown_positive_exit_is_not_a_crash_retry(setup, monkeypatch):
    control, _ = setup
    calls = []
    class Child:
        pid = 123456
        returncode = 1
        args = ['TEST']
        def poll(self): return self.returncode
        def wait(self, timeout): return self.returncode
        def terminate(self): pass
    def popen(command, **kwargs): calls.append(command); return Child()
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch, 'terminate_owned_group', lambda child: None)
    with pytest.raises(subprocess.CalledProcessError): launch.job(control)
    assert len(calls) == 2  # Awake + ONE controller. Unknown error, not retryable.
