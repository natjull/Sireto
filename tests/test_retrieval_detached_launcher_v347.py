"""Tiny launcher fixtures; not national retrieval or macOS survival proof."""
import json
from pathlib import Path
import pytest

from scripts import launch_retrieval_v347_detached as launch


def fixture(tmp_path, monkeypatch):
    from scripts import retrieval_v344_corroborated_population as driver
    repository = tmp_path/'repository'; repository.mkdir()
    runtime = repository/'.venv/bin/python'; runtime.parent.mkdir(parents=True)
    runtime.write_text('TEST_RUNTIME_NEVER_EXECUTED')
    doc = repository/'docs/retrieval_v347_detached_launch_contract.md'; doc.parent.mkdir()
    doc.write_text('TEST_CONTRACT_NEVER_EXECUTED')
    control_base = tmp_path/'controls'; control_base.mkdir()
    generation_base = tmp_path/'generation'; generation_base.mkdir()
    panel = tmp_path/'panel.json'; panel.write_text('{}')
    monkeypatch.setattr(launch, 'ROOT', repository)
    monkeypatch.setattr(launch, 'CONTROL_BASE', control_base)
    monkeypatch.setattr(launch, 'GENERATION_BASE', generation_base)
    monkeypatch.setattr(launch, 'PANEL_MANIFEST', launch.reference(panel))
    monkeypatch.setattr(driver.population, 'sealed', lambda root: ({}, {}, {}))
    monkeypatch.setattr(driver, 'implementation', lambda: {'TEST_ONLY': launch.reference(panel)})
    return control_base/'test_job', generation_base/'test_run'


def test_exact_detached_job_command(tmp_path):
    assert launch.command(tmp_path)[-3:] == ['job', '--control', str(tmp_path)]
    assert launch.THREADS['RAYON_NUM_THREADS'] == '1'


def test_claim_exclusive_before_any_producer(tmp_path):
    launch.claim(tmp_path)
    with pytest.raises(FileExistsError):
        launch.claim(tmp_path)


def test_progress_never_infers_recall_or_counts_unadmitted_query_logs(tmp_path):
    log = tmp_path/'workflow.log'
    log.write_text('\n'.join(['{"position":3,"candidates":100}', '{"generated":4}',
        '{"generated":true}', '{"generated":2001}', '{"primary":{"queries":2000}}', 'bad_json']))
    assert launch.progress(log, 0) == 4
    assert launch.progress(log, 12) == 12


def test_prepare_load_refuse_reuse_and_reference_drift(tmp_path, monkeypatch):
    control, generation = fixture(tmp_path, monkeypatch)
    original = launch.prepare(control, generation)
    assert launch.load(control) == original
    with pytest.raises(ValueError, match='Fresh'):
        launch.prepare(control, generation)
    Path(original['panel_manifest']['path']).write_text('DRIFT')
    with pytest.raises(ValueError, match='drift'):
        launch.load(control)


@pytest.mark.parametrize('value', ['parent', 'nested'])
def test_scope_is_exact_single_task_directory(tmp_path, monkeypatch, value):
    monkeypatch.setattr(launch, 'CONTROL_BASE', tmp_path)
    bad = tmp_path if value == 'parent' else tmp_path/'job'/'nested'
    with pytest.raises(ValueError):
        launch.scoped(bad, launch.CONTROL_BASE)


def test_changed_detached_transport_or_environment_refused(tmp_path, monkeypatch):
    control, generation = fixture(tmp_path, monkeypatch)
    original = launch.prepare(control, generation)
    original['start_new_session'] = False
    launch.write(control/'launch.json', original)
    with pytest.raises(ValueError, match='detached launch'):
        launch.load(control)


def test_starter_returns_only_after_exact_child_state_and_never_retries(tmp_path, monkeypatch):
    control, generation = fixture(tmp_path, monkeypatch)
    launch.prepare(control, generation)
    calls = []
    class Child:
        pid = 123456
        def poll(self): return None
    def popen(args, **kwargs):
        calls.append((args, kwargs))
        launch.write(control/'state.json', dict(status='RUNNING_V347_NOT_COMPLETE',
            job_pid=123456, workflow_pid=234567))
        return Child()
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    result = launch.start(control)
    assert result['child_pid'] == 123456 and not result['own_child_exit_confirmed']
    assert not result['recall_admitted'] and not result['automatic_retry']
    assert calls[0][1]['start_new_session'] is True and calls[0][1]['close_fds'] is True
    assert calls[0][1]['stdin'] == launch.subprocess.DEVNULL
    with pytest.raises(FileExistsError):
        launch.start(control)


def test_job_only_starts_a_new_session_and_records_real_failure(tmp_path, monkeypatch):
    control, generation = fixture(tmp_path, monkeypatch)
    launch.prepare(control, generation)
    calls = []
    class Child:
        pid = 123456
        returncode = 1
        args = ['TEST_DUMMY']
        def poll(self): return 1
        def wait(self, timeout): return 1
        def terminate(self): calls.append('terminate_awake')
    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return Child()
    monkeypatch.setattr(launch.subprocess, 'Popen', popen)
    monkeypatch.setattr(launch, 'terminate_owned_group', lambda child: calls.append('owned_cleanup'))
    with pytest.raises(launch.subprocess.CalledProcessError):
        launch.job(control)
    assert len([c for c in calls if isinstance(c, tuple)]) == 2  # caffeinate+ONE workflow
    actual_workflow = calls[1]
    assert actual_workflow[1]['start_new_session'] is True
    assert actual_workflow[1]['stdin'] == launch.subprocess.DEVNULL
    state = json.loads((control/'state.json').read_text())
    assert state['status'].startswith('STOP') and state['exit_code'] == 1
    assert state['own_workflow_exit_confirmed'] and not state['recall_admitted']
    with pytest.raises(FileExistsError):
        launch.job(control)
