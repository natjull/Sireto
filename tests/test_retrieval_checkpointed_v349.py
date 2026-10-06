"""Synthetic files/process receipts; real frozen validators, NOT recall evidence."""
import inspect
import json
import os

import pytest

from scripts import retrieval_v349_checkpointed as driver
from src.xgb_matcher.retrieval_checkpoints_v348 import atomic_json
from src.xgb_matcher.retrieval_name_pages_v151 import fingerprint
from src.xgb_matcher.retrieval_v20_workbench import record


@pytest.fixture
def setup(tmp_path, monkeypatch):
    qs = [dict(query_id='control'+str(i), fold=2+i%3, crm_name='NAME', crm_street='ROAD',
          crm_number='1', crm_suffix='', crm_postcode='75001', crm_insee='75056') for i in range(2000)]
    launch = dict(queries=qs)
    atomic_json(tmp_path/'launch.json', launch)
    for name in ('attempts', 'checkpoints', 'controllers'): (tmp_path/name).mkdir()
    source = tmp_path/'source.txt'; source.write_text('TEST ONLY SOURCE')
    epochs = [dict(reference=record(source), fingerprint=list(fingerprint(source)))]
    monkeypatch.setattr(driver, 'load', lambda root: launch)
    monkeypatch.setattr(driver.worker, 'load', lambda root: launch)
    monkeypatch.setattr(driver.time, 'sleep', lambda seconds: None)
    preflight = dict(policy_id=driver.worker.POLICY, pass_system_quiet_policy=True,
                     worker_swap_budget_changed=True, quiet_seconds=30)
    monkeypatch.setattr(driver.worker, 'quiet', lambda *args: preflight)
    receipt = dict(policy_id=driver.worker.POLICY, limits=driver.worker.LIMITS,
        exit_code=0, own_child_exit_confirmed=True, child_pid=123456,
        peak_rss_bytes=1000, minimum_system_free_percent=75, sample_count=2,
        swapout_pages_delta=0, global_swap_is_worker_attributed=False, wall_seconds=1., SIMULATED=True)
    report = dict(receipt, pass_resource_policy=True, pressure_levels=[1])
    calls = []

    def full(directory, batch):
        rows = []
        for p in driver.worker.positions(batch):
            rows.append(dict(position=p, query_id=qs[p]['query_id'], fold=qs[p]['fold'],
                seconds=dict(primary=1.), results=dict(primary=[dict(siret='10000000000001',
                siren='100000000', retrieval_rank=1, channel='joint_proof', source_score=1.)]),
                primary_trace=dict(policy='ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS',
                fixed_reservoir=None, score_after_fixed_pool=False, initial_cap=100,
                downstream_models_changed=False, label_lookup=False, identifiers_as_features=False,
                recall_measured=False, native_leaf_sites=43896818, full_appended_leaf_universe=False,
                exact_global_fused_score_top100=True, source_admitted={'joint_proof': 1}, geometry={})))
        atomic_json(directory/'payload.json', dict(rows=rows, source_epochs=epochs,
            used_sources_full_hash_verified_pre_and_post=True, native_segments_full_hash_verified=False,
            **driver.FALSE))
        atomic_json(directory/'manifest.json', dict(status='QUARANTINED_V344_BATCH',
            launch=record(directory.parent/'launch.json'), preflight=record(directory/'preflight.json'),
            positions=driver.worker.positions(batch), payload=record(directory/'payload.json'),
            resources=report, **driver.FALSE))

    def supervise(command, log, env, stop_path):
        calls.append(command)
        batch = int(command[-1]); full(stop_path.parent, batch)
        return receipt
    monkeypatch.setattr(driver.worker, 'supervise', supervise)
    return tmp_path, calls, full, receipt, preflight


def test_first_success_checkpoint_reused_without_compute(setup):
    root, calls, _, _, _ = setup
    cp, payload = driver.execute(root, 0)
    assert cp['attempt'] == 0 and len(payload['rows']) == 4
    assert driver.execute(root, 0) == (cp, payload)
    assert len(calls) == 1
    assert calls[0][3] == 'scripts.retrieval_v344_corroborated_population'


def test_declared_timeout_retries_fresh_then_keeps_first_pass(setup, monkeypatch):
    root, calls, full, receipt, _ = setup
    def producer(command, log, env, stop_path):
        calls.append(command)
        if len(calls) == 1:
            atomic_json(stop_path.parent/'worker_stop.json', dict(reason='STOP_CORROBORATED_QUERY_COST'))
            atomic_json(stop_path, dict(own_child_exit_confirmed=True, exit_code=1))
            raise RuntimeError('declared timeout')
        full(stop_path.parent, 0); return receipt
    monkeypatch.setattr(driver.worker, 'supervise', producer)
    cp, payload = driver.execute(root, 0)
    assert cp['attempt'] == 1 and len(cp['failures']) == 1 and len(calls) == 2
    assert calls[0][-3] != calls[1][-3]
    assert len(payload['rows']) == 4
    assert driver.execute(root, 0)[0] == cp and len(calls) == 2


def test_three_failures_stop_never_filter_query(setup, monkeypatch):
    root, calls, _, _, _ = setup
    def producer(command, log, env, stop_path):
        calls.append(command)
        atomic_json(stop_path.parent/'worker_stop.json', dict(reason='STOP_CORROBORATED_QUERY_COST'))
        atomic_json(stop_path, dict(own_child_exit_confirmed=True, exit_code=1))
        raise RuntimeError('timeout')
    monkeypatch.setattr(driver.worker, 'supervise', producer)
    with pytest.raises(RuntimeError, match='RETRIES_EXHAUSTED'): driver.execute(root, 0)
    assert len(calls) == 3 and not driver.checkpoint_path(root, 0).exists()
    with pytest.raises(FileNotFoundError): driver.admission(root)


@pytest.mark.parametrize('error', [ValueError('unknown integrity failure'), KeyboardInterrupt(), SystemExit(2)])
def test_unknown_exception_or_human_cancel_never_retries(setup, monkeypatch, error):
    root, calls, _, _, _ = setup
    def producer(command, *args):
        calls.append(command); raise error
    monkeypatch.setattr(driver.worker, 'supervise', producer)
    with pytest.raises(type(error)): driver.execute(root, 0)
    assert len(calls) == 1 and not driver.checkpoint_path(root, 0).exists()


def test_crash_after_own_exit_before_checkpoint_no_recompute(setup):
    root, calls, full, receipt, preflight = setup
    shard = driver.attempt_root(root, 0, 0); shard.mkdir(parents=True)
    atomic_json(shard/'launch.json', driver.load(root))
    directory = shard/'batch000'; directory.mkdir()
    atomic_json(directory/'preflight.json', preflight)
    full(directory, 0); atomic_json(directory/'supervision.json', receipt)
    assert driver.execute(root, 0)[0]['attempt'] == 0 and calls == []


def test_manifest_without_real_receipt_never_reused(setup):
    root, calls, full, _, preflight = setup
    shard = driver.attempt_root(root, 0, 0); shard.mkdir(parents=True)
    atomic_json(shard/'launch.json', driver.load(root))
    directory = shard/'batch000'; directory.mkdir()
    atomic_json(directory/'preflight.json', preflight); full(directory, 0)
    cp, _ = driver.execute(root, 0)
    assert cp['attempt'] == 1 and len(calls) == 1
    assert cp['failures'][0]['kind'].startswith('UNCONFIRMED')


@pytest.mark.parametrize('kind', ['source', 'payload', 'supervision', 'checkpoint'])
def test_checkpoint_drift_is_fatal_not_recompute(setup, kind):
    root, calls, _, _, _ = setup
    cp, _ = driver.execute(root, 0)
    path = root/'source.txt' if kind == 'source' else (
        driver.checkpoint_path(root, 0) if kind == 'checkpoint' else driver.worker.verify(cp[kind]))
    path.write_text('CHANGED')
    with pytest.raises(ValueError): driver.execute(root, 0)
    assert len(calls) == 1


def test_preflight_retry_receipt_also_pinned(setup, monkeypatch):
    root, _, _, _, preflight = setup
    events = []
    def quiet(*args):
        events.append(1)
        if len(events) == 1: raise RuntimeError('STOP_ACTUAL_MEMORY_PRESSURE')
        return preflight
    monkeypatch.setattr(driver.worker, 'quiet', quiet)
    cp, _ = driver.execute(root, 0)
    assert cp['attempt'] == 1
    ref = next(r for r in cp['failures'][0]['receipts'] if r['path'].endswith('preflight_stop.json'))
    driver.worker.verify(ref).write_text('CHANGED')
    with pytest.raises(ValueError): driver.execute(root, 0)


def test_manual_resume_without_owned_group_cleanup_authority_refused(setup):
    root, calls, _, _, _ = setup
    atomic_json(root/'controllers'/'run00.json', dict(pid=123456))
    with pytest.raises(FileNotFoundError, match='restart01'): driver.run(root)
    assert not calls


def test_crash_immediately_after_atomic_launch_recovers_empty_directories(setup, monkeypatch):
    root, calls, _, _, _ = setup
    for name in ('attempts', 'checkpoints'): (root/name).rmdir()
    def execute(*args):
        assert (root/'attempts').is_dir() and (root/'checkpoints').is_dir()
        calls.append(args); raise RuntimeError('TEST_TERMINAL_STOP')
    monkeypatch.setattr(driver, 'execute', execute)
    with pytest.raises(RuntimeError, match='TEST_TERMINAL_STOP'): driver.run(root)
    assert len(calls) == 1


@pytest.mark.parametrize('valid', [False, True])
def test_only_owned_cleaned_parent_can_restart_controller(setup, monkeypatch, valid):
    root, calls, _, _, _ = setup
    atomic_json(root/'controllers'/'run00.json', dict(pid=123456))
    (root/'restarts').mkdir()
    atomic_json(root/'restarts'/'restart01.json', dict(launch=record(root/'launch.json'),
        supervisor_pid=os.getppid() if valid else -1, previous_controller_pid=123456,
        own_child_exit_confirmed=True, owned_group_cleanup_completed=True, exit_code=-9))
    def execute(*args): calls.append(args); raise RuntimeError('TEST_TERMINAL_STOP')
    monkeypatch.setattr(driver, 'execute', execute)
    with pytest.raises(RuntimeError if valid else ValueError): driver.run(root)
    assert len(calls) == int(valid)
    assert (root/'controllers'/'run01.json').exists() is valid


def test_partial_population_blocks_authority_before_targets(setup):
    from src.xgb_matcher import retrieval_evaluation_authority_v350 as authority
    root, _, _, _, _ = setup
    driver.execute(root, 0)
    with pytest.raises(FileNotFoundError): authority.authorize(root)


def test_all500_checkpoints_revalidated_and_last_missing_blocks_seal(setup):
    root, calls, _, _, _ = setup
    for b in range(500): driver.execute(root, b)
    value = driver.admission(root)
    assert len(value['rows']) == 2000 and len(value['pairs']) == 500
    assert value['retry_summary']['failed_attempts'] == 0
    atomic_json(root/'outputs.json', dict(rows=value['rows'], source_epochs=value['source_epochs'], **driver.FALSE))
    manifest = dict(status=driver.STATUS, queries=2000, population=2000,
        all2000_outputs_sealed=True, initial100_emitted=True, full_appended_leaf_universe=False,
        launch=record(root/'launch.json'), outputs=record(root/'outputs.json'), pairs=value['pairs'],
        peak_worker_rss_bytes=value['peak_worker_rss_bytes'], worker_seconds=value['worker_seconds'],
        retry_summary=value['retry_summary'], **driver.FALSE)
    atomic_json(root/'manifest.json', manifest)
    assert len(driver.sealed(root)[2]['rows']) == 2000
    manifest['population'] = 2000.; atomic_json(root/'manifest.json', manifest)
    with pytest.raises(ValueError): driver.sealed(root)
    manifest['population'] = 2000; atomic_json(root/'manifest.json', manifest)
    driver.checkpoint_path(root, 499).unlink()
    with pytest.raises(FileNotFoundError): driver.sealed(root)
    assert len(calls) == 500


def test_frozen_models_resources_query_count_and_no_target_loader():
    assert driver.config()['resources'] == driver.worker.LIMITS
    assert driver.config()['queries'] == 2000
    code = inspect.getsource(driver)
    assert 'human_ground_truth' not in code and 'fit(' not in code
    from scripts import evaluate_retrieval_v351_postseal as evaluation
    assert evaluation.config()['generation_config']['path'] == str(driver.CONFIG)
