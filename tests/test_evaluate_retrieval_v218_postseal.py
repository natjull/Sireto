"""Tiny real Parquet/hash tests; authority/process/national counts explicitly simulated."""
from copy import deepcopy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import evaluate_retrieval_v218_postseal as work
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_retrieval_population_audit_v197 import controls
from test_seal_retrieval_v175_address_parts import gate as old_gate, supervision as old_supervision, resource_report as old_report

def adapted(value):
    value.update(policy_id=work.POLICY)
    if 'limits' in value: value['limits'] = work.LIMITS
    return value

def gate(): return adapted(old_gate())
def supervision(): return adapted(old_supervision())
def resource_report(): return adapted(old_report())


def tables(tmp_path, monkeypatch):
    # Tiny128, not national23571; simulated side authority explicitly declared.
    monkeypatch.setattr(work, 'ROOT', tmp_path)
    policy = tmp_path / 'docs/siret_operational_equivalence_policy.md'; policy.parent.mkdir()
    write_json(policy, {'simulation_not_national_business_policy': True})
    control = controls(); scope = {r['query_id']: r['fold'] for r in control}
    labels = [dict(query_id=r['query_id'], fold=r['fold'], human_ground_truth_siret=f'{i+1:014d}',
                   human_ground_truth_siren=f'{(i+1)//100000:09d}', label_kind='HUMAN', qualification_v2='UNRESOLVED')
              for i, r in enumerate(control)]
    official = [dict(query_id=r['query_id'], fold=r['fold'], gt_siret=r['human_ground_truth_siret'],
        gt_siren=r['human_ground_truth_siren'], official_catalog_present=True, exact_state='A') for r in labels]
    acceptable = [dict(query_id=r['query_id'], fold=r['fold'], siret=r['human_ground_truth_siret'],
        exact_truth=True, same_site_evidence=True, administrative_state='A') for r in labels]
    files = {}
    for name, rows in (('labels.parquet', labels), ('official_catalog_sidecar.parquet', official),
                       ('acceptable_operational.parquet', acceptable)):
        # Foreign, fold0/1 twins must not reach Python rows even when IDs coincide.
        extra = [dict(rows[0], fold=0), dict(rows[0], fold=1), dict(rows[0], query_id='FOREIGN')]
        p = tmp_path / name; pq.write_table(pa.Table.from_pylist([*rows, *extra]), p); files[name] = record(p)
    side_files = {k: v for k, v in files.items() if k != 'labels.parquet'}
    side = tmp_path / 'side.json'
    write_json(side, dict(ledger=files['labels.parquet'], policy=record(policy), files=side_files,
        queries=23571, never_used_in_retrieval=True, simulation_not_national_authority=True))
    evaluation = tmp_path / 'old_evaluation.json'
    write_json(evaluation, dict(label_side=record(side), files=side_files,
        excluded_missing_human_labels=16, simulation_not_national_authority=True))
    cfg = dict(evaluation=record(evaluation), ledger=files['labels.parquet'])
    return control, labels, official, acceptable, cfg, side


def test_all128_and_safe_folds_filtered_in_arrow_with_unknown_segments(tmp_path, monkeypatch):
    control, labels, official, acceptable, c, unused = tables(tmp_path, monkeypatch)
    got = work.targets(c, {'controls': control})
    assert got[0] == labels and got[1] == official and got[2] == acceptable
    assert got[-1] == 16 and len(got[3]) == len(got[4]) == 6
    assert len(got[0]) == 128 and all(r['fold'] in (2, 3, 4) for r in got[0])
    assert all('qualification_v3' not in r for r in got[0])


@pytest.mark.parametrize('change', ['size', 'bool_size', 'hash', 'extra_field'])
def test_target_reference_is_not_only_a_self_reported_path(tmp_path, change):
    p = tmp_path / 'own.json'; write_json(p, {'real_hash_only': True}); ref = record(p)
    if change == 'size': ref['size_bytes'] += 1
    elif change == 'bool_size': ref['size_bytes'] = True
    elif change == 'hash': ref['sha256'] = '0'*64
    else: ref['foreign_field'] = True
    with pytest.raises(ValueError): work.checked(ref)


@pytest.mark.parametrize('change', ['not_target_only', 'ledger', 'omitted_file', 'file_binding', 'count', 'policy'])
def test_sealed_official_target_side_required_not_hits_or_current_inference(tmp_path, monkeypatch, change):
    control, unused, unused_o, unused_a, c, side_path = tables(tmp_path, monkeypatch)
    side = json.loads(side_path.read_text())
    if change == 'not_target_only': side['never_used_in_retrieval'] = False
    elif change == 'ledger': side['ledger'] = c['evaluation']
    elif change == 'omitted_file': side['files'].pop('acceptable_operational.parquet')
    elif change == 'file_binding': side['files']['acceptable_operational.parquet'] = side['files']['official_catalog_sidecar.parquet']
    elif change == 'count': side['queries'] = True
    else: side['policy'] = c['evaluation']
    write_json(side_path, side)
    evpath = work.checked(c['evaluation']); ev = json.loads(evpath.read_text()); ev['label_side'] = record(side_path)
    write_json(evpath, ev); c['evaluation'] = record(evpath)
    with pytest.raises(ValueError): work.targets(c, {'controls': control})


def test_current_catalog_presence_exact_and_separate_from_old_sidecar(tmp_path, monkeypatch):
    control, labels, unused_o, unused_a, unused_c, unused_s = tables(tmp_path, monkeypatch)
    class Keys:
        sites = 1000
        def site_positions(self, values):
            assert values.dtype == np.uint64 and values.tolist() == list(range(1, 129))
            return np.array([work.MISSING if i % 3 else i for i in range(128)], dtype=np.uint32)
    present = work.current_membership(Keys(), labels, control)
    assert len(present) == 128 and sum(present.values()) == 43
    assert all(type(v) is bool for v in present.values())
    labels[0]['human_ground_truth_siret'] = 1
    with pytest.raises(ValueError): work.current_membership(Keys(), labels, control)


def test_permission_failure_prevents_keys_targets_and_any_gt_reader(monkeypatch):
    def deny(output): raise ValueError('No actual supervised complete214 permission')
    def forbidden(*args): raise AssertionError('GT/source opened before actual permission')
    monkeypatch.setattr(work, 'permitted', deny); monkeypatch.setattr(work, 'targets', forbidden)
    monkeypatch.setattr(work, 'RotatedKeys', forbidden)
    with pytest.raises(ValueError): work.evaluate(None)


def phase_fixture(tmp_path, monkeypatch, phase='authority'):
    out = tmp_path / 'own218'; out.mkdir(); dest = out / phase; dest.mkdir()
    generation = tmp_path / 'simulated_generation.json'; write_json(generation, {'simulation_not_national': True})
    launch = dict(generation=record(generation), **work.FALSE_FLAGS)
    write_json(out / 'launch.json', launch); write_json(dest / 'preflight.json', gate())
    write_json(dest / 'supervision.json', supervision())  # Explicitly simulated parent evidence.
    artifact = 'permission' if phase == 'authority' else 'report'
    write_json(dest / (artifact+'.json'), {'simulation_not_national': True})
    m = dict(status='QUARANTINED_V218_'+phase.upper(), launch=record(out / 'launch.json'),
        generation=launch['generation'], preflight=record(dest / 'preflight.json'), resources=resource_report(),
        artifacts={artifact: record(dest / (artifact+'.json'))}, truth_opened=phase == 'metrics',
        recall_measured=phase == 'metrics', **work.FALSE_FLAGS)
    write_json(dest / 'manifest.json', m); monkeypatch.setattr(work, 'load', lambda *args: launch)
    return out, dest, m


@pytest.mark.parametrize('change', ['missing_parent', 'parent_nonzero', 'rss', 'truth', 'generation', 'artifact_hash'])
def test_no_targets_on_unsigned_or_failed_permission_worker(tmp_path, monkeypatch, change):
    out, dest, m = phase_fixture(tmp_path, monkeypatch)
    if change == 'missing_parent': (dest / 'supervision.json').unlink()
    elif change == 'parent_nonzero':
        s = supervision(); s['exit_code'] = 1; write_json(dest / 'supervision.json', s)
    elif change == 'rss': m['resources']['peak_rss_bytes'] = 2**31+1
    elif change == 'truth': m['truth_opened'] = True
    elif change == 'generation': m['generation'] = m['launch']
    else: m['artifacts']['permission']['sha256'] = '0'*64
    write_json(dest / 'manifest.json', m)
    def forbidden(*args): raise AssertionError('Permission recheck must follow actual successful producer')
    monkeypatch.setattr(work.authority, 'recheck', forbidden)
    with pytest.raises(ValueError): work.permitted(out)


def test_live_or_stop_generation_never_creates_evaluation_or_reads_gt(tmp_path, monkeypatch):
    monkeypatch.setattr(work, 'config', lambda: {'output_root': str(tmp_path)})
    def deny(root): raise ValueError('LIVE or STOP generation')
    monkeypatch.setattr(work.authority, 'header', deny)
    output = tmp_path / 'must_not_exist'
    with pytest.raises(ValueError): work.run(tmp_path / 'generation', output)
    assert not output.exists()


@pytest.mark.parametrize('change', ['queries', 'resources', 'target', 'ledger', 'unknown'])
def test_real_config_pins_all128_backend_and_target_specs_without_opening_targets(tmp_path, monkeypatch, change):
    c = work.config()
    if change == 'queries': c['queries'] = 127
    elif change == 'resources': c['resources']['max_seconds'] += 1
    elif change == 'target': c['evaluation']['sha256'] = '0'*64
    elif change == 'ledger': c['ledger']['path'] = 'FOREIGN'
    else: c['positions'] = [0]
    p = tmp_path / 'cfg.json'; write_json(p, c); monkeypatch.setattr(work, 'CONFIG', p)
    with pytest.raises(ValueError): work.config()


def test_actual_real_config_does_not_verify_any_target_before_permission(monkeypatch):
    original = work.verify
    def guarded(ref):
        if ref['path'] in (work.TARGET_LEDGER['path'], work.TARGET_EVALUATION['path']):
            raise AssertionError('A target spec is not authorization to open it')
        return original(ref)
    monkeypatch.setattr(work, 'verify', guarded)
    c = work.config(); assert c['queries'] == 128


def test_end_to_end_tiny_after_simulated_permission_keeps_allhuman_exact_and_current_absence_separate(tmp_path, monkeypatch):
    control, labels, unused_o, unused_a, c, unused_side = tables(tmp_path, monkeypatch)
    generated = tmp_path / 'initial100.json'
    rows = [dict(query_id=r['query_id'], fold=r['fold'], position=i,
                 candidates=[{'siret': r['human_ground_truth_siret']}] if i % 2 == 0 else [])
            for i, r in enumerate(labels)]
    for i, row in enumerate(rows):
        found = row.pop('candidates')
        for rank, item in enumerate(found, 1):
            item.update(siren=item['siret'][:9], retrieval_rank=rank, channel='joint_proof', source_score=1.)
        row.update(results={'primary': found}, seconds={'primary': 1.})
    write_json(generated, {'rows': rows})
    generation = tmp_path / 'simulated_generation.json'; write_json(generation, {'simulation_not_national': True})
    permission_path = tmp_path / 'simulated_permission.json'; write_json(permission_path, {'simulation_not_national': True})
    native_path = tmp_path/'native_keys.npy'
    np.save(native_path, np.array([1, *range(3, 130)], dtype=np.uint64), allow_pickle=False)
    permission = dict(controls=control, outputs=record(generated), generation=record(generation),
        current_catalog={'simulated': True}, native_keys=record(native_path), native_leaf_sites=128, available_catalog_sites=128)
    order = []
    def permitted(output): order.append('ACTUAL_AUTHORITY_SIMULATED_IN_THIS_TEST'); return permission, record(permission_path)
    monkeypatch.setattr(work, 'permitted', permitted); monkeypatch.setattr(work, 'config', lambda: c)
    monkeypatch.setattr(work.authority, 'recheck', lambda p: order.append('RECHECK_AFTER_TARGETS'))
    class Keys:
        sites = 128
        def __init__(self, reference):
            assert order == ['ACTUAL_AUTHORITY_SIMULATED_IN_THIS_TEST']; order.append('CURRENT_KEYS')
        def site_positions(self, values):
            assert values.dtype == np.uint64 and values.tolist() == list(range(1, 129))
            positions = np.arange(128, dtype=np.uint32); positions[1] = work.MISSING; return positions
        def verify_after(self): order.append('CURRENT_KEYS_VERIFIED')
        def close(self): order.append('KEYS_CLOSED')
    monkeypatch.setattr(work, 'RotatedKeys', Keys)
    result = work.evaluate(None)
    assert result['status'] == work.STATUS and result['primary']['queries'] == 128
    assert result['primary']['exact_hits'] == result['primary']['operational_hits'] == 64
    assert result['primary']['exact_recall'] == .5 and result['decision'] == 'PIVOT_GENERATION'
    assert result['loss_counts'] == {'HIT': 64, 'CURRENT_INDEX_CATALOG_ABSENT': 1, 'NOT_RETRIEVED_AT100': 63}
    assert result['segments']['qualification_v3']['UNKNOWN']['queries'] == 128
    assert result['segments']['qualification_v2']['UNRESOLVED']['queries'] == 128
    assert all(result[k] is False for k in ('ready_for_fold1', 'full_oof_population', 'slo_validated', 'fine_loss_attribution_measured'))
    assert result['truth_opened'] and result['current_catalog_membership_checked']
    assert order == ['ACTUAL_AUTHORITY_SIMULATED_IN_THIS_TEST', 'CURRENT_KEYS', 'CURRENT_KEYS_VERIFIED', 'KEYS_CLOSED', 'RECHECK_AFTER_TARGETS']


def test_permission_payload_bound_to_this_original_generation(tmp_path, monkeypatch):
    out, dest, m = phase_fixture(tmp_path, monkeypatch)
    write_json(dest / 'permission.json', {'generation': m['launch']})
    m['artifacts']['permission'] = record(dest / 'permission.json'); write_json(dest / 'manifest.json', m)
    def forbidden(*args): raise AssertionError('Foreign payload must be rejected before source epoch recheck')
    monkeypatch.setattr(work.authority, 'recheck', forbidden)
    with pytest.raises(ValueError): work.permitted(out)


def test_required_target_columns_do_not_get_fabricated(tmp_path, monkeypatch):
    control, unused_l, unused_o, unused_a, c, unused_s = tables(tmp_path, monkeypatch)
    with pytest.raises(ValueError): work.selected(c['ledger'], control, ('query_id', 'fold', 'invented_column'))
    with pytest.raises(ValueError): work.selected(c['ledger'], control[:127], ('query_id', 'fold'))


@pytest.mark.parametrize('phase', ['authority', 'metrics'])
def test_worker_publishes_only_quarantined_hash_bound_artifact_before_parent_admission(tmp_path, monkeypatch, phase):
    out = tmp_path / 'own'; out.mkdir(); dest = out / phase; dest.mkdir()
    generation = tmp_path / 'generation.json'; write_json(generation, {'simulation_not_national': True})
    launch = dict(generation=record(generation), **work.FALSE_FLAGS)
    write_json(out / 'launch.json', launch); write_json(dest / 'preflight.json', gate())
    monkeypatch.setattr(work, 'load', lambda *args: launch)
    class SimulatedBudget:
        def __enter__(self): return self
        def __exit__(self, *args): self.elapsed = 1
        def report(self): return resource_report()
    monkeypatch.setattr(work, 'Budget', lambda limits: SimulatedBudget())
    value = dict(simulation_not_national=True, generation=launch['generation'])
    if phase == 'authority':
        monkeypatch.setattr(work.authority, 'authorize', lambda root: dict(value, truth_opened=False))
        def forbidden(*args): raise AssertionError('Authority worker cannot call metric/target reader')
        monkeypatch.setattr(work, 'evaluate', forbidden)
    else: monkeypatch.setattr(work, 'evaluate', lambda output: dict(value, truth_opened=True))
    work.worker(out, phase)
    with pytest.raises(ValueError): work.stage(out, phase)  # No actual parent receipt yet.
    write_json(dest / 'supervision.json', supervision())  # Explicitly SIMULATED, not a real producer proof.
    m = work.stage(out, phase)
    assert m['truth_opened'] is (phase == 'metrics')
    assert m['generation'] == launch['generation']
    assert m['artifacts'] == {('permission' if phase == 'authority' else 'report'):
        record(dest / (('permission' if phase == 'authority' else 'report')+'.json'))}
    assert all(m[k] is False for k in work.SAFE_FALSE)
    assert not (out / 'manifest.json').exists()
    with pytest.raises(ValueError): work.worker(out, phase)  # Never overwrite/retry.


@pytest.mark.parametrize('phase', ['authority', 'metrics'])
def test_worker_failure_never_publishes_admitted_result_or_retries(tmp_path, monkeypatch, phase):
    out = tmp_path / 'own'; out.mkdir(); dest = out / phase; dest.mkdir()
    generation = tmp_path / 'generation.json'; write_json(generation, {'simulation_not_national': True})
    launch = dict(generation=record(generation), **work.FALSE_FLAGS)
    monkeypatch.setattr(work, 'load', lambda *args: launch)
    class SimulatedBudget:
        def __enter__(self): return self
        def __exit__(self, *args): self.elapsed = 1
        def report(self): return resource_report()
    monkeypatch.setattr(work, 'Budget', lambda limits: SimulatedBudget())
    def deny(*args): raise ValueError('Actual producer/target cost or integrity error simulated')
    monkeypatch.setattr(work.authority, 'authorize', deny); monkeypatch.setattr(work, 'evaluate', deny)
    with pytest.raises(ValueError): work.worker(out, phase)
    stop = json.loads((dest / 'worker_stop.json').read_text())
    assert stop['recall_admitted'] is False and stop['truth_may_have_been_opened'] is (phase == 'metrics')
    assert not (dest / 'manifest.json').exists() and not (out / 'manifest.json').exists()
    with pytest.raises(ValueError): work.worker(out, phase)

