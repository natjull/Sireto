"""Tiny complete native source with fake pressure/process, not national metrics."""
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.probe_retrieval_v179_paged_addresses as workbench
from src.xgb_matcher import retrieval_memory_policy_v153 as policy
from src.xgb_matcher.retrieval_paged_names_v170 import DiskVector
from src.xgb_matcher.retrieval_v20_workbench import record, verify, write_json
from test_retrieval_paged_addresses_v178 import full_index, scalar
from test_seal_retrieval_v175_address_parts import gate, supervision


def setup(full_index, tmp_path, monkeypatch):
    output, ref = full_index
    m = json.loads(verify(ref).read_text())
    cfg = json.loads(workbench.CONFIG.read_text())
    qpath = tmp_path / 'query_only.json'
    rows = [dict(query_id=str(i), fold=2+i%3, crm_name='NAME', crm_number='', crm_suffix='',
                 crm_street='RUE JEAN JAURES', crm_postcode='75001', crm_insee='75056') for i in range(128)]
    write_json(qpath, dict(queries=rows, truth_opened=False, old_candidates_opened=False,
                          activity_inputs={'proofs': {'path': 'LABEL_SIDE_DO_NOT_FOLLOW', 'sha256': 'f'*64}}))
    cfg.update(address_access_path=ref['path'], source_launch=record(output / 'launch.json'),
               **{k: m[k] for k in ('aliases', 'sites', 'facts', 'postings')},
               query_only_launch=record(qpath), chunk=3, output_root=str(tmp_path))
    path = tmp_path / 'v179_config.json'
    write_json(path, cfg)
    monkeypatch.setattr(workbench, 'CONFIG', path)
    monkeypatch.setattr(workbench, 'quiet', gate)
    def supervised(args, log, env, stop_path):
        workbench.worker(Path(args[args.index('--output')+1]))
        return supervision()
    monkeypatch.setattr(workbench, 'supervise', supervised)
    return cfg, tmp_path / 'probe'


def test_complete_address_probe_no_truth_and_only_owned_success_scratch_removed(full_index, tmp_path, monkeypatch):
    cfg, out = setup(full_index, tmp_path, monkeypatch)
    workbench.run(out)
    m = json.loads((out / 'manifest.json').read_text())
    rows = json.loads(verify(m['outputs']).read_text())['rows']
    assert m['status'] == workbench.STATUS and m['queries'] == 1
    assert rows[0]['query_id'] == '0' and rows[0]['fold'] == 2
    cov, ex = scalar('RUE JEAN JAURES')
    g = rows[0]['gather']
    assert g['aliases'] == cfg['aliases'] == 12 and g['positive'] == np.count_nonzero(cov)
    assert g['exact'] == np.count_nonzero(ex) and len(g['sha256']) == 64
    assert g['every_alias_gathered_once'] and g['every_immutable_query_vector_full_hash_verified']
    assert not list((out / 'queries').rglob('*.npy'))
    assert rows[0]['successful_owned_query_scratch_removed']
    assert rows[0]['total_query_and_full_gather_seconds'] >= rows[0]['trace']['total_query_seconds']
    assert m['every_address_pack_full_hash_verified_before_and_after'] and m['query_features_float64_unchanged']
    assert not m['initial100_emitted'] and not m['recall_measured'] and not m['complete_retrieval_index']
    assert not m['truth_opened'] and not m['fold0_opened'] and not m['fold1_opened'] and not m['ready_for_fold1']
    assert not m['old_candidates_opened'] and not m['downstream_models_changed']
    frozen = json.loads((out / 'launch.json').read_text())['config']['addresses']
    assert frozen == record(cfg['address_access_path'])
    with pytest.raises(FileExistsError):
        workbench.run(out)
    with pytest.raises(ValueError, match='no retry'):
        workbench.worker(out)


@pytest.mark.parametrize('change', ['fold0', 'fold1', 'truth', 'old_hits', 'extra_field', 'late_type', 'float_fold'])
def test_all128_query_rows_validated_before_fixed_first_query(full_index, tmp_path, monkeypatch, change):
    cfg, out = setup(full_index, tmp_path, monkeypatch)
    path = verify(cfg['query_only_launch'])
    m = json.loads(path.read_text())
    row = m['queries'][-1]
    if change == 'fold0':
        row['fold'] = 0
    elif change == 'fold1':
        row['fold'] = 1
    elif change == 'truth':
        m['truth_opened'] = True
    elif change == 'old_hits':
        m['old_candidates_opened'] = True
    elif change == 'extra_field':
        row['target_siret'] = 'never a feature'
    elif change == 'late_type':
        row['crm_street'] = 123
    else:
        row['fold'] = 2.
    write_json(path, m)
    cfg['query_only_launch'] = record(path)
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(ValueError, match='Frozen query-only'):
        workbench.run(out)
    assert not out.exists()


def test_unfinished_source_does_not_launch_worker(full_index, tmp_path, monkeypatch):
    cfg, out = setup(full_index, tmp_path, monkeypatch)
    cfg['address_access_path'] = str(tmp_path / 'unfinished/manifest.json')
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(FileNotFoundError):
        workbench.run(out)
    assert not out.exists()


@pytest.mark.parametrize('key', ['aliases', 'sites', 'facts', 'postings'])
def test_source_denominators_cannot_shrink(full_index, tmp_path, monkeypatch, key):
    cfg, out = setup(full_index, tmp_path, monkeypatch)
    cfg[key] -= 1
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(ValueError, match='Complete exact pinned'):
        workbench.run(out)
    assert not out.exists()


def test_no_completed_pass_without_confirmed_parent_exit(full_index, tmp_path, monkeypatch):
    cfg, out = setup(full_index, tmp_path, monkeypatch)
    original = workbench.supervise
    def disappear(*args):
        value = original(*args)
        value['own_child_exit_confirmed'] = False
        return value
    monkeypatch.setattr(workbench, 'supervise', disappear)
    with pytest.raises(ValueError, match='UNCONFIRMED'):
        workbench.run(out)
    assert (out / 'status.json').exists()


def test_query_failure_preserves_files_and_no_success_manifest(full_index, tmp_path, monkeypatch):
    cfg, out = setup(full_index, tmp_path, monkeypatch)
    def stop(self, text, directory, *args):
        directory.mkdir(parents=True)
        DiskVector(directory / 'failed.npy', 12, np.float64)
        raise TimeoutError('STOP_TEST_QUERY_COST')
    monkeypatch.setattr(workbench.PagedAddresses, 'snapshot', stop)
    with pytest.raises(TimeoutError):
        workbench.run(out)
    assert (out / 'queries/query0000/failed.npy').exists()
    assert (out / 'worker_stop.json').exists() and (out / 'status.json').exists()
    assert not (out / 'manifest.json').exists()


def test_worker_budget_failure_prevents_completed_manifest(full_index, tmp_path, monkeypatch):
    cfg, out = setup(full_index, tmp_path, monkeypatch)
    original = policy.Budget.__exit__
    def stop(self, *args):
        original(self, *args)
        raise RuntimeError('STOP_TEST_RESOURCE_POLICY')
    monkeypatch.setattr(policy.Budget, '__exit__', stop)
    with pytest.raises(RuntimeError, match='RESOURCE_POLICY'):
        workbench.run(out)
    assert not (out / 'manifest.json').exists() and (out / 'worker_stop.json').exists()


@pytest.mark.parametrize('key,value', [('max_rss_bytes', 2147483648), ('min_system_free_percent', 1), ('required_pressure_level', 2)])
def test_no_implicit_resource_relaxation(full_index, tmp_path, monkeypatch, key, value):
    cfg, out = setup(full_index, tmp_path, monkeypatch)
    cfg['resources'][key] = value
    write_json(workbench.CONFIG, cfg)
    with pytest.raises(ValueError, match='authorized memory policy'):
        workbench.run(out)
    assert not out.exists()
