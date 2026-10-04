"""Small real files, simulated resource/process authorities, NOT national recall."""
from copy import deepcopy
import inspect
import json
import time

import pytest

from scripts import retrieval_v229_corroborated_population as driver
from src.xgb_matcher.retrieval_memory_policy_v214 import LIMITS
from src.xgb_matcher.retrieval_name_pages_v151 import fingerprint
from src.xgb_matcher.retrieval_v20_workbench import record, write_json


def queries():
    return [dict(query_id='control'+str(i), fold=2+i%3, crm_name='NAME', crm_street='ROAD',
                 crm_number='1', crm_suffix='', crm_postcode='75001', crm_insee='75056') for i in range(128)]


def outcome(qs, p):
    candidate = dict(siret='10000000000001', siren='100000000', retrieval_rank=1, channel='joint_proof', source_score=1.)
    return dict(position=p, query_id=qs[p]['query_id'], fold=qs[p]['fold'], results=dict(primary=[candidate]),
                seconds=dict(primary=1.), primary_trace=dict(policy='ONE_NATIVE_GLOBAL_TOP100_CORROBORATED_BOUNDED_PROOFS',
                    fixed_reservoir=None, score_after_fixed_pool=False, initial_cap=100,
                    downstream_models_changed=False, label_lookup=False, identifiers_as_features=False,
                    recall_measured=False, native_leaf_sites=43896818, full_appended_leaf_universe=False,
                    exact_global_fused_score_top100=True, source_admitted={'joint_proof': 1}, geometry={}))


@pytest.mark.parametrize('bad', [None, True, -1, 16, 1., '1'])
def test_exact_physical_pair_no_coercion(bad):
    with pytest.raises(ValueError):
        driver.positions(bad)


def test_all128_positions_once_and_typed_complete_inputs():
    assert [p for b in range(16) for p in driver.positions(b)] == list(range(128))
    qs = queries()
    driver.valid_rows([outcome(qs, i) for i in range(128)], qs, list(range(128)))
    qs[-1]['fold'] = 1
    with pytest.raises(ValueError):
        driver.valid_rows([outcome(qs, 0)], qs, [0])


@pytest.mark.parametrize('change', ['missing', 'wrong_order', 'fold', 'cap101', 'duplicate',
                                  'rank', 'source_score', 'pool', 'rerank', 'lie_about_append', 'seconds'])
def test_reject_partial_filtered_or_recapped_outputs(change):
    qs = queries()
    rows = [outcome(qs, 0), outcome(qs, 1)]
    r = rows[-1]
    if change == 'missing': rows.pop()
    elif change == 'wrong_order': rows.reverse()
    elif change == 'fold': r['fold'] = 0
    elif change == 'cap101': r['results']['primary'] *= 101
    elif change == 'duplicate': r['results']['primary'] *= 2
    elif change == 'rank': r['results']['primary'][0]['retrieval_rank'] = True
    elif change == 'source_score': r['results']['primary'][0]['source_score'] = float('nan')
    elif change == 'pool': r['primary_trace']['fixed_reservoir'] = 2000
    elif change == 'rerank': r['primary_trace']['score_after_fixed_pool'] = True
    elif change == 'lie_about_append': r['primary_trace']['full_appended_leaf_universe'] = True
    else: r['seconds']['primary'] = 20.1
    with pytest.raises(ValueError):
        driver.valid_rows(rows, qs, [0, 1])


def fixture(tmp_path, monkeypatch):
    qs = queries()
    launch = dict(queries=qs)
    write_json(tmp_path/'launch.json', launch)
    source = tmp_path/'source.txt'
    source.write_text('Synthetic source authority ONLY')
    epochs = [dict(reference=record(source), fingerprint=list(fingerprint(source)))]
    monkeypatch.setattr(driver, 'load', lambda root: launch)
    # These two simulated guards are NOT evidence of national process success.
    monkeypatch.setattr(driver, 'parent_exit', lambda dest: None)
    monkeypatch.setattr(driver, 'validate_report', lambda value: None)
    for b in range(16):
        dest = tmp_path/f'batch{b:03d}'
        dest.mkdir()
        write_json(dest/'preflight.json', dict(policy_id=driver.POLICY, pass_system_quiet_policy=True,
                                              worker_swap_budget_changed=True, quiet_seconds=30))
        payload = dict(rows=[outcome(qs, p) for p in driver.positions(b)], source_epochs=epochs,
                       used_sources_full_hash_verified_pre_and_post=True, native_segments_full_hash_verified=False,
                       **driver.FALSE)
        write_json(dest/'payload.json', payload)
        write_json(dest/'manifest.json', dict(status='QUARANTINED_V229_BATCH', launch=record(tmp_path/'launch.json'),
            preflight=record(dest/'preflight.json'), positions=driver.positions(b), payload=record(dest/'payload.json'),
            resources=dict(peak_rss_bytes=1000, wall_seconds=1., SIMULATED=True), **driver.FALSE))
    write_json(tmp_path/'cost_gate.json', dict(queries=16, mean_seconds=1., passed=None,
                                             truth_opened=False, production_slo_validated=False, eliminatory=False))
    return qs, driver.canonical_epochs(epochs)


def test_admission_all64_not_successful_fragments(tmp_path, monkeypatch):
    qs, epochs = fixture(tmp_path, monkeypatch)
    value = driver.admission(tmp_path)
    assert len(value['rows']) == 128 and len(value['pairs']) == 16 and value['source_epochs'] == epochs
    (tmp_path/'batch015'/'manifest.json').unlink()
    with pytest.raises(FileNotFoundError):
        driver.admission(tmp_path)


def test_slower_mean_is_descriptive_not_quality_gate(tmp_path, monkeypatch):
    qs, epochs = fixture(tmp_path, monkeypatch)
    for b in range(16):
        dest = tmp_path/f'batch{b:03d}'
        payload = json.loads((dest/'payload.json').read_text())
        for row in payload['rows']:
            row['seconds']['primary'] = 7.
        write_json(dest/'payload.json', payload)
        manifest = json.loads((dest/'manifest.json').read_text())
        manifest['payload'] = record(dest/'payload.json')
        write_json(dest/'manifest.json', manifest)
    write_json(tmp_path/'cost_gate.json', dict(queries=16, mean_seconds=7., passed=None,
        truth_opened=False, production_slo_validated=False, eliminatory=False))
    assert len(driver.admission(tmp_path)['rows']) == 128


def test_no_self_reported_seal_without_all_real_bound_outputs(tmp_path, monkeypatch):
    qs, epochs = fixture(tmp_path, monkeypatch)
    value = driver.admission(tmp_path)
    write_json(tmp_path/'outputs.json', dict(rows=value['rows'], source_epochs=epochs, **driver.FALSE))
    write_json(tmp_path/'manifest.json', dict(status=driver.STATUS, queries=128, population=128,
        all128_outputs_sealed=True, initial100_emitted=True, launch=record(tmp_path/'launch.json'),
        full_appended_leaf_universe=False, outputs=record(tmp_path/'outputs.json'), pairs=value['pairs'],
        peak_worker_rss_bytes=1000, **driver.FALSE))
    assert len(driver.sealed(tmp_path)[2]['rows']) == 128
    with open(tmp_path/'batch015'/'payload.json', 'a') as f:
        f.write(' ')
    with pytest.raises(ValueError):
        driver.sealed(tmp_path)


def test_cost_gate_and_epoch_drift_stop_whole_admission(tmp_path, monkeypatch):
    qs, epochs = fixture(tmp_path, monkeypatch)
    write_json(tmp_path/'cost_gate.json', dict(queries=16, mean_seconds=1., passed=False,
                                             truth_opened=False, production_slo_validated=False, eliminatory=False))
    with pytest.raises(ValueError, match='cost observation'):
        driver.admission(tmp_path)
    (tmp_path/'source.txt').write_text('CHANGED')
    with pytest.raises(ValueError, match='epoch'):
        driver.admission(tmp_path)


@pytest.mark.parametrize('kind', ['live', 'stop'])
def test_real_header_refuses_live_and_stop_before_any_target_call(tmp_path, monkeypatch, kind):
    cfg = dict(output_root=str(tmp_path), resources=LIMITS)
    root = tmp_path/'run'
    root.mkdir()
    monkeypatch.setattr(driver, 'config', lambda: cfg)
    if kind == 'stop': write_json(root/'status.json', dict(status='STOP'))
    with pytest.raises((ValueError, FileNotFoundError)):
        driver.sealed(root)


def test_each_query_engine_closed_before_next_query_and_fullhash(tmp_path, monkeypatch):
    qs = queries()
    dest = tmp_path/'batch000'
    dest.mkdir()
    (dest/'preflight.json').write_text('{}')
    launch = dict(queries=qs, index_documents=1, geo_queries={'UNUSED_TEST_REFERENCE': True})
    write_json(tmp_path/'launch.json', launch)
    monkeypatch.setattr(driver, 'load', lambda root: launch)
    monkeypatch.setattr(driver.backend, 'inputs', lambda: ({}, {}, {'geo_queries': launch['geo_queries']}, qs,
                                                         {q['query_id']: {} for q in qs}))
    monkeypatch.setattr(driver, 'validate_report', lambda report: None)
    monkeypatch.setattr(driver, 'serving', lambda files, queries: ({str(f): dict(weights=[1.]*11, alpha=0.) for f in (2,3,4)}, {q['query_id']: [] for q in queries}))
    events = []
    class Budget:
        failure = None
        def __init__(self, limits): assert limits == LIMITS
        def __enter__(self):
            self.start = time.monotonic()
            return self
        def __exit__(self, *args):
            events.append('budget_exit')
            self.elapsed = 1.
        def report(self): return {'SIMULATED': True}
    class Files:
        entries = {}
        def __init__(self, guard): pass
        def verify_after(self):
            assert events.count('closed') == 8 and 'evidence_closed' in events
            assert not (dest/'payload.json').exists()
            events.append('hash')
    class Evidence:
        def __init__(self, source, files): pass
        def close(self): events.append('evidence_closed')
    class Native:
        searcher = type('Searcher', (), {'num_docs': 1})()
        def __init__(self, *args):
            assert events.count('opened') == events.count('closed')
            events.append('opened')
        def __del__(self): events.append('closed')
    def search(native, evidence, q, g, weights, terms, alpha, max_seconds):
        assert 0 < max_seconds <= 20
        r = outcome(qs, 0)
        return r['results']['primary'], r['primary_trace']
    monkeypatch.setattr(driver, 'Budget', Budget)
    monkeypatch.setattr(driver, 'SourceFiles', Files)
    monkeypatch.setattr(driver, 'Evidence', Evidence)
    monkeypatch.setattr(driver, 'TantivyBackend', Native)
    monkeypatch.setattr(driver, 'search', search)
    driver.worker(tmp_path, 0)
    assert events == ['opened', 'closed']*8 + ['evidence_closed', 'hash', 'budget_exit']
    assert len(json.loads((dest/'payload.json').read_text())['rows']) == 8


def test_generation_has_no_target_loader_or_downstream_fit():
    code = inspect.getsource(driver)
    assert 'human_ground_truth' not in code and 'evaluate(' not in code
    assert 'read_table' not in code and 'fit(' not in code
    cfg = driver.config()
    assert cfg['resources'] == LIMITS and cfg['batch_width'] == 8 and cfg['mean16_seconds'] is None

