"""Driver lifecycle with simulated sources/budgets, NOT national RSS evidence."""
import inspect
import json
import time

from scripts import retrieval_v211_structural_channels as driver
from src.xgb_matcher.retrieval_memory_policy_v153 import LIMITS
from src.xgb_matcher.retrieval_v20_workbench import record


def test_query_engine_closed_before_postread_hash_and_outputs_after_budget(tmp_path, monkeypatch):
    events = []
    cfg = dict(query_seconds=20)
    q = dict(query_id='control', fold=2, crm_name='NAME', crm_street='ROAD', crm_number='1',
             crm_suffix='', crm_postcode='75001', crm_insee='75056')
    queries = [q]*128
    meta = tmp_path/'meta.json'
    meta.write_text('{}')
    launch = dict(config=cfg, queries=queries, inventory={}, positions=[24], implementation={},
                  index_documents=1, index_manifest=record(meta), index_meta=record(meta))
    (tmp_path/'launch.json').write_text(json.dumps(launch))
    monkeypatch.setattr(driver, 'inputs', lambda: (cfg, {}, {}, queries, {'control': {}}))
    monkeypatch.setattr(driver.v38, 'inventory', lambda: {})
    monkeypatch.setattr(driver, 'validate_report', lambda r: None)  # simulated, explicitly not an authority

    class Budget:
        failure = None
        def __init__(self, limits):
            assert limits == LIMITS
        def __enter__(self):
            self.start = time.monotonic()
            return self
        def __exit__(self, *exc):
            events.append('budget_exit')
            self.elapsed = 1.
        def report(self):
            return {'SIMULATED': True}
    class Files:
        entries = {}
        def __init__(self, guard):
            self.guard = guard
        def verify_after(self):
            assert 'engine_closed' in events and 'evidence_closed' in events
            assert not (tmp_path/'batch.json').exists()
            events.append('full_hash')
    class Evidence:
        def __init__(self, cfg, files):
            pass
        def close(self):
            events.append('evidence_closed')
    class Backend:
        searcher = type('Searcher', (), {'num_docs': 1})()
        def __init__(self, *a):
            pass
        def __del__(self):
            events.append('engine_closed')
    monkeypatch.setattr(driver, 'Budget', Budget)
    monkeypatch.setattr(driver, 'SourceFiles', Files)
    monkeypatch.setattr(driver, 'OfficialEvidence', Evidence)
    monkeypatch.setattr(driver, 'TantivyBackend', Backend)
    def search(*args, **kwargs):
        events.append('search')
        assert kwargs['max_seconds'] == 20
        return [], dict(seconds=1.)
    monkeypatch.setattr(driver, 'search', search)
    driver.worker(tmp_path, [24])
    assert events.index('search') < events.index('engine_closed') < events.index('full_hash') < events.index('budget_exit')
    batch = json.loads((tmp_path/'batch.json').read_text())
    assert batch['rows'][0]['position'] == 24 and batch['truth_opened'] is False
    assert batch['resources'] == {'SIMULATED': True}


def test_probe_stays_cost_only_limits_identical_and_no_target_reads():
    code = inspect.getsource(driver)
    assert 'evaluation_authorized=False' in code and 'full128_outputs_sealed=False' in code
    assert 'human_ground_truth' not in code and 'evaluate(' not in code
    cfg = json.loads(driver.CONFIG.read_text())
    assert cfg['resources'] == LIMITS and cfg['query_seconds'] == 20 and cfg['quiet_seconds'] == 30
    assert cfg['mean16_seconds'] == 4
