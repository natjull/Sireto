"""Real small-file hashes and exact merge; simulated source/process fixtures."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_retrieval_v188_compact_merge as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_probe_retrieval_v187_staged_initial import stages_fixture


def compact_fixture(tmp_path, monkeypatch):
    out, source_cfg = stages_fixture(tmp_path)
    names = ('score000', 'score001', 'audit-facts', 'audit-postings', 'audit-context')
    cfg = dict(origin_launch=record(out / 'launch.json'),
               stages={n: record(out/n/'manifest.json') for n in names})
    # This fixture uses tiny validated187 phases; source constructor/input
    # wiring is simulated explicitly, never a national resource certificate.
    monkeypatch.setattr(workbench, 'NAMES', names)
    monkeypatch.setattr(workbench, 'origin', lambda c: (out, source_cfg))
    source_cfg.update(catalog={'fixture': True}, addresses={'path': str(tmp_path / 'address' / 'manifest.json')})
    events = []
    class TinyKeys:
        sites = 9
        closed = False
        def site_positions(self, raw):
            return raw.astype(np.uint32)
        def verify_after(self):
            events.append('keys_verified')
        def close(self):
            self.closed = True
            events.append('keys_closed')
    keys = TinyKeys()
    monkeypatch.setattr(workbench.original, 'RotatedKeys', lambda ref: keys)
    def full_ledger(path):
        assert keys.closed  # Regression: full ledger after maps are closed.
        events.append('full_ledger_after_reduction')
    monkeypatch.setattr(workbench.backend, 'load', full_ledger)
    monkeypatch.setattr(workbench.original, 'inputs', lambda cfg: ({}, [{'query_id': 'CRM_ONLY', 'fold': 3}], {}, {}))
    return out, cfg, events


def test_all_audits_before_keys_and_same_entire_exact_reduction_after_map_close(tmp_path, monkeypatch):
    out, cfg, events = compact_fixture(tmp_path, monkeypatch)
    result, captured = workbench.compact_merge(out, cfg, events.append)
    assert result['candidates'] == []  # Tiny world has no positive scores.
    assert result['trace']['sites_completed'] == 9 and result['trace']['address_facts'] == 17
    assert result['trace']['physical_partitions'] == 2
    assert not result['truth_opened'] and not result['recall_measured']
    assert len(captured) == 3  # Original union once, NOT copied per score.
    assert events.index('REQUIRE_ALL_SIX_ACTUAL_SUCCESSFUL_PHASES_AND_FULL_HASH_RECEIPTS') < events.index('LOAD_EXACT_ORIGINAL_CATALOG')
    assert events.index('keys_verified') < events.index('keys_closed') < events.index('full_ledger_after_reduction')


def test_missing_full_audit_never_opens_identity_maps(tmp_path, monkeypatch):
    out, cfg, events = compact_fixture(tmp_path, monkeypatch)
    write_json(out / 'audit-context' / 'supervisor_stop.json', {'status': 'STOP'})
    with pytest.raises(ValueError, match='STOP_STAGE'):
        workbench.compact_merge(out, cfg, events.append)
    assert 'LOAD_EXACT_ORIGINAL_CATALOG' not in events


def test_original_full_ledger_failure_propagates_and_never_returns_output(tmp_path, monkeypatch):
    out, cfg, events = compact_fixture(tmp_path, monkeypatch)
    def failed(path):
        raise ValueError('FULL_ORIGINAL_LEDGER_DRIFT')
    monkeypatch.setattr(workbench.backend, 'load', failed)
    with pytest.raises(ValueError, match='FULL_ORIGINAL_LEDGER'):
        workbench.compact_merge(out, cfg, events.append)
    assert 'keys_closed' in events


@pytest.mark.parametrize('change', ['missing_phase', 'rss', 'wall', 'quiet'])
def test_no_source_or_resource_scope_relaxation(tmp_path, monkeypatch, change):
    cfg = json.loads(workbench.CONFIG.read_text())
    if change == 'missing_phase':
        cfg['stages'].pop('score002')
    elif change == 'rss':
        cfg['resources']['max_rss_bytes'] += 1
    elif change == 'wall':
        cfg['resources']['max_seconds'] += 1
    else:
        cfg['quiet_seconds'] = 0
    p = tmp_path / 'config.json'
    write_json(p, cfg)
    monkeypatch.setattr(workbench, 'CONFIG', p)
    with pytest.raises(ValueError):
        workbench.config()
