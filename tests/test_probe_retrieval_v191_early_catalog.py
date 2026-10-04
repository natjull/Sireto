"""Exact190 gate functions preserved; tiny real hashes/simulated source wiring."""
import ast
import inspect

import pytest

from scripts import probe_retrieval_v190_staged_merge as original
from scripts import probe_retrieval_v191_early_catalog as workbench
from test_probe_retrieval_v188_compact_merge import compact_fixture
from src.xgb_matcher.retrieval_v20_workbench import write_json


def setup(tmp_path, monkeypatch):
    out, cfg, events = compact_fixture(tmp_path, monkeypatch)
    source_cfg = dict(partitions=2, sites=9, facts=17, catalog={'fixture': True},
                      addresses={'path': str(tmp_path / 'unused' / 'manifest.json')})
    monkeypatch.setattr(workbench, 'origin', lambda c: (out, source_cfg))
    monkeypatch.setattr(workbench, 'NAMES', ('score000', 'score001', 'audit-facts', 'audit-postings', 'audit-context'))
    return out, cfg, events


def test_early_identical_catalog_then_all_audits_before_same_exact_reduction(tmp_path, monkeypatch):
    out, cfg, events = setup(tmp_path, monkeypatch)
    result, captured = workbench.compact_merge(out, cfg, events.append)
    assert result['trace']['sites_completed'] == 9 and result['trace']['address_facts'] == 17
    assert events.index('EARLY_LOAD_EXACT_ORIGINAL_CATALOG') < events.index('REQUIRE_ALL_SIX_ACTUAL_SUCCESSFUL_PHASES_AND_FULL_HASH_RECEIPTS')
    assert events.index('REQUIRE_ALL_SIX_ACTUAL_SUCCESSFUL_PHASES_AND_FULL_HASH_RECEIPTS') < events.index('SAME_SCORE_FULL_NATIONAL_TOP100_REDUCTION')
    assert len(captured) == 3 and 'keys_closed' in events


def test_failed_audit_still_closes_owned_catalog_and_never_reduces(tmp_path, monkeypatch):
    out, cfg, events = setup(tmp_path, monkeypatch)
    write_json(out / 'audit-facts/supervisor_stop.json', {'status': 'STOP'})
    with pytest.raises(ValueError, match='STOP_STAGE'):
        workbench.compact_merge(out, cfg, events.append)
    assert 'keys_closed' in events and 'SAME_SCORE_FULL_NATIONAL_TOP100_REDUCTION' not in events


def test_all_source_ledger_worker_admission_and_controller_code_only_version_changed():
    for name in ('config', 'origin', 'load', 'ledger', 'worker', 'stage', 'admission', 'run'):
        a = ast.parse(inspect.getsource(getattr(original, name)))
        b = ast.parse(inspect.getsource(getattr(workbench, name)).replace(
            'v191_early_catalog', 'v190_staged_merge').replace('191', '190'))
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    assert workbench.LIMITS == original.LIMITS
