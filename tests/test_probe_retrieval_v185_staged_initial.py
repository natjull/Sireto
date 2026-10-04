"""Regression of the actual184 wrapper bug plus simulated prefix control."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

import scripts.probe_retrieval_v184_staged_initial as original
import scripts.probe_retrieval_v185_staged_initial as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_probe_retrieval_v184_staged_initial import phase_fixture


def test_actual_rotation_wrapper_regression_keeps_results_and_only_own_maps(tmp_path, monkeypatch):
    values = np.array([100001, 100003, 100007], dtype=np.uint64)
    path = tmp_path / 'sites.npy'
    np.save(path, values)
    class TinyKeys:
        def __init__(self, unused):
            self.site = np.load(path, mmap_mode='r')
            self.sites = len(values)
            self.metadata = {}
            self.previous = SimpleNamespace()
        def reconstruct(self, ids, grain):
            return np.array(self.site[ids], copy=True)
        def site_positions(self, raw):
            return np.searchsorted(self.site, raw).astype(np.uint32)
    monkeypatch.setattr(original, 'AppendKeys', TinyKeys)
    with pytest.raises(AttributeError, match='remap'):
        original.RotatedKeys({'fixture': True})
    monkeypatch.setattr(workbench, 'AppendKeys', TinyKeys)
    keys = workbench.RotatedKeys({'fixture': True})
    outside = np.load(path, mmap_mode='r')
    before = keys.keys.site
    try:
        ids = np.array([2, 0], dtype=np.uint32)
        got = keys.reconstruct(ids, 'site')
        np.testing.assert_array_equal(got, values[ids])
        assert got.flags.owndata and before._mmap.closed
        assert keys.keys.site is not before and not keys.keys.site.flags.writeable
        np.testing.assert_array_equal(keys.site_positions(got), ids)
        assert not outside._mmap.closed
        np.testing.assert_array_equal(outside, values)
    finally:
        keys.close()
        outside._mmap.close()


def prefix_fixture(tmp_path):
    out, dest, m, _ = phase_fixture(tmp_path)
    cfg = {'fixture_query_and_whole_source': True}
    old = dict(config={**cfg, 'output_root': str(out)}, implementation={}, old_candidates_opened=False, **workbench.FLAGS)
    write_json(out / 'launch.json', old)
    m.update(launch=record(out / 'launch.json'), initial100_emitted=False,
             trace={'gather': {'candidates_emitted': False}})
    write_json(dest / 'manifest.json', m)
    # The containing run stopped in a DIFFERENT stage; never relabel it.
    write_json(out / 'status.json', {'status': 'STOP', 'reason': 'score wrapper AttributeError'})
    return dict(**cfg, output_root='new_fresh_root', proof_prefix=dict(
        launch=record(out / 'launch.json'), names=record(dest / 'manifest.json'))), out, dest, m


def test_only_actual_successful_proof_component_reused_without_relabeling_stopped_run(tmp_path):
    cfg, out, dest, m = prefix_fixture(tmp_path)
    got, ref = workbench.prefix_stage(cfg, 'names')
    assert got == m and ref == record(dest / 'manifest.json')
    assert json.loads((out / 'status.json').read_text())['status'] == 'STOP'
    with pytest.raises(ValueError, match='prefix'):
        workbench.prefix_stage(cfg, 'score000')


@pytest.mark.parametrize('change', ['configuration', 'hash', 'proof_candidates', 'emitted100', 'stop'])
def test_no_partial_or_candidate_prefix_or_changed_query_source_admitted(tmp_path, change):
    cfg, out, dest, m = prefix_fixture(tmp_path)
    if change == 'configuration':
        cfg['fixture_query_and_whole_source'] = False
    elif change == 'hash':
        cfg['proof_prefix']['names']['sha256'] = '0'*64
    elif change == 'proof_candidates':
        m['trace']['gather']['candidates_emitted'] = True
    elif change == 'emitted100':
        m['initial100_emitted'] = True
    else:
        write_json(dest / 'worker_stop.json', {'status': 'STOP'})
    if change in ('proof_candidates', 'emitted100'):
        write_json(dest / 'manifest.json', m)
        cfg['proof_prefix']['names'] = record(dest / 'manifest.json')
    with pytest.raises(ValueError):
        workbench.prefix_stage(cfg, 'names')
