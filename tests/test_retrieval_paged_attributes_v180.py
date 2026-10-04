"""Tiny complete audited structured source; no national resource/recall claim."""
import numpy as np
import pytest

import scripts.compile_retrieval_v139_site_attributes as workbench
import src.xgb_matcher.retrieval_paged_attributes_v180 as paged
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from src.xgb_matcher.retrieval_initial_collector_v146 import score_sites
from src.xgb_matcher.retrieval_site_attributes_v139 import StructuredSites
from src.xgb_matcher.retrieval_v20_workbench import record
from test_compile_retrieval_v139_site_attributes import setup as original_setup, finish


def setup(tmp_path, monkeypatch):
    out = original_setup(tmp_path, monkeypatch)
    finish(out)
    workbench.seal(out)
    return record(out / 'manifest.json')


def close_original(source):
    for arrays in (source.a, source.reused):
        for a in arrays.values():
            a._mmap.close()


def test_exact_flags_xy_and_leaf_scores_match139_all_sites_and_unknowns(tmp_path, monkeypatch):
    ref = setup(tmp_path, monkeypatch)
    old = StructuredSites(ref)
    new = paged.PagedStructuredSites(ref)
    ids = np.arange(8, dtype=np.uint32)
    try:
        for cp, ins, num in (('', '', ''), ('69001', '2a001', '07 bis'), ('75001', '75056', '123456789'),
                             ('97101', '97101', '①②③④⑤⑥⑦⑧⑨'), ('[ND]', '[ND]', '[ND]')):
            expected = old.flags(ids, cp, ins, num)
            actual = new.flags(ids, cp, ins, num)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
                assert a.flags.owndata
            xy = new.coordinates(ids)
            np.testing.assert_array_equal(xy, old.coordinates(ids))
            assert xy.flags.owndata and np.isnan(xy[2:]).all()
            q = Query(postcode=cp, insee=ins, number=num, xy=(650000., 6800000.), reliable_geo=True)
            names = np.linspace(0., 1., 8)
            w = np.arange(1, 31, dtype=np.float64)/30
            a, sa = score_sites(ids, q, w, names, np.zeros(8), old, [], activity={0: .6})
            b, sb = score_sites(ids, q, w, names, np.zeros(8), new, [], activity={0: .6})
            np.testing.assert_array_equal(a, b)
            assert sa == sb and not sb['closed_or_unknown_sites_filtered']
        new.verify_after()
    finally:
        new.close()
        close_original(old)


def test_only_own_touched_readonly_maps_rotated_outputs_independent(tmp_path, monkeypatch):
    ref = setup(tmp_path, monkeypatch)
    new = paged.PagedStructuredSites(ref)
    previous = {('a', k): a for k, a in new.a.items()}
    previous.update({('reused', k): a for k, a in new.reused.items()})
    try:
        flags = new.flags(np.array([0, 1, 7], dtype=np.uint32), insee='75056', number='07 BIS')
        expected_rotated = {('a', 'insee_codes'), ('a', 'number_codes'),
                            *[('reused', k) for k in ('headquarters_codes', 'nomenclature_codes', 'original_activity_codes')]}
        for key, a in previous.items():
            current = (new.a if key[0] == 'a' else new.reused)[key[1]]
            assert a._mmap.closed == (key in expected_rotated)
            assert (a is current) == (key not in expected_rotated)
            assert not current.flags.writeable
        assert flags[0].tolist() == [1., 0., 0.]
        flags[0][:] = 99
        assert new.flags(np.array([0, 1, 7], dtype=np.uint32), insee='75056')[0].tolist() == [1., 0., 0.]
        xy = new.coordinates(np.array([0, 7], dtype=np.uint32))
        assert previous['reused', 'site_xy']._mmap.closed
        new.close()
        assert np.isfinite(xy[0]).all() and np.isnan(xy[1]).all()
    finally:
        new.close()


def test_untouched_source_epoch_drift_fails_before_any_query(tmp_path, monkeypatch):
    ref = setup(tmp_path, monkeypatch)
    new = paged.PagedStructuredSites(ref)
    original = paged.fingerprint
    changed = new.specs['reused', 'creation_days']['path']
    def drift(path):
        epoch = original(path)
        return (*epoch[:-1], epoch[-1]+1) if str(path) == str(changed) else epoch
    monkeypatch.setattr(paged, 'fingerprint', drift)
    try:
        with pytest.raises(ValueError, match='epoch drift'):
            new.coordinates(np.array([0], dtype=np.uint32))
        assert not new.valid
    finally:
        new.close()


def test_verify_after_rehashes_every_array_and_metadata(tmp_path, monkeypatch):
    ref = setup(tmp_path, monkeypatch)
    new = paged.PagedStructuredSites(ref)
    seen = []
    original = paged.verify
    def tracking(ref):
        seen.append(ref['path'])
        return original(ref)
    monkeypatch.setattr(paged, 'verify', tracking)
    try:
        new.verify_after()
        assert set(seen) == {r['path'] for r in new.references}
        assert sum(path.endswith('.npy') for path in seen) == 10
    finally:
        new.close()


@pytest.mark.parametrize('ids', [np.array([8], dtype=np.uint32), np.array([0.]), np.array([-1], dtype=np.int32)])
def test_malformed_actual_positions_fail_closed(tmp_path, monkeypatch, ids):
    ref = setup(tmp_path, monkeypatch)
    new = paged.PagedStructuredSites(ref)
    try:
        with pytest.raises(ValueError):
            new.flags(ids)
        assert not new.valid
        with pytest.raises(ValueError, match='Closed'):
            new.coordinates(np.array([0], dtype=np.uint32))
    finally:
        new.close()
