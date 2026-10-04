"""Exact owned-vector arithmetic only, synthetic tests, no business data."""
import numpy as np
import pytest

from src.xgb_matcher import retrieval_grouped_proof_updates_v204 as grouped
from src.xgb_matcher.retrieval_paged_names_v170 import DiskVector


@pytest.mark.parametrize('dtype', [np.uint32, np.int64])
@pytest.mark.parametrize('quantum', [1, 3, 131072])
def test_duplicate_unsorted_updates_and_multiple_calls_are_bit_exact(tmp_path, monkeypatch, dtype, quantum):
    monkeypatch.setattr(grouped, 'MAX_UPDATE_CHUNK', quantum)
    n = 13; rng = np.random.default_rng(204)
    old = DiskVector(tmp_path / 'old.npy', n, np.float64)
    new = grouped.GroupedProofVector(tmp_path / 'new.npy', n, np.float64)
    a = DiskVector(tmp_path / 'old_flags.npy', n, np.bool_)
    b = grouped.GroupedProofVector(tmp_path / 'new_flags.npy', n, np.bool_)
    try:
        for length in (0, 1, 30, 97):
            ids = rng.integers(0, n, length).astype(dtype); values = rng.integers(0, 19, length).astype(np.float64)/19
            flags = rng.integers(0, 2, length).astype(np.bool_)
            old.maximum(ids, values); new.maximum(ids, values); a.flags(ids, flags); b.flags(ids, flags)
            assert old.read(0, n).tobytes() == new.read(0, n).tobytes()
            assert a.read(0, n).tobytes() == b.read(0, n).tobytes()
    finally:
        for v in (old, new, a, b): v.close()


def test_last_large_partial_block_and_no_source_or_position_cap(tmp_path):
    n = grouped.MAX_UPDATE_CHUNK+7; rng = np.random.default_rng(0)
    ids = np.r_[np.arange(n, dtype=np.uint32)[::-1], rng.integers(0, n, n, dtype=np.uint32)]
    values = rng.integers(0, 31, len(ids)).astype(np.float64)/31
    old = DiskVector(tmp_path / 'old.npy', n, np.float64)
    new = grouped.GroupedProofVector(tmp_path / 'new.npy', n, np.float64)
    try:
        old.maximum(ids, values); new.maximum(ids, values)
        assert old.read(0, n).tobytes() == new.read(0, n).tobytes()
    finally: old.close(); new.close()


@pytest.mark.parametrize('values', [[-0., 0., -0.], [0., -0., 0.], [-0., -0., 0.]])
def test_signed_zero_keeps_old_tie_bits(tmp_path, values):
    old = DiskVector(tmp_path / 'old.npy', 1, np.float64)
    new = grouped.GroupedProofVector(tmp_path / 'new.npy', 1, np.float64)
    ids = np.array([0, 0, 0], dtype=np.uint32); scores = np.array(values, dtype=np.float64)
    try:
        old.maximum(ids, scores); new.maximum(ids, scores)
        assert old.read(0, 1).tobytes() == new.read(0, 1).tobytes()
    finally: old.close(); new.close()


@pytest.mark.parametrize('change', ['out_of_range_false', 'negative_false', 'dtype', 'shape', 'nan', 'negative', 'over1', 'closed_false'])
def test_all_references_and_owned_vectors_checked_even_for_noop(change, tmp_path):
    v = grouped.GroupedProofVector(tmp_path / 'v.npy', 3, np.bool_ if 'false' in change else np.float64)
    ids = np.array([0], dtype=np.uint32)
    values = np.array([False], dtype=np.bool_) if 'false' in change else np.array([.5], dtype=np.float64)
    if change == 'out_of_range_false': ids[0] = 3
    elif change == 'negative_false': ids = np.array([-1], dtype=np.int64)
    elif change == 'dtype': values = values.astype(np.float32)
    elif change == 'shape': values = np.array([.1, .5], dtype=np.float64)
    elif change == 'nan': values[0] = np.nan
    elif change == 'negative': values[0] = -.1
    elif change == 'over1': values[0] = 1.1
    else: v.close()
    try:
        with pytest.raises(ValueError):
            (v.flags if 'false' in change else v.maximum)(ids, values)
    finally: v.close()
