"""Complete tiny native packs with simulated pressure/processes, no national claim."""
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.compile_retrieval_v177_packed_addresses as backend
import src.xgb_matcher.retrieval_paged_addresses_v178 as reader
from src.xgb_matcher.retrieval_address_reverse_v147 import AliasSnapshot, ReversePart
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_paged_names_v170 import DiskVector
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_compile_retrieval_v177_packed_addresses import setup as source_setup

TEXTS = ['RUE JEAN JAURES', 'JEAN JEAN JEAN', 'JEANNETTE', 'CH 390865069', '390865069', 'AAAAAAA'] * 2


@pytest.fixture(scope='module')
def full_index(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp('whole_address_v178')
    cfg, output, staged, original = source_setup(root, patch)
    backend.init(output)
    backend.remaining(output)
    backend.seal(output)
    try:
        yield output, record(output / 'manifest.json')
    finally:
        patch.undo()


def scalar(text):
    query = channels(text)
    cov = np.array([max(len(a & b) / len(a) if a else 0. for a, b in zip(query, channels(s))) for s in TEXTS])
    exact = np.array([bool(normalized(text)) and normalized(s) == normalized(text) for s in TEXTS], dtype=np.bool_)
    return cov, exact


@pytest.mark.parametrize('text', ['RUE JEAN JAURES', 'JEAN', 'CH 390865069', '390865069',
                                  'AAAAAAA', 'JEAN JEAN JEAN', '', ' '.join('MOT'+str(i) for i in range(300))])
def test_whole_immutable_snapshot_matches_exact_scalar_all_aliases(full_index, tmp_path, text):
    output, ref = full_index
    index = reader.PagedAddresses(ref)
    snapshot = None
    try:
        snapshot, stats = index.snapshot(text, tmp_path / 'query', chunk=3)
        cov, exact = scalar(text)
        assert isinstance(snapshot, AliasSnapshot) and snapshot.aliases == 12
        np.testing.assert_array_equal(snapshot.coverage[:], cov)
        np.testing.assert_array_equal(snapshot.exact[:], exact)
        assert not snapshot.coverage.flags.writeable and not snapshot.exact.flags.writeable
        assert not stats['candidates_emitted'] and not stats['recall_measured'] and not stats['slo_validated']
        assert stats['total_query_seconds'] >= stats['seconds']
        assert stats['every_original_alias_checked'] and stats['postings_or_alias_cap'] is None
        assert stats['query_vectors_file_backed'] and not (tmp_path / 'query/token_counts.npy').exists()
        ids = np.array([11, 0, 11, 2], dtype=np.uint32)
        np.testing.assert_array_equal(snapshot.coverage[ids], cov[ids])
        values = snapshot.coverage[ids]
        assert values.flags.owndata
        values[:] = 99
        np.testing.assert_array_equal(snapshot.coverage[ids], cov[ids])
        with pytest.raises(ValueError, match='implicitly'):
            np.asarray(snapshot.coverage)
        index.verify_after()
        with pytest.raises(FileExistsError):
            index.snapshot(text, tmp_path / 'query')
    finally:
        if snapshot is not None:
            snapshot.close(remove=True)
        index.close()


def test_readonly_snapshot_drives_unchanged_reverse147_coherent_facts(full_index, tmp_path):
    _, ref = full_index
    index = reader.PagedAddresses(ref)
    snapshot, _ = index.snapshot('JEAN', tmp_path / 'query', chunk=4)
    try:
        facts = ReversePart.__new__(ReversePart)
        facts.stats = dict(start=0, stop=2, aliases=12)
        facts.a = dict(site_offsets=np.array([0, 2, 3], dtype=np.uint64),
                       alias_position=np.array([0, 2, 1], dtype=np.uint32),
                       site_position=np.array([0, 0, 1], dtype=np.uint32),
                       current_exact=np.array([True, False, True]),
                       postcode_code=np.array([123, 234, 345], dtype=np.uint32),
                       insee_code=np.array([456, 567, 678], dtype=np.uint32),
                       number_codes=np.array([1, 2, 0], dtype=np.uint32))
        facts.book = ['', '12', '24 BIS']
        batches = list(facts.batches(np.array([0, 1], dtype=np.uint32), snapshot, chunk=1))
        cov, ex = scalar('JEAN')
        assert len(batches) == 3
        for i, alias in enumerate((0, 2, 1)):
            b = batches[i]
            np.testing.assert_array_equal(b.coverage, [cov[alias]])
            np.testing.assert_array_equal(b.alias_exact, [ex[alias]])
            np.testing.assert_array_equal(b.current_exact, [facts.a['current_exact'][i]])
            np.testing.assert_array_equal(b.postcode_codes, [facts.a['postcode_code'][i]])
            np.testing.assert_array_equal(b.insee_codes, [facts.a['insee_code'][i]])
            assert b.number_texts == (facts.book[facts.a['number_codes'][i]],)
        index.close()  # Snapshot must own its query vectors independently of source maps.
        np.testing.assert_array_equal(snapshot.coverage[:], cov)
    finally:
        snapshot.close(remove=True)
        index.close()


@pytest.mark.parametrize('change', ['access', 'phases', 'aliases', 'postings', 'source', 'truth', 'ready', 'status'])
def test_no_reader_from_partial_or_changed_seal(full_index, change):
    output, _ = full_index
    path = output / 'manifest.json'
    original = path.read_bytes()
    m = json.loads(original)
    if change == 'access':
        m['access'].pop()
    elif change == 'phases':
        m['phase_authorities'].pop()
    elif change in ('aliases', 'postings'):
        m[change] -= 1
    elif change == 'source':
        m['source_store'] = m['dictionary']
    elif change == 'truth':
        m['truth_opened'] = True
    elif change == 'ready':
        m['ready_for_fold1'] = True
    else:
        m['status'] = 'PARTIAL'
    try:
        write_json(path, m)
        with pytest.raises(ValueError):
            reader.PagedAddresses(record(path))
    finally:
        path.write_bytes(original)


def test_actual_supervised_seal_exit_required(full_index):
    output, ref = full_index
    path = output / 'seal/supervision.json'
    original = path.read_bytes()
    m = json.loads(original)
    m['own_child_exit_confirmed'] = False
    try:
        write_json(path, m)
        with pytest.raises(ValueError, match='UNCONFIRMED'):
            reader.PagedAddresses(ref)
    finally:
        path.write_bytes(original)


def test_all_arrays_rehashed_after_queries_and_public_postings_owned(full_index, monkeypatch):
    _, ref = full_index
    index = reader.PagedAddresses(ref)
    try:
        values = index.posting(0, 'JEAN')
        assert values.flags.owndata and len(values)
        expected = values.copy()
        values[:] = 0
        np.testing.assert_array_equal(index.posting(0, 'JEAN'), expected)
        seen = []
        original = reader.verify
        def tracking(ref):
            seen.append(ref['path'])
            return original(ref)
        monkeypatch.setattr(reader, 'verify', tracking)
        index.verify_after()
        assert set(r['path'] for r in index.references) <= set(seen)
        assert len(index.references) == 1024
    finally:
        index.close()


def test_unconsumed_pack_epoch_drift_refuses_query(full_index, tmp_path, monkeypatch):
    _, ref = full_index
    index = reader.PagedAddresses(ref)
    original = reader.fingerprint
    changed = next(iter(index.files[2, 63].values()))['path']
    def drift(path):
        value = original(path)
        return (*value[:-1], value[-1] + 1) if Path(path) == changed else value
    monkeypatch.setattr(reader, 'fingerprint', drift)
    try:
        with pytest.raises(ValueError, match='epoch'):
            index.snapshot('JEAN', tmp_path / 'query')
        assert not index.valid
    finally:
        index.close()


@pytest.mark.parametrize('kind', ['range', 'nan', 'exact_without_whole_match'])
def test_every_alias_snapshot_validated_not_only_matches(tmp_path, kind):
    cov = DiskVector(tmp_path / 'cov.npy', 12, np.float64)
    ex = DiskVector(tmp_path / 'exact.npy', 12, np.bool_)
    values = np.zeros(12)
    values[-1] = float('nan') if kind == 'nan' else 2. if kind == 'range' else .5
    cov.assign(0, values)
    if kind == 'exact_without_whole_match':
        ex.assign(0, np.array([False] * 11 + [True]))
    try:
        with pytest.raises(ValueError, match='coherent'):
            reader.PagedAliasSnapshot(cov, ex, lambda: None, 3)
    finally:
        cov.close()
        ex.close()


def test_query_vectors_cannot_mutate_after_snapshot_seal(tmp_path):
    cov = DiskVector(tmp_path / 'cov.npy', 4, np.float64)
    ex = DiskVector(tmp_path / 'exact.npy', 4, np.bool_)
    snapshot = reader.PagedAliasSnapshot(cov, ex, lambda: None, 2)
    try:
        assert not cov.valid and not ex.valid
        with pytest.raises(ValueError):
            cov.assign(0, np.ones(4))
        with pytest.raises(ValueError, match='readonly'):
            snapshot.coverage._vector.assign(0, np.ones(4))
        np.testing.assert_array_equal(snapshot.coverage[:], np.zeros(4))
        snapshot.verify_after()
        snapshot.close(remove=True)
        assert not cov.path.exists() and not ex.path.exists()
    finally:
        snapshot.close()


def test_snapshot_full_hash_catches_external_write_even_if_epoch_restored(tmp_path):
    import os
    cov = DiskVector(tmp_path / 'cov.npy', 4, np.float64)
    ex = DiskVector(tmp_path / 'exact.npy', 4, np.bool_)
    snapshot = reader.PagedAliasSnapshot(cov, ex, lambda: None, 2)
    stat = cov.path.stat()
    a = np.load(cov.path, mmap_mode='r+')
    a[:] = 1.
    a.flush()
    a._mmap.close()
    os.utime(cov.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    try:
        snapshot.assert_live()  # Epoch check alone is explicitly insufficient here.
        with pytest.raises(ValueError):
            snapshot.verify_after()
        with pytest.raises(ValueError):
            snapshot.close(remove=True)
        assert cov.path.exists()
    finally:
        snapshot.close()


def test_total_query_timeout_and_closed_reader_publish_nothing(full_index, tmp_path):
    _, ref = full_index
    index = reader.PagedAddresses(ref)
    try:
        with pytest.raises(TimeoutError, match='COMPLETE_ADDRESS_QUERY'):
            index.snapshot('JEAN', tmp_path / 'failed', max_seconds=1e-12)
        assert not index.valid and not index.in_coverage and not (tmp_path / 'failed/alias_coverage.npy').exists()
        index.close()
        with pytest.raises(ValueError):
            index.snapshot('JEAN', tmp_path / 'closed')
        assert not (tmp_path / 'closed').exists()
    finally:
        index.close()
