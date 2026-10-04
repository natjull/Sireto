"""Real tiny native packs with simulated supervision/pressure; no national metric."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.xgb_matcher import retrieval_deferred_addresses_v196 as workbench
from src.xgb_matcher import retrieval_paged_addresses_v178 as original
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_retrieval_paged_addresses_v178 import full_index, scalar


@pytest.mark.parametrize('text', ['RUE JEAN JAURES', 'JEAN', '390865069', 'AAAAAAA', '', 'JEAN JEAN JEAN'])
def test_same_whole_alias_arithmetic_and_all_full_hashes_after_query(full_index, tmp_path, monkeypatch, text):
    root, ref = full_index
    seen = []
    def tracking(ref):
        seen.append(ref['path'])
        return real(ref)
    real = original.verify
    monkeypatch.setattr(workbench, 'verify', tracking)
    monkeypatch.setattr(original, 'verify', tracking)
    index = workbench.DeferredHashAddresses(ref)
    snap = None
    try:
        assert len(index.references) == 1024 and len(index.access) == 256
        paths = {r['path'] for r in index.references}
        assert not paths.intersection(seen)  # No initial full data hash, NOT an omitted target.
        assert index.pending_full_source_hash
        snap, stats = index.snapshot(text, tmp_path / 'query', chunk=3)
        cov, ex = scalar(text)
        np.testing.assert_array_equal(snap.coverage[:], cov)
        np.testing.assert_array_equal(snap.exact[:], ex)
        assert stats['every_original_alias_checked'] and stats['postings_or_alias_cap'] is None
        assert stats['source_full_hash_audit_pending'] and not stats['source_full_hash_admitted_here']
        assert not stats['source_hashes_checked_at_open_not_per_query']
        assert not stats['candidates_emitted'] and not stats['recall_measured']
        snap.verify_after()
        index.verify_after()  # Every original FULL hash really read here.
        assert paths <= set(seen) and not index.pending_full_source_hash
        values = index.posting(0, 'JEAN')
        assert values.flags.owndata and index.pending_full_source_hash
        index.verify_after()
    finally:
        if snap is not None: snap.close(remove=True)
        index.close()


def test_same_inode_size_mtime_payload_corruption_never_gets_full_source_admission(full_index):
    root, ref = full_index
    seal = json.loads((root / 'manifest.json').read_text())
    data = next(a['artifacts']['post_ids.npy'] for a in seal['access'] if a['postings'] >= 2)
    path = Path(data['path']); raw = path.read_bytes(); stat = path.stat()
    index = None
    try:
        a = np.load(path, mmap_mode='r+')
        a[0] = (int(a[0])+1) % seal['aliases']; a.flush(); a._mmap.close()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        index = workbench.DeferredHashAddresses(ref)
        index.check_live()  # Demonstrates why neither epoch nor header is a full hash.
        assert index.pending_full_source_hash
        with pytest.raises(ValueError): index.verify_after()
        assert not index.valid
    finally:
        if index is not None: index.close()
        path.write_bytes(raw)  # Restore only OWN pytest fixture, never a user source.
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


@pytest.mark.parametrize('change', ['shape', 'dtype', 'offset_start', 'offset_end'])
def test_every_typed_csr_header_and_endpoint_checked_before_read(full_index, change):
    root, ref = full_index
    seal = json.loads((root / 'manifest.json').read_text())
    name = 'post_ids.npy' if change in ('shape', 'dtype') else 'post_offsets.npy'
    chosen = next(a for a in seal['access'] if a['postings'] >= 2)
    path = Path(chosen['artifacts'][name]['path']); raw = path.read_bytes(); stat = path.stat()
    try:
        a = np.load(path, allow_pickle=False)
        if change == 'shape': a = a[:-1]
        elif change == 'dtype': a = a.view(np.int32)
        elif change == 'offset_start': a[0] = 1
        else: a[-1] -= 1
        np.save(path, a, allow_pickle=False)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with pytest.raises(ValueError): workbench.DeferredHashAddresses(ref)
    finally:
        path.write_bytes(raw)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


@pytest.mark.parametrize('change', ['access', 'phases', 'aliases', 'postings', 'source', 'truth', 'ready', 'status'])
def test_no_deferred_reader_from_incomplete_or_wrong_official_source(full_index, change):
    root, ref = full_index
    path = root / 'manifest.json'; raw = path.read_bytes(); m = json.loads(raw)
    if change == 'access': m['access'].pop()
    elif change == 'phases': m['phase_authorities'].pop()
    elif change in ('aliases', 'postings'): m[change] -= 1
    elif change == 'source': m['source_store'] = m['dictionary']
    elif change == 'truth': m['truth_opened'] = True
    elif change == 'ready': m['ready_for_fold1'] = True
    else: m['status'] = 'PARTIAL'
    try:
        write_json(path, m)
        with pytest.raises(ValueError): workbench.DeferredHashAddresses(record(path))
    finally:
        path.write_bytes(raw)


def test_actual_original_parent_even_in_unqueried_last_pack_required(full_index):
    root, ref = full_index
    path = root / 'access/channel3/bucket063/supervision.json'
    raw = path.read_bytes(); m = json.loads(raw); m['own_child_exit_confirmed'] = False
    try:
        write_json(path, m)
        with pytest.raises(ValueError, match='UNCONFIRMED'): workbench.DeferredHashAddresses(ref)
    finally:
        path.write_bytes(raw)


def test_query_after_full_hash_rearms_quarantine_and_timeout_or_closed_publish_nothing(full_index, tmp_path):
    _, ref = full_index
    index = workbench.DeferredHashAddresses(ref)
    snap = None
    try:
        index.verify_after(); assert not index.pending_full_source_hash
        snap, stats = index.snapshot('JEAN', tmp_path / 'query', chunk=3)
        assert index.pending_full_source_hash and stats['source_full_hash_audit_pending']
        with pytest.raises(TimeoutError): index.snapshot('JEAN', tmp_path / 'failed', max_seconds=1e-12)
        assert not index.valid
        index.close()
        with pytest.raises(ValueError): index.snapshot('JEAN', tmp_path / 'closed')
        assert not (tmp_path / 'closed').exists()
    finally:
        if snap is not None: snap.close(remove=True)
        index.close()
