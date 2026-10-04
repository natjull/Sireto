"""Actual tiny V160/V147 data; simulated V178 audited control, not national."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.resume_retrieval_v160_reverse_store as store
import src.xgb_matcher.retrieval_certified_address_facts_v181 as workbench
from src.xgb_matcher.retrieval_address_reverse_v147 import AliasSnapshot, ReversePart
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query
from src.xgb_matcher.retrieval_initial_collector_v146 import score_sites
from src.xgb_matcher.retrieval_paged_addresses_v178 import PagedAddresses, PagedAliasSnapshot
from src.xgb_matcher.retrieval_paged_names_v170 import DiskVector
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_resume_retrieval_v160_reverse_store import setup as original_setup, finish
from test_seal_retrieval_v175_address_parts import supervision
from test_retrieval_initial_collector_v146 import Attributes


def fixture(tmp_path, monkeypatch):
    prefix, output, stop, orphan = original_setup(tmp_path, monkeypatch)
    store.init(output)
    finish(output)
    m = json.loads((output / 'manifest.json').read_text())
    refs = [record(output / 'manifest.json'), m['launch'], m['catalog'], m['dictionary'],
            m['reverse_staged'], m['prefix_staged'], m['original_stop'], *m['phase_authorities']]
    for st, ref in zip(m['intervals'], m['phase_authorities']):
        path = Path(ref['path']).parent / 'supervision.json'
        write_json(path, supervision())
        refs.extend([record(path), *st['artifacts'].values(), st['numbers'], st['ordered_source'], *st['sources']])
    unique = {ref['path']: ref for ref in refs}
    epochs = [dict(reference=ref, fingerprint=list(workbench.fingerprint(ref['path']))) for ref in unique.values()]
    class FixtureIndex(PagedAddresses):
        def __init__(self):
            self.valid = True
            self.backend = SimpleNamespace(STATUS='SEALED_V177_ALL_OFFICIAL_ADDRESS_CHANNEL_ACCESS')
            self.source = record(output / 'manifest.json')
            self.metadata = dict(status=self.backend.STATUS, **{k: m[k] for k in ('aliases', 'sites', 'facts', 'catalog', 'dictionary')},
                                 **{k: False for k in PROTECTED})
            self.epochs = {'epochs': epochs}
        def check_live(self):
            if not self.valid:
                raise ValueError('Closed index')
            for entry in self.epochs['epochs']:
                if tuple(entry['fingerprint']) != workbench.fingerprint(entry['reference']['path']):
                    raise ValueError('Fixture entire source epoch drift')
    return FixtureIndex(), m, output


def snapshot(tmp_path, size=2):
    values = np.array([1., .25])
    flags = np.array([True, False])
    cov = DiskVector(tmp_path / 'cov.npy', size, np.float64)
    ex = DiskVector(tmp_path / 'ex.npy', size, np.bool_)
    cov.assign(0, values[:size])
    ex.assign(0, flags[:size])
    return PagedAliasSnapshot(cov, ex, lambda: None, 1)


def detached(batch):
    return type(batch)(*(a.copy() for a in (batch.site_positions, batch.coverage, batch.alias_exact,
                       batch.current_exact, batch.postcode_codes, batch.insee_codes)), batch.number_texts)


def test_all_original_intervals_facts_and_leaf_scores_equal147_not_global_raw_reaudit(tmp_path, monkeypatch):
    index, m, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(store, 'prefix', lambda *args: pytest.fail('No monolithic raw-prefix rehash'))
    facts = workbench.CertifiedAddressFacts(index)
    snap = snapshot(tmp_path)
    query = Query(address='RUE TEST', postcode='75001', number='7')
    expected = []
    flat_cov = snap.coverage[:]
    flat_ex = snap.exact[:]
    flat_cov.flags.writeable = flat_ex.flags.writeable = False
    for st in m['intervals']:
        part = ReversePart(st)
        try:
            ids = np.arange(st['start'], st['stop'], dtype=np.uint32)
            expected.extend(detached(b) for b in part.batches(ids, AliasSnapshot(flat_cov, flat_ex), chunk=1))
        finally:
            part.close()
    try:
        for chunk in (1, 3, facts.sites):
            bound = facts.bind(snap, query, fact_chunk=1)
            observed = []
            for lo in range(0, facts.sites, chunk):
                ids = np.arange(lo, min(facts.sites, lo+chunk), dtype=np.uint32)
                observed.extend(detached(b) for b in bound.batches(ids, query))
            bound.finish()
            assert bound.consumed == facts.facts == 2 and bound.cursor == facts.sites == 11
            assert facts.active is None and len(observed) == len(expected)
            for a, b in zip(observed, expected):
                for k in ('site_positions', 'coverage', 'alias_exact', 'current_exact', 'postcode_codes', 'insee_codes'):
                    np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
                assert a.number_texts == b.number_texts
            ids = np.arange(facts.sites, dtype=np.uint32)
            nc = np.linspace(0., 1., facts.sites)
            w = np.ones(30)
            a, _ = score_sites(ids, query, w, nc, np.zeros(11), Attributes(11), observed)
            b, _ = score_sites(ids, query, w, nc, np.zeros(11), Attributes(11), expected)
            np.testing.assert_array_equal(a, b)
        facts.verify_after()
    finally:
        facts.close()
        snap.close(remove=True)


@pytest.mark.parametrize('change', ['missing_epoch', 'wrong_hash', 'wrong_catalog', 'wrong_count', 'partial', 'closed_index', 'wrong_status'])
def test_no_facts_from_partial_or_uncertified_namespace(tmp_path, monkeypatch, change):
    index, m, _ = fixture(tmp_path, monkeypatch)
    if change == 'missing_epoch':
        selected = m['intervals'][-1]['numbers']['path']
        index.epochs['epochs'] = [e for e in index.epochs['epochs'] if e['reference']['path'] != selected]
    elif change == 'wrong_hash':
        index.epochs['epochs'][-1]['reference'] = {**index.epochs['epochs'][-1]['reference'], 'sha256': '0'*64}
    elif change == 'wrong_catalog':
        index.metadata['catalog'] = index.metadata['dictionary']
    elif change == 'wrong_count':
        index.metadata['facts'] += 1
    elif change == 'partial':
        index.metadata['truth_opened'] = True
    elif change == 'closed_index':
        index.valid = False
    else:
        index.metadata['status'] = 'PARTIAL'
    with pytest.raises(ValueError):
        workbench.CertifiedAddressFacts(index)


def test_actual_phase_parent_exit_and_resource_report_required(tmp_path, monkeypatch):
    index, m, _ = fixture(tmp_path, monkeypatch)
    path = Path(m['phase_authorities'][0]['path']).parent / 'supervision.json'
    value = json.loads(path.read_text())
    value['own_child_exit_confirmed'] = False
    write_json(path, value)
    # Refresh only the fixture epoch: a genuine actual-exit check must still fail.
    for e in index.epochs['epochs']:
        if e['reference']['path'] == str(path):
            e.update(reference=record(path), fingerprint=list(workbench.fingerprint(path)))
    with pytest.raises(ValueError, match='UNCONFIRMED'):
        workbench.CertifiedAddressFacts(index)


def test_complete_traversal_and_snapshot_immutable_hash_required(tmp_path, monkeypatch):
    index, m, _ = fixture(tmp_path, monkeypatch)
    facts = workbench.CertifiedAddressFacts(index)
    snap = snapshot(tmp_path)
    q = Query()
    try:
        bound = facts.bind(snap, q, 1)
        list(bound.batches(np.arange(5, dtype=np.uint32), q))
        with pytest.raises(ValueError, match='Every original'):
            bound.finish()
        list(bound.batches(np.arange(5, 11, dtype=np.uint32), q))
        before = snap.verify_after
        called = []
        def verified():
            called.append(True)
            before()
        monkeypatch.setattr(snap, 'verify_after', verified)
        bound.finish()
        assert called == [True]
        monkeypatch.setattr(snap, 'verify_after', lambda: (_ for _ in ()).throw(ValueError('SNAPSHOT_HASH_DRIFT')))
        with pytest.raises(ValueError, match='HASH_DRIFT'):
            bound.finish()
    finally:
        facts.close()
        snap.close()


def test_epoch_drift_midinterval_releases_own_map_and_never_finishes(tmp_path, monkeypatch):
    index, m, _ = fixture(tmp_path, monkeypatch)
    facts = workbench.CertifiedAddressFacts(index)
    snap = snapshot(tmp_path)
    bound = facts.bind(snap, Query(), 1)
    gen = bound.batches(np.arange(11, dtype=np.uint32), bound.query)
    try:
        next(gen)
        assert facts.active is not None
        path = Path(next(iter(facts.checked[facts.starts.index(facts.active.stats['start'])]['epochs'])))
        path.write_bytes(path.read_bytes()+b'corrupt')
        with pytest.raises(ValueError, match='epoch drift'):
            list(gen)
        assert facts.active is None and not facts.valid
    finally:
        gen.close()
        facts.close()
        snap.close()


@pytest.mark.parametrize('scope,name', [('root', 'seal_source_stop.json'),
                                      ('phase', 'supervisor_stop.json'),
                                      ('phase', 'worker_stop.json')])
def test_new_stop_cannot_hide_behind_unchanged_audited_file_epochs(tmp_path, monkeypatch, scope, name):
    index, m, output = fixture(tmp_path, monkeypatch)
    facts = workbench.CertifiedAddressFacts(index)
    target = output if scope == 'root' else Path(m['phase_authorities'][-1]['path']).parent
    write_json(target / name, {'status': 'STOP'})
    with pytest.raises(ValueError, match='STOP_ORIGINAL'):
        facts.assert_live()
    assert facts.valid is False
    with pytest.raises(ValueError, match='STOP_ORIGINAL'):
        workbench.CertifiedAddressFacts(index)
