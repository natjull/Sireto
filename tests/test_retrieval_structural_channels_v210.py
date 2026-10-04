"""Synthetic source fixtures test semantics only, never business recall."""
from collections import deque
import json

import numpy as np
import pytest
import tantivy

from scripts.build_hierarchical_retrieval_index import _schema
from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from src.xgb_matcher.retrieval_structural_channels_v210 import (
    SourceFiles, OfficialEvidence, NativeCursor, interleave, scoped_query)
from src.xgb_matcher.retrieval_v20_workbench import record


def row(i, parent=None):
    s = f'{100000000+i:09d}00001'
    return dict(siret=s, siren=parent or s[:9], score=1.)


class Stream:
    def __init__(self, values):
        self.values = deque(values)
    def next(self):
        return self.values.popleft() if self.values else None


def test_direct100_no_source_cut_or_rescoring_and_duplicate_turn_continues():
    streams = dict(a=Stream([row(i) for i in range(170)]),
                   b=Stream([row(i) for i in range(120)]+[row(200)]))
    result, stats = interleave(streams, lambda: None)
    assert len(result) == 100 and len({r['siret'] for r in result}) == 100
    assert stats['fixed_reservoir'] is None and stats['score_after_fixed_pool'] is False
    assert result[0]['siret'] == row(0)['siret'] and result[1]['siret'] == row(1)['siret']
    assert [r['retrieval_rank'] for r in result] == list(range(1, 101))


def test_empty_channels_return_turns_and_more_than100_duplicates_do_not_cut_source():
    result, stats = interleave(dict(a=Stream([row(i) for i in range(140)]), empty=Stream([]),
                                    b=Stream([row(i) for i in range(140)]+[row(200)])), lambda: None)
    assert len(result) == 100 and 'empty' not in stats['source_admitted']
    # A later channel can consume >100 duplicate results in one turn, without
    # source-rank cap, even when the requested direct output is only two.
    a = Stream([row(0)])
    b = Stream([row(0), row(1)])
    result, stats = interleave(dict(a=a, b=b), lambda: None, limit=2)
    assert stats['source_consumed']['b'] == 2 and len(result) == 2


@pytest.mark.parametrize('bad', [dict(siret='10000000000001', siren='200000000', score=1.),
                                 dict(siret='10000000000001', siren='100000000', score=float('nan')),
                                 dict(siret='10000000000001', siren='100000000', score=True)])
def test_bad_identity_and_scores_stop(bad):
    with pytest.raises(ValueError):
        interleave(dict(a=Stream([bad])), lambda: None)


def test_duplicate_inside_source_is_failure_not_silent_dedup():
    with pytest.raises(ValueError, match='pagination'):
        interleave(dict(a=Stream([row(0), row(0)])), lambda: None)


def test_deadline_never_publishes_partial100():
    def stop():
        raise TimeoutError('fixed budget')
    with pytest.raises(TimeoutError):
        interleave(dict(a=Stream([row(0)])), stop)


def array(tmp_path, name, values, dtype):
    path = tmp_path/(name+'.npy')
    np.save(path, np.array(values, dtype=dtype), allow_pickle=False)
    return record(path)


def test_source_hash_and_epoch_not_metadata_only(tmp_path):
    ref = array(tmp_path, 'payload', [1, 2, 3], np.uint32)
    files = SourceFiles(lambda: None)
    mapped = files.mapped(ref, np.uint32, (3,))
    assert not mapped.flags.writeable
    mapped._mmap.close()
    with open(ref['path'], 'r+b') as f:
        f.seek(-4, 2)
        f.write(np.uint32(4).tobytes())
    with pytest.raises(ValueError):
        files.verify_after()


def owner_fixture(tmp_path):
    evidence = OfficialEvidence.__new__(OfficialEvidence)
    evidence.files = SourceFiles(lambda: None)
    evidence.items = [dict(bucket=0, name_start=0, names=3, facts=3, artifacts={
        k+'.npy': array(tmp_path, k, v, d) for k, v, d in (
            ('name_fact_offsets', [0, 1, 2, 3], np.uint64),
            ('owner_key', [100000000, 10000000100001, 100000002], np.uint64),
            ('parent_key', [100000000, 100000001, 100000002], np.uint32),
            ('subject_code', [1, 2, 1], np.uint8), ('scope_code', [2, 3, 1], np.uint8))})]
    # Whole-variant official IDs: words must intersect on the SAME alias.
    evidence.lookup = lambda channel, text: np.array({
        (3, 'ALPHA BETA'): [], (0, 'ALPHA'): [0, 1, 2], (0, 'BETA'): [0, 1]
    }.get((channel, text), []), dtype=np.uint32)
    return evidence


def test_whole_variant_conjunction_and_site_name_not_lifted_to_parent(tmp_path):
    evidence = owner_fixture(tmp_path)
    parents, sites, stats = evidence.name_owners('ALPHA BETA')
    assert parents == {'100000000'} and sites == {'10000000100001'}
    assert '100000001' not in parents and '100000002' not in parents
    assert stats['aliases'] == 2 and stats['owner_facts'] == 2 and stats['owner_cap'] is None
    evidence.files.verify_after()


def test_masks_and_identifier_only_names_have_no_alias_channel(tmp_path):
    evidence = owner_fixture(tmp_path)
    for text in ('[ND]', '100000000', '10000000000001'):
        parents, sites, _ = evidence.name_owners(text)
        assert not parents and not sites


def graph_fixture(tmp_path):
    evidence = OfficialEvidence.__new__(OfficialEvidence)
    evidence.files = SourceFiles(lambda: None)
    # A->B, B->C and a raw self-link. Name on A expands B, never C.
    a, b, c = [10000000000001, 10000000100001, 10000000200001]
    data = dict(endpoint_keys=([a, b, c], np.uint64), endpoint_offsets=([0, 3, 5, 6], np.uint64),
                endpoint_incidences=([0, 2, 5, 1, 3, 4], np.uint32),
                parent_keys=([100000000, 100000001, 100000002], np.uint32),
                parent_endpoint_offsets=([0, 1, 2, 3], np.uint64),
                predecessor_keys=([a, b, a], np.uint64), successor_keys=([b, c, a], np.uint64))
    evidence.graph = dict(stats=dict(endpoints=3, parents=3, observations=3, oriented_incidences=6),
                          artifacts={k+'.npy': array(tmp_path, k, v, d) for k, (v, d) in data.items()})
    return evidence


def test_all_bidirectional_links_and_self_context_not_twohop(tmp_path):
    evidence = graph_fixture(tmp_path)
    found, stats = evidence.related({'100000000'}, set())
    assert found == {'10000000100001'} and stats['hops'] == 1 and stats['observations'] == 3
    found, _ = evidence.related(set(), {'10000000100001'})
    assert found == {'10000000000001', '10000000200001'}
    evidence.files.verify_after()


def test_spatial_all_radius_not_nearest100_and_weak_ban_expands(tmp_path):
    evidence = OfficialEvidence.__new__(OfficialEvidence)
    evidence.files = SourceFiles(lambda: None)
    evidence.sites = 152
    coords = [[0., 0.]]*150+[[800., 0.], [np.nan, np.nan]]
    evidence.xy_ref = array(tmp_path, 'xy', coords, np.float64)
    evidence.keys_ref = array(tmp_path, 'keys', [int(row(i)['siret']) for i in range(152)], np.uint64)
    geo = dict(spatial_crs_compatible=True, ban_x_l93=0., ban_y_l93=0.,
               ban_uncertainty_m=300., ban_declared_weak=False)
    sites, trace = evidence.nearby(geo, chunk=7)
    assert len(sites) == 150 and trace['sites_scanned'] == 152 and trace['radius_m'] == 300
    sites, trace = evidence.nearby({**geo, 'ban_declared_weak': True}, chunk=7)
    assert len(sites) == 151 and trace['radius_m'] == 900 and trace['truncated'] is False
    sites, trace = evidence.nearby({'spatial_crs_compatible': False})
    assert not sites and trace['compatible'] is False
    evidence.files.verify_after()


def native_backend(tmp_path):
    schema = _schema(tantivy)
    directory = tmp_path/'tantivy'
    directory.mkdir()
    index = tantivy.Index(schema, path=str(directory))
    writer = index.writer(heap_size=15_000_000, num_threads=1)
    rows = [dict(siret='10000000000001', siren='100000000', addresses='RUE BLEUE', insee='75056', state='F'),
            dict(siret='10000000000002', siren='100000000', addresses='RUE VERTE', insee='99999', state='A'),
            dict(siret='10000000100001', siren='100000001', addresses='RUE BLEUE', insee='75056', state='A')]
    for r in rows:
        writer.add_document(tantivy.Document.from_dict(dict(document_type='siret', **r), schema))
    writer.commit()
    writer.wait_merging_threads()
    index.reload()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(build_scope='fixture')))
    return TantivyBackend(tmp_path)


def test_real_tantivy_parent_all_sites_closed_and_nonlocal_not_filtered(tmp_path):
    backend = native_backend(tmp_path)
    q = RetrievalQuery(name='ANCIEN NOM', address='RUE BLEUE', insee='75056')
    parsed = scoped_query(backend, q, {'100000000'}, set(), v38.ADDRESSES, q.address)
    cursor = NativeCursor(backend, parsed, lambda: None, page_size=1)
    result, _ = interleave(dict(name=cursor), lambda: None)
    assert {r['siret'] for r in result} == {'10000000000001', '10000000000002'}
    assert result[0]['siret'] == '10000000000001'


def test_real_tantivy_site_scope_does_not_spread_to_sibling(tmp_path):
    backend = native_backend(tmp_path)
    q = RetrievalQuery(name='', address='RUE BLEUE')
    parsed = scoped_query(backend, q, set(), {'10000000000001'}, v38.ADDRESSES, q.address)
    result, _ = interleave(dict(name=NativeCursor(backend, parsed, lambda: None)), lambda: None)
    assert [r['siret'] for r in result] == ['10000000000001']


def test_probe_is_not_permission_to_open_truth():
    from scripts import retrieval_v210_structural_channels as driver
    import inspect
    code = inspect.getsource(driver)
    assert 'evaluation_authorized=False' in code and 'full128_outputs_sealed=False' in code
    assert 'human_ground_truth' not in code and 'evaluate(' not in code
    assert 'models]' not in code and 'old_candidates_opened=False' in code

