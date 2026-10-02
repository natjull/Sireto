import json

import pytest
import tantivy

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_history_site_v75 import document, grouped, query, schema


def row(name='ALPHA', suffix='00001', address='10 RUE BETA'):
    return dict(siren='123456789', siret='123456789'+suffix, normalized_name=name,
        name_kind='SITE', valid_from='2010-01-01', valid_to='2011-01-01',
        historical_state='F', current_state='F', evidence_id='official:'+name,
        is_headquarters=True, number='10', number_suffix='', postcode='75001', insee='75056',
        geo_scaffold_address=address, geo_scaffold_source='SIRENE_CURRENT_ESTABLISHMENT', geo_scaffold=True)


def index(tmp_path, records):
    i = tantivy.Index(schema(), path=str(tmp_path)); w = i.writer(heap_size=15000000, num_threads=1)
    for values in grouped(records, '12').values():
        w.add_document(document(values))
    w.commit(); w.wait_merging_threads(); i.reload()
    return i


def test_closed_history_is_indexed_with_periods_and_current_scaffold_not_historical_address(tmp_path):
    i = index(tmp_path, [row(), row('ANCIEN NOM')])
    assert i.searcher().num_docs == 1
    w = [0.] * 11; w[2] = 3.
    found = i.searcher().search(query(i, RetrievalQuery(name='ANCIEN NOM'), w), limit=100).hits
    assert len(found) == 1 and found[0][0] == 3.
    d = i.searcher().doc(found[0][1]).to_dict(); p = json.loads(d['provenance'][0])
    assert not p['historical_addresses_claimed'] and len(p['observations']) == 2
    assert p['current_scaffold_source'] == 'SIRENE_CURRENT_ESTABLISHMENT'
    assert {r['valid_to'] for r in p['observations']} == {'2011-01-01'}


def test_identifiers_not_similarity_features_and_no_geography_filter(tmp_path):
    i = index(tmp_path, [row()]); w = [0.] * 11; w[2] = 1.
    q = RetrievalQuery(name='ALPHA 12345678900001', insee='99999', postcode='99999')
    found = i.searcher().search(query(i, q, w), limit=100).hits
    assert len(found) == 1 and found[0][0] == 1.
    assert not i.searcher().search(query(i, RetrievalQuery(name='12345678900001'), w), limit=100).hits


def test_no_cross_alias_exact_or_fictitious_scaffold(tmp_path):
    i = index(tmp_path, [row('ALPHA'), row('BETA')]); w = [0.] * 11; w[2] = 1.
    assert not i.searcher().search(query(i, RetrievalQuery(name='ALPHA BETA'), w), limit=100).hits
    with pytest.raises(ValueError, match='Conflicting'):
        document([row(), row('ALPHA', address='OTHER ADDRESS')])


def test_wrong_source_partition_or_identity_refused():
    with pytest.raises(ValueError, match='identity'):
        grouped([row()], '99')
    r = row(); r['siret'] = '99999999900001'
    with pytest.raises(ValueError, match='identity'):
        grouped([r], '12')
    r = row(); r['geo_scaffold_source'] = 'HISTORICAL_ADDRESS'
    with pytest.raises(ValueError, match='provenance'):
        grouped([r], '12')


def test_actual_native_search_cap_without_fixed_candidate_reranking(tmp_path):
    i = index(tmp_path, [row(suffix=f'{j:05d}') for j in range(120)])
    w = [0.] * 11; w[2] = 1.
    found = i.searcher().search(query(i, RetrievalQuery(name='ALPHA'), w), limit=100).hits
    assert len(found) == 100
    assert len({i.searcher().doc(loc).to_dict()['siret'][0] for _,loc in found}) == 100


@pytest.mark.parametrize('weights', [[0.]*10, [float('nan')]+[0.]*10, [-1.]+[0.]*10])
def test_invalid_transferred_metric_rejected(tmp_path, weights):
    i = index(tmp_path, [row()])
    with pytest.raises(ValueError, match='metric'):
        query(i, RetrievalQuery(name='ALPHA'), weights)
