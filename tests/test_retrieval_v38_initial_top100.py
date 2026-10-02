import json

import pytest
import tantivy

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend


def backend(tmp_path, rows):
    b = tantivy.SchemaBuilder()
    for f in ('document_type', 'siren', 'siret', 'insee', 'postcode'):
        b.add_text_field(f, stored=True, tokenizer_name='raw')
    for f in (*v38.NAMES, *v38.ADDRESSES, 'current_name_ngrams', 'address_ngrams'):
        b.add_text_field(f, stored=True)
    schema = b.build(); root = tmp_path/'tantivy'; root.mkdir()
    idx = tantivy.Index(schema, path=str(root)); writer = idx.writer(heap_size=15_000_000, num_threads=1)
    for i, row in enumerate(rows):
        siren = str(100000000+i)
        raw = {**dict(document_type='siret', siren=siren, siret=siren+'00001', insee='75056', postcode='75001'), **row}
        for f, source in (('current_name_ngrams', 'legal_names'), ('address_ngrams', 'addresses')):
            raw[f] = ' '.join(v38.tokens(raw.get(source, ''), True))
        writer.add_document(tantivy.Document.from_dict(raw, schema))
    writer.commit(); writer.wait_merging_threads(); idx.reload()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(build_scope='test_fixture')))
    return TantivyBackend(tmp_path)


def test_text_required_and_no_siren_document(tmp_path):
    b = backend(tmp_path, [dict(legal_names='ALPHA'), dict(document_type='siren', legal_names='ALPHA')])
    assert v38.search(b, RetrievalQuery(name='', insee='75056'), 'primary') == []
    found = v38.search(b, RetrievalQuery(name='ALPHA'), 'primary')
    assert len(found) == 1
    assert found[0]['siret'] == '10000000000001'


def test_geo_not_required_and_history_retrievable(tmp_path):
    b = backend(tmp_path, [dict(historical_names='ANCIENNE ENSEIGNE')])
    query = RetrievalQuery(name='ANCIENNE ENSEIGNE', insee='99999', postcode='99999')
    assert len(v38.search(b, query, 'primary')) == 1
    assert v38.search(b, query, 'comparator') == []


def test_address_can_retrieve_without_name_agreement(tmp_path):
    b = backend(tmp_path, [dict(legal_names='TOTALLY DIFFERENT', addresses='12 RUE BLEUE')])
    assert len(v38.search(b, RetrievalQuery(name='ZZZZZZZ', address='12 RUE BLEUE'), 'primary')) == 1


def test_output_is_always_at_most_100(tmp_path):
    b = backend(tmp_path, [dict(legal_names='ALPHA') for _ in range(130)])
    assert len(v38.search(b, RetrievalQuery(name='ALPHA'), 'primary')) == 100


def test_geographic_bonus_is_bounded_not_double_counted(tmp_path):
    b = backend(tmp_path, [dict(legal_names='ALPHA')])
    none = v38.search(b, RetrievalQuery(name='ALPHA'), 'primary')[0]['score']
    one = v38.search(b, RetrievalQuery(name='ALPHA', insee='75056'), 'primary')[0]['score']
    both = v38.search(b, RetrievalQuery(name='ALPHA', insee='75056', postcode='75001'), 'primary')[0]['score']
    assert one-none == pytest.approx(1., abs=1e-5)
    assert one == both


def test_comparator_requires_locality_and_joint_text(tmp_path):
    b = backend(tmp_path, [dict(legal_names='ALPHA', addresses='12 RUE BLEUE')])
    q = RetrievalQuery(name='ALPHA', address='12 RUE BLEUE', insee='75056')
    assert len(v38.search(b, q, 'comparator')) == 1
    assert not v38.search(b, RetrievalQuery(name=q.name, address=q.address), 'comparator')


def test_invalid_mode_and_absent_positive_metric():
    with pytest.raises(ValueError, match='recipe'):
        v38.query_for(None, RetrievalQuery(name='a'), 'oracle')
    result = v38.summarize([dict(primary_exact=False, primary_operational=False)], 'primary')
    assert result['queries'] == 1 and result['exact_recall'] == 0
