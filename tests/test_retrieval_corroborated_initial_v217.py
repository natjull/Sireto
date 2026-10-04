import pytest
from types import SimpleNamespace
import tantivy

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from src.xgb_matcher.retrieval_corroborated_initial_v217 import alias_fragments, compose


def test_complete_context_spans_not_an_alias_dictionary():
    spans = alias_fragments('COMMUNE DE ALPHA ECOLE BETA')
    assert 'COMMUNE DE ALPHA' in spans and 'ECOLE BETA' in spans
    assert 'DE' not in spans


def test_legal_particle_only_spans_not_promoted():
    spans = alias_fragments('SA SOCIETE DE XYZ')
    assert 'SA SOCIETE' not in spans and 'DE' not in spans
    assert 'SA SOCIETE DE XYZ' in spans


def test_single_long_name_but_not_short_acronym_fragment():
    spans = alias_fragments('ABC ALPHABET SITE')
    assert 'ALPHABET' in spans and 'ABC' not in spans


def test_identifiers_cannot_become_name_features():
    assert alias_fragments('ALPHA 123456789 12345678900001') == ['ALPHA']


def test_all_spans_reachable_not_top_k():
    text = 'ALPHA BETA GAMMA DELTA EPSILON ZETA ETA THETA IOTA KAPPA LAMBDA'
    assert text in alias_fragments(text) and 'KAPPA LAMBDA' in alias_fragments(text)


def test_frozen_head_shape_before_native_access():
    with pytest.raises(ValueError): compose(None, None, [1.], [], 0., set(), set(), set(), [])


def tiny_native():
    # Synthetic engine semantics fixture ONLY, not business evidence/recall.
    builder = tantivy.SchemaBuilder()
    for f in (*v38.NAMES, *v38.ADDRESSES, 'current_name_ngrams', 'address_ngrams'):
        builder.add_text_field(f, stored=True)
    for f in ('siret', 'siren', 'document_type', 'insee', 'postcode', 'number', 'is_siege',
              'legal_names_exact', 'trade_names_exact', 'site_names_exact', 'addresses_exact'):
        builder.add_text_field(f, stored=True, tokenizer_name='raw')
    index = tantivy.Index(builder.build()); writer = index.writer()
    for i, site, place in ((1, 'ECOLE BETA', '11111'), (2, 'ECOLE GAMMA', '11111'),
                           (3, 'ECOLE BETA', '22222')):
        writer.add_document(tantivy.Document.from_dict(dict(document_type='siret',
            siret=f'100000000{i:05d}', siren='100000000', legal_names='COMMUNE DE ALPHA',
            site_names=site, legal_names_exact='COMMUNE DE ALPHA', site_names_exact=site,
            postcode=place, insee=place, addresses='ROAD', number='1', is_siege='0'), index.schema))
    writer.commit(); index.reload()
    return SimpleNamespace(_tantivy=tantivy, index=index, searcher=index.searcher(), _first=TantivyBackend._first)


def test_real_native_parent_site_corroboration_and_contained_alias():
    b = tiny_native(); q = RetrievalQuery(name='COMMUNE DE ALPHA ECOLE BETA', address='WRONG',
                                          postcode='11111', insee='11111')
    parsed = compose(b, q, [1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0.], [], 0., set(), set(), set(), [])
    hits = [(s, b._first(b.searcher.doc(loc).to_dict(), 'siret')) for s, loc in b.searcher.search(parsed, limit=100).hits]
    assert len(hits) == 3 and hits[0][1] == '10000000000001'
    assert hits[0][0] > max(s for s, identity in hits if identity != hits[0][1])


def test_geo_scope_is_soft_no_global_locality_filter():
    b = tiny_native(); q = RetrievalQuery(name='ALPHA', address='ROAD', postcode='11111', insee='11111')
    parsed = compose(b, q, [1.]*11, [], 0., set(), set(), set(), [({'10000000000001'}, .8)])
    hits = b.searcher.search(parsed, limit=100).hits
    identities = {b._first(b.searcher.doc(loc).to_dict(), 'siret') for score, loc in hits}
    assert identities == {'10000000000001', '10000000000002', '10000000000003'}
