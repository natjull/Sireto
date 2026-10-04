import pytest
import tantivy
from types import SimpleNamespace
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery,TantivyBackend
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_compound_initialisms_v220 import membership,tokens,component
from tests.test_retrieval_corroborated_initial_v217 import tiny_native


@pytest.mark.parametrize('query,alias',[
    ('AZB','AZIMUTH BRAVO'),('AZB','ALPHA ZULU BRAVO'),
    ('ABCD','ALPHA BETA CHARLIE DELTA'),('ABCD','ABACUS CHARLIE DELTA'),
    ('ABCD','ALPHA BRAVO CDROM'),('ABCD','ABACUS CDROM'),
    ('SAS ABCD SITE','SAS ALPHA DE BRAVO CHARLIE DELTA SARL')])
def test_all_compound_partitions_and_legal_particles(query,alias):
    assert membership(query,[alias])


@pytest.mark.parametrize('query,aliases',[
    ('AB',['ALPHA','BRAVO']),('AB',['ALPHA CHARLIE','DELTA BRAVO']),
    ('AB',['ALPHA BRAVO CHARLIE']),('AB',['ABACUS']),
    ('AB',['BRAVO ALPHA']),('AB',['[ND]']),('123456789',['ALPHA BRAVO'])])
def test_no_cross_alias_trailing_context_or_identifier_proof(query,aliases):
    assert not membership(query,aliases)


def test_no_token_top_k_or_manual_crm_dictionary():
    assert tokens('AB CD EF GH IJ KL 12345678900001') == ['AB','CD','EF','GH','IJ','KL']


def engine():
    b=tiny_native()
    # Synthetic native fixture ONLY, not measured business evidence.
    w=b.index.writer()
    for i,alias,place in [(4,'AZIMUTH BRAVO','11111'),(5,'ALPHA ZULU BRAVO','11111'),
                          (6,'AZIMUTH BRAVO','22222'),(7,'ALPHA BRAVO','11111')]:
        w.add_document(tantivy.Document.from_dict(dict(document_type='siret',
            siret=f'100000000{i:05d}',siren='100000000',legal_names=alias,
            legal_names_exact=alias,insee=place,postcode=place), b.index.schema))
    w.commit(); b.index.reload()
    return SimpleNamespace(_tantivy=tantivy,index=b.index,searcher=b.index.searcher(),_first=TantivyBackend._first)


def test_real_native_regex_matches_partitioned_whole_alias_and_place():
    b=engine(); q=RetrievalQuery(name='AZB',insee='11111')
    hits=b.searcher.search(component(b,q,components(b,q)),limit=100,count=False).hits
    assert {b._first(b.searcher.doc(loc).to_dict(),'siret') for _,loc in hits} == {
        '10000000000004','10000000000005'}
    assert all(score == 1. for score,_ in hits)


def test_no_location_does_not_invent_location():
    b=engine();q=RetrievalQuery(name='AZB')
    assert component(b,q,components(b,q)) is None
