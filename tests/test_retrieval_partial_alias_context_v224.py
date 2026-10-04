"""Synthetic engine semantics ONLY, never national quality evidence."""
import tantivy
from types import SimpleNamespace

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery,TantivyBackend
from src.xgb_matcher.retrieval_corroborated_initial_v224 import compose
from tests.test_retrieval_corroborated_initial_v217 import tiny_native


def test_address_context_does_not_promote_partial_alias_back_to_one():
    b=tiny_native();writer=b.index.writer()
    for i,name in [(8,'SERVICE'),(9,'SERVICE ALPHA')]:
        writer.add_document(tantivy.Document.from_dict(dict(document_type='siret',
            siret=f'100000000{i:05d}',siren='100000000',legal_names=name,
            legal_names_exact=name,addresses='ROAD',addresses_exact='ROAD',
            insee='11111',postcode='11111'),b.index.schema))
    writer.commit();b.index.reload()
    b=SimpleNamespace(_tantivy=tantivy,index=b.index,searcher=b.index.searcher(),_first=TantivyBackend._first)
    weights=[0.]*10+[1.]
    q=RetrievalQuery(name='SERVICE ALPHA',address='ROAD',insee='11111')
    hits=b.searcher.search(compose(b,q,weights,[],0.,set(),set(),set(),[]),limit=100,count=False).hits
    scores={b._first(b.searcher.doc(loc).to_dict(),'siret'):score for score,loc in hits}
    assert scores == {'10000000000008':.5,'10000000000009':1.}


def test_legal_particles_do_not_dilute_partial_content_confidence():
    b=tiny_native();writer=b.index.writer()
    writer.add_document(tantivy.Document.from_dict(dict(document_type='siret',
        siret='10000000000008',siren='100000000',legal_names='SERVICE',
        legal_names_exact='SERVICE'),b.index.schema))
    writer.commit();b.index.reload()
    b=SimpleNamespace(_tantivy=tantivy,index=b.index,searcher=b.index.searcher(),_first=TantivyBackend._first)
    weights=[0.,0.,1.]+[0.]*8
    q=RetrievalQuery(name='SOCIETE DE SERVICE ALPHA')
    hits=b.searcher.search(compose(b,q,weights,[],0.,set(),set(),set(),[]),limit=100,count=False).hits
    assert len(hits) == 1 and hits[0][0] == .5
