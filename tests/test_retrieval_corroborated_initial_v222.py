import tantivy
from types import SimpleNamespace
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery,TantivyBackend
from src.xgb_matcher.retrieval_corroborated_initial_v222 import compose
from tests.test_retrieval_compound_initialisms_v222 import engine
from tests.test_retrieval_corroborated_initial_v217 import tiny_native


def test_compound_alias_name_is_bounded_max_not_duplicate_bonus():
    b=engine();q=RetrievalQuery(name='AZB',insee='11111')
    w=[1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.]
    hit=b.searcher.search(compose(b,q,w,[],0.,set(),set(),set(),[]),limit=100,count=False).hits
    scores={b._first(b.searcher.doc(loc).to_dict(),'siret'):s for s,loc in hit}
    assert scores['10000000000004'] == scores['10000000000005'] == 2.
    assert scores.get('10000000000006',0.) < 2.  # Outside remains eligible, not globally filtered.


def test_singleton_partial_alias_confidence_and_full_name_preserved():
    b=tiny_native();writer=b.index.writer()
    for i,name in [(8,'SERVICE'),(9,'SERVICE ALPHA')]:
        writer.add_document(tantivy.Document.from_dict(dict(document_type='siret',
            siret=f'100000000{i:05d}',siren='100000000',legal_names=name,
            legal_names_exact=name),b.index.schema))
    writer.commit();b.index.reload()
    b=SimpleNamespace(_tantivy=tantivy,index=b.index,searcher=b.index.searcher(),_first=TantivyBackend._first)
    weights=[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.]
    def scores(name):
        q=RetrievalQuery(name=name)
        hits=b.searcher.search(compose(b,q,weights,[],0.,set(),set(),set(),[]),limit=100,count=False).hits
        return {b._first(b.searcher.doc(loc).to_dict(),'siret'):s for s,loc in hits}
    s=scores('SERVICE ALPHA')
    assert s['10000000000008'] == .5 and s['10000000000009'] == 1.
    assert scores('SERVICE')['10000000000008'] == 1.


def test_global_paths_unchanged_without_locality():
    b=tiny_native();q=RetrievalQuery(name='ALPHA',address='ROAD')
    hits=b.searcher.search(compose(b,q,[1.]*11,[],0.,set(),set(),set(),[]),limit=100,count=False).hits
    assert len(hits) == 3
