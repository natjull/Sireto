import inspect
from scripts import preflight_retrieval_v67_stored_initialisms as v67
from src.xgb_matcher.retrieval_initialisms_v67 import membership,acronym_components
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from tests.test_retrieval_proof_v56 import backend


def test_exact_terms_not_stored_but_current_role_aliases_are(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA BETA')])
    _,loc=b.searcher.search(b._tantivy.Query.all_query(),limit=1,count=False).hits[0]
    d=b.searcher.doc(loc).to_dict()
    assert not d.get('legal_names_exact') and d['legal_names']==['ALPHA BETA']
    assert membership('AB',d['legal_names'])==1
    assert b.searcher.search(acronym_components(b,RetrievalQuery(name='AB'))[0],limit=10,count=False).hits


def test_offline_alias_normalization_matches_index_including_numeric_tokens(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA 123456789 BETA')])
    assert membership('AB',['ALPHA 123456789 BETA'])==0
    assert not b.searcher.search(acronym_components(b,RetrievalQuery(name='AB'))[0],limit=10,count=False).hits


def test_repair_guard_is_exhaustive_not_just_small_zero_parity_sample():
    source=inspect.getsource(v67.worker)
    assert "('legal_names','trade_names','site_names')" in source
    assert "exact!=native[2]" in source and 'current_exact_audited+=1' in source
    assert 'positive_initialisms' in source and 'negative_initialisms' in source
