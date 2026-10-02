import inspect
import numpy as np
from tests.test_retrieval_proof_v56 import backend
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_native_v57 import search as baseline
from src.xgb_matcher.retrieval_proof_v56 import point_features
from src.xgb_matcher.retrieval_sparse_query_v59 import semantic_score
from src.xgb_matcher.retrieval_sparse_native_v60 import search

def test_alpha_zero_exactly_preserves_native_baseline(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA',addresses='12 RUE BLEUE')])
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',insee='75056'); w=np.ones(11)
    assert search(b,q,w,[('commune',1.)],0.)[0]==baseline(b,q,w)[0]

def test_learned_official_words_can_retrieve_site_without_original_name_overlap(tmp_path):
    b=backend(tmp_path,[dict(legal_names='COMMUNE DE VILLE'),dict(legal_names='COMMUNE DE VILLE',insee='99999',postcode='99999')])
    q=RetrievalQuery(name='MAIRIE',insee='75056'); w=np.zeros(11); w[0]=1.
    found,trace=search(b,q,w,[('commune',1.)],2.)
    assert found==[dict(siret='10000000000001',siren='100000000',score=2.)]
    assert trace['fixed_reservoir'] is None

def test_compiled_joint_score_matches_same_two_native_components(tmp_path):
    fields=dict(legal_names=['ALPHA COMMUNE'],site_names=['COLLEGE'],addresses=['12 RUE BLEUE'])
    b=backend(tmp_path,[fields]); q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',insee='75056')
    w=np.arange(1,12,dtype=float)*.1; terms=[('commune',.4),('college',.6)]
    _,x=point_features(b,q,['10000000000001']); found,_=search(b,q,w,terms,2.)
    expected=x[0]@w+2.*semantic_score({k:fields[k] for k in ('legal_names','site_names')},terms,True)
    assert np.isclose(found[0]['score'],expected,atol=1e-5)

def test_serving_does_not_open_training_vectors_labels_or_old_candidates():
    from scripts.retrieval_v60_sparse_initial import run,worker,encode
    source=inspect.getsource(run)+inspect.getsource(worker)+inspect.getsource(encode)
    assert all(term not in source for term in ('training_rows','human_ground_truth','outer_controls','by_siret','BASELINE'))
