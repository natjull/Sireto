import numpy as np
import pytest
from scipy.spatial import cKDTree
from tests.test_retrieval_national_paths_v99 import sources
from tests.test_retrieval_proof_v56 import backend
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_national_paths_v99 import CompactWordPostings,search as old_search
from src.xgb_matcher.retrieval_national_paths_v101 import QueryWordPostings,search
from src.xgb_matcher.retrieval_scoped_paths_v90 import literal_scores


def test_query_projection_exact_full_df_scores_and_unknown_request_stop(tmp_path):
    stats,_,_,_,_,_=sources(tmp_path)
    stats['vocabulary_words']=sum(b['words'] for b in stats['buckets'])
    full=CompactWordPostings(stats); names=['PHARMACIE CENTRE','PORT DU CENTRE','ABSENT','FORBIDDEN CONTEXT']
    projected=QueryWordPostings(stats,names)
    assert set(projected.terms)==set(full.terms)&projected.allowed
    assert all(projected.terms[t]==full.terms[t] for t in projected.terms)
    for name in names:
        a,ta=literal_scores(full,name); b,tb=literal_scores(projected,name)
        assert np.array_equal(a,b) and ta==tb
    with pytest.raises(ValueError,match='Unloaded query word'):
        projected.scores({'OTHER':1.})


@pytest.mark.parametrize('weak,available',[(False,True),(True,True),(False,False)])
def test_projected_memory_and_larger_completion_same_certified_initial100(tmp_path,weak,available):
    stats,owners,graph,_,_,ids=sources(tmp_path)
    stats['vocabulary_words']=sum(b['words'] for b in stats['buckets'])
    q=RetrievalQuery(name='PHARMACIE CENTRE',address='12 RUE BLEUE',insee='75056')
    full=CompactWordPostings(stats); projected=QueryWordPostings(stats,[q.name])
    root=tmp_path/'native'; root.mkdir()
    b=backend(root,[dict(siret=s,siren=s[:9],legal_names='OTHER',addresses='12 RUE BLEUE') for s in ids])
    keys=np.array(sorted(map(int,ids)),dtype=np.uint64); parents=(keys//100000).astype(np.uint32)
    xy=np.random.default_rng(101).uniform(0,1000,(len(ids),2)); tree=cKDTree(xy)
    g=dict(spatial_crs_compatible=available,ban_x_l93=20.,ban_y_l93=30.,ban_uncertainty_m=100.,ban_declared_weak=weak)
    old,to=old_search(b,tree,keys,parents,full,owners,graph,q,g,np.ones(11),[],0.)
    new,tn=search(b,tree,keys,parents,projected,owners,graph,q,g,np.ones(11),[],0.)
    assert new==old and tn['certified'] and tn['fixed_reservoir'] is None
    assert tn['word_index']==to['word_index'] and tn['collector_cap']==100
