import numpy as np
import pytest

from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_joint_v48 import index_query,search
from src.xgb_matcher.retrieval_joint_v46 import geo_features
from src.xgb_matcher.retrieval_spatial_v43 import build_tree
from tests.test_retrieval_v38_initial_top100 import backend


@pytest.mark.parametrize('w',[[.2,.04,1.5,.4,0,0,1.4,0],[0,0,1.2,.15,0,0,1.8,0]])
def test_termset_eligibility_preserves_every_native_score_and_document(tmp_path,w):
    rows=[dict(legal_names='ALPHA' if i%3 else 'BETA',addresses='RUE BLEUE' if i%2 else 'RUE VERTE',
        insee='75056' if i%5 else '99999',postcode='75001' if i%5 else '99999') for i in range(350)]
    b=backend(tmp_path,rows); q=RetrievalQuery(name='ALPHA',address='RUE BLEUE',insee='75056',postcode='75001')
    scores=[]
    for parsed in (v42.index_query(b,q,w),index_query(b,q,w)):
        scores.append({b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s)
            for s,loc in b.searcher.search(parsed,limit=350,count=False).hits})
    assert scores[0]==scores[1]


@pytest.mark.parametrize('w',[[.2,.04,1.5,.4,0,0,1.4,0,1.7,.3],[0,0,1.2,.15,0,0,1.8,0,1.2,.55]])
def test_joint_matches_full_native_oracle_and_keeps_nonlocal_spatial_sites(tmp_path,w):
    rows=[dict(legal_names='ALPHA' if i%3 else 'BETA',addresses='RUE BLEUE',
        insee='75056' if i%5 else '99999',postcode='75001' if i%5 else '99999') for i in range(350)]
    b=backend(tmp_path,rows); q=RetrievalQuery(name='ALPHA',address='RUE BLEUE',insee='75056',postcode='75001')
    ids=[f'{100000000+i}00001' for i in range(350)]
    keys=np.array([int(s) for s in ids],dtype=np.uint64); parents=(keys//100000).astype(np.uint32)
    xy=np.random.default_rng(48).uniform(0,500,(350,2)); tree=build_tree(xy)
    g=dict(spatial_crs_compatible=True,ban_x_l93=100.,ban_y_l93=100.,ban_uncertainty_m=25.,ban_declared_weak=False)
    found,trace=search(b,tree,keys,parents,q,g,w)
    full={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s)
        for s,loc in b.searcher.search(v42.index_query(b,q,w[:8]),limit=350,count=False).hits}
    geo=geo_features(keys,xy,ids,g)[:,0]*(w[8]+w[9])
    values={s:full.get(s,0.)+geo[i] for i,s in enumerate(ids)}
    expected=sorted(ids,key=lambda s:(-values[s],s))[:100]
    assert [r['siret'] for r in found]==expected and [r['score'] for r in found]==[values[s] for s in expected]
    assert trace['certified'] and trace['fixed_reservoir'] is None
