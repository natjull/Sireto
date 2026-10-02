import numpy as np
import pytest

from scripts import retrieval_v42_indexed_metric as v42
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_joint_v46 import collect,search,geo_features,fit_metric
from src.xgb_matcher.retrieval_spatial_v43 import build_tree
from tests.test_retrieval_v38_initial_top100 import backend


class FixtureStream:
    def __init__(self,values): self.values=values; self.offset=0; self.exhausted=False
    def fetch(self,n):
        rows=self.values[self.offset:self.offset+n]; self.offset+=len(rows)
        self.exhausted=self.offset==len(self.values)
        return rows


def fixture_scores(x):
    ids=[f'{100000000+i}00001' for i in range(len(x))]
    streams={str(j):FixtureStream([dict(siret=ids[i],siren=ids[i][:9],score=float(x[i,j]))
        for i in np.argsort(-x[:,j],kind='stable')]) for j in range(x.shape[1])}
    lookup=dict(zip(ids,np.sum(x,axis=1)))
    rows,trace=collect(streams,lambda rs:{r['siret']:lookup[r['siret']] for r in rs})
    expected=sorted(ids,key=lambda s:(-lookup[s],s))[:100]
    return rows,trace,expected


def test_additive_winners_can_be_beyond100_in_both_sources():
    x=np.vstack([np.tile([10.,0.],(100,1)),np.tile([6.,6.],(100,1)),np.tile([0.,10.],(100,1))])
    rows,trace,expected=fixture_scores(x)
    assert [r['siret'] for r in rows]==expected
    assert trace['unique_documents_visited']>200 and trace['fixed_reservoir'] is None
    assert trace['certified'] and trace['score_cutoff']>=trace['unseen_score_upper_bound']


@pytest.mark.parametrize('seed',[1,2,3,4])
def test_threshold_matches_full2000_document_exhaustive_metric(seed):
    x=np.random.default_rng(seed).uniform(0,10,(2000,2))
    rows,trace,expected=fixture_scores(x)
    assert [r['siret'] for r in rows]==expected and len(rows)==100
    assert trace['certified']


def test_missing_coordinates_are_zero_feature_not_document_filter():
    keys=np.array([10000000000001],dtype=np.uint64); xy=np.array([[0.,0.]])
    g=dict(spatial_crs_compatible=True,ban_x_l93=0.,ban_y_l93=0.,ban_uncertainty_m=25.,ban_declared_weak=True)
    assert geo_features(keys,xy,['10000000000001','99999999900001'],g).tolist()==[[1.,0.],[0.,0.]]
    g['spatial_crs_compatible']=False
    assert not geo_features(keys,xy,['10000000000001'],g).any()


def test_native_joint_search_matches_exhaustive_fixture_and_keeps_no_xy(tmp_path):
    rows=[dict(legal_names='ALPHA' if i%3 else 'BETA',addresses='RUE BLEUE' if i%2 else 'RUE VERTE',
        insee='75056' if i%5 else '99999',postcode='75001' if i%5 else '99999') for i in range(350)]
    b=backend(tmp_path,rows); q=RetrievalQuery(name='ALPHA',address='RUE BLEUE',insee='75056',postcode='75001')
    ids=[f'{100000000+i}00001' for i in range(len(rows))]
    # A genuine source missingness case: only the first200 fixture sites have XY.
    keys=np.array([int(s) for s in ids[:200]],dtype=np.uint64); parents=(keys//100000).astype(np.uint32)
    xy=np.random.default_rng(46).uniform(0,500,(200,2)); tree=build_tree(xy)
    g=dict(spatial_crs_compatible=True,ban_x_l93=100.,ban_y_l93=100.,ban_uncertainty_m=25.,ban_declared_weak=False)
    w=np.array([.3,.2,1.1,.4,.05,.03,.7,0.,4.,2.])
    found,trace=search(b,tree,keys,parents,q,g,w)
    tv=b._tantivy
    full=b.searcher.search(v42.index_query(b,q,w[:8]),limit=len(rows),count=False).hits
    lexical={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(score) for score,loc in full}
    gf=geo_features(keys,xy,ids,g)[:,0]*(w[8]+w[9])
    scores={s:lexical.get(s,0.)+gf[i] for i,s in enumerate(ids)}
    expected=sorted(ids,key=lambda s:(-scores[s],s))[:100]
    assert [r['siret'] for r in found]==expected
    assert [r['score'] for r in found]==[scores[s] for s in expected]
    assert trace['certified'] and trace['fixed_reservoir'] is None


def test_strong_name_can_win_without_source_coordinates(tmp_path):
    b=backend(tmp_path,[dict(legal_names='UNIQUEFOO' if i==349 else 'OTHER') for i in range(350)])
    keys=np.array([int(f'{100000000+i}00001') for i in range(200)],dtype=np.uint64)
    parents=(keys//100000).astype(np.uint32); xy=np.zeros((200,2),dtype=np.float64)
    geo=dict(spatial_crs_compatible=True,ban_x_l93=0.,ban_y_l93=0.,ban_uncertainty_m=25.,ban_declared_weak=True)
    rows,trace=search(b,build_tree(xy),keys,parents,RetrievalQuery(name='UNIQUEFOO'),geo,
        [1,0,0,0,0,0,0,0,.01,0])
    assert rows[0]['siret']=='10000034900001' and trace['certified']


def test_zero_nonlocal_lexical_pruning_does_not_remove_nearby_nonlocal_sites(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA',insee='99999' if i==149 else '75056',
        postcode='99999' if i==149 else '75001') for i in range(150)])
    keys=np.array([int(f'{100000000+i}00001') for i in range(150)],dtype=np.uint64)
    parents=(keys//100000).astype(np.uint32)
    xy=np.full((150,2),500.,dtype=np.float64); xy[-1]=[0.,0.]
    geo=dict(spatial_crs_compatible=True,ban_x_l93=0.,ban_y_l93=0.,ban_uncertainty_m=25.,ban_declared_weak=False)
    q=RetrievalQuery(name='ALPHA',insee='75056',postcode='75001')
    rows,trace=search(b,build_tree(xy),keys,parents,q,geo,[0,0,.1,0,0,0,.1,0,20,0])
    assert trace['zero_nonlocal_lexical_pruning']
    assert rows[0]['siret']=='10000014900001' and trace['certified']


def test_fit_excludes_heldout_queries_and_negative_entities_exactly():
    import copy
    rng=np.random.default_rng(46); registry={str(f):[f'{f}00000000'] for f in (2,3,4)}
    scenes=[dict(query_id=str(i),fold=2+i%3,positive_siren=f'{2+i%3}00000000',
        positive_features=rng.uniform(0,5,10).tolist(),negatives=[
            dict(siren='999999999',features=rng.uniform(0,5,10).tolist()),
            dict(siren='200000000',features=rng.uniform(0,5,10).tolist())]) for i in range(60)]
    first=fit_metric(scenes,2,registry); changed=copy.deepcopy(scenes)
    for r in changed:
        if r['fold']==2: r['positive_features']=[900.]*10; r['negatives']=[]
        else: r['negatives'][1]['features']=[900.]*10
    second=fit_metric(changed,2,registry)
    np.testing.assert_array_equal(first['weights'],second['weights'])
    assert first['training_queries']==40 and first['training_pairs']==40
    assert first['training_folds']==[3,4] and not first['identifiers_as_features']


def test_timeout_never_returns_an_uncertified_shortcut():
    stream=FixtureStream([dict(siret='10000000000001',siren='100000000',score=1.)])
    with pytest.raises(TimeoutError):
        collect({'fixture':stream},lambda rs:{r['siret']:1. for r in rs},max_seconds=1e-100)
