"""Exhaustive independent initial retrieval oracles; no business synthetic data."""
from types import SimpleNamespace
import numpy as np
import pytest

from src.xgb_matcher.retrieval_activity_initial_v115 import (
    collect,OfficialActivityGrades,native_roundoff,UncertifiedActivityBudget)


class Stream:
    def __init__(self,rows,maximum):
        self.rows=sorted(rows,key=lambda r:(-r['score'],r['siret']));self.maximum=maximum;self.offset=0
        self.exhausted=not len(rows)
    def fetch(self,count):
        result=self.rows[self.offset:self.offset+count];self.offset+=len(result);self.exhausted=self.offset==len(self.rows)
        return result


def sources(native,geometry):
    rows=[dict(siret=f'{100000000+i:09d}00001',siren=f'{100000000+i:09d}') for i in range(len(native))]
    a=Stream([dict(r,score=float(v)) for r,v in zip(rows,native) if v>0],max(native,default=0))
    b=Stream([dict(r,score=float(v)) for r,v in zip(rows,geometry) if v>0],max(geometry,default=0))
    return rows,dict(native=a,geometry=b)


@pytest.mark.parametrize('cap,page',[(100,128),(100,7),(7,16),(1,1),(25,256)])
def test_exact_global_oracle_with_document_specific_activity(cap,page):
    rng=np.random.default_rng(115);n=rng.random(450);g=rng.random(450);a=rng.random(450)
    values=n*(.5+.5*a)+g*(.2+.8*a);rows,channels=sources(n,g)
    exact={r['siret']:float(v) for r,v in zip(rows,values)}
    result,trace=collect(channels,lambda rs:{r['siret']:exact[r['siret']] for r in rs},cap=cap,page_size=page)
    wanted=sorted(rows,key=lambda r:(-exact[r['siret']],r['siret']))[:cap]
    assert [r['siret'] for r in result]==[r['siret'] for r in wanted]
    assert all(r['score']==exact[r['siret']] for r in result)
    assert trace['certified'] and trace['collector_cap']==cap and trace['fixed_reservoir'] is None
    assert trace['live_ranked_groups']<=100


def test_winner_below100_in_both_sources_is_not_truncated():
    n=np.r_[np.ones(200),np.zeros(200),.8];g=np.r_[np.zeros(200),np.ones(200),.8]
    rows,channels=sources(n,g);exact={r['siret']:float(n[i]+g[i]) for i,r in enumerate(rows)}
    winner=rows[-1]['siret']
    assert all(winner not in {r['siret'] for r in s.rows[:100]} for s in channels.values())
    result,trace=collect(channels,lambda rs:{r['siret']:exact[r['siret']] for r in rs},page_size=16)
    assert result[0]['siret']==winner and result[0]['score']==1.6
    assert all(v>100 for v in trace['source_rows'].values())


def test_global_ties_exhaust_and_are_canonical_not_visited_only():
    rows,channels=sources(np.ones(325),np.zeros(325));exact={r['siret']:1. for r in rows}
    result,trace=collect(channels,lambda rs:{r['siret']:exact[r['siret']] for r in rs},page_size=17)
    assert [r['siret'] for r in result]==sorted(exact)[:100]
    assert trace['source_rows']['native']==325 and trace['unseen_score_upper_bound']==0


def test_zero_and_missing_channel_not_denominator_filter():
    _,channels=sources(np.zeros(3),np.zeros(3));result,trace=collect(channels,lambda rs:{})
    assert result==[] and trace['certified']
    rows,channels=sources(np.arange(1.,6.),np.zeros(5))
    exact={r['siret']:float(i+1) for i,r in enumerate(rows)}
    result,trace=collect(channels,lambda rs:{r['siret']:exact[r['siret']] for r in rs})
    assert len(result)==5 and trace['source_calls']['geometry']==0


def test_whole_completion_unsafe_scores_and_nonprogress_stop():
    _,channels=sources([1.],[1.])
    with pytest.raises(ValueError,match='coverage'):collect(channels,lambda rs:{})
    _,channels=sources([1.],[1.])
    with pytest.raises(ValueError,match='bound'):collect(channels,lambda rs:{r['siret']:3. for r in rs})
    class Stalled:
        exhausted=False;maximum=1.
        def fetch(self,count):return []
    with pytest.raises(ValueError,match='Nonprogressing'):collect({'native':Stalled()},lambda rs:{})
    with pytest.raises(UncertifiedActivityBudget) as exc:collect({'native':Stalled()},lambda rs:{},max_seconds=1e-12)
    assert not exc.value.trace['certified']


def test_parent_conflict_and_unsorted_bound_fail():
    rows,channels=sources([1.],[1.]);channels['geometry'].rows[0]['siren']='999999999'
    with pytest.raises(ValueError):collect(channels,lambda rs:{r['siret']:.2 for r in rs})
    rows,channels=sources([1.,.5],[0.,0.]);channels['native'].rows.reverse()
    with pytest.raises(ValueError,match='Sorted'):collect(channels,lambda rs:{r['siret']:.1 for r in rs})


def test_exact_activity_version_join_unknown_kept_missing_site_stops():
    keys=np.array([10000000000001,10000000100001,10000000200001],np.uint64)
    nom=np.array([0,1,1],np.uint8);codes=np.array([0,0,1],np.uint16)
    mapping=np.array([[0,-1],[1,-1]],np.int32);grades=np.array([.3,.9])
    store=OfficialActivityGrades(keys,nom,codes,mapping,grades)
    values=store.values([f'{s:014d}' for s in keys])
    np.testing.assert_array_equal(values,[.3,.9,0])
    assert store.maximum==.9 and store.unknown_joins==1
    with pytest.raises(ValueError,match='Missing'):store.values(['99999999900001'])
    with pytest.raises(ValueError,match='identity'):store.values(['１００００００００００００１'])
    with pytest.raises(ValueError):OfficialActivityGrades(keys,nom,codes,mapping,np.array([np.nan,.5]))
    nom[0]=2
    with pytest.raises(ValueError,match='codebook'):store.values([f'{keys[0]:014d}'])


def test_float32_native_error_envelope_not_hidden_score_clip():
    q=SimpleNamespace(name='UNE SOCIETE TEST',address='12 RUE DU TEST');weights=np.linspace(0,64,11)
    padding,operations=native_roundoff(q,weights)
    assert padding>0 and operations>=512
    rng=np.random.default_rng(8)
    for _ in range(200):
        f=rng.random(11);f[8]*=2
        ideal=f@weights
        terms=f.astype(np.float32)*weights.astype(np.float32)
        native=float(np.sum(terms,dtype=np.float32))
        assert abs(native-ideal)<padding


@pytest.mark.parametrize('mode',['reliable','weak','missing'])
def test_real_tantivy_and_xy_activity_search_matches_entire_fixture_corpus(tmp_path,mode):
    from tests.test_retrieval_proof_v56 import backend
    from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
    from src.xgb_matcher.retrieval_proof_v56 import point_features,geo_features
    from src.xgb_matcher.retrieval_spatial_v43 import build_tree
    from src.xgb_matcher.retrieval_activity_joint_v114 import vector
    from src.xgb_matcher.retrieval_activity_initial_v115 import search
    b=backend(tmp_path,[dict(legal_names='ALPHA' if i%3 else 'BETA',addresses='12 RUE BLEUE',
        insee='99999' if i%5==0 else '75056',postcode='99999' if i%5==0 else '75001') for i in range(245)])
    ids=[f'{100000000+i}00001' for i in range(245)];keys=np.array([int(s) for s in ids],np.uint64)
    nom=np.array([i%2 for i in range(245)],np.uint8);codes=np.array([i%3 for i in range(245)],np.uint16)
    store=OfficialActivityGrades(keys,nom,codes,np.array([[-1,1,2],[0,-1,2]],np.int32),np.array([.1,.7,.9]))
    geo_keys=keys[:180];parents=(geo_keys//100000).astype(np.uint32)
    xy=np.random.default_rng(55).uniform(0,10000,(180,2));tree=build_tree(xy)
    g=dict(spatial_crs_compatible=mode!='missing',ban_x_l93=100.,ban_y_l93=100.,ban_uncertainty_m=25.,ban_declared_weak=mode=='weak')
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',number='12',insee='75056',postcode='75001')
    weights=np.random.default_rng(116).random(30)*2
    ordered,native=point_features(b,q,ids);geom=geo_features(geo_keys,xy,ordered,g);grades=store.values(ordered)
    values={s:float(vector(np.r_[native[i],geom[i]],grades[i])@weights) for i,s in enumerate(ordered)}
    expected=sorted(ordered,key=lambda s:(-values[s],s))[:100]
    result,trace=search(b,tree,geo_keys,parents,q,g,weights,store)
    assert [r['siret'] for r in result]==expected
    assert all(r['score']==values[r['siret']] for r in result)
    assert trace['certified'] and trace['fixed_reservoir'] is None and trace['native_rounding_padding']>0
    assert trace['official_unknown_activity_visits']>0
    if mode=='missing':assert trace['source_calls']['geometry']==0


def test_blind_controller_fixed_budget_and_no_evaluation_before_seal():
    import inspect,json
    from scripts.retrieval_v115_activity_initial import CONFIG,run,worker
    cfg=json.loads(CONFIG.read_text());source=inspect.getsource(run)+inspect.getsource(worker)
    assert cfg['collector_cap']==100 and cfg['fixed_reservoir'] is None
    assert cfg['primary_mean16_seconds']==4 and cfg['resources']['max_seconds']==80
    assert cfg['resources']['max_swapout_pages_delta']==0 and cfg['controller_seconds']==900
    assert 'evaluate(' not in source and 'human_ground_truth' not in source and 'label_side' not in source
