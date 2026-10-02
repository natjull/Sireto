import numpy as np
import pytest
from scripts import retrieval_v42_indexed_metric as v42
from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from tests.test_retrieval_v38_initial_top100 import backend


def test_baseline_weights_equal_existing_initial_search(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA',addresses='12 RUE BLEUE'),
                        dict(legal_names='ALPHA',addresses='12 RUE BLEUE',insee='99999',postcode='99999'),
                        dict(legal_names='BETA',addresses='12 RUE BLEUE')])
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',insee='75056',postcode='75001')
    old=v38.search(b,q,'primary'); new=v42.search(b,q,v42.BASE)
    assert [r['siret'] for r in old]==[r['siret'] for r in new]
    assert [r['score'] for r in old]==pytest.approx([r['score'] for r in new],rel=1e-6)


def test_compiled_index_score_equals_linear_score_over_full_fixture(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA',addresses='12 RUE BLEUE'),
                        dict(legal_names='ALPHA',addresses='RUE VERTE',insee='99999',postcode='99999'),
                        dict(legal_names='BETA',addresses='12 RUE BLEUE')])
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',insee='75056',postcode='75001')
    ids=[f'{100000000+i}00001' for i in range(3)]
    found_ids,x=v42.training_features(b,q,ids)
    w=np.array([.3,.6,1.,2.,.5,3.,8.,0.])
    expected=sorted(zip(found_ids,x@w),key=lambda r:(-r[1],r[0]))
    found=v42.search(b,q,w)
    assert [r['siret'] for r in found]==[r[0] for r in expected]
    assert [r['score'] for r in found]==pytest.approx([r[1] for r in expected],rel=1e-6)


def test_no_text_no_candidates_and_cap_is_hard(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA') for _ in range(150)])
    assert v42.search(b,RetrievalQuery(name='',insee='75056'),np.ones(8))==[]
    assert len(v42.search(b,RetrievalQuery(name='ALPHA'),np.ones(8)))==100
    for w in (np.ones(7),np.zeros(8),np.full(8,-1.),np.full(8,np.nan)):
        with pytest.raises(ValueError): v42.search(b,RetrievalQuery(name='ALPHA'),w)


def test_fit_heldout_scenes_and_entities_cannot_change_weights():
    rng=np.random.default_rng(42); scenes=[]
    registry={str(f):[f'{f}00000000'] for f in (2,3,4)}
    for i in range(60):
        fold=2+i%3
        scenes.append(dict(query_id=str(i),fold=fold,positive_siren=f'{fold}00000000',
            positive_features=rng.uniform(0,5,8).tolist(),
            negatives=[dict(siren='999999999',features=rng.uniform(0,5,8).tolist()),
                       dict(siren='200000000',features=rng.uniform(0,5,8).tolist())]))
    a=v42.fit_metric(scenes,2,registry)
    import copy
    mutated=copy.deepcopy(scenes)
    for r in mutated:
        if r['fold']==2: r['positive_features']=[500.]*8; r['negatives']=[]
        else: r['negatives'][1]['features']=[500.]*8
    b=v42.fit_metric(mutated,2,registry)
    np.testing.assert_array_equal(a['weights'],b['weights'])
    assert a['training_queries']==40 and a['training_pairs']==40
    assert a['training_folds']==[3,4] and a['objective_after']<=a['objective_before']
    with pytest.raises(ValueError): v42.fit_metric(scenes,1,registry)
