import copy
from src.xgb_matcher.retrieval_factored_native_v105 import query,BoundedNativeStream
import numpy as np
import pytest
from scipy.spatial import cKDTree

from tests.test_retrieval_national_paths_v99 import sources
from tests.test_retrieval_proof_v56 import backend
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_initialisms_v67 import FEATURES,acronym_components,membership
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_sparse_query_v59 import compiled_component
from src.xgb_matcher.retrieval_national_paths_v105 import QueryWordPostings,search
from src.xgb_matcher.retrieval_scoped_paths_v90 import literal_scores
from scripts.retrieval_v105_exact_intervals import validate_models


@pytest.mark.parametrize('weak,available',[(False,True),(True,True),(False,False)])
def test_actual_initialism_positive_full_corpus_native_score_and_joint_top100(tmp_path,weak,available):
    stats,owners,graph,facts,edges,ids=sources(tmp_path)
    stats['vocabulary_words']=sum(b['words'] for b in stats['buckets'])
    q=RetrievalQuery(name='PC PHARMACIE',address='12 RUE BLEUE',number='12',insee='75056')
    index=QueryWordPostings(stats,[q.name]); names,_=literal_scores(index,q.name)
    root=tmp_path/'native'; root.mkdir()
    aliases={s:('PHARMACIE CENTRE' if i%7==0 else 'OTHER') for i,s in enumerate(ids)}
    b=backend(root,[dict(siret=s,siren=s[:9],legal_names=aliases[s],addresses='12 RUE BLEUE') for s in ids])
    w=np.arange(1,17,dtype=np.float64)*.05; terms=[('other',1.)]
    parsed=query(b,q,w,terms)
    native={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(v)
        for v,loc in b.searcher.search(parsed,limit=len(ids),count=False).hits}
    # Independent component sum on EVERY document, with genuinely positive
    # initialism values; not an all-zero parity test.
    component_values={s:np.zeros(16) for s in ids}
    parts=[*components(b,q),*acronym_components(b,q),compiled_component(b,q,terms)]
    for j,part in enumerate(parts):
        if part is not None:
            for score,loc in b.searcher.search(part,limit=len(ids),count=False).hits:
                component_values[b._first(b.searcher.doc(loc).to_dict(),'siret')][j]=float(score)
    assert sum(v[11]>0 for v in component_values.values())>0
    assert all(v[11]==membership(q.name,[aliases[s]]) for s,v in component_values.items())
    assert all(np.isclose(native.get(s,0.),v@w,atol=1e-6) for s,v in component_values.items())
    keys=np.array(sorted(map(int,ids)),dtype=np.uint64); parents=(keys//100000).astype(np.uint32)
    xy=np.random.default_rng(104).uniform(0,1000,(len(ids),2)); tree=cKDTree(xy)
    g=dict(spatial_crs_compatible=available,ban_x_l93=20.,ban_y_l93=30.,ban_uncertainty_m=100.,ban_declared_weak=weak)
    # Independent full-source name projection with every direct/parent target
    # and every typed edge, rather than trusting the random-access adapter.
    path_scores={s:0. for s in ids}
    grades={'CURRENT_ADDRESS_GEO_CONCORDANT_NOT_HISTORICAL_PROOF':1.,
        'CURRENT_LOCATION_DIFFERS_OR_GEO_DISAGREES':.5,'CURRENT_SITE_BINDING_INCOMPLETE':.25}
    for i,fact in enumerate(facts):
        if not names[i]: continue
        anchors=([s for s in ids if int(s[:9])==fact['official_parent']] if fact['subject_code']==1
                 else [s for s in ids if int(s)==fact['owner_key']])
        for anchor in anchors:
            path_scores[anchor]=max(path_scores[anchor],float(names[i]))
            for edge in edges:
                for left,right in (('predecessor_siret','successor_siret'),('successor_siret','predecessor_siret')):
                    if edge[left]==anchor:
                        path_scores[edge[right]]=max(path_scores[edge[right]],float(names[i])*grades[edge['site_binding']])
    site={s:(1./(1.+float(np.sqrt(np.sum((xy[i]-[20.,30.])**2)))/100.)*(.5 if weak else 1.)) if available else 0.
        for i,s in enumerate(ids)}
    full={s:max(native.get(s,0.),64.*path_scores[s]*site[s]) for s in ids}
    wanted=sorted((s for s in ids if full[s]>0),key=lambda s:(-full[s],s))[:100]
    output,trace=search(b,tree,keys,parents,index,owners,graph,q,g,w,terms,float(w[15]))
    assert [r['siret'] for r in output]==wanted
    assert all(r['score']==full[r['siret']] for r in output)
    assert trace['certified'] and trace['fixed_reservoir'] is None and len(output)<=100
    with pytest.raises(ValueError,match='coefficient'):
        search(b,tree,keys,parents,index,owners,graph,q,g,w,terms,0.)


def test_closed_fold_and_changed_model_recipe_fail_closed():
    models={str(f):dict(features=list(FEATURES),family='HEAD_FULL',heldout_fold=f,
        training_folds=sorted({2,3,4}-{f}),weights=[1.]*16,alpha=1.) for f in (2,3,4)}
    validate_models(models)
    bad=copy.deepcopy(models); bad['2']['alpha']=2.
    with pytest.raises(ValueError,match='recipe'): validate_models(bad)
    bad=copy.deepcopy(models); bad['2']['training_folds']=[1,3]
    with pytest.raises(ValueError,match='recipe'): validate_models(bad)
    bad=copy.deepcopy(models); bad['2']['family']='HEAD_WORD'
    with pytest.raises(ValueError,match='recipe'): validate_models(bad)


@pytest.mark.parametrize('place,number',[(True,True),(True,False),(False,True),(False,False)])
@pytest.mark.parametrize('weights',[
    [0.]*11+[1.,0.,0.,0.,0.],
    [0.]*11+[0.,1.,0.,0.,0.],
    [0.]*11+[0.,0.,1.,0.,0.],
    [0.]*11+[0.,0.,0.,1.,0.],
    (np.arange(16)*.173).tolist(),
])
def test_factorization_full_fixture_components_and_sorted_random_scores_identical(tmp_path,place,number,weights):
    aliases=['PHARMACIE CENTRE','SAS PHARMACIE DU CENTRE','PC','PHARMACIE AUTRE','OTHER',
             'PHARMACIE CENTRE OTHER','PHARMACIE DE CENTRE','PHARMACIE BLEUE']
    b=backend(tmp_path,[dict(legal_names=a,trade_names='PHARMACIE CENTRE' if i==4 else '',
        site_names='PHARMACIE CENTRE' if i==7 else '',addresses='12 RUE BLEUE' if i%2 else 'OTHER',
        insee='75056' if i%3 else '99999',postcode='75001' if i%3 else '99999',number='12' if i%4 else '13')
        for i,a in enumerate(aliases)])
    q=RetrievalQuery(name='PC',address='12 RUE BLEUE',number='12' if number else '',insee='75056' if place else '')
    terms=[('other',1.)]; w=np.asarray(weights); parts=[*components(b,q),*acronym_components(b,q),compiled_component(b,q,terms)]
    ids=[f'{100000000+i}00001' for i in range(len(aliases))]; values={s:np.zeros(16) for s in ids}
    for j,p in enumerate(parts):
        if p is not None:
            for score,loc in b.searcher.search(p,limit=len(ids),count=False).hits:
                values[b._first(b.searcher.doc(loc).to_dict(),'siret')][j]=score
    parsed=query(b,q,w,terms)
    full={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(score)
        for score,loc in b.searcher.search(parsed,limit=len(ids),count=False).hits}
    assert all(np.isclose(full.get(s,0.),v@w,atol=1e-6,rtol=1e-6) for s,v in values.items())
    point=BoundedNativeStream(b,parsed).complete(ids)
    assert point=={s:full.get(s,0.) for s in ids}
    subset=BoundedNativeStream(b,parsed).complete(ids[1:4])
    assert subset=={s:full.get(s,0.) for s in ids[1:4]}
    stream=BoundedNativeStream(b,parsed); seen=[]
    while not stream.exhausted:
        batch=stream.fetch(2); seen.extend(batch)
        for s in ids:
            bound=stream.bound(s)
            assert full.get(s,0.)<=bound['native_upper_bound']
            if bound['native_score'] is not None: assert bound['native_score']==full.get(s,0.)
    assert len(seen)==len(full)
