import json

import numpy as np
import pytest
import tantivy

from scripts.build_hierarchical_retrieval_index import _schema
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery,TantivyBackend,normalize_text
from src.xgb_matcher.retrieval_proof_v56 import components,point_features,index_query,geo_features,search,LEXICAL,FEATURES,signal
from src.xgb_matcher.retrieval_spatial_v43 import build_tree
from scripts import retrieval_v38_initial_top100 as v38


def backend(tmp_path,rows):
    schema=_schema(tantivy); root=tmp_path/'tantivy'; root.mkdir(); index=tantivy.Index(schema,path=str(root))
    writer=index.writer(heap_size=15_000_000,num_threads=1)
    for i,r in enumerate(rows):
        s=f'{100000000+i}00001'; row={**dict(document_type='siret',siren=s[:9],siret=s,insee='75056',postcode='75001',number='12',is_siege='0'),**r}
        for target,source in [('legal_names_exact','legal_names'),('trade_names_exact','trade_names'),('site_names_exact','site_names'),('addresses_exact','addresses')]:
            values=row.get(source,[]); values=values if isinstance(values,list) else [values]
            row[target]=[normalize_text(v) for v in values]
        for target,source in [('current_name_ngrams','legal_names'),('address_ngrams','addresses')]:
            values=row.get(source,[]); values=values if isinstance(values,list) else [values]
            row[target]=' '.join(t for v in values for t in v38.tokens(v,True))
        writer.add_document(tantivy.Document.from_dict(row,schema))
    writer.commit(); writer.wait_merging_threads(); index.reload()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(build_scope='test_fixture')))
    return TantivyBackend(tmp_path)


def test_exact_proofs_distinct_from_bag_and_frequency_does_not_change_coverage(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA BETA',addresses='12 RUE BLEUE'),dict(legal_names='ALPHA '*50,addresses='ZZZ')])
    ids,x=point_features(b,RetrievalQuery(name='ALPHA BETA',address='12 RUE BLEUE'),['10000000000001','10000000100001'])
    assert x[0,2]==x[0,3]==1. and x[1,2]==0.
    assert x[0,0]==1. and x[1,0]<=1.


def test_site_evidence_without_any_name_or_address_text_is_not_masked(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ZZZ',is_siege='1')]); q=RetrievalQuery(name='',number='12',insee='75056')
    _,x=point_features(b,q,['10000000000001']); assert x[0,4]==x[0,5]==x[0,9]==1.
    w=np.ones(len(LEXICAL)); hits=b.searcher.search(index_query(b,q,w),limit=1,count=False).hits
    assert hits and np.isclose(hits[0][0],x[0]@w)


def test_every_compiled_native_score_matches_same_training_components(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA',addresses='12 RUE BLEUE'),dict(legal_names='BETA',addresses='13 RUE VERTE',postcode='99999',insee='99999')])
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',number='12',insee='75056'); w=np.arange(1,len(LEXICAL)+1)*.25
    ids,x=point_features(b,q,['10000000000001','10000000100001'])
    native={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s) for s,loc in b.searcher.search(index_query(b,q,w),limit=2,count=False).hits}
    assert all(np.isclose(native.get(s,0.),x[i]@w,atol=1e-5) for i,s in enumerate(ids))


def test_joint_direct_search_matches_entire_corpus_oracle(tmp_path):
    b=backend(tmp_path,[dict(legal_names='ALPHA' if i%3 else 'BETA',addresses='12 RUE BLEUE',insee='99999' if i%5==0 else '75056',postcode='99999' if i%5==0 else '75001') for i in range(350)])
    q=RetrievalQuery(name='ALPHA',address='12 RUE BLEUE',number='12',insee='75056',postcode='75001')
    ids=[f'{100000000+i}00001' for i in range(350)]; keys=np.array([int(s) for s in ids],dtype=np.uint64); parents=(keys//100000).astype(np.uint32)
    xy=np.random.default_rng(55).uniform(0,10000,(350,2)); tree=build_tree(xy)
    g=dict(spatial_crs_compatible=True,ban_x_l93=100.,ban_y_l93=100.,ban_uncertainty_m=25.,ban_declared_weak=False)
    w=np.array([0.,0.,0.,0.,.1,0.,.2,.1,0.,0.,0.,1.,2.,3.,4.,.1,.2,.3,.4])
    found,trace=search(b,tree,keys,parents,q,g,w)
    native={b._first(b.searcher.doc(loc).to_dict(),'siret'):float(s) for s,loc in b.searcher.search(index_query(b,q,w[:len(LEXICAL)]),limit=350,count=False).hits}
    gf=geo_features(keys,xy,ids,g); geom=sum((w[len(LEXICAL)+i]+w[len(LEXICAL)+i+4])*gf[:,i] for i in range(4))
    scores={s:native.get(s,0.)+geom[i] for i,s in enumerate(ids)}; expected=sorted(ids,key=lambda s:(-scores[s],s))[:100]
    assert [r['siret'] for r in found]==expected and all(r['score']==scores[r['siret']] for r in found)
    assert trace['certified'] and trace['fixed_reservoir'] is None and trace['zero_nonlocal_lexical_pruning']
    assert any(s in expected for i,s in enumerate(ids) if i%5==0)


def test_weak_ban_and_missing_coordinates_do_not_create_positive_geo_features():
    keys=np.array([10000000000001],dtype=np.uint64); xy=np.array([[1.,2.]])
    g=dict(spatial_crs_compatible=True,ban_x_l93=1.,ban_y_l93=2.,ban_uncertainty_m=5.,ban_declared_weak=True)
    values=geo_features(keys,xy,['10000000000001','10000000100001'],g)
    assert np.array_equal(values[0,:4],np.ones(4)) and not values[:,4:].any() and not values[1].any()


def test_identifiers_are_not_query_features():
    assert signal('ALPHA 123456789 12345678900001 75001')=='ALPHA 75001'


def test_controller_import_frozen_caps_and_no_serving_training_rows_read():
    import inspect
    from scripts.retrieval_v56_exact_fields import CONFIG,run,worker
    cfg=json.loads(CONFIG.read_text())
    assert cfg['primary_mean16_seconds']==4 and cfg['max_query_search_seconds']==20
    assert 'training_rows' not in inspect.getsource(run)+inspect.getsource(worker)


def test_empty_legacy_names_exact_is_not_used_in_v56(tmp_path):
    from src.xgb_matcher.retrieval_proof_v55 import point_features as legacy
    b=backend(tmp_path,[dict(legal_names='ALPHA',trade_names='BETA',site_names='GAMMA')])
    for name in ('ALPHA','BETA','GAMMA'):
        q=RetrievalQuery(name=name)
        assert point_features(b,q,['10000000000001'])[1][0,2]==1.
        assert legacy(b,q,['10000000000001'])[1][0,2]==0.


def test_cached_patch_preserves_every_other_coordinate_and_rejects_misalignment():
    from scripts.retrieval_v56_exact_fields import patch_scene
    values=np.arange(len(FEATURES),dtype=float); values[[2,10]]=0.
    row=dict(query_id='q',fold=2,positive_siren='100000000',positive_siret='10000000000001',negatives=[dict(siren='200000000',siret='20000000000001')])
    scene=dict(query_id='q',fold=2,positive_siren='100000000',positive_features=values.tolist(),negatives=[dict(siren='200000000',features=values.tolist())])
    patched=patch_scene(row,scene,{'10000000000001':1.})
    expected=values.copy(); expected[[2,10]]=1.
    assert np.array_equal(patched['positive_features'],expected)
    assert patched['negatives']==scene['negatives']
    with pytest.raises(ValueError,match='alignment'):
        patch_scene({**row,'query_id':'other'},scene,{})
    with pytest.raises(ValueError,match='parent'):
        patch_scene(row,{**scene,'negatives':[dict(siren='wrong',features=values.tolist())]},{})


def test_historical_supporting_names_do_not_gain_unavailable_exact_proof(tmp_path):
    b=backend(tmp_path,[dict(historical_names='ALPHA',supporting_names='BETA')])
    for name in ('ALPHA','BETA'):
        _,x=point_features(b,RetrievalQuery(name=name),['10000000000001'])
        assert x[0,0]>0. and x[0,2]==0.

