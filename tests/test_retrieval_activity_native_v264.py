"""Synthetic source/engine fixtures ONLY, not national recall or producer proof."""
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_activity_native_v264 import ActivityModels,search
from src.xgb_matcher.retrieval_structural_channels_v210 import SourceFiles
from src.xgb_matcher.retrieval_v20_workbench import record
from test_retrieval_corroborated_initial_v217 import tiny_native


def fixture(tmp_path):
    def saved(name,value):
        p=tmp_path/name
        if name.endswith('.npy'):np.save(p,value)
        else:p.write_text(json.dumps(value))
        return record(p)
    labels=saved('labels.json',dict(rows=[dict(nomenclature='NAF',code=str(i)) for i in range(2140)]))
    nomenclature=saved('nom.npy',np.zeros(4,np.uint8));activity=saved('act.npy',np.zeros(4,np.uint16))
    catalog=saved('catalog.json',dict(status='SEALED_COMPLETE_EXPLICIT_NATIONAL_NOMENCLATURE',
        stats=dict(sites=4,nomenclatures=['NAF'],activity_book=[str(i) for i in range(2140)]),official_labels=labels))
    encoder=saved('encoder.json',{'FIXTURE_ONLY':True})
    source=dict(nomenclatures=['NAF'],activities=[str(i) for i in range(2140)],nomenclature=nomenclature,
        original_activity=activity,labels=labels,model_refs={'encoder':encoder},model_path='FIXTURE_ONLY',max_length=128)
    launch=saved('launch.json',dict(inputs=source,implementation={'encoder':encoder}))
    docs=np.zeros((2140,384));docs[:,0]=1.
    desc=saved('docs.npy',docs);projection=np.zeros((385,384));projection[0,0]=1.
    p=saved('projection.npy',projection)
    models={str(f):dict(heldout_fold=f,training_folds=sorted({2,3,4}-{f}),query_ids=['TRAIN'],
        family='SEMANTIC_RIDGE',ridge=1,identifiers_as_features=False,catalog=catalog,descriptions=desc,parameter=p) for f in (2,3,4)}
    model_ref=saved('models.json',models)
    semantic=dict(status='SEALED_TRAINING_ONLY_OFFICIAL_ACTIVITY_SEMANTIC_PROBE',resources={'pass_resource_policy':True},
        no_crm_alias_dictionary=True,launch=launch,models=model_ref,encoding=encoder,
        **{k:False for k in ('recall_measured','fold0_opened','fold1_opened','identifiers_as_features',
            'output_candidates','national_search_performed','downstream_models_changed')})
    sem=saved('semantic.json',semantic)
    qs=[dict(query_id='q'+str(i),fold=2+i%3) for i in range(128)]
    vectors=np.zeros((128,384));vectors[:,0]=1.
    v=saved('queries.npy',vectors)
    expansions=saved('expansions.json',dict(truth_opened=False,resources={'pass_resource_policy':True},rows=qs,vectors=v))
    bundle=saved('bundle.json',dict(model_refs=source['model_refs'],model_path=source['model_path'],max_length=128,encoder_prefix='nom'))
    old=dict(expansions=expansions,bundle=bundle)
    structured=dict(activity_catalog=catalog,stats=dict(sites=4),
        reused_arrays=dict(nomenclature_codes=nomenclature,original_activity_codes=activity))
    heads={str(f):dict(query_ids=['TRAIN']) for f in (2,3,4)}
    return saved,sem,old,qs,heads,structured,models,model_ref,semantic


def test_full_query_only_models_and_exact_versioned_mapping(tmp_path):
    _,sem,old,qs,heads,structured,*_=fixture(tmp_path);files=SourceFiles(lambda:None)
    a=ActivityModels(files,sem,old,qs,heads,structured)
    assert a.mapping.shape==(1,2140)
    assert all(v.shape==(128,2140) and np.all(v==1.) for v in a.grades.values())
    files.verify_after()


@pytest.mark.parametrize('kind',['heldfold','trainingfold','pilot_overlap','protected_training','family','ridge','identifiers','description_mismatch'])
def test_model_protection_failure_not_silently_used(tmp_path,kind):
    saved,sem,old,qs,heads,structured,models,model_ref,semantic=fixture(tmp_path)
    m=models['2']
    if kind=='heldfold':m['heldout_fold']=1
    elif kind=='trainingfold':m['training_folds']=[2,3]
    elif kind=='pilot_overlap':m['query_ids']=[qs[0]['query_id']];heads['2']['query_ids']=m['query_ids']
    elif kind=='protected_training':m['query_ids']=['OUTSIDE_SAFE_HEAD']
    elif kind=='family':m['family']='TRAIN_FREQUENCY'
    elif kind=='ridge':m['ridge']=10
    elif kind=='identifiers':m['identifiers_as_features']=True
    else:models['3']['descriptions']=saved('other.npy',np.ones((2140,384)))
    semantic['models']=saved('changed_models.json',models);sem=saved('changed_sem.json',semantic)
    with pytest.raises(ValueError):ActivityModels(SourceFiles(lambda:None),sem,old,qs,heads,structured)


def test_activity_fused_before_one_collector_not_score_completion(monkeypatch):
    from src.xgb_matcher import retrieval_activity_native_v264 as module
    b=tiny_native();native=b.searcher;calls=[]
    def queried(p,**kw):calls.append(kw);return native.search(p,**kw)
    b.searcher=SimpleNamespace(search=queried,doc=native.doc)
    guard=lambda:None
    e=SimpleNamespace(files=SimpleNamespace(guard=guard),name_owners=lambda t:(set(),set(),{}),
        related=lambda p,s:(set(),{}),radial_proofs=lambda geo:([],{}))
    monkeypatch.setattr(module,'collect_groups',lambda *args:({.9:{'10000000000002'}},dict(scanned=3)))
    rows,t=search(b,e,RetrievalQuery(name='ALPHA',postcode='11111'),{},[1.]*11,[],0.,[.2],np.array([[0]],np.int32))
    assert calls==[dict(limit=100,count=False)] and rows[0]['siret']=='10000000000002'
    assert e.files.guard is guard and t['activity_coefficient']==1.
    assert not t['metric_equivalent_to_v115_v116'] and not t['score_after_fixed_pool']
    assert t['unknown_and_offzone_sites_remain_reachable'] and len(rows)==3
