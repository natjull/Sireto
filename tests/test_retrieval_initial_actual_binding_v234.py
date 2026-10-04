"""Synthetic serving authority/mutation tests, not business retrieval."""
from copy import deepcopy
import pytest
from scripts.retrieval_v234_corroborated_population import checked_update
from src.xgb_matcher import retrieval_corroborated_initial_v234 as kernel


def fixture():
    old={str(f):dict(weights=[1.]*11,query_ids=['TRAIN'],alpha=2.,vocabulary={'IMMUTABLE':True}) for f in (2,3,4)}
    new=deepcopy(old)
    for m in new.values():
        m['weights']=[1.5]*11
        m['native_training']=dict(query_ids=['TRAIN'],changed_native_indices=list(range(11)),
            actual_native_kernel_features=True,semantic_owners_relations_xy_fixed=True,
            parameters_selected_on_outer_controls=False,heldout_entity_overlap=0)
    return old,new


def test_only_index_native_weights_changed_fixed_evidence_preserved():
    old,new=fixture();assert checked_update(old,new)==new


@pytest.mark.parametrize('bad',['alpha','vocab','query','entity','negative','nonfinite','partial','approximate','outer_tune','fixed'])
def test_any_other_model_or_protection_change_stops(bad):
    old,new=fixture();m=new['2'];train=m['native_training']
    if bad=='alpha':m['alpha']=3.
    elif bad=='vocab':m['vocabulary']={'FOREIGN':True}
    elif bad=='query':train['query_ids']=['PILOT']
    elif bad=='entity':train['heldout_entity_overlap']=1
    elif bad=='negative':m['weights'][4]=-1.
    elif bad=='nonfinite':m['weights'][0]=float('nan')
    elif bad=='approximate':train['actual_native_kernel_features']=False
    elif bad=='outer_tune':train['parameters_selected_on_outer_controls']=True
    elif bad=='fixed':train['semantic_owners_relations_xy_fixed']=False
    else:new.pop('3')
    with pytest.raises(ValueError):checked_update(old,new)


def test_one_native_search_no_candidate_rescore_or_perquery_training(monkeypatch):
    candidates=[dict(siret='12345678900001')];calls=[]
    def original(*a,**k):calls.append((a,k));return candidates,dict(initial_cap=100,fixed_reservoir=None)
    monkeypatch.setattr(kernel,'original_search',original)
    result,trace=kernel.search('NATIVE',weights=[1.]*11)
    assert result is candidates and len(calls)==1 and not trace['query_time_training']
    assert trace['actual_native_kernel_training_features'] and trace['changed_initial_weight_indices']==list(range(11))
