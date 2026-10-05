"""Control fixtures only: no national quality conclusion."""
from copy import deepcopy
import pytest
from scripts.retrieval_v301_corroborated_population import checked_update


def fixture():
    old={str(f):dict(weights=[1.]*11,query_ids=['TRAIN'],alpha=2.,vocabulary={'IMMUTABLE':'REFERENCE'}) for f in (2,3,4)}
    new=deepcopy(old)
    for m in new.values():
        m['weights'][4]=1.5
        m['place_training']=dict(query_ids=['TRAIN'],changed_indices=[4],heldout_entity_overlap=0)
    return old,new


def test_only_trained_initial_place_changes():
    old,new=fixture();assert checked_update(old,new)==new


@pytest.mark.parametrize('bad',['weight','vocab','alpha','query','entity','negative','nonfinite','partial'])
def test_any_unrelated_model_or_protection_change_stops(bad):
    old,new=fixture();m=new['2']
    if bad=='weight':m['weights'][0]=3.
    elif bad=='vocab':m['vocabulary']={'FOREIGN':True}
    elif bad=='alpha':m['alpha']=3.
    elif bad=='query':m['place_training']['query_ids']=['PILOT']
    elif bad=='entity':m['place_training']['heldout_entity_overlap']=1
    elif bad=='negative':m['weights'][4]=-1.
    elif bad=='nonfinite':m['weights'][4]=float('nan')
    else:new.pop('3')
    with pytest.raises(ValueError):checked_update(old,new)
