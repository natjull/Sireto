from copy import deepcopy
import pytest
from scripts.retrieval_v254_corroborated_population import checked_update
from src.xgb_matcher.retrieval_crossfield_proofs_v250 import FEATURES


def fixture():
    old={str(f):dict(weights=[1.]*11,query_ids=['TRAIN'],alpha=2.,vocabulary={'IMMUTABLE':'REFERENCE'}) for f in (2,3,4)}
    updated=deepcopy(old)
    for m in updated.values():
        m['weights'][4]=1.5;m['place_training']=dict(query_ids=['TRAIN'],changed_indices=[4],heldout_entity_overlap=0)
        m['extra_weights']=[.2,.3,.4];m['extra_proof_training']=dict(query_ids=['TRAIN'],features=list(FEATURES),
            heldout_entity_overlap=0,original11_fixed=True,conditional_native_parity_verified=True,parameters_selected_on_outer_controls=False)
    return old,updated


def test_original11_except_prior_place_unchanged_and_new3_only():
    old,updated=fixture();assert checked_update(old,updated)==updated


@pytest.mark.parametrize('bad',['original_weight','alpha','vocab','extra_width','extra_nonfinite','extra_negative','features','query','held','prior_held','parity','selection','fixed','partial'])
def test_any_unrelated_change_or_wrong_protection_stops(bad):
    old,updated=fixture();m=updated['2'];t=m['extra_proof_training']
    if bad=='original_weight':m['weights'][0]=2.
    elif bad=='alpha':m['alpha']=3.
    elif bad=='vocab':m['vocabulary']={}
    elif bad=='extra_width':m['extra_weights']=[1.]
    elif bad=='extra_nonfinite':m['extra_weights'][0]=float('nan')
    elif bad=='extra_negative':m['extra_weights'][0]=-1.
    elif bad=='features':t['features']=['FOREIGN']*3
    elif bad=='query':t['query_ids']=['PILOT']
    elif bad=='held':t['heldout_entity_overlap']=1
    elif bad=='prior_held':m['place_training']['heldout_entity_overlap']=1
    elif bad=='parity':t['conditional_native_parity_verified']=False
    elif bad=='selection':t['parameters_selected_on_outer_controls']=True
    elif bad=='fixed':t['original11_fixed']=False
    else:updated.pop('4')
    with pytest.raises(ValueError):checked_update(old,updated)
