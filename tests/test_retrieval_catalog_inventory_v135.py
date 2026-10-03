import numpy as np
import pytest

from src.xgb_matcher.retrieval_catalog_inventory_v135 import inventory,present


def test_all_source_references_missing_union_and_overlap(tmp_path):
    keys=np.array([0,2,4,8],dtype=np.uint64)
    streams=[('history',np.array([2,3,3,9],dtype=np.uint64)),('history',np.array([0,9],dtype=np.uint64)),
        ('graph',np.array([3,4,5],dtype=np.uint64)),('owners',np.array([8],dtype=np.uint64))]
    s=inventory(keys,streams,tmp_path)
    assert s['sources']['history']==dict(references=6,present_references=2,absent_references=4,unique_absent_siret=2)
    assert s['union_unique_absent_siret']==3 and np.load(tmp_path/'all_absent_siret.npy').tolist()==[3,5,9]
    assert np.load(tmp_path/'owners_absent.npy').tolist()==[]
    assert not s['candidate_catalog_changed'] and not s['human_denominator_filtered'] and not s['recall_measured']


def test_unsigned_keys_exact_and_out_of_range_safe():
    base=2**53+1;keys=np.array([base,base+2],dtype=np.uint64)
    assert present(keys,np.array([base-1,base,base+1,base+2,base+3],dtype=np.uint64)).tolist()==[False,True,False,True,False]
    assert not np.any(present(np.array([],dtype=np.uint64),np.array([0,2],dtype=np.uint64)))


def test_not_float_or_unsorted_identifiers(tmp_path):
    with pytest.raises(ValueError,match='Typed unsigned'):present(np.array([1],dtype=np.uint64),np.array([1.]))
    with pytest.raises(ValueError,match='sorted unique'):inventory(np.array([2,1],dtype=np.uint64),[],tmp_path)
