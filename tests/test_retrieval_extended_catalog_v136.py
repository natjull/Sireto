import numpy as np
import pytest

from src.xgb_matcher.retrieval_extended_catalog_v136 import MISSING,build,positions


def test_current_positions_stable_all_official_extras_parent_member_block_exact(tmp_path):
    sites=np.array([100001,300001,500001],dtype=np.uint64);parents=np.array([1,3,5],dtype=np.uint32)
    extras=np.array([100002,200001,200002,900001,900002],dtype=np.uint64)
    s=build(sites,parents,extras,2,2,tmp_path)
    assert s['sites']==8 and s['parents']==5 and s['blocks']==5
    assert s['every_current_site_position_unchanged'] and s['every_extra_member_and_block_csr_exact']
    assert np.load(tmp_path/'extra_parent_keys.npy').tolist()==[2,9]
    assert np.load(tmp_path/'extra_site_parent_positions.npy').tolist()==[0,3,3,4,4]
    assert np.load(tmp_path/'extra_site_blocks.npy').tolist()==[2,2,3,3,4]
    assert np.load(tmp_path/'extra_block_offsets.npy').tolist()==[0,2,4,5]
    assert np.load(tmp_path/'bridge_parent_positions.npy').tolist()==[0,3,4]
    assert np.load(tmp_path/'extra_parent_members.npy').tolist()==[3,4,5,6,7]
    assert np.load(tmp_path/'extra_parent_member_offsets.npy').tolist()==[0,1,3,5]
    assert np.load(tmp_path/'extra_parent_blocks.npy').tolist()==[2,2,3,3,4]
    assert np.load(tmp_path/'extra_parent_block_offsets.npy').tolist()==[0,1,3,5]
    assert positions(sites,extras,np.array([100001,100002,200002,900002,999999],dtype=np.uint64)).tolist()==[0,3,5,7,MISSING]


def test_empty_extension_and_empty_base_have_exact_csr(tmp_path):
    s=build(np.array([],dtype=np.uint64),np.array([],dtype=np.uint32),np.array([],dtype=np.uint64),0,128,tmp_path)
    assert s['sites']==0 and s['extra_blocks']==0
    assert np.load(tmp_path/'extra_parent_member_offsets.npy').tolist()==[0]
    assert np.load(tmp_path/'extra_block_offsets.npy').tolist()==[0]


def test_no_member_or_alias_cap_even_for_large_same_parent(tmp_path):
    extras=np.arange(12345678900001,12345678902002,dtype=np.uint64)
    s=build(np.array([],dtype=np.uint64),np.array([],dtype=np.uint32),extras,0,128,tmp_path)
    assert s['sites']==2001 and s['parents']==1 and s['extra_blocks']==16 and s['source_or_site_or_member_cap'] is None
    assert np.load(tmp_path/'extra_parent_member_offsets.npy').tolist()==[0,2001]


@pytest.mark.parametrize('sites,parents,extras,pattern',[
    ([100001],[1],[100001],'must not duplicate'),
    ([100002,100001],[1],[],'sorted unique'),
    ([100001],[1],[100003,100002],'sorted unique'),
    ([100001],[1],[10**14],'sorted unique'),
])
def test_invalid_or_duplicate_source_stops_not_prunes(tmp_path,sites,parents,extras,pattern):
    with pytest.raises(ValueError,match=pattern):build(np.array(sites,dtype=np.uint64),np.array(parents,dtype=np.uint32),np.array(extras,dtype=np.uint64),1,128,tmp_path)


def test_scalar_and_vector_unsigned_positions_exact():
    current=np.array([2**53+1],dtype=np.uint64);extras=np.array([2**53+3],dtype=np.uint64)
    assert positions(current,extras,np.array([2**53+1,2**53+2,2**53+3],dtype=np.uint64)).tolist()==[0,MISSING,1]
    with pytest.raises(ValueError,match='same unsigned'):positions(current,extras,np.array([1.]))
