import numpy as np
import pytest

from src.xgb_matcher.retrieval_extended_catalog_v136 import positions,MISSING
from src.xgb_matcher.retrieval_append_catalog_v141 import append_build,append_positions


class ToyKeys:
    def __init__(self):
        self.site=np.array([100001,300001,500001],dtype=np.uint64)
        self.extra_site=np.array([100002,200001,200002,900001,900002],dtype=np.uint64)
        self.parent=np.array([1,3,5],dtype=np.uint32);self.extra_parent=np.array([2,9],dtype=np.uint32)
        self.sites=8;self.parents=5
    def site_positions(self,keys):return positions(self.site,self.extra_site,keys)
    def parent_positions(self,keys):return positions(self.parent,self.extra_parent,keys)
    def reconstruct(self,pos,grain):
        return np.concatenate((self.site,self.extra_site) if grain=='site' else (self.parent,self.extra_parent))[pos]


def test_append_never_resorts_old_extras_or_parents_and_full_membership_exact(tmp_path):
    c=ToyKeys();keys=np.array([100003,200009,400001,900003],dtype=np.uint64)
    s=append_build(c,keys,5,2,tmp_path)
    assert (s['sites'],s['parents'],s['blocks'])==(12,6,7)
    assert s['every_previous_site_and_parent_position_unchanged'] and s['every_exported_array_exact']
    assert np.load(tmp_path/'new_parent_keys.npy').tolist()==[4]
    assert np.load(tmp_path/'new_site_parent_positions.npy').tolist()==[0,3,5,4]
    assert np.load(tmp_path/'new_site_blocks.npy').tolist()==[5,5,6,6]
    assert np.load(tmp_path/'new_block_offsets.npy').tolist()==[0,2,4]
    assert np.load(tmp_path/'bridge_parent_positions.npy').tolist()==[0,3,4,5]
    assert np.load(tmp_path/'new_parent_members.npy').tolist()==[8,9,11,10]
    assert np.load(tmp_path/'new_parent_member_offsets.npy').tolist()==[0,1,2,3,4]
    assert np.load(tmp_path/'new_parent_blocks.npy').tolist()==[5,5,6,6]
    assert np.load(tmp_path/'new_parent_block_offsets.npy').tolist()==[0,1,2,3,4]
    allkeys=np.array([900002,100003,200002,400001,500001,777777],dtype=np.uint64)
    assert append_positions(8,keys,allkeys,c.site_positions(allkeys)).tolist()==[7,8,5,10,2,MISSING]


def test_all_members_no_2000_cap_and_empty_extension(tmp_path):
    c=ToyKeys();keys=np.arange(700001,702502,dtype=np.uint64)
    s=append_build(c,keys,5,128,tmp_path)
    assert s['additional_sites']==2501 and s['additional_blocks']==20
    assert np.load(tmp_path/'new_parent_members.npy').tolist()==list(range(8,2509))
    assert s['site_or_parent_or_member_cap'] is None and not s['identifiers_as_features']
    empty=tmp_path/'empty';empty.mkdir();s=append_build(c,np.empty(0,dtype=np.uint64),5,128,empty)
    assert s['sites']==8 and s['parents']==5 and s['blocks']==5
    assert np.load(empty/'new_parent_member_offsets.npy').tolist()==[0]


@pytest.mark.parametrize('additions,pattern',[
    ([100001],'ALL previous'),([100002],'ALL previous'),([100003,100003],'sorted unique'),
    ([400001,100003],'sorted unique'),([10**14],'sorted unique')])
def test_duplicate_or_bad_identity_stops_never_filters(tmp_path,additions,pattern):
    with pytest.raises(ValueError,match=pattern):append_build(ToyKeys(),np.array(additions,dtype=np.uint64),5,128,tmp_path)


def test_unsigned_boundary_validation_without_float_or_missing_wrap():
    values=np.array([10**14-1],dtype=np.uint64);extra=values.copy()
    assert append_positions(8,extra,values,np.array([MISSING],dtype=np.uint32)).tolist()==[8]
    with pytest.raises(ValueError,match='previous positions'):
        append_positions(8,extra,values,np.array([8],dtype=np.uint32))
    with pytest.raises(ValueError,match='previous positions'):
        append_positions(8,extra,values,np.array([1.]))
