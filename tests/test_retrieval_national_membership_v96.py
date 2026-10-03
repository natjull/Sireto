from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_national_membership_v96 import build,NationalMembership


def write_source(path,rows):
    pq.write_table(pa.Table.from_pylist(rows,schema=pa.schema([
        ('site_key',pa.uint64()),('parent_key',pa.uint32()),('administrative_state',pa.string())])),path)


def test_all_members_states_original_ordinals_and_uncapped_fanout_match_independent_registry(tmp_path,monkeypatch):
    rows=[dict(site_key=123456700001+i,parent_key=1234567,administrative_state='F' if i%2 else 'A') for i in range(2501)]
    rows.extend([dict(site_key=22222222200001,parent_key=222222222,administrative_state=None),
                 dict(site_key=33333333300001,parent_key=333333333,administrative_state='OTHER')])
    np.random.default_rng(96).shuffle(rows)
    path=tmp_path/'source.parquet'; write_source(path,rows)
    stats=build(path,tmp_path); assert stats['sites']==2503 and stats['max_sites_per_parent']==2501
    registry=NationalMembership(tmp_path); expected=defaultdict(list)
    sorted_rows=sorted(enumerate(rows),key=lambda pair:pair[1]['site_key'])
    for pos,(ordinal,row) in enumerate(sorted_rows):
        assert registry.endpoint_position(f'{row["site_key"]:014d}')==pos
        assert registry.a['endpoint_parents'][pos]==row['parent_key']
        assert registry.a['endpoint_source_observations'][pos]==ordinal
        expected_state=0 if row['administrative_state'] is None else 1 if row['administrative_state']=='A' else 2 if row['administrative_state']=='F' else 3
        assert registry.a['endpoint_states'][pos]==expected_state
        expected[row['parent_key']].append(pos)
    original=np.searchsorted; seen=[]
    def spy(keys,value,*args,**kwargs):
        assert isinstance(value,np.generic) and value.dtype==keys.dtype; seen.append(str(value.dtype))
        return original(keys,value,*args,**kwargs)
    monkeypatch.setattr(np,'searchsorted',spy)
    for parent,positions in expected.items(): assert registry.parent_endpoint_positions(f'{parent:09d}').tolist()==positions
    assert registry.parent_endpoint_positions('999999999').tolist()==[]
    assert registry.endpoint_position('00123456799999') is None
    assert set(seen)=={'uint64','uint32'}
    with pytest.raises(ValueError,match='Invalid site'): registry.endpoint_position('123')
    with pytest.raises(ValueError,match='Invalid parent'): registry.parent_endpoint_positions('ABC')


@pytest.mark.parametrize('failure',['duplicate','parent'])
def test_unverified_membership_or_duplicate_cannot_pass(tmp_path,failure):
    rows=[dict(site_key=11111111100001,parent_key=111111111,administrative_state='A')]
    if failure=='duplicate': rows*=2
    else: rows[0]['parent_key']=222222222
    path=tmp_path/'source.parquet'; write_source(path,rows)
    with pytest.raises(ValueError): build(path,tmp_path)
