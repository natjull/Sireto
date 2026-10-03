from datetime import date,datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_all_succession_v133 import SOURCE,build

SCHEMA=pa.schema(list(zip(SOURCE,[pa.string(),pa.string(),pa.date32(),pa.bool_(),pa.bool_(),pa.timestamp('us')])))


def fixture(tmp_path,rows):
    p=tmp_path/'source.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),p,row_group_size=2)
    out=tmp_path/'mapped';out.mkdir();return p,out


def row(pred,succ,continuity=None,transfer=None,day=None,updated=None):
    return dict(zip(SOURCE,(pred,succ,day,transfer,continuity,updated)))


def test_every_raw_observation_orientation_null_and_self_context_retained(tmp_path):
    rows=[row('00000000100001','00000000200002',True,False,date(2000,1,1),datetime(2024,3,1)),
        row('00000000200002','00000000100003',False,True),
        row('00000000100001','00000000100001'),
        row('00000000100001','00000000200002',True,False,date(1999,1,1))]
    source,out=fixture(tmp_path,rows);s=build(source,out,4,batch_rows=1)
    assert s['observations']==4 and s['oriented_incidences']==8 and s['self_context_links']==1
    assert s['continuity_counts']=={'0':1,'1':1,'2':2}
    assert s['every_original_field_reconstructed_exactly'] and s['every_oriented_incidence_once_and_exact']
    assert s['continuity_filtered'] is False and s['observations_deduplicated'] is False and s['self_links_expanded'] is False
    keys=np.load(out/'endpoint_keys.npy');offsets=np.load(out/'endpoint_offsets.npy');inc=np.load(out/'endpoint_incidences.npy')
    assert keys.tolist()==[100001,100003,200002] and offsets[-1]==8 and sorted(inc.tolist())==list(range(8))
    assert np.load(out/'parent_keys.npy').tolist()==[1,2]
    assert np.load(out/'parent_endpoint_offsets.npy').tolist()==[0,2,3]


def test_popular_endpoint_has_no_cap(tmp_path):
    rows=[row('12345678900001',f'{i:09d}00001') for i in range(1,1002)]
    source,out=fixture(tmp_path,rows);s=build(source,out,len(rows),batch_rows=71)
    assert s['max_endpoint_incidences']==1001 and s['endpoint_or_relation_cap'] is None


@pytest.mark.parametrize('value',[None,'123','１２３４５６７８９００００１','1234567890000X'])
def test_invalid_original_endpoint_stops_not_filters(tmp_path,value):
    source,out=fixture(tmp_path,[row(value,'12345678900001')])
    with pytest.raises(ValueError,match='ASCII14'):build(source,out,1)


def test_wrong_whole_source_count_rejected(tmp_path):
    source,out=fixture(tmp_path,[row('12345678900001','12345678900002')])
    with pytest.raises(ValueError,match='Every original'):build(source,out,2)
