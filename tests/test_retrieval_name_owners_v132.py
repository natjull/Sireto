import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_name_owners_v132 import SCHEMA,FIELDS,build


def fixture(tmp_path,rows):
    source=tmp_path/'facts.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),source,row_group_size=1)
    output=tmp_path/'mapped';output.mkdir();return source,output


def row(fact,name,parent,owner,subject,scope,current=False):
    return dict(fact_id=fact,name_id=name,parent_key=parent,owner_key=owner,subject_code=subject,scope_code=scope,current_exact=current)


def test_all_names_owners_scopes_and_current_retained(tmp_path):
    rows=[row(0,0,0,0,1,1),row(1,0,1,1,1,2,True),row(2,0,1,100001,2,3),
        row(3,1,123456789,123456789,1,1),row(4,1,123456789,12345678900001,2,3,True)]
    source,output=fixture(tmp_path,rows);stats=build(source,output,2,5,batch_rows=2)
    assert stats['every_original_fact_reconstructed_exactly'] and stats['owners_outside_candidate_catalog_preserved']
    assert stats['subject_counts']=={'1':3,'2':2} and stats['current_exact_true']==2
    offsets=np.load(output/'name_fact_offsets.npy');assert offsets.tolist()==[0,3,5]
    for key,dtype in FIELDS.items():
        values=np.load(output/(key+'.npy'),mmap_mode='r');assert values.dtype==dtype and values.tolist()==[r[key] for r in rows]


def test_uncapped_popular_name_and_single_row_batches(tmp_path):
    rows=[row(i,0,i,i,1,1) for i in range(10001)]
    source,output=fixture(tmp_path,rows);stats=build(source,output,1,len(rows),batch_rows=257)
    assert stats['facts']==10001 and stats['owner_or_name_or_period_cap'] is None
    assert np.load(output/'name_fact_offsets.npy').tolist()==[0,10001]


def test_empty_source_is_exact_empty_csr(tmp_path):
    source,output=fixture(tmp_path,[]);stats=build(source,output,0,0)
    assert stats['facts']==0 and np.load(output/'name_fact_offsets.npy').tolist()==[0]


@pytest.mark.parametrize('rows,names,facts,pattern',[
    ([row(1,0,1,1,1,1)],1,1,'contiguous'),
    ([row(0,1,1,1,1,1)],1,1,'position'),
    ([row(0,0,1,1,1,1)],2,1,'Every raw name'),
    ([row(0,0,1,1,1,1)],1,2,'Whole typed'),
    ([row(0,0,1,1,1,3)],1,1,'owner/scope'),
    ([row(0,0,1,200001,2,3)],1,1,'owner/scope'),
    ([row(0,0,2,2,1,1),row(1,0,1,1,1,1)],1,2,'sorted order'),
    ([row(0,0,1,1,1,1),row(1,0,1,1,1,1)],1,2,'sorted order'),
])
def test_drift_and_invalid_full_source_rejected(tmp_path,rows,names,facts,pattern):
    source,output=fixture(tmp_path,rows)
    with pytest.raises(ValueError,match=pattern):build(source,output,names,facts)


def test_original_order_boundary_checked(tmp_path):
    source,output=fixture(tmp_path,[row(0,0,2,2,1,1),row(1,0,1,1,1,1)])
    with pytest.raises(ValueError,match='across batches'):build(source,output,1,2,batch_rows=1)


def test_null_original_owner_rejected(tmp_path):
    r=row(0,0,1,1,1,1);r['owner_key']=None;source,output=fixture(tmp_path,[r])
    with pytest.raises(ValueError,match='present'):build(source,output,1,1)
