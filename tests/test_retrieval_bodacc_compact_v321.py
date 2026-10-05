"""Serialization equality fixtures only, not resource/recall authority."""
from datetime import date
import inspect
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts import retrieval_v320_bodacc_relations as old
from scripts import retrieval_v321_bodacc_relations as new
from test_retrieval_bodacc_relations_v320 import row,original


def test_every_source_field_and_date_null_equal320_projection(tmp_path):
    values=[row(),row(kind='LEGAL_UNIT_SUCCESSION',origin='SIRENE_SUCCESSION'),row(owner='100000003',target='100000004')]
    values[0]['effective_date']=date(2000,2,29)
    p=original(tmp_path/'source.parquet',values); out=tmp_path/'projection.parquet'
    expected,oldstats=old.project(p,lambda:None,expected_rows=3)
    stats=new.project(p,out,lambda:None,expected_rows=3)
    assert stats==oldstats
    got=[dict(ordinal=r['ordinal'],fields=old.raw({k:r[k] for k in old.FIELDS})) for r in new.projected_rows(out)]
    assert got==expected
    assert new.roundtrip(p,out,lambda:None,expected_rows=3)==2
    f=pq.ParquetFile(out)
    assert f.schema_arrow.field('observed_at').type==pa.date32()
    assert f.schema_arrow.field('effective_date').type==pa.date32()
    assert f.schema_arrow.field('ordinal').type==pa.uint64()


@pytest.mark.parametrize('change',['ordinal','field','date','missing','extra'])
def test_every_original_row_roundtrip_required(tmp_path,change):
    p=original(tmp_path/'source.parquet',[row(),row(owner='100000003',target='100000004')]);out=tmp_path/'out.parquet'
    new.project(p,out,lambda:None,expected_rows=2)
    f=pq.ParquetFile(out);values=list(new.projected_rows(out))
    if change=='ordinal':values[0]['ordinal']=1
    elif change=='field':values[0]['relation_id']='different-source-proof'
    elif change=='date':values[0]['observed_at']=None
    elif change=='missing':values.pop()
    else:values.append(values[0])
    pq.write_table(pa.Table.from_pylist(values,schema=f.schema_arrow),out)
    with pytest.raises(ValueError):new.roundtrip(p,out,lambda:None,expected_rows=2)


def test_zero_eligible_bodacc_preserves_full_source_denominator(tmp_path):
    p=original(tmp_path/'source.parquet',[row(kind='LEGAL_UNIT_SUCCESSION',origin='SIRENE_SUCCESSION')]);out=tmp_path/'out.parquet'
    s=new.project(p,out,lambda:None,expected_rows=1)
    assert s['records']==1 and s['bodacc_records']==0
    assert list(new.projected_rows(out))==[]
    assert new.roundtrip(p,out,lambda:None,expected_rows=1)==0


def test_all151_records_not_top100_and_same_input_limits_no_gt(tmp_path):
    p=original(tmp_path/'source.parquet',[row(owner=f'{100000001+i:09d}',target=f'{200000001+i:09d}') for i in range(151)])
    out=tmp_path/'out.parquet';s=new.project(p,out,lambda:None,expected_rows=151)
    assert s['bodacc_records']==151 and len(list(new.projected_rows(out)))==151
    assert new.roundtrip(p,out,lambda:None,expected_rows=151)==151
    with pytest.raises(ValueError):new.roundtrip(p,out,lambda:None,expected_rows=150)
    assert new.INPUT==old.INPUT and new.DOSSIER==old.DOSSIER
    assert new.LIMITS==old.LIMITS and new.MAX_OUTPUT_BYTES==old.MAX_OUTPUT_BYTES
    code=inspect.getsource(new)
    assert 'human_ground_truth' not in code and 'crm_name' not in code and 'evaluate(' not in code


def test_equal_python_values_not_enough_original_arrow_types_required(tmp_path):
    p=original(tmp_path/'source.parquet',[row()]);out=tmp_path/'out.parquet'
    new.project(p,out,lambda:None,expected_rows=1)
    t=pq.read_table(out)
    index=t.schema.get_field_index('source_priority')
    t=t.set_column(index,'source_priority',t['source_priority'].cast(pa.int32()))
    pq.write_table(t,out)
    with pytest.raises(ValueError,match='Arrow field type'):
        new.roundtrip(p,out,lambda:None,expected_rows=1)
