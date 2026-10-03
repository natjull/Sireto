"""Lossless raw-source context, never a recall or synthetic-training test."""
import datetime as dt

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_official_context_v110 import (
    DATE_NULL,OfficialContext,build,codebook,coarse_state,encode,join_keys,positions)


def fixture(tmp_path, rows=None, registry_rows=None):
    if rows is None:
        rows=[('00000000100001','000000001','00.98','F',False,None,dt.date(1970,1,1)),
              ('00000000100002','000000001','47.64Z','A',True,dt.date(2020,1,1),None),
              ('00000000200001','000000002','51.3T','F',None,dt.date(1900,1,1),dt.date(1995,3,4)),
              ('00000000300001','000000003',None,None,None,None,None),
              ('00000000400001','000000004','','',False,None,None),
              ('00000000500001','000000005','  aBc  ','OTHER',True,None,None),
              ('00000000600001','000000006','74.04','F',False,None,None)]
    names=['siret','siren','main_activity','administrative_state','is_headquarters','creation_date','current_period_start']
    types=[pa.string()]*4+[pa.bool_(),pa.date32(),pa.date32()]
    table=pa.Table.from_arrays([pa.array([row[i] for row in rows],type=t) for i,t in enumerate(types)],names=names)
    source=tmp_path/'source.parquet';pq.write_table(table,source,row_group_size=2)
    registry=tmp_path/'registry';registry.mkdir()
    ordered=sorted(rows if registry_rows is None else registry_rows,key=lambda r:int(r[0]))
    np.save(registry/'endpoint_keys.npy',np.array([int(r[0]) for r in ordered],dtype=np.uint64))
    np.save(registry/'endpoint_parents.npy',np.array([int(r[1]) for r in ordered],dtype=np.uint32))
    np.save(registry/'endpoint_states.npy',coarse_state(pa.array([r[3] for r in ordered],type=pa.string())).astype(np.uint8))
    output=tmp_path/'out';output.mkdir()
    return rows,table,source,registry,output


def test_lossless_full_source_and_shuffled_alignment(tmp_path):
    rows,table,source,registry,output=fixture(tmp_path)
    # Reorder independently from V96 ordinals and across row groups.
    pq.write_table(table.take(pa.array([5,0,6,1,4,2,3])),source,row_group_size=2)
    stats=build(source,registry,output,batch_size=2)
    context=OfficialContext(output,stats)
    for pos,row in enumerate(sorted(rows,key=lambda r:int(r[0]))):
        actual=context.at(np.uint32(pos))
        assert actual['main_activity']==row[2]
        assert actual['administrative_state']==row[3]
        assert actual['is_headquarters'] is row[4]
        for raw,dest in ((row[5],'creation_days'),(row[6],'current_period_start_days')):
            assert actual[dest]==(None if raw is None else (raw-dt.date(1970,1,1)).days)
    assert sum(stats['counts']['main_activity'])==len(rows)
    assert stats['counts']['main_activity'][:2]==[1,1]
    assert stats['counts']['headquarters']==[2,3,2]
    assert stats['null_and_empty_distinct']
    assert not stats['closed_sites_removed']
    assert not stats['activity_taxonomy_inferred']
    assert not stats['context_serving_integrated']
    assert not stats['identifiers_as_features']
    assert not stats['dates_are_historical_validity']
    assert set(p.name for p in output.iterdir())=={
        'activity_codes.npy','state_codes.npy','headquarters_codes.npy',
        'creation_days.npy','current_period_start_days.npy'}
    assert stats['null_date_sentinel']==DATE_NULL
    assert stats['persisted_context_roundtrip_rows']==len(rows)


def test_raw_enum_no_normalization_or_nomenclature_merge():
    raw=[None,'','A','a',' A ','74.04','74.04Z','00.98','é']
    book=codebook(set(raw))
    coded=encode(pa.array(raw,type=pa.string()),book)
    assert [book[int(x)] for x in coded]==raw
    assert coded.dtype==np.uint16
    assert book[:2]==[None,'']
    assert book==codebook(set(reversed(raw)))
    with pytest.raises(ValueError,match='codebook drift'):
        encode(pa.array(['not present']),book)


def test_codebook_overflow_fails_not_truncates():
    with pytest.raises(ValueError,match='uint16'):
        codebook({str(i) for i in range(65535)})


@pytest.mark.parametrize('value',[None,'0000000010000','000000001000001','0000000010000x',
                                 '００００００００１００００１',' 0000000100001','00000000100001\n'])
def test_invalid_join_keys(value):
    with pytest.raises(ValueError,match='ASCII'):
        join_keys(pa.array([value],type=pa.string()),14,np.uint64)


def test_typed_vector_search_no_whole_registry_conversion(monkeypatch):
    keys=np.array([100001,200001],dtype=np.uint64)
    parents=np.array([1,2],dtype=np.uint32);seen=np.zeros(2,dtype=bool)
    original=np.searchsorted
    def checked(haystack,needle):
        assert haystack is keys
        assert haystack.dtype==needle.dtype==np.uint64
        return original(haystack,needle)
    monkeypatch.setattr(np,'searchsorted',checked)
    assert positions(keys,parents,keys[::-1],parents[::-1],seen).tolist()==[1,0]
    assert seen.all()


@pytest.mark.parametrize('mutation,reason',[
    ('duplicate','Duplicate'),('duplicate_between','Duplicate'),('absent','absent'),
    ('parent','parent'),('state','state differs'),('cardinality','cardinality'),
    ('order','order'),('dtype','dtype')])
def test_corrupt_registry_or_source_stop(tmp_path,mutation,reason):
    rows,table,source,registry,output=fixture(tmp_path)
    if mutation in ('duplicate','duplicate_between'):
        index=[0,0,2,3,4,5,6] if mutation=='duplicate' else [0,1,2,3,4,5,0]
        pq.write_table(table.take(pa.array(index)),source)
    elif mutation=='absent':
        idx=table.schema.get_field_index('siret')
        values=table['siret'].to_pylist();values[-1]='00000009900001'
        table=table.set_column(idx,'siret',pa.array(values));pq.write_table(table,source)
    elif mutation=='parent':
        idx=table.schema.get_field_index('siren')
        values=table['siren'].to_pylist();values[-1]='000000099'
        table=table.set_column(idx,'siren',pa.array(values));pq.write_table(table,source)
    elif mutation=='state':
        idx=table.schema.get_field_index('administrative_state')
        values=table['administrative_state'].to_pylist();values[0]='A'
        table=table.set_column(idx,'administrative_state',pa.array(values));pq.write_table(table,source)
    else:
        path=registry/'endpoint_keys.npy';keys=np.load(path)
        if mutation=='cardinality':keys=keys[:-1]
        elif mutation=='order':keys[[1,2]]=keys[[2,1]]
        elif mutation=='dtype':keys=keys.astype(np.int64)
        np.save(path,keys)
    with pytest.raises(ValueError,match=reason):
        build(source,registry,output,batch_size=2)


@pytest.mark.parametrize('position',[-1,7,True,1.5,'1',None])
def test_reader_invalid_position_fails(tmp_path,position):
    _,_,source,registry,output=fixture(tmp_path)
    stats=build(source,registry,output,batch_size=2)
    with pytest.raises(ValueError,match='position'):
        OfficialContext(output,stats).at(position)


def test_unknown_state_and_null_headquarters_retained(tmp_path):
    _,_,source,registry,output=fixture(tmp_path)
    stats=build(source,registry,output,batch_size=1)
    context=OfficialContext(output,stats)
    assert context.at(2)['is_headquarters'] is None
    assert context.at(3)['administrative_state'] is None
    assert context.at(4)['administrative_state']==''
    assert context.at(5)['administrative_state']=='OTHER'


def test_source_schema_drift_fails(tmp_path):
    _,table,source,registry,output=fixture(tmp_path)
    i=table.schema.get_field_index('is_headquarters')
    table=table.set_column(i,'is_headquarters',pa.array([0]*len(table),type=pa.int8()))
    pq.write_table(table,source)
    with pytest.raises(ValueError,match='schema'):
        build(source,registry,output)
