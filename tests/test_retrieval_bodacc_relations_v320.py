"""Tiny source projection fixtures, not actual national producer authority."""
from datetime import date
import inspect
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts import retrieval_v320_bodacc_relations as source


def row(kind='ASSET_TRANSFER', owner='100000001', target='100000002', origin='BODACC'):
    return dict(relation_id='original-proof',source=origin,source_record_id='source-event',
        relation_type=kind,from_kind='SIREN',from_identifier=owner,to_kind='SIREN',to_identifier=target,
        effective_date=None,observed_at=date(2020,1,2),source_priority=200 if origin=='BODACC' else 500)


def original(path, values):
    pq.write_table(pa.Table.from_pylist(values).select(source.FIELDS),path)
    return path


def test_all_rows_all_provenance_fields_original_ordinal_and_dates(tmp_path):
    values=[row(),row(kind='LEGAL_UNIT_SUCCESSION',origin='SIRENE_SUCCESSION'),row(owner='100000003',target='100000004')]
    p=original(tmp_path/'source.parquet',values)
    selected,stats=source.project(p,lambda:None,expected_rows=3)
    assert [r['ordinal'] for r in selected]==[0,2]
    assert selected[0]['fields']==source.raw(values[0])
    assert selected[0]['fields']['effective_date'] is None
    assert selected[0]['fields']['observed_at']=='2020-01-02'
    assert stats['records']==3 and stats['bodacc_records']==2
    assert sum(x['records'] for x in stats['relation_counts'])==3
    source.roundtrip(p,selected,lambda:None,expected_rows=3)


@pytest.mark.parametrize('change',['ordinal','field','missing','extra'])
def test_no_omitted_or_rewritten_source_relation(tmp_path,change):
    p=original(tmp_path/'source.parquet',[row(),row(owner='100000003',target='100000004')])
    selected,_=source.project(p,lambda:None,expected_rows=2)
    if change=='ordinal':selected[0]['ordinal']=1
    elif change=='field':selected[0]['fields']['source_record_id']='invented'
    elif change=='missing':selected.pop()
    else:selected.append(selected[0])
    with pytest.raises(ValueError):source.roundtrip(p,selected,lambda:None,expected_rows=2)


@pytest.mark.parametrize('change',['identifier','priority','self','kind','date','whitespace'])
def test_invalid_fields_not_silently_normalized_or_repaired(change):
    r=row()
    if change=='identifier':r['from_identifier']='100000001X'
    elif change=='priority':r['source_priority']=300
    elif change=='self':r['to_identifier']=r['from_identifier']
    elif change=='kind':r['from_kind']='SIRET'
    elif change=='date':r['observed_at']='not a date'
    else:r['source_record_id']=' source-event '
    with pytest.raises(ValueError):source.typed(r)


def test_complete_denominator_no_cap_no_query_or_truth(tmp_path):
    p=original(tmp_path/'source.parquet',[row(owner=f'{100000001+i:09d}',target=f'{200000001+i:09d}') for i in range(151)])
    selected,stats=source.project(p,lambda:None,expected_rows=151)
    assert len(selected)==151 and stats['records']==151
    with pytest.raises(ValueError):source.project(p,lambda:None,expected_rows=150)
    code=inspect.getsource(source)
    assert 'human_ground_truth' not in code and 'crm_name' not in code and 'evaluate(' not in code
    assert all(v is False for v in source.FALSE.values())
