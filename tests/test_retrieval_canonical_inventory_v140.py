import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_canonical_inventory_v140 import classify,inventory
from test_retrieval_proof_bridge_v137 import keys


def rows():
    return dict(subject_kind=['SIRET','SIRET','SIRET','SIRET','SIREN','SIRET',None,'SIRET'],
        siren=['000000001','000000002','000000002','000000001','000000009','000000002',None,'000000003'],
        siret=['00000000100001','00000000200001','00000000200009',' 00000000100003 ','00000000900009','00000000300001',None,'００００００００３００００１'])


def table(data):return pa.Table.from_pydict(data,schema=pa.schema([(k,pa.string()) for k in data]))


def test_explicit_site_grain_only_parent_addresses_and_malformed_context_not_sites():
    t=table(rows());a,parent,context=classify(t.to_batches()[0])
    assert a.tolist()==[100001,200001,200009,100003] and parent==1 and context==3


def test_whole_source_including_masked_or_no_text_rows_and_duplicates(tmp_path):
    t=table(rows()).append_column('raw_value',pa.array(['[ND]']*8))
    p=tmp_path/'source.parquet';pq.write_table(t,p,row_group_size=2)
    result=inventory(p,keys(),tmp_path,8,batch_rows=3)
    assert result['rows']==8 and result['explicit_site_observations']==4
    assert result['present_site_observations']==2 and result['absent_site_observations']==2
    assert result['unique_absent_sites']==2 and not result['lexical_mask_or_family_filter']
    assert np.load(tmp_path/'absent_site_keys.npy').tolist()==[100003,200009]
    assert not result['candidate_catalog_changed'] and not result['human_denominator_filtered']


def test_all_observations_counted_but_absent_identity_dedup_complete(tmp_path):
    raw=rows();one={k:[v[2]]*2003 for k,v in raw.items()};p=tmp_path/'source.parquet';pq.write_table(table(one),p)
    s=inventory(p,keys(),tmp_path,2003,batch_rows=17)
    assert s['absent_site_observations']==2003 and s['unique_absent_sites']==1 and s['identity_or_observation_cap'] is None


def test_no_site_identity_in_parent_or_context_only_file(tmp_path):
    raw=dict(subject_kind=['SIREN',None],siren=['000000001',None],siret=['00000000100003',None])
    p=tmp_path/'source.parquet';pq.write_table(table(raw),p)
    s=inventory(p,keys(),tmp_path,2)
    assert s['explicit_site_observations']==0 and s['unique_absent_sites']==0 and np.load(tmp_path/'absent_site_keys.npy').dtype==np.uint64


def test_incomplete_source_or_bad_grain_schema_stops(tmp_path):
    p=tmp_path/'source.parquet';pq.write_table(table(rows()),p)
    with pytest.raises(ValueError,match='Whole'):inventory(p,keys(),tmp_path,9)
    b=pa.RecordBatch.from_arrays([pa.array(['SIRET']),pa.array([1]),pa.array(['00000000100001'])],names=['subject_kind','siren','siret'])
    with pytest.raises(ValueError,match='grain schema'):classify(b)
