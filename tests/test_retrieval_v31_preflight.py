from dataclasses import asdict

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.retrieval_v31_replay_preflight import input_rows, normalize_input, normalized_rows, SAFE
from scripts.retrieval_v31_channel_preflight import compare


def test_projection_excludes_truth_and_forbidden_folds(tmp_path):
    rows=[dict(query_id=str(f),oof_fold=f,crm_name='ACME',crm_address='12 RUE TEST',
      postcode='75001',insee='75101',human_ground_truth_siret='12345678901234') for f in (0,1,2,3,4)]
    path=tmp_path/'input.parquet';pq.write_table(pa.Table.from_pylist(rows),path)
    actual=input_rows(path)
    assert [r['oof_fold'] for r in actual]==[2,3,4]
    assert all(set(r)==set(SAFE) for r in actual)


def test_truth_candidate_and_identity_fields_cannot_affect_query():
    base=dict(crm_name='ACME',crm_address='12 RUE TEST',postcode='75001',insee='75101')
    polluted=dict(base,query_id='other',oof_fold=4,number='999',ground_truth_siret='12345678901234',
      candidate_postcode='69001',candidate_name='FORBIDDEN',acceptable_sirets_operational=['12345678901234'])
    assert normalize_input(base)==normalize_input(polluted)
    assert normalize_input(base).number=='12'


def test_empty_query_is_not_dropped():
    result=normalized_rows([dict(query_id='q',oof_fold=2,crm_name='',crm_address='',postcode='',insee='')])
    assert len(result)==1 and result[0]['query_id']=='q'
    assert asdict(normalize_input({}))==dict(name='',address='',number='',postcode='',insee='')


def test_comparison_detects_membership_and_rank_change(tmp_path):
    import duckdb
    c=duckdb.connect();c.register('chosen',pa.table({'query_id':['q'],'fold':[2]}))
    a=tmp_path/'a.parquet';b=tmp_path/'b.parquet'
    pq.write_table(pa.table({'query_id':['q'],'fold':[2],'siret':['A'],'rank':[1]}),a)
    pq.write_table(pa.table({'query_id':['q'],'fold':[2],'siret':['A'],'rank':[2]}),b)
    assert compare(c,a,b,['query_id','fold','siret'])['pass_parity']
    assert not compare(c,a,b,['query_id','fold','siret','rank'])['pass_parity']
    c.close()
