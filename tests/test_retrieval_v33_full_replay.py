import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.retrieval_v33_full_replay import ordered_inputs, same, RAW_SCHEMA, PARENTS_SCHEMA, PRESENCE_SCHEMA, JOINT_SCHEMA
from scripts.retrieval_v31_replay_preflight import normalize_input


def row(q='q',fold=2):
    return dict(query_id=q,oof_fold=fold,crm_name='ATELIER',crm_address='12 RUE TEST',postcode='75001',insee='75101')


def test_order_is_population_only_and_stable():
    rows=[row(str(i),2+i%3) for i in range(100)]
    assert ordered_inputs(rows)==ordered_inputs(list(reversed(rows)))
    assert len(ordered_inputs(rows))==100


@pytest.mark.parametrize('bad',[0,1,5,None])
def test_forbidden_folds(bad):
    with pytest.raises(ValueError,match='Forbidden fold'):ordered_inputs([row(fold=bad)])


def test_duplicate_query_rejected():
    with pytest.raises(ValueError,match='Duplicate'):ordered_inputs([row(),row()])


def test_truth_and_identity_not_query_features():
    base=row();other=dict(base,query_id='another',oof_fold=4,ground_truth_siret='12345678900001',cohort_kind='old_failure')
    assert normalize_input(base)==normalize_input(other)


def test_empty_candidate_artifacts_have_explicit_schema():
    for schema in (RAW_SCHEMA,PARENTS_SCHEMA,PRESENCE_SCHEMA,JOINT_SCHEMA):
        t=pa.Table.from_pylist([],schema=schema)
        assert t.num_rows==0 and 'query_id' in t.column_names


def test_comparison_detects_rank_and_duplicate_drift(tmp_path):
    a=tmp_path/'a.parquet';b=tmp_path/'b.parquet'
    pq.write_table(pa.table({'siret':['a','b'],'rank':[1,2]}),a)
    pq.write_table(pa.table({'siret':['b','a'],'rank':[2,1]}),b)
    c=duckdb.connect();same(c,str(a),str(b))
    pq.write_table(pa.table({'siret':['a','b'],'rank':[2,1]}),b)
    with pytest.raises(ValueError,match='Nonreproducible'):same(c,str(a),str(b))
    duplicate=tmp_path/'duplicate.parquet'
    pq.write_table(pa.table({'siret':['a','b','a'],'rank':[1,2,1]}),duplicate)
    with pytest.raises(ValueError,match='Nonreproducible'):same(c,str(a),str(duplicate))


def test_generator_does_not_read_historical_candidates_or_labels():
    import inspect
    from scripts.retrieval_v33_full_replay import shard,legacy,predict,batch
    for fn in (shard,legacy,predict,batch):
        source=inspect.getsource(fn)
        assert 'human_ground_truth' not in source
        assert 'SceneIRCache' not in source
        assert 'raw_union_train234' not in source
    # Inference is guarded by an explicit full-population scope, not by removing
    # the historical predict() target guard.
    assert "Foreign inference target" in inspect.getsource(predict)


def test_hash_edge_selection_equals_or_join_including_duplicates_and_nulls(tmp_path):
    from scripts.retrieval_v33_full_replay import select_edges
    p=tmp_path/'edges.parquet'
    pq.write_table(pa.table({'predecessor_siret':['a','x','a','a',None,'z'],
      'successor_siret':['b','b','b','c','b','z'],'proof':[1,2,1,3,4,5]}),p)
    c=duckdb.connect();c.execute("CREATE TABLE raw AS SELECT 'a' siret,1 union_rank UNION ALL SELECT 'b',250 UNION ALL SELECT 'z',251")
    select_edges(c,[str(p)])
    c.execute('''CREATE TABLE reference AS SELECT DISTINCT e.* FROM read_parquet(?) e SEMI JOIN
      (SELECT siret FROM raw WHERE union_rank<=250) r ON e.predecessor_siret=r.siret OR e.successor_siret=r.siret''',[str(p)])
    assert c.execute('SELECT count(*) FROM ((TABLE edges EXCEPT ALL TABLE reference) UNION ALL (TABLE reference EXCEPT ALL TABLE edges))').fetchone()[0]==0
