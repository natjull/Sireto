import duckdb
import numpy as np
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_v20_workbench import negative_positions,inner_validation,choose_master
from scripts.run_retrieval_v20_master import build_tables


def test_sampler_equal_budget_covers_tail_deterministically():
    for n in (1,180,512,513,2000,5713):
        a=negative_positions(n,'HISTORICAL_HEAD512');b=negative_positions(n,'HEAD256_PLUS_STRATIFIED_TAIL256')
        assert len(a)==len(b)==min(n,512)
        assert len(b)==len(set(b)) and np.all(b>=0) and np.all(b<n)
        assert np.array_equal(b,negative_positions(n,'HEAD256_PLUS_STRATIFIED_TAIL256'))
        if n>512:
            assert np.array_equal(b[:256],np.arange(256))
            assert b[-1]>=n-max(2,n//256)


def test_inner_split_is_parent_grouped_and_order_independent():
    parents=['123456789','987654321','123456789']
    assert inner_validation(parents[0])==inner_validation(parents[2])
    assert list(map(inner_validation,reversed(parents)))==list(reversed(list(map(inner_validation,parents))))


def test_master_never_displaces_baseline_or_breaks_cap():
    rows=[dict(siret=f'{i:014}',base_rank=i+1 if i<2000 else None,rescue_score=float(i),direct_rrf=0.,parent_rrf=0.) for i in range(2600)]
    selected=choose_master(rows)
    assert len(selected)==len({r['siret'] for r in selected})==2000
    assert {r['siret'] for r in rows[:1750]}<={r['siret'] for r in selected}
    assert selected[-1]['siret']==rows[2350]['siret']
    with pytest.raises(ValueError,match='Duplicate'):choose_master(rows+[rows[0]])


def sql_fixture(v17_rank=1):
    c=duckdb.connect()
    c.register('q',pa.Table.from_pylist([dict(query_id='q',fold=2,crm_insee='11111',crm_postcode='11100',crm_number='1',crm_suffix='',crm_street='RUE A')]))
    c.register('raw',pa.Table.from_pylist([dict(query_id='q',fold=2,siren='100000001',siret='10000000100001',base_rank=1)]))
    c.register('signals',pa.Table.from_pylist([
        dict(query_id='q',fold=2,siren='100000001',source_siret='10000000100001',channel='legacy_site_pool',channel_rank=1),
        dict(query_id='q',fold=2,siren='200000001',source_siret='20000000100001',channel='siret_site_name_word',channel_rank=2),
        dict(query_id='q',fold=2,siren='200000001',source_siret='',channel='v17_union',channel_rank=v17_rank)]))
    c.register('sites',pa.Table.from_pylist([
        dict(siret='10000000100001',siren='100000001',insee='11111',postcode='11100',street_number='1',street_number_suffix='',street_normalized='RUE A',administrative_state='A'),
        dict(siret='20000000100001',siren='200000001',insee='99999',postcode='99900',street_number='99',street_number_suffix='',street_normalized='OTHER',administrative_state='F'),
        dict(siret='20000000100002',siren='200000001',insee='11111',postcode='11100',street_number='1',street_number_suffix='',street_normalized='RUE A',administrative_state='A')]))
    return c


def test_sql_preserves_direct_witness_outside_geography_and_adds_sibling():
    c=sql_fixture();build_tables(c)
    rows=c.execute('SELECT siret,selection_reason FROM master ORDER BY candidate_rank').fetchall()
    assert rows[0]==('10000000100001','PRESERVED_BASE1750')
    assert {r[0] for r in rows}=={'10000000100001','20000000100001','20000000100002'}
    assert c.execute("SELECT count(*) FROM witnesses WHERE siret='20000000100001'").fetchone()[0]==1
    assert c.execute("SELECT direct_rrf FROM candidates WHERE siret='20000000100002'").fetchone()[0]==0
    c.close()


def test_artificial_v17_rank_cannot_change_any_new_score():
    snapshots=[]
    for rank in (1,9000):
        c=sql_fixture(rank);build_tables(c)
        snapshots.append(c.execute('SELECT siret,rescue_score,parent_rrf,direct_rrf,candidate_rank FROM master ORDER BY siret').fetchall())
        c.close()
    assert snapshots[0]==snapshots[1]
