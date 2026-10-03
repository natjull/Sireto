import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.compile_retrieval_v81_succession_bindings import compile_bindings
from scripts.compile_retrieval_v81_streaming import compile_streaming
from scripts.compile_retrieval_v81_partitioned import compile_partitioned


def fixture(tmp_path,duplicate_site=False):
    a='11111111100001'; b='22222222200001'; c='33333333300001'; d='44444444400001'
    def edge(p,s,continuity=True):
        return dict(siretEtablissementPredecesseur=p,siretEtablissementSuccesseur=s,
            dateLienSuccession='2020-01-01',transfertSiege=False,
            continuiteEconomique=continuity,dateDernierTraitementLienSuccession='2021-01-01')
    edges=[edge(a,b),edge(a,b),edge(a,c),edge(a,d),edge(a,'55555555500001'),
        edge('66666666600001',b),edge('66666666600001','55555555500001'),edge(a,a),
        edge(a,b,False),edge('bad',b),edge(None,b),edge(a,b,None)]
    sites=[]
    for sid,address,state in ((a,'10 RUE A','A'),(b,'10 RUE A','F'),(c,'99 RUE B','A'),
                              (d,None,'F'),('22222222200002','10 RUE A','A')):
        sites.append(dict(siret=sid,siren=sid[:9],administrative_state=state,is_headquarters=False,
            insee='75056',postcode='75001',street_number='10',current_address_normalized=address))
    if duplicate_site: sites.append(dict(sites[0],current_address_normalized='CONFLICT'))
    ep=tmp_path/'source.parquet'; sp=tmp_path/'sites.parquet'
    pq.write_table(pa.Table.from_pylist(edges),ep); pq.write_table(pa.Table.from_pylist(sites),sp)
    return ep,sp


def test_full_source_coverage_closed_sites_no_sibling_no_transitive_alias(tmp_path):
    ep,sp=fixture(tmp_path); c=duckdb.connect(); r=compile_bindings(c,ep,sp,tmp_path); c.close()
    paths=pq.read_table(tmp_path/'succession_bindings.parquet').to_pylist()
    coverage={x['status']:x['observations'] for x in r['coverage']}
    assert r['source_rows']==12 and sum(coverage.values())==12
    assert coverage==dict(BOUND_OFFICIAL_ENDPOINTS=4,BOTH_ENDPOINTS_ABSENT=1,
        PREDECESSOR_ABSENT=1,SUCCESSOR_ABSENT=1,SELF_RELATION=1,
        INVALID_IDENTIFIER=2,NO_EXPLICIT_ECONOMIC_CONTINUITY=2)
    assert len(paths)==4 and {p['successor_siret'] for p in paths}=={'22222222200001','33333333300001','44444444400001'}
    assert all(p['predecessor_siret']=='11111111100001' and p['economic_continuity'] for p in paths)
    assert sum(p['successor_state']=='F' for p in paths)==3
    assert {p['site_binding'] for p in paths}=={'CURRENT_SITE_BINDING_INCOMPLETE',
        'CURRENT_ADDRESS_GEO_CONCORDANT_NOT_HISTORICAL_PROOF','CURRENT_LOCATION_DIFFERS_OR_GEO_DISAGREES'}
    assert all(not p['historical_site_coexistence_proven'] for p in paths)
    assert not r['names_transferred'] and not r['global_relation_deduplication_claimed']


def test_conflicting_official_endpoint_fails_closed(tmp_path):
    ep,sp=fixture(tmp_path,True); c=duckdb.connect()
    with pytest.raises(ValueError,match='identity duplicate'):
        compile_bindings(c,ep,sp,tmp_path)
    c.close()


def test_streaming_plan_is_identical_to_materialized_plan(tmp_path):
    ep,sp=fixture(tmp_path); a=tmp_path/'materialized'; b=tmp_path/'streaming'; a.mkdir(); b.mkdir()
    c=duckdb.connect(); old=compile_bindings(c,ep,sp,a); c.close()
    c=duckdb.connect(); new=compile_streaming(c,ep,sp,b); c.close()
    for name in ('succession_bindings.parquet','coverage.parquet'):
        left=pq.read_table(a/name).to_pylist(); right=pq.read_table(b/name).to_pylist()
        assert sorted(map(str,left))==sorted(map(str,right))
    assert old['coverage']==new['coverage'] and old['site_binding_classes']==new['site_binding_classes']
    assert old['source_rows']==new['source_rows']==12
    assert pq.ParquetFile(b/'all_eligible_endpoint_observations.parquet').metadata.num_rows==7


def test_all_physical_partitions_equal_full_logical_oracle(tmp_path):
    ep,sp=fixture(tmp_path)
    edges=pq.read_table(ep).to_pylist(); sites=pq.read_table(sp).to_pylist()
    for bucket in range(16):
        predecessor=f'777777777{bucket:05d}'; successor=f'888888888{bucket:05d}'
        edges.append(dict(edges[0],siretEtablissementPredecesseur=predecessor,siretEtablissementSuccesseur=successor))
        sites.extend([dict(sites[0],siret=sid,siren=sid[:9]) for sid in (predecessor,successor)])
    pq.write_table(pa.Table.from_pylist(edges),ep); pq.write_table(pa.Table.from_pylist(sites),sp)
    a=tmp_path/'full'; b=tmp_path/'buckets'; a.mkdir(); b.mkdir()
    c=duckdb.connect(); old=compile_bindings(c,ep,sp,a); c.close()
    c=duckdb.connect(); new=compile_partitioned(c,ep,sp,b); c.close()
    for name in ('succession_bindings.parquet','coverage.parquet'):
        assert sorted(map(str,pq.read_table(a/name).to_pylist()))==sorted(map(str,pq.read_table(b/name).to_pylist()))
    parts=list(b.glob('all_endpoint_observations_part*.parquet'))
    assert len(parts)==16 and sum(pq.ParquetFile(p).metadata.num_rows for p in parts)==23
    assert all(pq.ParquetFile(p).metadata.num_rows>=1 for p in parts)
    assert old['coverage']==new['coverage'] and new['source_rows']==28
