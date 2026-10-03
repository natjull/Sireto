import hashlib

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.audit_retrieval_v80_succession_site import stats


def test_real_sql_pair_sample_and_current_scope_not_temporal_identity(tmp_path):
    edges=[]; sites=[]
    for i in range(70):
        p=f'{100000000+i:09d}00001'; s=f'{200000000+i:09d}00001'
        edges.append(dict(siretEtablissementPredecesseur=p,siretEtablissementSuccesseur=s,
            dateLienSuccession='2020-01-01',dateDernierTraitementLienSuccession='2021-01-01',
            continuiteEconomique=True,transfertSiege=False))
        for sid,parent in ((p,p[:9]),(s,s[:9])):
            sites.append(dict(siret=sid,siren=parent,insee='75056',postcode='75001',
                current_address_normalized='10 RUE TEST',administrative_state='F'))
    # Repeated source dates do not create another sampled endpoint pair.
    edges.append(dict(edges[0],dateLienSuccession='2022-01-01'))
    edges.append(dict(edges[1],siretEtablissementPredecesseur='bad',continuiteEconomique=None))
    ep=tmp_path/'edges.parquet'; sp=tmp_path/'sites.parquet'
    pq.write_table(pa.Table.from_pylist(edges),ep); pq.write_table(pa.Table.from_pylist(sites),sp)
    c=duckdb.connect(); r=stats(c,ep,sp,64)
    expected=sorted({(e['siretEtablissementPredecesseur'],e['siretEtablissementSuccesseur'])
        for e in edges if e['continuiteEconomique'] is True},
        key=lambda pair:(hashlib.sha256(('|'.join(pair)).encode()).hexdigest(),pair))[:64]
    actual=c.execute('SELECT predecessor,successor FROM sample_keys').fetchall()
    assert set(actual)==set(expected)
    assert r['raw_source']['raw_rows']==72 and r['raw_source']['invalid_identifier_rows']==1
    assert r['sample_current_binding']['sampled_pairs']==64
    assert r['sample_current_binding']['current_address_geo_compatible_pairs']==64
    assert r['sample_current_binding']['cross_parent_pairs']==64
    assert not r['historical_site_coexistence_proven'] and r['sample_is_not_search_corpus']
    assert r['sample_current_binding']['sampled_source_rows']==64+int((edges[0]['siretEtablissementPredecesseur'],edges[0]['siretEtablissementSuccesseur']) in expected)
    c.close()


def test_diagnostic_cannot_reduce_fixed_population():
    with pytest.raises(ValueError,match='scope drift'):
        stats(duckdb.connect(),'not_opened','not_opened',32)
