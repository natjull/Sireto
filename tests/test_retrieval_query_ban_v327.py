"""Tiny synthetic BAN fixture only; no real source/run/retrieval/GT."""
from copy import deepcopy

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher import retrieval_query_ban_v327 as work


def queries():
    return [dict(query_id='Q'+str(i),fold=2+i,crm_name='ACME',crm_number='1' if i!=1 else '',
        crm_suffix='',crm_street='RUE TEST',crm_postcode='75001',crm_insee='75056' if i!=2 else '97101') for i in range(3)]


def test_exact_frozen_serving_surface_and_empty_number_not_repaired(tmp_path):
    qs=queries();old=deepcopy(qs)
    assert work.surface(qs).to_pylist()==[dict(query_id=q['query_id'],fold=q['fold'],crm_number=q['crm_number'],
        crm_street_norm='RUE TEST',crm_postcode=q['crm_postcode'],crm_insee=q['crm_insee']) for q in qs]
    path=tmp_path/'ban.parquet'
    pq.write_table(pa.Table.from_pylist([dict(numero='1',longitude=2.35,latitude=48.85,nom_voie='RUE TEST',
        code_postal='75001',code_commune='75056',type_position='bâtiment',source_position='commune',certification_commune=True)]),path)
    class Files:
        def file(self,ref):assert ref==work.BAN;return path
    with duckdb.connect() as c:
        c.execute('SET threads=1')
        result=work.resolve(c,qs,Files())
    assert len(result)==3 and result[0]['ban_resolved'] and result[0]['spatial_crs_compatible']
    assert not result[1]['ban_resolved'] and result[1]['ban_x_l93'] is None and not result[1]['spatial_crs_compatible']
    assert result[2]['ban_resolved'] and not result[2]['spatial_crs_compatible']
    assert qs==old


@pytest.mark.parametrize('bad',['fold0','fold1','duplicate','label','numeric','empty'])
def test_no_closed_fold_alias_target_or_missing_denominator(bad):
    qs=queries()
    if bad=='fold0':qs[0]['fold']=0
    elif bad=='fold1':qs[0]['fold']=1
    elif bad=='duplicate':qs[-1]=deepcopy(qs[0])
    elif bad=='label':qs[0]['human_ground_truth_siret']='10000000000001'
    elif bad=='numeric':qs[0]['crm_number']=1
    else:qs=[]
    with pytest.raises(ValueError):work.surface(qs)


def test_misaligned_geometry_rejected():
    with pytest.raises(ValueError):work.compatible(queries()[0],dict(query_id='WRONG',fold=2))
