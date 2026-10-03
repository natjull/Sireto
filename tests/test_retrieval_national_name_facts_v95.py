from collections import defaultdict

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.project_retrieval_v94_virtual_national_names import projection_sql
from src.xgb_matcher.retrieval_national_name_facts_v95 import partition,factor_bucket,invert,FIELDS
from src.xgb_matcher.retrieval_name_facts_v85 import FactObservations


def test_every_original_tuple_and_provenance_including_null_empty_context_closed_scope(tmp_path):
    rows=[]
    def add(subject,parent,site,kind,text):
        rows.append(dict(subject_kind=subject,siren=parent,siret=site,name_kind=kind,normalized_value=text,
            source='BODACC',raw_value=text,evidence_id=str(len(rows)),valid_from=None,valid_to=None))
    add('SIREN','111111111','','LEGAL','PARENT A'); add('SIREN','111111111','','LEGAL','PARENT A')
    add('SIREN','111111111','','SIGN','SITE SCOPE UNKNOWN')
    add('SIRET','111111111','11111111100002','SIGN','SIBLING OWN NAME')
    add('SIRET','222222222','22222222200001','SIGN','')
    add('SIRET','222222222','22222222200001','SIGN',None)
    add('SIREN','333333333','','LEGAL','UNRELATED')
    source=tmp_path/'source.parquet'; pq.write_table(pa.Table.from_pylist(rows),source)
    c=duckdb.connect(); stats=partition(c,projection_sql(source),tmp_path,len(rows)); c.close()
    parts=[]; offset=0
    for bucket in range(16):
        c=duckdb.connect(); part=factor_bucket(c,tmp_path,bucket,offset,len(rows),stats['bucket_observations'][str(bucket)]); c.close()
        if part['facts']: parts.append(part); offset+=part['facts']
    result=invert(tmp_path,parts,len(rows)); assert result['observations_indexed']==7
    mapping=np.load(tmp_path/'observation_facts.npy'); assert mapping[0]==mapping[1] and mapping[4]!=mapping[5]
    facts=[r for p in parts for r in pq.read_table(p['path']).to_pylist()]
    assert len(facts)==6
    expected=defaultdict(list)
    for i,r in enumerate(rows):
        subject=1 if r['subject_kind']=='SIREN' else 2
        scope=1 if r['name_kind']=='LEGAL' else (3 if subject==2 and r['normalized_value'] else 0)
        key=(subject,int(r['siren'] if subject==1 else r['siret']),int(r['siren']),scope,r['normalized_value'])
        expected[key].append(i)
    inverse=FactObservations(tmp_path)
    for i,fact in enumerate(facts):
        key=tuple(fact[k] for k in FIELDS)
        assert inverse.all(i).tolist()==expected[key]
        assert all(mapping[j]==i for j in inverse.all(i))
    assert sum(len(inverse.all(i)) for i in range(len(facts)))==len(rows)
    with pytest.raises(ValueError,match='Nonunique'):
        c=duckdb.connect(); first=parts[0]; factor_bucket(c,tmp_path,first['bucket'],first['fact_start'],len(rows),first['observations'])
    c.close()


def test_invalid_parent_does_not_become_lexical_fact(tmp_path):
    rows=[dict(subject_kind='SIRET',siren='111111111',siret='22222222200001',name_kind='SIGN',normalized_value='BAD')]
    source=tmp_path/'source.parquet'; pq.write_table(pa.Table.from_pylist(rows),source)
    c=duckdb.connect()
    with pytest.raises(ValueError,match='grain/scope'): partition(c,projection_sql(source),tmp_path,1)
    c.close()
