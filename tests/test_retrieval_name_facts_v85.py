from collections import Counter,defaultdict
from datetime import date

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_name_facts_v85 import FactObservations,SCOPE,factorize,invert


def fixture(tmp_path):
    rows=[]
    def add(owner,parent,subject,scope,text,source='RNE',kind='LEGAL',date_value=None):
        rows.append(dict(siren=parent,siret=owner if subject=='SIRET' else '',subject_kind=subject,
            normalized_value=text,lexical_scope=scope,source=source,name_kind=kind,
            valid_from=date_value,valid_to=None,evidence_id=f'e{len(rows)}',source_record_id=f'r{len(rows)}',
            alias_of_related_target=False,name_relation_co_validity_proven=False))
    add('001234567','001234567','SIREN','LEGAL_PERSON_NAME','PORT DU CENTRE')
    add('001234567','001234567','SIREN','LEGAL_PERSON_NAME','PORT DU CENTRE','BODACC',date_value=date(2000,1,1))
    add('001234567','001234567','SIREN','LEGAL_PERSON_COMMERCIAL_NAME','PORT DU CENTRE',kind='TRADE')
    add('222222222','222222222','SIREN','LEGAL_PERSON_NAME','PORT DU CENTRE')
    add('00123456700001','001234567','SIRET','EXPLICIT_ENDPOINT_NAME','CLINIQUE PORT',kind='SIGN')
    add('00123456700001','001234567','SIRET','EXPLICIT_ENDPOINT_NAME','PHARMACIE CENTRE',kind='SIGN')
    add('00123456700002','001234567','SIRET','EXPLICIT_ENDPOINT_NAME','CLINIQUE PORT',kind='SIGN')
    add('001234567','001234567','SIREN','CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED','NEVER A SITE ALIAS',kind='SIGN')
    add('00123456700001','001234567','SIRET','EXPLICIT_ENDPOINT_NAME','CLINIQUE PORT','SIRENE_CURRENT','USUAL')
    path=tmp_path/'source.parquet'; pq.write_table(pa.Table.from_pylist(rows),path,row_group_size=2)
    return path,rows


def test_full_key_facts_and_all_provenance_ordinals_equal_independent_oracle(tmp_path):
    source,rows=fixture(tmp_path); out=tmp_path/'catalog'; out.mkdir()
    c=duckdb.connect(); c.execute('SET threads=2'); c.execute('SET preserve_insertion_order=false')
    stats=factorize(c,source,out); c.close(); stats['inverse']=invert(out,stats)
    oracle=defaultdict(list)
    for i,row in enumerate(rows):
        key=(1 if row['subject_kind']=='SIREN' else 2,int(row['siren'] if row['subject_kind']=='SIREN' else row['siret']),
            int(row['siren']),SCOPE[row['lexical_scope']],row['normalized_value'])
        oracle[key].append(i)
    actual={}; index=FactObservations(out); mapping=np.load(out/'observation_facts.npy')
    for part in stats['parts']:
        for i,row in enumerate(pq.read_table(part['path']).to_pylist()):
            key=tuple(row[k] for k in ('subject_code','owner_key','official_parent','scope_code','name_text'))
            fact=part['fact_start']+i; obs=index.all(fact).tolist()
            assert np.all(mapping[obs]==fact)
            actual[key]=obs
    assert dict(oracle)==actual and len(actual)==7
    assert stats['observations']==len(rows)==9 and stats['inverse']['observations_indexed']==9
    assert stats['all_observations_mapped_once'] and stats['inverse']['alias_observation_cap'] is None
    assert Counter(x for indices in actual.values() for x in indices)==Counter(range(len(rows)))
    assert not any(key[-1]=='CLINIQUE CENTRE' for key in actual) # No cross-alias word bag.


@pytest.mark.parametrize('field,value',[('official_parent',None),('alias_of_related_target',True),
                                       ('lexical_scope','INVENTED'),('siren','999999999')])
def test_invalid_scope_alias_or_membership_fails_closed(tmp_path,field,value):
    source,rows=fixture(tmp_path)
    if field=='official_parent': rows[0]['siren']=None
    elif field=='siren': rows[4][field]=value
    else: rows[0][field]=value
    pq.write_table(pa.Table.from_pylist(rows),source); out=tmp_path/'catalog'; out.mkdir(); c=duckdb.connect()
    with pytest.raises(ValueError,match='drift'): factorize(c,source,out)
    c.close()
