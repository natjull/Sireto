import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.project_retrieval_v82_scoped_succession_names import project


def fixture(tmp_path,conflict=False):
    bindings=[dict(predecessor_siret='11111111100001',predecessor_siren='111111111',
        successor_siret='22222222200001',successor_siren='222222222')]*3
    if conflict: bindings.append(dict(bindings[0],predecessor_siren='222222222'))
    names=[]
    def name(subject,parent,site,kind,text):
        return dict(subject_kind=subject,siren=parent,siret=site,name_kind=kind,normalized_value=text,
            raw_value=text,source='SIRENE_CURRENT',evidence_id=f'e{len(names)}',source_record_id=f'r{len(names)}',
            valid_from=None,valid_to=None,observed_at=None,is_current=True,source_priority=400)
    names.append(name('SIREN','111111111','','LEGAL','PARENT A'))
    names.append(name('SIREN','222222222','','TRADE','PARENT B COMMERCIAL'))
    names.append(name('SIRET','111111111','11111111100001','SIGN','SITE A'))
    names.append(name('SIRET','111111111','11111111100002','SIGN','SIBLING MUST NOT LEAK'))
    names.append(name('SIRET','222222222','11111111100001','LEGAL','WRONG PARENT MUST NOT LEAK'))
    names.append(name('SIREN','333333333','','LEGAL','UNRELATED MUST NOT LEAK'))
    names.append(name('SIREN','111111111','','SIGN','SITE SCOPE UNVERIFIED'))
    names.append(name('SIREN','111111111','','LEGAL','PARENT A'))
    names.append(name('SIRET','222222222','22222222200001','USUAL','SITE B'))
    bp=tmp_path/'bindings.parquet'; np=tmp_path/'names.parquet'
    pq.write_table(pa.Table.from_pylist(bindings),bp); pq.write_table(pa.Table.from_pylist(names),np)
    return bp,np


def test_names_are_factorized_scoped_and_preserve_all_observations(tmp_path):
    bp,np=fixture(tmp_path); c=duckdb.connect(); r=project(c,bp,np,tmp_path); c.close()
    names=pq.read_table(tmp_path/'scoped_name_observations.parquet').to_pylist()
    assert r['endpoint_sites']==2 and r['legal_owners']==2 and r['name_observations']==6
    assert len(names)==6 and sum(n['normalized_value']=='PARENT A' for n in names)==2
    assert not any('LEAK' in n['normalized_value'] for n in names)
    assert all(not n['alias_of_related_target'] and not n['name_relation_co_validity_proven'] for n in names)
    context=[n for n in names if n['name_kind']=='SIGN' and n['subject_kind']=='SIREN']
    assert len(context)==1 and context[0]['lexical_scope']=='CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED'
    endpoint=[n for n in names if n['subject_kind']=='SIRET']
    assert all(n['lexical_scope']=='EXPLICIT_ENDPOINT_NAME' for n in endpoint)
    assert len({n['evidence_id'] for n in names})==6
    assert r['repeated_observations_retained'] and not r['relation_aliases_created']


def test_conflicting_endpoint_parent_fails_before_names(tmp_path):
    bp,np=fixture(tmp_path,True); c=duckdb.connect()
    with pytest.raises(ValueError,match='Conflicting official parent'):
        project(c,bp,np,tmp_path)
    c.close()
