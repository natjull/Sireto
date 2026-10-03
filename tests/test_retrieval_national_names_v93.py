import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.project_retrieval_v93_national_names import project
from tests.test_retrieval_scoped_succession_names_v82 import fixture


def test_complete_names_and_sites_preserve_unrelated_sibling_closed_context_and_invalid(tmp_path):
    _,names=fixture(tmp_path)
    rows=pq.read_table(names).to_pylist()
    rows.append(dict(rows[0],normalized_value='',raw_value='',evidence_id='empty'))
    pq.write_table(pa.Table.from_pylist(rows),names)
    sites=[dict(siret='11111111100001',siren='111111111',administrative_state='A'),
           dict(siret='11111111100002',siren='111111111',administrative_state='F'),
           dict(siret='33333333300001',siren='333333333',administrative_state=None)]
    sp=tmp_path/'sites.parquet'; pq.write_table(pa.Table.from_pylist(sites),sp)
    output=tmp_path/'out'; output.mkdir(); c=duckdb.connect()
    stats=project(c,names,sp,output); c.close()
    observed=pq.read_table(output/'scoped_name_observations.parquet').to_pylist()
    assert stats['sites']==3 and stats['name_observations']==len(rows)==10
    assert sorted(x['source_observation_index'] for x in observed)==list(range(10))
    for i,x in enumerate(observed):
        for key,value in rows[x['source_observation_index']].items(): assert x[key]==value
        assert not x['alias_of_related_target'] and not x['name_relation_co_validity_proven']
    assert sum(x['normalized_value']=='PARENT A' for x in observed)==2
    sibling=next(x for x in observed if x['normalized_value']=='SIBLING MUST NOT LEAK')
    assert sibling['lexical_scope']=='EXPLICIT_ENDPOINT_NAME' and sibling['siret']=='11111111100002'
    assert sibling['lexical_eligible'] # Own site, not related-site alias.
    assert next(x for x in observed if x['normalized_value']=='UNRELATED MUST NOT LEAK')['lexical_eligible']
    invalid=next(x for x in observed if x['normalized_value']=='WRONG PARENT MUST NOT LEAK')
    assert not invalid['lexical_eligible'] and not invalid['name_grain_valid']
    assert not observed[-1]['lexical_eligible']
    context=next(x for x in observed if x['normalized_value']=='SITE SCOPE UNVERIFIED')
    assert not context['lexical_eligible']
    members=pq.read_table(output/'national_membership.parquet').to_pylist()
    assert len(members)==3 and any(x['administrative_state']=='F' for x in members)
    assert not stats['graph_membership_required'] and not stats['crm_label_or_query_selection']


@pytest.mark.parametrize('bad', ['duplicate','conflicting_parent','malformed'])
def test_registry_grain_and_uniqueness_are_required(tmp_path,bad):
    _,names=fixture(tmp_path)
    rows=[dict(siret='11111111100001',siren='111111111',administrative_state='A')]
    if bad=='duplicate': rows*=2
    elif bad=='conflicting_parent': rows[0]['siren']='222222222'
    else: rows[0]['siret']='INVALID'
    sp=tmp_path/'sites.parquet'; pq.write_table(pa.Table.from_pylist(rows),sp)
    output=tmp_path/'out'; output.mkdir(); c=duckdb.connect()
    with pytest.raises(ValueError,match='membership drift'): project(c,names,sp,output)
    c.close()
