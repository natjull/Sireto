import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_address_dictionary_v145 import partition,dictionary
from src.xgb_matcher.retrieval_address_projection_v142 import project,SCHEMA
from test_retrieval_address_projection_v142 import canonical,Keys


def projected(tmp_path):
    rows=[dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='RNE',raw_value='RUE A',postcode='75001',insee='75101',street_number='1',is_current=False),
        dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='BODACC',raw_value='RUE A',postcode='75001',insee='75101',street_number='1',is_current=True),
        dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='RNE',raw_value='RUE A',postcode='69001',insee='69381',street_number='9'),
        dict(siren='000000002',siret='00000000200009',subject_kind='SIRET',source='RNE',raw_value='RUE A',postcode='75001',insee='75101',street_number='1',is_current=False),
        dict(siren='000000004',subject_kind='SIREN',source='RNE',raw_value='ADRESSE PARENT'),
        dict(siren='000000001',siret='00000000100001',subject_kind='SIRET',source='RNE',raw_value='[ND]',normalized_value='ND ND')]
    t=project(canonical(rows),'official_addresses',Keys(),0,0);p=tmp_path/'source.parquet';pq.write_table(t,p);return p,t


def test_shared_text_not_shared_site_cp_number_or_current_metadata(tmp_path):
    source,t=projected(tmp_path);stage=tmp_path/'stage';stage.mkdir();c=duckdb.connect()
    s=partition(c,source,stage,0,6,4,1,12,6);c.close()
    assert s['original_observations']==6 and s['context_observations']==2 and s['all_eligible_rows_unique_and_exact']
    out=tmp_path/'dictionary';out.mkdir();c=duckdb.connect();s=dictionary(c,s['files'],out,0,1,4)
    assert s['addresses']==1 and s['facts']==3 and s['observations']==4 and s['every_observation_linked_once']
    facts=pq.read_table(out/'facts.parquet').to_pylist()
    assert {(r['site_position'],r['number_text']) for r in facts}=={(8,'1'),(8,'9'),(9,'1')}
    assert {(r['site_position'],r['number_text']):r['current_exact'] for r in facts}=={(8,'1'):True,(8,'9'):False,(9,'1'):False}
    assert sorted(pq.read_table(out/'observation_links.parquet')['observation_id'].to_pylist())==[0,1,2,3]
    assert not s['parent_address_spread_to_sites'] and not s['identifiers_as_features']
    c.close()


def test_physical_hash_collision_never_merges_different_whole_texts(tmp_path):
    rows=[dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='RNE',raw_value=v) for v in ('RUE A','RUE B')]
    p=tmp_path/'source.parquet';pq.write_table(project(canonical(rows),'official_addresses',Keys(),0,0),p)
    stage=tmp_path/'stage';stage.mkdir();c=duckdb.connect();s=partition(c,p,stage,0,2,2,1,12,6);c.close()
    out=tmp_path/'dict';out.mkdir();c=duckdb.connect();s=dictionary(c,s['files'],out,0,1,2);c.close()
    assert s['addresses']==2 and s['facts']==2
    assert pq.read_table(out/'addresses.parquet')['address_text'].to_pylist()==['RUE A','RUE B']


def test_all_occurrences_linked_no_2000_cap_and_no_latest_period_filter(tmp_path):
    rows=[dict(siren='000000001',siret='00000000100003',subject_kind='SIRET',source='BODACC',raw_value='VIEILLE RUE',is_current=False)]*2007
    p=tmp_path/'source.parquet';pq.write_table(project(canonical(rows),'official_addresses',Keys(),0,0),p)
    stage=tmp_path/'stage';stage.mkdir();c=duckdb.connect();s=partition(c,p,stage,0,2007,2007,1,12,6);c.close()
    out=tmp_path/'dict';out.mkdir();c=duckdb.connect();s=dictionary(c,s['files'],out,0,1,2007);c.close()
    assert s['observations']==2007 and s['addresses']==1 and s['facts']==1
    assert pq.read_table(out/'observation_links.parquet')['observation_id'].to_pylist()==list(range(2007))


def test_parent_only_empty_masked_scope_stays_in_source_not_fake_site_dictionary(tmp_path):
    rows=[dict(siren='000000004',subject_kind='SIREN',source='RNE',raw_value='ADRESSE DE SIEGE')]
    p=tmp_path/'source.parquet';pq.write_table(project(canonical(rows),'official_addresses',Keys(),0,0),p)
    stage=tmp_path/'stage';stage.mkdir();c=duckdb.connect();s=partition(c,p,stage,0,1,0,2,12,6);c.close()
    assert not s['files'] and s['context_observations']==1
    out=tmp_path/'dict';out.mkdir();c=duckdb.connect();s=dictionary(c,[],out,0,2,0);c.close()
    assert s['addresses']==0 and s['facts']==0 and s['observations']==0


def test_drift_in_scope_ordinal_or_expected_eligible_rows_is_stop_not_filter(tmp_path):
    p,t=projected(tmp_path);c=duckdb.connect()
    with pytest.raises(ValueError,match='grain/ordinal'):partition(c,p,tmp_path,0,6,3,1,12,6)
    c.close();bad=t.set_column(t.schema.get_field_index('lexical_eligible'),'lexical_eligible',pa.array([True]*6))
    pq.write_table(bad,p);c=duckdb.connect()
    with pytest.raises(ValueError,match='grain/ordinal'):partition(c,p,tmp_path,0,6,6,1,12,6)
    c.close()


@pytest.mark.parametrize('mutation',['duplicate_observation','missing_site','missing_flag','wrong_bucket'])
def test_dictionary_rejects_corrupt_staging_instead_of_silent_join_loss(tmp_path,mutation):
    p,t=projected(tmp_path);stage=tmp_path/'stage';stage.mkdir();c=duckdb.connect()
    s=partition(c,p,stage,0,6,4,1,12,6);c.close()
    table=pq.ParquetFile(s['files'][0]).read();rows=table.to_pylist()
    if mutation=='duplicate_observation':rows[1]['observation_id']=rows[0]['observation_id']
    elif mutation=='missing_site':rows[0]['site_position']=None
    elif mutation=='missing_flag':rows[0]['current_exact']=None
    else:rows[0]['bucket']=1
    corrupt=tmp_path/'corrupt.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),corrupt)
    out=tmp_path/'out';out.mkdir();c=duckdb.connect()
    with pytest.raises(ValueError,match='unique original'):dictionary(c,[corrupt],out,0,1,4)
    c.close()


def test_quoted_paths_are_sql_literals_and_not_interpolated_statements(tmp_path):
    quoted=tmp_path/"l'adresse";quoted.mkdir();p,t=projected(quoted)
    stage=quoted/'stage';stage.mkdir();c=duckdb.connect();s=partition(c,p,stage,0,6,4,1,12,6);c.close()
    out=quoted/'dictionary';out.mkdir();c=duckdb.connect();s=dictionary(c,s['files'],out,0,1,4);c.close()
    assert s['observations']==4 and s['facts']==3


@pytest.mark.parametrize('bucket,buckets,expected',[(1,1,0),(0,0,0),(0,1,-1),(True,1,0)])
def test_invalid_bucket_or_count_is_stop(tmp_path,bucket,buckets,expected):
    c=duckdb.connect()
    with pytest.raises(ValueError,match='range'):dictionary(c,[],tmp_path,bucket,buckets,expected)
    c.close()
