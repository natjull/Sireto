import collections

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_variant_dictionary_v130 import dictionary,partition
from src.xgb_matcher.retrieval_variants_v118 import OBS_SCHEMA,observations


def fixture(tmp_path):
    rows=[]
    for i,(role,row) in enumerate([
        ('history_ul',dict(siren='001234567',denominationUniteLegale='École Alpha',dateDebut='2000-01-01',dateFin='2009-01-01')),
        ('history_ul',dict(siren='001234567',denominationUniteLegale='École Alpha',dateDebut='2009-01-01',dateFin='2015-01-01')),
        ('current_ul',dict(siren='001234567',denominationUniteLegale='École Alpha')),
        ('current_site',dict(siren='001234567',siret='00123456700001',enseigne1Etablissement='École Alpha')),
        ('current_site',dict(siren='001234567',siret='00123456700002',enseigne1Etablissement='Autre')),
        ('current_ul',dict(siren='001234567',denominationUniteLegale='[ND]')),
        ('history_ul',dict(siren='bad',denominationUniteLegale='Alpha')),
        ('official_names',dict(siren='001234567',subject_kind='SIREN',source='BODACC',name_kind='LEGAL',raw_value='Autre',is_current=None)),
        ('official_names',dict(siren='001234567',subject_kind='SIREN',source='RNE',name_kind='LEGAL',raw_value='École Alpha',is_current=False)),
    ]):rows.extend(observations(row,role,i))
    path=tmp_path/'original.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=OBS_SCHEMA),path)
    return path,rows


def test_all_rows_global_text_dedup_scope_and_every_period_reconstruct(tmp_path):
    path,rows=fixture(tmp_path);stage=tmp_path/'stage';stage.mkdir();eligible=sum(r['lexical_eligible'] for r in rows)
    with duckdb.connect() as c:stats=partition(c,path,stage,100,len(rows),eligible,4)
    assert stats['lexical_observations']==eligible and stats['context_observations']==2
    names=collections.Counter();linked=[];facts=[]
    for bucket in range(4):
        inputs=list((stage/'staging'/f'bucket={bucket}').glob('*.parquet'));out=tmp_path/f'd{bucket}';out.mkdir()
        count=sum(pq.ParquetFile(p).metadata.num_rows for p in inputs)
        with duckdb.connect() as c:result=dictionary(c,inputs,out,bucket,4,count)
        n=pq.read_table(out/'names.parquet').to_pylist();f=pq.read_table(out/'facts.parquet').to_pylist();links=pq.read_table(out/'observation_links.parquet').to_pylist()
        names.update(r['name_text'] for r in n);names_by_id={r['name_id']:r['name_text'] for r in n};by_fact={r['fact_id']:r for r in f}
        for link in links:
            original=rows[link['observation_id']-100];fact=by_fact[link['fact_id']]
            assert names_by_id[fact['name_id']]==original['name_text']
            assert fact['owner_key']==int(original['siren'] if original['subject_kind']=='SIREN' else original['siret'])
            assert str(fact['parent_key']).zfill(9)==original['siren']
            assert str(fact['owner_key']).zfill(9 if fact['subject_code']==1 else 14)==(original['siren'] if fact['subject_code']==1 else original['siret'])
            linked.append(link['observation_id']);facts.append((names_by_id[fact['name_id']],fact))
        assert result['every_owner_and_name_reconstructed_exactly']
    assert all(v==1 for v in names.values()) and names['ECOLE ALPHA']==1
    assert sorted(linked)==[100+i for i,r in enumerate(rows) if r['lexical_eligible']]
    assert len([1 for name,f in facts if name=='ECOLE ALPHA' and f['subject_code']==1])==4
    assert all(f['current_exact'] for name,f in facts if name=='ECOLE ALPHA')
    assert any(not f['current_exact'] for name,f in facts if name=='AUTRE' and f['subject_code']==1)


def test_storage_hash_collision_never_merges_distinct_names(tmp_path):
    path,rows=fixture(tmp_path);stage=tmp_path/'stage';stage.mkdir();eligible=sum(r['lexical_eligible'] for r in rows)
    with duckdb.connect() as c:partition(c,path,stage,0,len(rows),eligible,1)
    out=tmp_path/'dict';out.mkdir();files=list((stage/'staging'/'bucket=0').glob('*.parquet'))
    with duckdb.connect() as c:stats=dictionary(c,files,out,0,1,eligible)
    assert stats['names']==len({r['name_text'] for r in rows if r['lexical_eligible']})


@pytest.mark.parametrize('change',[{'lexical_scope':None},{'siren':'bad'},{'siret':None},{'name_text':'bad é'},{'lexical_eligible':None}])
def test_invalid_admissible_source_cannot_be_silently_cast(tmp_path,change):
    path,rows=fixture(tmp_path);rows[0].update(change);pq.write_table(pa.Table.from_pylist(rows,schema=OBS_SCHEMA),path)
    stage=tmp_path/'stage';stage.mkdir()
    with duckdb.connect() as c,pytest.raises(ValueError,match='scope/text/count'):partition(c,path,stage,0,len(rows),sum(r['lexical_eligible'] is True for r in rows),4)


def test_wrong_whole_source_count_or_bucket_fails(tmp_path):
    path,rows=fixture(tmp_path);stage=tmp_path/'stage';stage.mkdir();eligible=sum(r['lexical_eligible'] for r in rows)
    with duckdb.connect() as c,pytest.raises(ValueError,match='scope/text/count'):partition(c,path,stage,0,len(rows),eligible-1,4)
    with duckdb.connect() as c:partition(c,path,stage,0,len(rows),eligible,1)
    files=list((stage/'staging'/'bucket=0').glob('*.parquet'));out=tmp_path/'dict';out.mkdir()
    with duckdb.connect() as c,pytest.raises(ValueError,match='whole-bucket'):dictionary(c,files,out,1,2,eligible)


def test_all_context_rows_preserved_without_fake_lexical_facts(tmp_path):
    rows=list(observations(dict(siren='123456789',denominationUniteLegale='[ND]'),'current_ul',0))
    path=tmp_path/'masked.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=OBS_SCHEMA),path)
    out=tmp_path/'stage';out.mkdir()
    with duckdb.connect() as c:result=partition(c,path,out,0,len(rows),0,4)
    assert result['files']==[] and result['context_observations']==len(rows)
    dest=tmp_path/'dictionary';dest.mkdir()
    with duckdb.connect() as c:stats=dictionary(c,[],dest,0,4,0)
    assert stats['names']==stats['facts']==stats['observations']==0
