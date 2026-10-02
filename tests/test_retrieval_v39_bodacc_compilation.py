import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.compile_retrieval_v39_bodacc import compile_paths


def test_only_same_announcement_establishment_can_ground_transfer(tmp_path):
    """Trap SIREN-only joins, siège addresses, siblings and active preference."""
    def save(key, data):
        path=tmp_path/(key+'.parquet')
        pq.write_table(pa.Table.from_pylist(data),path)
        return path

    relations=[]
    for i in range(1,6):
        relations.append(dict(source='BODACC',relation_type='ASSET_TRANSFER',from_kind='SIREN',to_kind='SIREN',
            from_identifier='111111111',to_identifier='222222222',relation_id=f'r{i}',
            source_record_id=f'event:{i}:predecessor:0',effective_date='2020-01-01',observed_at=None))
    addresses=[]
    for event,kind,address in [(1,'establishment','10 RUE A'),(2,'person','10 RUE A'),
                               (3,'establishment','99 RUE B'),(4,'establishment','10 RUE A'),
                               (4,'establishment','99 RUE B'),(5,'establishment','10 RUE A')]:
        addresses.append(dict(source='BODACC',siren='222222222',siret='',subject_kind='SIREN',
            source_record_id=f'event:{event}:{kind}:{len(addresses)}',evidence_id=str(len(addresses)),
            normalized_value=address,postcode='' if event==5 else '75001',insee=''))
    sites=[dict(siren='222222222',siret=f'2222222220000{i}',administrative_state=state,
        current_address_normalized=address,postcode='75001',insee='75101')
        for i,state,address in [(1,'A','10 RUE A'),(2,'F','10 RUE A'),(3,'A','77 RUE C')]]
    names=[dict(siren='111111111',subject_kind=kind,normalized_value=text,
        source='SIRENE_CURRENT',name_kind='LEGAL',evidence_id=f'n{i}',source_record_id='',
        valid_from=None,valid_to=None) for i,kind,text in [(1,'SIREN','ANCIEN OPERATEUR'),(2,'SIRET','AUTRE SITE')]]
    names.append(dict(names[0],evidence_id='repeated-publication',source_record_id='another-announcement'))
    sources=dict(relations=save('relations',relations),addresses=save('addresses',addresses),
                 sites=save('sites',sites),names=save('names',names))
    conn=duckdb.connect(); stats=compile_paths(conn,sources,tmp_path); conn.close()
    proofs=pq.read_table(tmp_path/'proof_paths.parquet').to_pylist()
    assert {p['siret'] for p in proofs}=={'22222222200001','22222222200002'}
    assert {p['relation_id'] for p in proofs}=={'r1'}
    assert all([n['text'] for n in p['anchor_names']]==['ANCIEN OPERATEUR'] for p in proofs)
    coverage={r['relation_id']:r for r in pq.read_table(tmp_path/'coverage.parquet').to_pylist()}
    assert coverage['r2']['status']=='NO_EVENT_ESTABLISHMENT_ADDRESS'
    assert coverage['r3']['status']=='NO_COMPATIBLE_CURRENT_OFFICIAL_SITE_OR_GEO'
    assert coverage['r4']['status']=='MULTIPLE_EVENT_ADDRESSES_MAPPING_UNVERIFIED'
    assert coverage['r5']['status']=='NO_COMPATIBLE_CURRENT_OFFICIAL_SITE_OR_GEO'
    assert stats['valid_unique_relations']==5
    assert sum(r['row_count'] for r in stats['coverage'])==5


def test_conflicting_relation_id_fails(tmp_path):
    import pytest
    data=[dict(source='BODACC',relation_type='ASSET_TRANSFER',from_kind='SIREN',to_kind='SIREN',
        from_identifier='111111111',to_identifier='222222222',relation_id='same',
        source_record_id=f'event:{i}:predecessor:0',effective_date='2020-01-01',observed_at=None) for i in (1,2)]
    path=tmp_path/'relations.parquet'; pq.write_table(pa.Table.from_pylist(data),path)
    conn=duckdb.connect()
    with pytest.raises(ValueError,match='Conflicting'):
        compile_paths(conn,dict(relations=path),tmp_path)
    conn.close()
