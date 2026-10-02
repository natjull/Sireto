import pyarrow as pa
import pyarrow.parquet as pq
import tantivy

from scripts.retrieval_v39_bodacc_index import build,search


def test_direct_relation_search_has_one_document_per_site_and_cap100(tmp_path):
    rows=[]
    for i in range(110):
        for proof in (1,2):
            rows.append(dict(siret=f'222222222{i:05d}',target_siren='222222222',target_address='10 RUE ALPHA',
                target_postcode='75001',target_insee='75101',anchor_names=[dict(text='ANCIEN COMMERCE')],
                relation_id=f'r{i}_{proof}',relation_record_id=f'event{i}',event_id=f'event{i}',
                anchor_siren='111111111',address_evidence_id=f'a{i}',address_record_id=f'event{i}:establishment:0',
                reported_date='2020-01-01',date_semantics='UNVERIFIED',direction='FORWARD',site_binding='OFFICIAL'))
    path=tmp_path/'proofs.parquet'; pq.write_table(pa.Table.from_pylist(rows),path)
    counts=build(path,tmp_path/'index')
    assert counts==dict(sites=110,proof_paths=220)
    index=tantivy.Index.open(str(tmp_path/'index'))
    result=search(index,'ancien commerce','10 rue alpha','75001','75101')
    assert len(result)==100 and len({r['siret'] for r in result})==100
    assert all(len(r['proofs'])==2 for r in result)
    assert search(index,'222222222','10 rue alpha')==[]
