import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_name_access_v131 import pack as original
from src.xgb_matcher.retrieval_packed_names_v134 import pack,run_positions


def data(tmp_path,rows):
    schema=pa.schema([('alias_id',pa.uint32()),('token',pa.string()),('channel',pa.uint8()),('term_bucket',pa.uint8())])
    path=tmp_path/'posts.parquet';pq.write_table(pa.Table.from_pylist(rows,schema=schema),path);return path


def test_every_array_bit_identical_to_original_semantics(tmp_path):
    rows=[dict(alias_id=i,token=token,channel=0,term_bucket=0) for token in ('A','AA','A_B','B') for i in range(17)]
    source=data(tmp_path,rows);a=tmp_path/'original';b=tmp_path/'packed';a.mkdir();b.mkdir()
    with duckdb.connect() as c:old=original(c,[source],a,0,0,1,len(rows),17)
    with duckdb.connect() as c:new=pack(c,[source],b,0,0,1,len(rows),17)
    assert new['every_original_posting_and_token_reconstructed_exactly'] and new['source_semantics_unchanged']
    for name in ('post_ids','post_offsets','token_data','token_offsets'):
        assert (a/(name+'.npy')).read_bytes()==(b/(name+'.npy')).read_bytes()
    assert all(new[k]==v for k,v in old.items())


def test_unsigned_vector_search_never_scalar_promotes(monkeypatch):
    actual=np.searchsorted;calls=[]
    def spy(a,b,*args,**kwargs):
        calls.append(np.asarray(b).dtype);return actual(a,b,*args,**kwargs)
    monkeypatch.setattr(np,'searchsorted',spy)
    offsets=np.array([0,3,6,10],dtype=np.uint64)
    assert run_positions(offsets,2,np.array([1,3,6],dtype=np.int32)).tolist()==[0,1,1]
    assert calls==[np.dtype(np.uint64)]


def test_uint64_positions_beyond_float_exact_range():
    base=2**53+1;offsets=np.array([0,base,base+1,base+4],dtype=np.uint64)
    assert run_positions(offsets,base,np.array([1,2,4],dtype=np.int32)).tolist()==[1,2,2]


def test_large_vocabulary_every_text_and_alias_reconstructed(tmp_path):
    rows=[dict(alias_id=i,token=f'TERM{i:07d}',channel=3,term_bucket=0) for i in range(150001)]
    source=data(tmp_path,rows);out=tmp_path/'mapped';out.mkdir()
    with duckdb.connect() as c:s=pack(c,[source],out,3,0,1,len(rows),len(rows))
    assert s['terms']==150001 and s['postings']==150001 and s['every_original_posting_and_token_reconstructed_exactly']
    offsets=np.load(out/'token_offsets.npy');raw=np.load(out/'token_data.npy')
    for i in (0,131071,131072,150000):assert raw[int(offsets[i]):int(offsets[i+1])].tobytes()==rows[i]['token'].encode('ascii')


def test_popular_posting_continues_across_batches_without_cap(tmp_path):
    rows=[dict(alias_id=i,token='POPULAR',channel=0,term_bucket=0) for i in range(150001)]
    source=data(tmp_path,rows);out=tmp_path/'mapped';out.mkdir()
    with duckdb.connect() as c:s=pack(c,[source],out,0,0,1,len(rows),len(rows))
    assert s['terms']==1 and s['postings']==150001 and s['no_posting_or_owner_cap']
    assert np.load(out/'post_offsets.npy').tolist()==[0,150001]


def test_empty_channel_has_exact_empty_arrays(tmp_path):
    out=tmp_path/'mapped';out.mkdir()
    with duckdb.connect() as c:s=pack(c,[],out,0,0,1,0,0)
    assert s['postings']==0 and np.load(out/'post_offsets.npy').tolist()==[0]


def test_wrong_whole_posting_count_rejected(tmp_path):
    source=data(tmp_path,[dict(alias_id=0,token='A',channel=0,term_bucket=0)]);out=tmp_path/'mapped';out.mkdir()
    with duckdb.connect() as c,pytest.raises(ValueError,match='Whole channel'):pack(c,[source],out,0,0,1,2,1)
