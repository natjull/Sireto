from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_name_access_v131 import stage,pack
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized


def decode(root):
    data=np.load(root/'token_data.npy',mmap_mode='r');offs=np.load(root/'token_offsets.npy',mmap_mode='r')
    posts=np.load(root/'post_offsets.npy',mmap_mode='r');ids=np.load(root/'post_ids.npy',mmap_mode='r')
    return {data[int(offs[i]):int(offs[i+1])].tobytes().decode('ascii'):ids[int(posts[i]):int(posts[i+1])].tolist() for i in range(len(offs)-1)}


def test_all_channels_whole_variants_exact_feature_collisions_and_masks(tmp_path):
    names=['ALPHA','ALPHA 123456789','AB','COMMUNE DE TEST','AAAAAA','123456789','ABC 123456789 123456789 DEF']
    table=pa.table(dict(name_id=pa.array(np.arange(len(names)),pa.uint32()),name_text=names));root=tmp_path/'stage';root.mkdir()
    with duckdb.connect() as c:stats=stage(c,table,root,100,0,4)
    assert stats['names']==len(names) and stats['feature_empty']==1 and stats['all_scalar_channel_sets_exact']
    features=pq.read_table(root/'features.parquet').to_pylist()
    assert [r['feature_text'] for r in features]==[normalized(s) for s in names]
    found={j:{} for j in range(4)}
    for channel in range(4):
        for bucket in range(4):
            files=sorted((root/'staging'/f'term_bucket={bucket}').glob('*.parquet'));out=tmp_path/f'{channel}_{bucket}';out.mkdir()
            expected=sum(sum(row['channel']==channel for row in pq.ParquetFile(p).read().to_pylist()) for p in files)
            with duckdb.connect() as c:result=pack(c,files,out,channel,bucket,4,expected,100+len(names))
            found[channel].update(decode(out));assert result['every_original_posting_and_token_reconstructed_exactly']
    expected={j:{} for j in range(4)}
    for i,name in enumerate(names):
        tokens=[*channels(name),{normalized(name)} if normalized(name) else set()]
        for channel,values in enumerate(tokens):
            for token in values:expected[channel].setdefault(token,[]).append(100+i)
    assert found==expected
    assert found[3]['ALPHA']==[100,101]  # Same FEATURE text, different raw names.
    assert all(105 not in ids for values in found.values() for ids in values.values())


def test_hash_collisions_do_not_mix_channels_or_variants(tmp_path):
    table=pa.table(dict(name_id=pa.array([0,1,2],pa.uint32()),name_text=['ABC','ABCD','AAA']));root=tmp_path/'stage';root.mkdir()
    with duckdb.connect() as c:stats=stage(c,table,root,0,0,1)
    files=stats['files'];out=tmp_path/'packed';out.mkdir()
    with duckdb.connect() as c:pack(c,files,out,1,0,1,stats['channel_postings']['1'],3)
    assert decode(out)=={'AAA':[2],'ABC':[0,1],'BCD':[1]}


@pytest.mark.parametrize('names',[['lowercase'],['CAFÉ'],['[ND]'],[None],['ALPHA  BETA']])
def test_raw_dictionary_cannot_be_silently_reinterpreted(tmp_path,names):
    table=pa.table(dict(name_id=pa.array([0],pa.uint32()),name_text=pa.array(names,pa.string())))
    with duckdb.connect() as c,pytest.raises(ValueError,match='canonical ASCII'):stage(c,table,tmp_path,0,0,2)


def test_missing_or_displaced_raw_names_rejected(tmp_path):
    table=pa.table(dict(name_id=pa.array([0,2],pa.uint32()),name_text=['ALPHA','BETA']))
    with duckdb.connect() as c,pytest.raises(ValueError,match='position'):stage(c,table,tmp_path,0,0,2)


def test_wrong_expected_posting_count_rejected(tmp_path):
    table=pa.table(dict(name_id=pa.array([0],pa.uint32()),name_text=['ALPHA']));root=tmp_path/'stage';root.mkdir()
    with duckdb.connect() as c:stats=stage(c,table,root,0,0,2)
    files=stats['files'];out=tmp_path/'packed';out.mkdir()
    with duckdb.connect() as c,pytest.raises(ValueError,match='Whole channel'):pack(c,files,out,0,0,2,999,1)


def test_stream_boundary_and_empty_channel(tmp_path):
    # >1 sorted batch; all aliases share token AA, current/raw metadata is
    # untouched and no cap on references/term popularity is accepted.
    count=131073;path=tmp_path/'posts.parquet'
    table=pa.table(dict(alias_id=pa.array(np.arange(count,dtype=np.uint32)),token=['AA']*count,
        channel=pa.array(np.zeros(count,dtype=np.uint8)),term_bucket=pa.array(np.zeros(count,dtype=np.uint8))))
    pq.write_table(table,path);out=tmp_path/'packed';out.mkdir()
    with duckdb.connect() as c:pack(c,[path],out,0,0,1,count,count)
    assert decode(out)['AA']==list(range(count))
    empty=tmp_path/'empty';empty.mkdir()
    with duckdb.connect() as c:pack(c,[],empty,1,0,1,0,count)
    assert decode(empty)=={}
