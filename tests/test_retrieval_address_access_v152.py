import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_address_access_v152 import SCHEMA,physical_plan,stage_addresses
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_national_names_v138 import PackedTerms
from src.xgb_matcher.retrieval_packed_names_v134 import pack
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import record


def table(texts,start=0):
    return pa.Table.from_arrays([pa.array(np.arange(start,start+len(texts),dtype=np.uint32)),pa.array(texts,pa.string())],schema=SCHEMA)


def test_all_whole_address_channels_raw_feature_collisions_masks_and_no_token_bag(tmp_path):
    texts=['RUE ALPHA','RUE ALPHA 123456789','RUE BETA','ALPHA BETA','123456789','AB','AAAAAA']
    root=tmp_path/'stage';root.mkdir()
    with duckdb.connect() as c:stats=stage_addresses(c,table(texts),root,100,0,2)
    assert stats['addresses']==7 and stats['feature_empty']==1
    assert stats['all_raw_addresses_retained'] and stats['alias_or_proof_cap'] is None
    assert not stats['address_fact_attributes_or_scope_changed']
    assert pq.read_table(root/'features.parquet')['feature_text'].to_pylist()==[normalized(s) for s in texts]
    found={i:{} for i in range(4)}
    for channel in range(4):
        for bucket in range(2):
            out=tmp_path/f'{channel}_{bucket}';out.mkdir()
            files=sorted((root/'staging'/f'term_bucket={bucket}').glob('*.parquet'))
            count=sum(sum(row['channel']==channel for row in pq.ParquetFile(p).read().to_pylist()) for p in files)
            with duckdb.connect() as c:pack(c,files,out,channel,bucket,2,count,107)
            arrays={k:np.load(out/(k+'.npy'),mmap_mode='r') for k in ('post_ids','post_offsets','token_data','token_offsets')}
            terms=PackedTerms(arrays)
            for i in range(terms.terms):
                token=terms.key(i).decode('ascii');found[channel][token]=terms.lookup(token).tolist()
            for a in arrays.values():a._mmap.close()
    expected={i:{} for i in range(4)}
    for index,text in enumerate(texts):
        sets=[*channels(text),{normalized(text)} if normalized(text) else set()]
        for channel,values in enumerate(sets):
            for token in values:expected[channel].setdefault(token,[]).append(100+index)
    assert found==expected and found[3]['RUE ALPHA']==[100,101]
    assert all(104 not in ids for items in found.values() for ids in items.values())
    assert found[3]['ALPHA BETA']==[103]  # Not assembled from two different streets.


def test_physical_plan_covers_all_rowgroups_slices_and_empty_buckets(tmp_path):
    buckets=[];start=0
    for b,texts in enumerate([['RUE ALPHA','RUE BETA','RUE C'],[],['AVENUE D','CHEMIN E']]):
        p=tmp_path/f'addresses{b}.parquet';pq.write_table(table(texts),p,row_group_size=2)
        buckets.append(dict(bucket=b,alias_start=start,aliases=len(texts),artifacts={'addresses.parquet':record(p)}));start+=len(texts)
    plan=physical_plan(buckets,5,batch_rows=1)
    recovered=[]
    for spec in plan:
        raw=pq.ParquetFile(spec['source']['path']).read_row_group(spec['row_group'])
        value=raw.slice(spec['group_slice_start'],spec['group_slice_stop']-spec['group_slice_start'])
        assert value['address_id'].to_pylist()==list(range(spec['row_start'],spec['row_stop']))
        recovered.extend(spec['alias_start']+i for i in value['address_id'].to_pylist())
    assert recovered==list(range(5)) and [s['part_id'] for s in plan]==list(range(5))
    assert physical_plan([dict(bucket=0,alias_start=0,aliases=0,artifacts={'addresses.parquet':buckets[1]['artifacts']['addresses.parquet']})],0)==[]


@pytest.mark.parametrize('texts',[['lowercase'],['CAFÉ'],['[ND]'],[None],['RUE  ALPHA']])
def test_original_dictionary_text_never_silently_reinterpreted(tmp_path,texts):
    with duckdb.connect() as c,pytest.raises(ValueError,match='canonical ASCII'):
        stage_addresses(c,table(texts),tmp_path,0,0,2)


def test_shifted_alias_ids_or_changed_schema_stop(tmp_path):
    with duckdb.connect() as c,pytest.raises(ValueError,match='position'):
        stage_addresses(c,table(['RUE A'],1),tmp_path,0,0,2)
    bad=pa.table(dict(address_id=[0],address_text=['RUE A']))
    with duckdb.connect() as c,pytest.raises(ValueError,match='schema'):
        stage_addresses(c,bad,tmp_path,0,0,2)


@pytest.mark.parametrize('field,value',[('bucket',1),('alias_start',1),('aliases',2)])
def test_missing_or_displaced_dictionary_namespace_stop(tmp_path,field,value):
    p=tmp_path/'addresses.parquet';pq.write_table(table(['RUE A']),p)
    item=dict(bucket=0,alias_start=0,aliases=1,artifacts={'addresses.parquet':record(p)});item[field]=value
    with pytest.raises(ValueError,match='bucket|dictionary'):physical_plan([item],1)


@pytest.mark.parametrize('batch',[0,-1,True])
def test_invalid_physical_budget_is_not_a_source_cap(tmp_path,batch):
    with pytest.raises(ValueError,match='physical batch'):physical_plan([],0,batch)
