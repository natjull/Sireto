import numpy as np
import pytest

from src.xgb_matcher.retrieval_owner_blocks_v127 import build,OwnerBlocks


def test_all_pairs_exact_uncapped_and_unknown_geometry_not_needed(tmp_path):
    rng=np.random.default_rng(127);parents=rng.choice(np.array([123456789,987654321,111111111],dtype=np.uint32),size=1900)
    parent_keys=np.unique(parents);sites=rng.permutation(len(parents)).astype(np.uint32)
    offsets=np.r_[np.arange(0,len(sites),17),len(sites)].astype(np.uint64)
    stats=build(parents,parent_keys,sites,offsets,tmp_path);index=OwnerBlocks(tmp_path,parent_keys)
    expected={int(k):set() for k in parent_keys}
    for b in range(len(offsets)-1):
        for i in sites[int(offsets[b]):int(offsets[b+1])]:
            expected[int(parents[i])].add(b);assert index.site_blocks[i]==b
    assert stats['parent_block_references']==sum(map(len,expected.values()))
    for key,blocks in expected.items():assert list(index.parent_blocks(f'{key:09d}'))==sorted(blocks)
    assert list(index.parent_blocks('000000000'))==[]
    assert stats['alias_times_sites_materialized'] is False and stats['all_distinct_pairs_reconstructed_exactly']


def test_multi_site_parent_fanout_over100_and_singletons_all_retained(tmp_path):
    parents=np.array([123456789]*20000+[987654321],dtype=np.uint32);keys=np.unique(parents)
    sites=np.arange(len(parents),dtype=np.uint32);offsets=np.r_[np.arange(0,len(sites),128),len(sites)].astype(np.uint64)
    stats=build(parents,keys,sites,offsets,tmp_path);index=OwnerBlocks(tmp_path,keys)
    assert len(index.parent_blocks('123456789'))==157 and stats['max_blocks_per_parent']==157
    assert len(index.parent_blocks('987654321'))==1


@pytest.mark.parametrize('mutation',['repeat','out_of_range','missing_parent','unsorted_keys','bad_offsets'])
def test_no_identity_loss_or_silent_join_miss(tmp_path,mutation):
    parents=np.array([1,1,2],dtype=np.uint32);keys=np.array([1,2],dtype=np.uint32)
    sites=np.array([0,1,2],dtype=np.uint32);offsets=np.array([0,2,3],dtype=np.uint64)
    if mutation=='repeat':sites[-1]=sites[0]
    elif mutation=='out_of_range':sites[-1]=99
    elif mutation=='missing_parent':keys=keys[:1]
    elif mutation=='unsorted_keys':keys=keys[::-1]
    else:offsets[1]=0
    with pytest.raises(ValueError):build(parents,keys,sites,offsets,tmp_path)
