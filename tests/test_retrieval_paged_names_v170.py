"""Bit-exact tiny official-source fixtures, no national cost/recall claims."""
from pathlib import Path
from types import SimpleNamespace

import duckdb

import numpy as np
import pytest

import src.xgb_matcher.retrieval_paged_names_v170 as module
import src.xgb_matcher.retrieval_rebound_names_v150 as original
from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_national_names_v138 import NationalNames
from src.xgb_matcher.retrieval_proof_bridge_v137 import one_hop_names,MISSING
from src.xgb_matcher.retrieval_name_rebind_v149 import patched_positions
from src.xgb_matcher.retrieval_variants_v118 import normalized
from test_retrieval_rebound_names_v150 import setup,Index


def guard():pass


class TokenIndex:
    def __init__(self,texts):self.texts=texts;self.names=len(texts);self.calls=[]
    def posting(self,channel,token):
        self.calls.append((channel,token))
        return np.array([i for i,text in enumerate(self.texts)
            if token in channels(text)[channel] if channel<3],dtype=np.uint32) if channel<3 else np.array(
                [i for i,text in enumerate(self.texts) if normalized(text)==token],dtype=np.uint32)


def disk(path,values):
    a=module.DiskVector(path,len(values),values.dtype);a.assign(0,values);return a


@pytest.mark.parametrize('chunk',[1,3,131072,524288])
@pytest.mark.parametrize('text',['ALPHA BETA','RUE JEAN JAURES','A','', 'A & B SOCIETE'])
def test_complete_coverage_bit_equal_to_native_all_channels(tmp_path,chunk,text):
    texts=['ALPHA BETA','BETA ALPHA','ALPHA','RUE JEAN JAURES','RUE JEAN JAURES','A','', 'A ET B SOCIETE']
    index=TokenIndex(texts);a,b,stats=module.coverage(index,text,tmp_path,guard,chunk)
    native=TokenIndex(texts);x,y,old=NationalNames.coverage(native,text,chunk=chunk)
    np.testing.assert_array_equal(a.read(0,len(texts)),x);np.testing.assert_array_equal(b.read(0,len(texts)),y)
    assert stats['channels']==old['channels'] and stats['positive_aliases']==old['positive_aliases']
    assert stats['exact_aliases']==old['exact_aliases'] and stats['postings_or_alias_cap'] is None
    assert stats['float_score_precision_unchanged'] and not (tmp_path/'token_counts.npy').exists()
    assert index.calls==native.calls
    a.close(remove=True);b.close(remove=True)


def test_disk_vectors_fresh_zero_full_namespace_owned_copies_and_explicit_close(tmp_path):
    a=module.DiskVector(tmp_path/'v.npy',17,np.float64)
    assert not np.count_nonzero(a.read(0,17))
    a.maximum(np.array([16,16,0],dtype=np.uint32),np.array([.3,.8,.5]))
    result=a.take(np.array([16,0,16],dtype=np.uint32));assert result.tolist()==[.8,.5,.8]
    assert result.flags.owndata and not isinstance(result,np.memmap)
    with pytest.raises(ValueError,match='Fresh'):module.DiskVector(a.path,17,np.float64)
    with pytest.raises(ValueError):a.take(np.array([17],dtype=np.uint32))
    with a.mapped():
        with pytest.raises(ValueError):a.take(np.array([0],dtype=np.uint32))
        with pytest.raises(ValueError):a.close()
    a.close(remove=True)
    assert not a.path.exists() and result.tolist()==[.8,.5,.8]
    with pytest.raises(ValueError):a.read(0,1)


def test_posting_duplicates_wrong_dtype_and_outside_aliases_refused(tmp_path):
    class Bad(TokenIndex):
        def posting(self,*args):return self.values
    index=Bad(['ALPHA','BETA'])
    for i,values in enumerate((np.array([0,0],dtype=np.uint32),np.array([2],dtype=np.uint32),np.array([0],dtype=np.int64))):
        index.values=values
        with pytest.raises(ValueError,match='sorted unique'):module.coverage(index,'ALPHA',tmp_path/str(i),guard,1)


def test_sparse_overlay_byte_equal_to_original_bucket_copy(tmp_path):
    previous=np.array([0,MISSING,1,MISSING,MISSING],dtype=np.uint32)
    rows=np.array([1,4],dtype=np.uint32);positions=np.array([2,3],dtype=np.uint32)
    oracle=patched_positions(previous,rows,positions,2,4)
    for ids in (np.arange(5,dtype=np.int64),np.array([4,1,4,0],dtype=np.int64)):
        np.testing.assert_array_equal(module.overlay(previous,rows,positions,ids,2,4),oracle[ids])
    np.testing.assert_array_equal(previous,np.array([0,MISSING,1,MISSING,MISSING],dtype=np.uint32))
    with pytest.raises(ValueError):module.overlay(previous,rows,positions,np.array([-1]),2,4)
    with pytest.raises(ValueError):module.overlay(previous,rows[::-1],positions,np.array([0]),2,4)


@pytest.mark.parametrize('chunk',[1,3,32])
def test_frozen_graph_snapshot_duplicates_chain_selflinks_no_exact_or_two_hops(tmp_path,chunk):
    parent=np.array([1.,0.,0.,.3]);site=np.zeros(4)
    graph=dict(graph_site_positions=np.array([0,1,2,3],dtype=np.uint32),
        graph_parent_positions=np.array([0,1,2,3],dtype=np.uint32),
        predecessor_graph_positions=np.array([0,1,0,3,0],dtype=np.uint32),
        successor_graph_positions=np.array([1,2,1,3,3],dtype=np.uint32),continuity_codes=np.array([2,2,0,2,1],dtype=np.uint8))
    pc=disk(tmp_path/'pc.npy',parent);sc=disk(tmp_path/'sc.npy',site)
    def read(name,ids):return graph[name][ids].copy()
    actual=module.one_hop(pc,sc,read,4,5,tmp_path,guard,chunk)
    expected=one_hop_names(parent,site,graph['graph_site_positions'],graph['graph_parent_positions'],
        graph['predecessor_graph_positions'],graph['successor_graph_positions'],graph['continuity_codes'])
    np.testing.assert_array_equal(actual.read(0,4),expected)
    assert expected.tolist()==[.075,.5,0.,.25] and not (tmp_path/'endpoint_snapshot.npy').exists()


@pytest.mark.parametrize('chunk',[1,2,32,524288])
def test_loader_scatter_all_patches_snapshot_and_stats_bit_equal_to_native(tmp_path,monkeypatch,chunk):
    ref,out=setup(tmp_path,monkeypatch);oracle=original.ReboundNames(ref)
    # setup supplies in-memory fake keys to the original loader only. Reuse
    # that exact constructor symbol, without mutating measured source files.
    monkeypatch.setattr(module,'ReboundNames',original.ReboundNames)
    index=Index(oracle.source);expected,oldtrace=oracle.query(index,'fixture',chunk=chunk)
    def fixture_coverage(index,text,directory,guard,chunk):
        c,e,stats=index.coverage(text);return disk(directory/'alias_coverage.npy',c),disk(directory/'alias_exact.npy',e),stats
    monkeypatch.setattr(module,'coverage',fixture_coverage)
    reader=module.PagedReboundNames(ref)
    try:
        progress=[]
        actual,stats=reader.query(Index(reader.source),'fixture',tmp_path/'query',chunk=chunk,progress=progress.append)
        assert progress[0]['stage']=='LEXICAL_COMPLETE' and progress[-1]['stage']=='ONE_HOP_COMPLETE'
        ids=np.array([4,0,2,1,3,0],dtype=np.uint32)
        for x,y in zip(actual.gather(ids),expected.gather(ids)):np.testing.assert_array_equal(x,y)
        for a,b in zip(actual.vectors,(expected.snapshot.parent_cov,expected.snapshot.parent_exact,
            expected.snapshot.previous.site_cov,expected.snapshot.previous.site_exact,expected.snapshot.previous.relation_cov)):
            np.testing.assert_array_equal(a.read(0,len(a)),b)
        for k in ('references_processed','original_parent_proofs','appended_parent_proofs','appended_parent_exact',
            'direct_site_proofs','related_endpoint_proofs','appended_sites_with_name_proofs','appended_sites_with_direct_exact'):
            assert stats[k]==oldtrace[k]
        assert stats['no_alias_or_owner_or_member_cap'] and stats['float_score_precision_unchanged']
        reader.verify_after();actual.close(remove=True)
        with pytest.raises(ValueError,match='Closed'):actual.gather(ids)
    finally:reader.close();oracle.close()


def test_timeout_keeps_failed_evidence_but_returns_no_snapshot(tmp_path):
    index=TokenIndex(['ALPHA','BETA'])
    def fail():raise TimeoutError('STOP_PAGED_NAME_QUERY_COST')
    with pytest.raises(TimeoutError):module.coverage(index,'ALPHA',tmp_path,fail,1)
    assert (tmp_path/'alias_coverage.npy').exists() and (tmp_path/'alias_exact.npy').exists()


def test_rotate_only_touched_partition_all_other_postings_stay_reachable(tmp_path):
    index=SimpleNamespace(c=duckdb.connect(),buckets=2,access={},files={},valid=True)
    try:
        for b in range(2):
            path=tmp_path/f'posting{b}.npy';np.save(path,np.array([b,b+3],dtype=np.uint32),allow_pickle=False)
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            index.access[0,b]=SimpleNamespace(a=dict(post_ids=a))
            index.files[0,b]=dict(post_ids=dict(path=path,fingerprint=module.fingerprint(path),shape=a.shape,dtype=a.dtype))
        b=int(index.c.execute("SELECT hash('ALPHA') % 2").fetchone()[0]);untouched=index.access[0,1-b].a['post_ids']
        touched=index.access[0,b].a['post_ids'];module.rotate_posting(index,0,'ALPHA')
        assert index.access[0,1-b].a['post_ids'] is untouched and not untouched._mmap.closed
        assert touched._mmap.closed and index.access[0,b].a['post_ids'].tolist()==[b,b+3]
        assert index.valid
        path=index.files[0,b]['post_ids']['path'];path.write_bytes(path.read_bytes()+b'changed')
        with pytest.raises(ValueError,match='epoch drift'):module.rotate_posting(index,0,'ALPHA')
        assert not index.valid
    finally:
        for access in index.access.values():
            for a in access.a.values():a._mmap.close()
        index.c.close()
