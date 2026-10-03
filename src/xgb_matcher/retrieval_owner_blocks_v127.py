"""Exact all-parent→physical-block CSR, without expanding aliases to sites.

Identifiers are joins only. Source membership and layout define every pair;
no query, name, state, model, label, graph or source-family shortlist.
"""
from pathlib import Path
import time

import numpy as np


def build(parents,parent_keys,block_sites,block_offsets,root,progress=lambda row:None):
    root=Path(root);n=len(parents);np_=len(parent_keys);nb=len(block_offsets)-1;began=time.monotonic()
    if (parents.dtype!=np.uint32 or parent_keys.dtype!=np.uint32 or block_sites.dtype!=np.uint32
        or block_offsets.dtype!=np.uint64 or n!=len(block_sites) or not n or n>=2**32 or not np_
        or nb<=0 or nb>=2**32 or block_offsets[0]!=0 or block_offsets[-1]!=n
        or np.any(block_offsets[1:]<=block_offsets[:-1]) or np.any(parent_keys[1:]<=parent_keys[:-1])):
        raise ValueError('Typed complete membership/layout required')
    site_blocks=np.lib.format.open_memmap(root/'site_blocks.npy',mode='w+',dtype=np.uint32,shape=(n,))
    site_blocks[:]=np.iinfo(np.uint32).max
    for b in range(0,nb,1024):
        stop=min(nb,b+1024);lo=int(block_offsets[b]);hi=int(block_offsets[stop]);ids=block_sites[lo:hi]
        if np.any(ids>=n) or np.any(site_blocks[ids]!=np.iinfo(np.uint32).max) or len(np.unique(ids))!=len(ids):
            raise ValueError('Repeated or invalid official site position')
        site_blocks[ids]=np.repeat(np.arange(b,stop,dtype=np.uint32),np.diff(block_offsets[b:stop+1]).astype(np.int64))
    if np.any(site_blocks==np.iinfo(np.uint32).max):raise ValueError('Missing official site from physical blocks')
    site_blocks.flush();progress(dict(phase='all_sites_have_one_physical_block',sites=n,blocks=nb))
    pairs=np.empty(n,dtype=np.uint64)
    for lo in range(0,n,131072):
        hi=min(n,lo+131072);pairs[lo:hi]=(parents[lo:hi].astype(np.uint64)<<32)|site_blocks[lo:hi]
    pairs.sort(kind='quicksort')
    references=1+int(np.count_nonzero(pairs[1:]!=pairs[:-1]))
    targets=np.lib.format.open_memmap(root/'parent_blocks.npy',mode='w+',dtype=np.uint32,shape=(references,))
    counts=np.zeros(np_,dtype=np.uint64);offset=0;previous=None
    for lo in range(0,n,131072):
        chunk=pairs[lo:lo+131072];keep=np.r_[previous is None or chunk[0]!=previous,chunk[1:]!=chunk[:-1]]
        unique=chunk[keep];owners=(unique>>32).astype(np.uint32)
        values,multiplicity=np.unique(owners,return_counts=True);positions=np.searchsorted(parent_keys,values)
        if np.any(positions>=np_) or not np.array_equal(parent_keys[positions],values):raise ValueError('Unrecognized official parent join')
        counts[positions]+=multiplicity.astype(np.uint64)
        targets[offset:offset+len(unique)]=(unique&np.uint64(0xFFFFFFFF)).astype(np.uint32)
        offset+=len(unique);previous=chunk[-1]
    if offset!=references or np.any(counts==0):raise ValueError('Lost parent or block reference')
    offsets=np.empty(np_+1,dtype=np.uint64);offsets[0]=0;np.cumsum(counts,out=offsets[1:]);del counts
    np.save(root/'parent_block_offsets.npy',offsets,allow_pickle=False);targets.flush()
    # Full exact reconstruction against ALL distinct source pairs, chunked
    # by owners; not a count/checksum/random subset substituted for identity.
    cursor=0;previous=None
    for lo in range(0,n,131072):
        chunk=pairs[lo:lo+131072];keep=np.r_[previous is None or chunk[0]!=previous,chunk[1:]!=chunk[:-1]]
        expected=chunk[keep];positions=np.arange(cursor,cursor+len(expected),dtype=np.uint64)
        owner_positions=np.searchsorted(offsets,positions,side='right')-1
        actual=(parent_keys[owner_positions].astype(np.uint64)<<32)|targets[cursor:cursor+len(expected)]
        if not np.array_equal(actual,expected):raise ValueError('Exact parent/block reconstruction drift')
        cursor+=len(expected);previous=chunk[-1]
    if cursor!=references:raise ValueError('CSR full reference audit incomplete')
    stats=dict(sites=n,parents=np_,blocks=nb,parent_block_references=references,
        max_blocks_per_parent=int(np.max(np.diff(offsets))),max_sites_per_block=int(np.max(np.diff(block_offsets))),
        all_sites_assigned_once=True,all_parents_present=True,all_distinct_pairs_reconstructed_exactly=True,
        alias_times_sites_materialized=False,parent_or_site_or_alias_cap=None,closed_or_unknown_xy_filtered=False,
        selected_by_query_or_truth=False,identifiers_as_features=False,seconds=time.monotonic()-began)
    progress(dict(phase='all_parent_physical_block_csr_complete',**stats));return stats


class OwnerBlocks:
    def __init__(self,root,parent_keys):
        root=Path(root);self.keys=parent_keys
        self.offsets=np.load(root/'parent_block_offsets.npy',mmap_mode='r',allow_pickle=False)
        self.blocks=np.load(root/'parent_blocks.npy',mmap_mode='r',allow_pickle=False)
        self.site_blocks=np.load(root/'site_blocks.npy',mmap_mode='r',allow_pickle=False)
    def parent_blocks(self,siren):
        if not isinstance(siren,str) or len(siren)!=9 or not siren.isascii() or not siren.isdecimal():raise ValueError('Official parent join key required')
        key=np.uint32(int(siren));pos=int(np.searchsorted(self.keys,key))
        if pos==len(self.keys) or self.keys[pos]!=key:return self.blocks[:0]
        lo,hi=self.offsets[pos:pos+2];return self.blocks[int(lo):int(hi)]
