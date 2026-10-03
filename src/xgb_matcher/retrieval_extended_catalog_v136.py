"""Exact current+official-extra candidate namespace, without alias fanout.

Current site/parent/block positions remain unchanged. Extra sites have no
invented geometry, state, headquarters, activity or address. Keys are joins,
output identities and global boundary ties, never similarity features.
"""
from pathlib import Path
import json

import numpy as np

from src.xgb_matcher.retrieval_v20_workbench import verify

MISSING=np.iinfo(np.uint32).max


def positions(current,extra,values):
    if current.dtype!=extra.dtype or values.dtype!=current.dtype or current.dtype not in (np.dtype(np.uint32),np.dtype(np.uint64)):
        raise ValueError('Exact same unsigned key dtypes required')
    if current.ndim!=1 or extra.ndim!=1 or values.ndim!=1 or len(current)+len(extra)>=MISSING:
        raise ValueError('Whole virtual key position range required')
    result=np.full(len(values),MISSING,dtype=np.uint32);pos=np.searchsorted(current,values);valid=pos<len(current)
    rows=np.flatnonzero(valid);rows=rows[current[pos[rows]]==values[rows]];result[rows]=pos[rows].astype(np.uint32)
    unresolved=np.flatnonzero(result==MISSING);pos=np.searchsorted(extra,values[unresolved]);valid=pos<len(extra)
    chosen=np.flatnonzero(valid);chosen=chosen[extra[pos[chosen]]==values[unresolved[chosen]]]
    result[unresolved[chosen]]=(len(current)+pos[chosen]).astype(np.uint32);return result


def ordered(values,dtype,maximum):
    if values.dtype!=np.dtype(dtype) or values.ndim!=1 or np.any(values>=maximum) or np.any(values[1:]<=values[:-1]):
        raise ValueError('Whole sorted unique official typed keys required')


def save(output,name,value):
    np.save(Path(output)/(name+'.npy'),value,allow_pickle=False)


def build(current_sites,current_parents,extra_sites,base_blocks,block_size,output):
    ordered(current_sites,np.uint64,10**14);ordered(extra_sites,np.uint64,10**14);ordered(current_parents,np.uint32,10**9)
    if (not isinstance(base_blocks,int) or isinstance(base_blocks,bool) or base_blocks<0
        or not isinstance(block_size,int) or isinstance(block_size,bool) or block_size<=0
        or len(current_sites)+len(extra_sites)>=MISSING or len(current_parents)+len(extra_sites)>=MISSING):
        raise ValueError('Whole virtual namespace/block range required')
    if np.any(positions(current_sites,np.empty(0,dtype=np.uint64),extra_sites)!=MISSING):raise ValueError('Extra official sites must not duplicate current sites')
    raw_parents=(extra_sites//100000).astype(np.uint32);unique_parents=np.unique(raw_parents)
    missing=positions(current_parents,np.empty(0,dtype=np.uint32),unique_parents)==MISSING
    extra_parents=unique_parents[missing];parent_positions=positions(current_parents,extra_parents,raw_parents)
    if np.any(parent_positions==MISSING):raise ValueError('Every extra official site parent required')
    nb=(len(extra_sites)+block_size-1)//block_size
    if base_blocks+nb>=MISSING:raise ValueError('Every extra block position required')
    block_offsets=np.minimum(np.arange(nb+1,dtype=np.uint64)*block_size,len(extra_sites))
    blocks=(base_blocks+np.arange(len(extra_sites),dtype=np.uint32)//block_size).astype(np.uint32)
    # Compact parent→extra-members and parent→extra-blocks. No alias×sites.
    site_ids=(len(current_sites)+np.arange(len(extra_sites),dtype=np.uint32)).astype(np.uint32)
    member_pairs=(parent_positions.astype(np.uint64)<<np.uint64(32))|site_ids
    order=np.argsort(member_pairs,kind='stable');member_pairs=member_pairs[order]
    starts=np.r_[0,np.flatnonzero((member_pairs[1:]>>np.uint64(32))!=(member_pairs[:-1]>>np.uint64(32)))+1] if len(member_pairs) else np.empty(0,dtype=np.int64)
    bridge_parents=(member_pairs[starts]>>np.uint64(32)).astype(np.uint32)
    member_offsets=np.r_[starts,len(member_pairs)].astype(np.uint64)
    members=(member_pairs&np.uint64(MISSING)).astype(np.uint32)
    block_pairs=np.unique((parent_positions.astype(np.uint64)<<np.uint64(32))|blocks)
    starts=np.r_[0,np.flatnonzero((block_pairs[1:]>>np.uint64(32))!=(block_pairs[:-1]>>np.uint64(32)))+1] if len(block_pairs) else np.empty(0,dtype=np.int64)
    if not np.array_equal(bridge_parents,(block_pairs[starts]>>np.uint64(32)).astype(np.uint32)):
        raise ValueError('Every extra parent must have members AND blocks')
    bridge_offsets=np.r_[starts,len(block_pairs)].astype(np.uint64);bridge_blocks=(block_pairs&np.uint64(MISSING)).astype(np.uint32)
    arrays=dict(extra_parent_keys=extra_parents,extra_site_parent_positions=parent_positions,
        extra_site_blocks=blocks,extra_block_offsets=block_offsets,bridge_parent_positions=bridge_parents,
        extra_parent_member_offsets=member_offsets,extra_parent_members=members,
        extra_parent_block_offsets=bridge_offsets,extra_parent_blocks=bridge_blocks)
    for name,value in arrays.items():save(output,name,value)
    # Full source identities through the virtual lookup; not a sampled or
    # checksum oracle. Current physical positions must be bit-unchanged.
    for begin in range(0,len(current_sites),131072):
        end=min(len(current_sites),begin+131072)
        if not np.array_equal(positions(current_sites,extra_sites,current_sites[begin:end]),np.arange(begin,end,dtype=np.uint32)):
            raise ValueError('Every current site position must be unchanged')
    if not np.array_equal(positions(current_sites,extra_sites,extra_sites),site_ids):raise ValueError('Every extra official site must be admitted exactly once')
    recovered=np.empty(len(raw_parents),dtype=np.uint32);native=parent_positions<len(current_parents)
    recovered[native]=current_parents[parent_positions[native]];recovered[~native]=extra_parents[parent_positions[~native]-len(current_parents)]
    if not np.array_equal(recovered,raw_parents):raise ValueError('Every original extra site parent must reconstruct exactly')
    # Reconstruct every member tuple and every block tuple through each CSR.
    parent=np.repeat(bridge_parents,np.diff(member_offsets).astype(np.int64));local=members-len(current_sites)
    if (not np.array_equal(np.sort(members),site_ids) or not np.array_equal(parent,parent_positions[local])
        or np.any(blocks[local]<base_blocks)):raise ValueError('Every extra parent/member CSR tuple must reconstruct exactly')
    parent=np.repeat(bridge_parents,np.diff(bridge_offsets).astype(np.int64))
    reconstructed=(parent.astype(np.uint64)<<np.uint64(32))|bridge_blocks
    if not np.array_equal(reconstructed,block_pairs):raise ValueError('Every extra parent/block CSR tuple must reconstruct exactly')
    if (int(block_offsets[-1])!=len(extra_sites) or np.any(np.diff(block_offsets)>block_size)
        or np.any(blocks!=base_blocks+np.arange(len(extra_sites),dtype=np.uint32)//block_size)):
        raise ValueError('Every extra physical site/block position required')
    for name,value in arrays.items():
        exported=np.load(Path(output)/(name+'.npy'),mmap_mode='r',allow_pickle=False)
        if exported.dtype!=value.dtype or not np.array_equal(exported,value):raise ValueError('Every exported virtual catalog array must reconstruct exactly')
    return dict(current_sites=len(current_sites),extra_sites=len(extra_sites),sites=len(current_sites)+len(extra_sites),
        current_parents=len(current_parents),extra_parents=len(extra_parents),parents=len(current_parents)+len(extra_parents),
        current_blocks=base_blocks,extra_blocks=nb,blocks=base_blocks+nb,extra_parent_block_references=len(block_pairs),
        every_current_site_position_unchanged=True,every_extra_site_admitted_once=True,every_extra_parent_exact=True,
        every_extra_member_and_block_csr_exact=True,every_exported_array_exact=True,
        unknown_extra_attributes_not_inferred=True,source_or_site_or_member_cap=None,
        site_cartesian_product_materialized=False,identifiers_as_features=False,
        scope='CURRENT_V96_PLUS_EVERY_SEALED_V135_ABSENT_OFFICIAL_SITE_NOT_ALL_UNINVENTORIED_SOURCES')


class VirtualCatalog:
    """Readonly complete V136 namespace; source-authority checks at opening.

    This is an identity/membership/block bridge, NOT a query scorer. Opening
    hashes sources once; no per-query rehash, hidden pool or fanout cap.
    """
    def __init__(self,reference):
        path=verify(reference);m=json.loads(path.read_text());self.metadata=m
        if (m['status']!='SEALED_V136_CURRENT_PLUS_ALL_V135_OFFICIAL_SITE_NAMESPACE'
            or any(m.get(k) is not False for k in ('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened'))
            or m['complete_retrieval_index'] or not m['stats']['every_current_site_position_unchanged']
            or not m['stats']['every_extra_site_admitted_once']):raise ValueError('Protected exact virtual namespace required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
        self.current={k:np.load(verify(ref),mmap_mode='r',allow_pickle=False) for k,ref in m['input_arrays'].items()}
        self.extra={k.removesuffix('.npy'):np.load(verify(ref),mmap_mode='r',allow_pickle=False) for k,ref in m['artifacts'].items()}
        self.extra_keys=np.load(verify(m['extra_sites']),mmap_mode='r',allow_pickle=False)
        self.base_sites=m['stats']['current_sites'];self.base_parents=m['stats']['current_parents']
        self.sites=m['stats']['sites'];self.parents=m['stats']['parents'];self.blocks=m['stats']['blocks']
        if (len(self.current['endpoint_keys'])!=self.base_sites or len(self.current['parent_keys'])!=self.base_parents
            or len(self.extra_keys)+self.base_sites!=self.sites):raise ValueError('Whole virtual namespace coverage required')

    def site_positions(self,keys):return positions(self.current['endpoint_keys'],self.extra_keys,keys)
    def parent_positions(self,keys):return positions(self.current['parent_keys'],self.extra['extra_parent_keys'],keys)

    @staticmethod
    def identity(value,width):
        if not isinstance(value,str) or len(value)!=width or not value.isascii() or not value.isdecimal():raise ValueError('Explicit official identity grain required')
        return int(value)

    def endpoint_position(self,siret):
        p=int(self.site_positions(np.array([self.identity(siret,14)],dtype=np.uint64))[0]);return None if p==MISSING else p

    def parent_position(self,siren):
        p=int(self.parent_positions(np.array([self.identity(siren,9)],dtype=np.uint32))[0]);return None if p==MISSING else p

    @staticmethod
    def bounded(value,maximum):
        if not isinstance(value,(int,np.integer)) or isinstance(value,(bool,np.bool_)) or not 0<=int(value)<maximum:
            raise ValueError('Actual virtual position required')
        return int(value)

    def site_key(self,position):
        p=self.bounded(position,self.sites)
        return int(self.current['endpoint_keys'][p]) if p<self.base_sites else int(self.extra_keys[p-self.base_sites])

    def parent_key(self,position):
        p=self.bounded(position,self.parents)
        return int(self.current['parent_keys'][p]) if p<self.base_parents else int(self.extra['extra_parent_keys'][p-self.base_parents])

    def site_block(self,position):
        p=self.bounded(position,self.sites)
        return int(self.current['site_blocks'][p]) if p<self.base_sites else int(self.extra['extra_site_blocks'][p-self.base_sites])

    def site_parent_position(self,position):
        p=self.bounded(position,self.sites)
        if p>=self.base_sites:return int(self.extra['extra_site_parent_positions'][p-self.base_sites])
        return int(self.parent_positions(self.current['endpoint_parents'][p:p+1])[0])

    def state_code(self,position):
        p=self.bounded(position,self.sites)
        return int(self.current['endpoint_states'][p]) if p<self.base_sites else 0

    def extra_range(self,parent_position,prefix):
        keys=self.extra['bridge_parent_positions'];pos=int(np.searchsorted(keys,np.uint32(parent_position)))
        if pos==len(keys) or int(keys[pos])!=parent_position:return self.extra[prefix][:0]
        offsets=self.extra[prefix.removesuffix('s')+'_offsets'];lo,hi=map(int,offsets[pos:pos+2]);return self.extra[prefix][lo:hi]

    def members_for_parent(self,position):
        p=self.bounded(position,self.parents);native=self.current['parent_endpoints'][:0]
        if p<self.base_parents:
            lo,hi=map(int,self.current['parent_offsets'][p:p+2]);native=self.current['parent_endpoints'][lo:hi]
        return np.concatenate((native,self.extra_range(p,'extra_parent_members')))

    def blocks_for_parent(self,position):
        p=self.bounded(position,self.parents);native=self.current['parent_blocks'][:0]
        if p<self.base_parents:
            lo,hi=map(int,self.current['parent_block_offsets'][p:p+2]);native=self.current['parent_blocks'][lo:hi]
        return np.concatenate((native,self.extra_range(p,'extra_parent_blocks')))

    def sites_in_block(self,position):
        p=self.bounded(position,self.blocks);base=self.metadata['stats']['current_blocks']
        if p<base:
            lo,hi=map(int,self.current['block_offsets'][p:p+2]);return self.current['block_sites'][lo:hi]
        lo,hi=map(int,self.extra['extra_block_offsets'][p-base:p-base+2])
        return np.arange(self.base_sites+lo,self.base_sites+hi,dtype=np.uint32)
