"""Append all V140 official identities without renumbering existing proofs.

Three independently sorted key segments; positions are not globally sorted
by identity. They are opaque joins, never similarity features. New sites
have UNKNOWN attributes until an explicit site-scoped source provides them.
"""
import json
from pathlib import Path

import numpy as np

from src.xgb_matcher.retrieval_extended_catalog_v136 import MISSING,ordered,VirtualCatalog
from src.xgb_matcher.retrieval_proof_bridge_v137 import CatalogKeys,PROTECTED
from src.xgb_matcher.retrieval_v20_workbench import verify


def append_positions(base,extra,values,previous):
    """Lookup only unresolved values, preserving every old physical position."""
    if (values.dtype!=extra.dtype or previous.dtype!=np.uint32 or values.ndim!=1
        or previous.shape!=values.shape or not isinstance(base,int) or isinstance(base,bool)
        or base<0 or base+len(extra)>=MISSING or np.any((previous!=MISSING)&(previous>=base))):
        raise ValueError('Exact previous positions and typed append keys required')
    result=previous.copy();missing=np.flatnonzero(result==MISSING)
    p=np.searchsorted(extra,values[missing]);valid=p<len(extra)
    rows=np.flatnonzero(valid);rows=rows[extra[p[rows]]==values[missing[rows]]]
    result[missing[rows]]=(base+p[rows]).astype(np.uint32)
    return result


def append_build(catalog,additions,base_blocks,block_size,output):
    ordered(additions,np.uint64,10**14)
    if (not isinstance(base_blocks,int) or isinstance(base_blocks,bool) or base_blocks<0
        or not isinstance(block_size,int) or isinstance(block_size,bool) or block_size<=0
        or catalog.sites+len(additions)>=MISSING or catalog.parents+len(additions)>=MISSING):
        raise ValueError('Whole append namespace and physical block range required')
    if np.any(catalog.site_positions(additions)!=MISSING):
        raise ValueError('Every addition must be absent from ALL previous key segments')
    raw_parents=(additions//np.uint64(100000)).astype(np.uint32)
    unique=np.unique(raw_parents);new_parents=unique[catalog.parent_positions(unique)==MISSING]
    pp=append_positions(catalog.parents,new_parents,raw_parents,catalog.parent_positions(raw_parents))
    if np.any(pp==MISSING):raise ValueError('Every appended site parent required')
    nb=(len(additions)+block_size-1)//block_size
    if base_blocks+nb>=MISSING:raise ValueError('Whole appended physical block range required')
    offsets=np.minimum(np.arange(nb+1,dtype=np.uint64)*block_size,len(additions))
    blocks=(np.uint64(base_blocks)+np.arange(len(additions),dtype=np.uint64)//block_size).astype(np.uint32)
    members=(np.uint64(catalog.sites)+np.arange(len(additions),dtype=np.uint64)).astype(np.uint32)
    pairs=(pp.astype(np.uint64)<<np.uint64(32))|members
    pairs=np.sort(pairs,kind='stable')
    starts=(np.r_[0,np.flatnonzero((pairs[1:]>>np.uint64(32))!=(pairs[:-1]>>np.uint64(32)))+1]
        if len(pairs) else np.empty(0,dtype=np.int64))
    bridge=(pairs[starts]>>np.uint64(32)).astype(np.uint32)
    mo=np.r_[starts,len(pairs)].astype(np.uint64);mm=(pairs&np.uint64(MISSING)).astype(np.uint32)
    bpairs=np.unique((pp.astype(np.uint64)<<np.uint64(32))|blocks)
    bstarts=(np.r_[0,np.flatnonzero((bpairs[1:]>>np.uint64(32))!=(bpairs[:-1]>>np.uint64(32)))+1]
        if len(bpairs) else np.empty(0,dtype=np.int64))
    if not np.array_equal(bridge,(bpairs[bstarts]>>np.uint64(32)).astype(np.uint32)):
        raise ValueError('Every appended parent member/block group required')
    bo=np.r_[bstarts,len(bpairs)].astype(np.uint64);bb=(bpairs&np.uint64(MISSING)).astype(np.uint32)
    arrays=dict(new_parent_keys=new_parents,new_site_parent_positions=pp,new_site_blocks=blocks,
        new_block_offsets=offsets,bridge_parent_positions=bridge,new_parent_member_offsets=mo,
        new_parent_members=mm,new_parent_block_offsets=bo,new_parent_blocks=bb)
    dest=Path(output)
    for name,value in arrays.items():np.save(dest/(name+'.npy'),value,allow_pickle=False)
    exported={k:np.load(dest/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in arrays}
    for k,value in arrays.items():
        if exported[k].dtype!=value.dtype or not np.array_equal(exported[k],value):
            raise ValueError('Every appended exported array required unchanged')
    # Full inverse checks, not sampled/hash-only membership assertions.
    got=append_positions(catalog.sites,additions,additions,catalog.site_positions(additions))
    if not np.array_equal(got,members):raise ValueError('Every appended site admitted exactly once')
    parent=exported['new_site_parent_positions'];known=parent<catalog.parents
    recovered=np.empty(len(parent),dtype=np.uint32)
    recovered[known]=catalog.reconstruct(parent[known],'parent')
    recovered[~known]=exported['new_parent_keys'][parent[~known]-catalog.parents]
    if not np.array_equal(recovered,raw_parents):raise ValueError('Every appended site parent exact')
    recovered_parent=np.repeat(exported['bridge_parent_positions'],np.diff(exported['new_parent_member_offsets']).astype(np.int64))
    recovered_members=exported['new_parent_members']
    if (not np.array_equal(np.sort(recovered_members),members)
        or not np.array_equal(recovered_parent,parent[recovered_members-catalog.sites])):
        raise ValueError('Every appended member CSR tuple required')
    recovered_parent=np.repeat(exported['bridge_parent_positions'],np.diff(exported['new_parent_block_offsets']).astype(np.int64))
    if not np.array_equal((recovered_parent.astype(np.uint64)<<np.uint64(32))|exported['new_parent_blocks'],bpairs):
        raise ValueError('Every appended block CSR tuple required')
    if (exported['new_block_offsets'][0]!=0 or int(exported['new_block_offsets'][-1])!=len(additions)
        or np.any(np.diff(exported['new_block_offsets'])>block_size)
        or not np.array_equal(exported['new_site_blocks'],blocks)):
        raise ValueError('Every appended physical site/block required')
    # Check every old site AND parent position through the new three-segment
    # lookup. In particular, previously appended V135 identities cannot move.
    for grain,segments in (('site',(catalog.site,catalog.extra_site)),('parent',(catalog.parent,catalog.extra_parent))):
        base=catalog.sites if grain=='site' else catalog.parents
        extra=additions if grain=='site' else new_parents
        lookup=catalog.site_positions if grain=='site' else catalog.parent_positions
        start=0
        for segment in segments:
            for begin in range(0,len(segment),131072):
                end=min(len(segment),begin+131072);values=segment[begin:end]
                result=append_positions(base,extra,values,lookup(values))
                if not np.array_equal(result,np.arange(start+begin,start+end,dtype=np.uint32)):
                    raise ValueError('Every previous site AND parent position must stay unchanged')
            start+=len(segment)
    return dict(previous_sites=catalog.sites,previous_parents=catalog.parents,previous_blocks=base_blocks,
        additional_sites=len(additions),additional_parents=len(new_parents),additional_blocks=nb,
        sites=catalog.sites+len(additions),parents=catalog.parents+len(new_parents),blocks=base_blocks+nb,
        additional_parent_block_references=len(bpairs),every_previous_site_and_parent_position_unchanged=True,
        every_new_site_admitted_once=True,every_new_site_parent_exact=True,every_member_and_block_csr_exact=True,
        every_exported_array_exact=True,unknown_attributes_not_inferred=True,
        site_or_parent_or_member_cap=None,identifiers_as_features=False,site_cartesian_product_materialized=False)


class AppendKeys:
    """Three-segment typed identity lookups; existing V137 proofs stay valid."""
    def __init__(self,reference):
        m=json.loads(verify(reference).read_text());self.metadata=m
        if (m['status']!='SEALED_V141_APPEND_ALL_V140_OFFICIAL_SITE_IDENTITIES'
            or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index'] or m['ready_for_fold1']
            or not m['stats']['every_previous_site_and_parent_position_unchanged']
            or not m['stats']['every_new_site_admitted_once'] or not m['resources']['pass_resource_policy']):
            raise ValueError('Protected immutable complete append namespace required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
        self.previous=CatalogKeys(m['previous_catalog'])
        self.new_site=np.load(verify(m['additional_sites']),mmap_mode='r',allow_pickle=False)
        self.new_parent=np.load(verify(m['artifacts']['new_parent_keys.npy']),mmap_mode='r',allow_pickle=False)
        self.sites=m['stats']['sites'];self.parents=m['stats']['parents']
        if (self.sites!=self.previous.sites+len(self.new_site) or self.parents!=self.previous.parents+len(self.new_parent)
            or m['stats']['previous_sites']!=self.previous.sites or m['stats']['previous_parents']!=self.previous.parents):
            raise ValueError('Exact full append authority required')
        ordered(self.new_site,np.uint64,10**14);ordered(self.new_parent,np.uint32,10**9)
        if np.any(self.previous.site_positions(self.new_site)!=MISSING) or np.any(self.previous.parent_positions(self.new_parent)!=MISSING):
            raise ValueError('No duplicate previous identity permitted')
        self.references=(*self.previous.references,m['additional_sites'],m['artifacts']['new_parent_keys.npy'])

    def site_positions(self,keys):
        return append_positions(self.previous.sites,self.new_site,keys,self.previous.site_positions(keys))

    def parent_positions(self,keys):
        return append_positions(self.previous.parents,self.new_parent,keys,self.previous.parent_positions(keys))

    def reconstruct(self,pos,grain):
        if grain not in ('site','parent'):raise ValueError('Explicit identity grain required')
        base=self.previous.sites if grain=='site' else self.previous.parents
        extra=self.new_site if grain=='site' else self.new_parent
        if pos.dtype!=np.uint32 or pos.ndim!=1 or np.any(pos>=base+len(extra)):
            raise ValueError('Every actual appended position required')
        result=np.empty(len(pos),dtype=extra.dtype);old=pos<base
        result[old]=self.previous.reconstruct(pos[old],grain);result[~old]=extra[pos[~old]-base]
        return result


class AppendCatalog:
    """Readonly membership/layout wrapper, not a scorer or fixed candidate pool."""
    def __init__(self,reference):
        self.keys=AppendKeys(reference);self.metadata=self.keys.metadata
        self.previous=VirtualCatalog(self.metadata['previous_catalog'])
        self.new={k.removesuffix('.npy'):np.load(verify(ref),mmap_mode='r',allow_pickle=False)
            for k,ref in self.metadata['artifacts'].items()}
        self.sites=self.keys.sites;self.parents=self.keys.parents;self.blocks=self.metadata['stats']['blocks']
        if (self.previous.sites!=self.metadata['stats']['previous_sites']
            or self.previous.blocks!=self.metadata['stats']['previous_blocks']
            or self.new['new_site_parent_positions'].shape!=(len(self.keys.new_site),)):
            raise ValueError('Whole original membership and append block namespace required')

    def site_positions(self,keys):return self.keys.site_positions(keys)
    def parent_positions(self,keys):return self.keys.parent_positions(keys)
    def site_key(self,position):
        p=VirtualCatalog.bounded(position,self.sites)
        return int(self.keys.reconstruct(np.array([p],dtype=np.uint32),'site')[0])
    def parent_key(self,position):
        p=VirtualCatalog.bounded(position,self.parents)
        return int(self.keys.reconstruct(np.array([p],dtype=np.uint32),'parent')[0])
    def endpoint_position(self,siret):
        p=int(self.site_positions(np.array([VirtualCatalog.identity(siret,14)],dtype=np.uint64))[0])
        return None if p==MISSING else p
    def parent_position(self,siren):
        p=int(self.parent_positions(np.array([VirtualCatalog.identity(siren,9)],dtype=np.uint32))[0])
        return None if p==MISSING else p
    def site_block(self,position):
        p=VirtualCatalog.bounded(position,self.sites)
        return self.previous.site_block(p) if p<self.previous.sites else int(self.new['new_site_blocks'][p-self.previous.sites])
    def site_parent_position(self,position):
        p=VirtualCatalog.bounded(position,self.sites)
        return self.previous.site_parent_position(p) if p<self.previous.sites else int(self.new['new_site_parent_positions'][p-self.previous.sites])
    def state_code(self,position):
        p=VirtualCatalog.bounded(position,self.sites)
        return self.previous.state_code(p) if p<self.previous.sites else 0

    def new_range(self,parent_position,prefix):
        keys=self.new['bridge_parent_positions'];p=int(np.searchsorted(keys,np.uint32(parent_position)))
        if p==len(keys) or keys[p]!=parent_position:return self.new[prefix][:0]
        offsets=self.new[prefix.removesuffix('s')+'_offsets'];lo,hi=map(int,offsets[p:p+2])
        return self.new[prefix][lo:hi]

    def members_for_parent(self,position):
        p=VirtualCatalog.bounded(position,self.parents)
        old=self.previous.members_for_parent(p) if p<self.previous.parents else np.empty(0,dtype=np.uint32)
        return np.concatenate((old,self.new_range(p,'new_parent_members')))

    def blocks_for_parent(self,position):
        p=VirtualCatalog.bounded(position,self.parents)
        old=self.previous.blocks_for_parent(p) if p<self.previous.parents else np.empty(0,dtype=np.uint32)
        return np.concatenate((old,self.new_range(p,'new_parent_blocks')))

    def sites_in_block(self,position):
        p=VirtualCatalog.bounded(position,self.blocks)
        if p<self.previous.blocks:return self.previous.sites_in_block(p)
        b=p-self.previous.blocks;lo,hi=map(int,self.new['new_block_offsets'][b:b+2])
        return np.arange(self.previous.sites+lo,self.previous.sites+hi,dtype=np.uint32)
