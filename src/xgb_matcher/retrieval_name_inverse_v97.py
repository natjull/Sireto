"""Exact complete name provenance, isolated scratch per storage partition."""
from pathlib import Path

import numpy as np


def inverse_part(mapping_path,part,output,n):
    output=Path(output); mapping=np.load(mapping_path,mmap_mode='r',allow_pickle=False)
    if mapping.dtype!=np.uint32 or len(mapping)!=n: raise ValueError('Original mapping schema drift')
    lo=part['fact_start']; hi=lo+part['facts']; bucket=part['bucket']
    if not 0<=lo<hi<2**32: raise ValueError('Part fact range drift')
    chosen=np.flatnonzero((mapping>=np.uint32(lo))&(mapping<np.uint32(hi)))
    if len(chosen)!=part['observations']: raise ValueError('Full source part denominator drift')
    ordered=chosen[np.argsort(mapping[chosen],kind='stable')].astype(np.uint32)
    del chosen
    counts=np.bincount(mapping[ordered]-np.uint32(lo),minlength=part['facts'])
    if len(counts)!=part['facts'] or np.any(counts==0) or int(counts.sum())!=len(ordered):
        raise ValueError('Unreferenced fact or inverse denominator drift')
    offsets=np.r_[0,np.cumsum(counts,dtype=np.uint64)].astype(np.uint32)
    np.save(output/f'inverse_offsets{bucket:02d}.npy',offsets,allow_pickle=False)
    np.save(output/f'inverse_ordinals{bucket:02d}.npy',ordered,allow_pickle=False)
    return dict(bucket=bucket,facts=part['facts'],observations=len(ordered),max_observations_per_fact=int(counts.max()))


def assemble(mapping_path,parts,output,n):
    output=Path(output); mapping=np.load(mapping_path,mmap_mode='r',allow_pickle=False)
    nf=sum(p['facts'] for p in parts)
    if len(mapping)!=n or mapping.dtype!=np.uint32: raise ValueError('Original mapping schema drift')
    for start in range(0,n,1048576):
        if np.any(mapping[start:start+1048576]>=np.uint32(nf)): raise ValueError('Incomplete original source mapping')
    offsets=np.lib.format.open_memmap(output/'fact_offsets.npy',mode='w+',dtype=np.uint32,shape=(nf+1,))
    inverse=np.lib.format.open_memmap(output/'fact_observations.npy',mode='w+',dtype=np.uint32,shape=(n,))
    seen=np.zeros(n,dtype=bool); fact_start=0; obs_start=0; maximum=0
    for part in parts:
        if part['fact_start']!=fact_start: raise ValueError('Noncontiguous full fact catalog')
        bucket=part['bucket']; local=np.load(output/f'inverse_offsets{bucket:02d}.npy',mmap_mode='r',allow_pickle=False)
        ids=np.load(output/f'inverse_ordinals{bucket:02d}.npy',mmap_mode='r',allow_pickle=False)
        if (local.dtype!=np.uint32 or ids.dtype!=np.uint32 or len(local)!=part['facts']+1 or local[0]!=0
                or int(local[-1])!=part['observations'] or len(ids)!=part['observations'] or np.any(ids>=n)):
            raise ValueError('Inverse part schema/denominator drift')
        counts=np.diff(local)
        if np.any(counts==0): raise ValueError('Unreferenced fact')
        expected=np.repeat(np.arange(fact_start,fact_start+part['facts'],dtype=np.uint32),counts)
        if not np.array_equal(mapping[ids],expected): raise ValueError('Independent mapping/inverse mismatch')
        seen[ids]=True
        if np.count_nonzero(seen)!=obs_start+len(ids): raise ValueError('Duplicate or missing source ordinal')
        offsets[fact_start:fact_start+len(local)]=local+np.uint32(obs_start)
        inverse[obs_start:obs_start+len(ids)]=ids
        fact_start+=part['facts']; obs_start+=len(ids); maximum=max(maximum,int(counts.max()))
        del expected,counts,local,ids
    offsets.flush(); inverse.flush()
    if fact_start!=nf or obs_start!=n or not np.all(seen): raise ValueError('Complete original source coverage required')
    return dict(observations_indexed=n,facts=nf,max_observations_per_fact=maximum,alias_observation_cap=None,
        independent_inverse_mapping_equality=True,all_original_ordinals_exactly_once=True)
