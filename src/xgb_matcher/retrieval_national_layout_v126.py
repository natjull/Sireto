"""All-site, query-independent Morton block layout; no name or candidate selection."""
from pathlib import Path
import time

import numpy as np


def spread(value):
    value=np.asarray(value,dtype=np.uint64).copy()
    for shift,mask in ((16,0x0000FFFF0000FFFF),(8,0x00FF00FF00FF00FF),
                       (4,0x0F0F0F0F0F0F0F0F),(2,0x3333333333333333),(1,0x5555555555555555)):
        value=(value|(value<<shift))&np.uint64(mask)
    return value


def layout(xy,root,block_size=128,progress=lambda row:None):
    if not isinstance(block_size,int) or isinstance(block_size,bool) or block_size<=0:raise ValueError('Positive block size required')
    root=Path(root);n=len(xy)
    if not n or n>=2**32 or xy.shape!=(n,2):raise ValueError('National coordinate/position shape drift')
    known=np.flatnonzero(np.isfinite(xy[:,0]));unknown=np.flatnonzero(~np.isfinite(xy[:,0]));m=len(known)
    codes=np.empty(m,dtype=np.uint64);began=time.monotonic()
    for start in range(0,m,131072):
        ids=known[start:start+131072];points=np.asarray(xy[ids])
        if (not np.isfinite(points).all() or np.any(points[:,0]<0) or np.any(points[:,0]>1300000)
            or np.any(points[:,1]<6000000) or np.any(points[:,1]>7300000)):raise ValueError('Validated metropolitan L93 required')
        x=np.floor(points[:,0]).astype(np.uint64);y=np.floor(points[:,1]-6000000).astype(np.uint64)
        codes[start:start+len(ids)]=spread(x)|(spread(y)<<1)
    order=np.argsort(codes,kind='stable');del codes
    sites=np.lib.format.open_memmap(root/'block_sites.npy',mode='w+',dtype=np.uint32,shape=(n,))
    sites[:m]=known[order];sites[m:]=unknown;sites.flush();del known,unknown,order
    offsets=np.r_[np.arange(0,m,block_size,dtype=np.uint64),m+np.arange(0,n-m,block_size,dtype=np.uint64),np.uint64(n)]
    np.save(root/'block_offsets.npy',offsets,allow_pickle=False);nb=len(offsets)-1
    lo=np.lib.format.open_memmap(root/'block_lo.npy',mode='w+',dtype=np.float64,shape=(nb,2))
    hi=np.lib.format.open_memmap(root/'block_hi.npy',mode='w+',dtype=np.float64,shape=(nb,2))
    for b in range(0,nb,1024):
        stop=min(nb,b+1024);start=int(offsets[b]);end=int(offsets[stop]);points=np.asarray(xy[sites[start:end]])
        starts=(offsets[b:stop]-offsets[b]).astype(np.int64)
        lo[b:stop]=np.fmin.reduceat(points,starts,axis=0);hi[b:stop]=np.fmax.reduceat(points,starts,axis=0)
    lo.flush();hi.flush()
    # Exact once-only coverage, no probabilistic checksum substitutes.
    seen=np.zeros(n,dtype=np.bool_)
    for start in range(0,n,131072):
        ids=sites[start:start+131072]
        if np.any(ids>=n) or np.any(seen[ids]) or len(np.unique(ids))!=len(ids):raise ValueError('Duplicate/lost physical site')
        seen[ids]=True
    if not np.all(seen):raise ValueError('Incomplete physical site denominator')
    stats=dict(sites=n,known_xy=m,unknown_xy=n-m,blocks=nb,block_size=block_size,
        max_block_sites=int(np.max(np.diff(offsets))),all_sites_exactly_once=True,closed_sites_filtered=False,
        unknown_geometry_sites_filtered=False,selected_by_query_or_truth=False,layout='MORTON_FLOORED_METER_PHYSICAL_ONLY',
        score_coordinates_quantized=False,seconds=time.monotonic()-began)
    progress(dict(phase='all_site_physical_layout_complete',**stats));return stats


def read_coordinates(connection,source,keys,root,progress=lambda row:None):
    """One complete official source scan, exact membership join; never WHERE-filter."""
    root=Path(root);n=len(keys)
    if keys.dtype!=np.uint64 or np.any(keys[1:]<=keys[:-1]):raise ValueError('Sorted unique typed official keys required')
    xy=np.lib.format.open_memmap(root/'site_xy.npy',mode='w+',dtype=np.float64,shape=(n,2));xy[:]=np.nan
    seen=np.zeros(n,dtype=np.bool_);count=0;began=time.monotonic()
    sql='''WITH source AS (SELECT CAST(siret AS VARCHAR) raw_key,
        trim(coalesce(codeCommuneEtablissement,'')) insee,
        TRY_CAST(coordonneeLambertAbscisseEtablissement AS DOUBLE) x,
        TRY_CAST(coordonneeLambertOrdonneeEtablissement AS DOUBLE) y FROM read_parquet(?))
        SELECT TRY_CAST(raw_key AS UBIGINT) site_key,regexp_full_match(raw_key,'[0-9]{14}') valid_key,
        CASE WHEN insee<>'' AND NOT starts_with(insee,'97') AND NOT starts_with(insee,'98')
            AND isfinite(x) AND isfinite(y) AND x>=0 AND x<=1300000 AND y>=6000000 AND y<=7300000 THEN x END x,
        CASE WHEN insee<>'' AND NOT starts_with(insee,'97') AND NOT starts_with(insee,'98')
            AND isfinite(x) AND isfinite(y) AND x>=0 AND x<=1300000 AND y>=6000000 AND y<=7300000 THEN y END y FROM source'''
    reader=connection.execute(sql,[str(source)]).fetch_record_batch(131072)
    for batch in reader:
        if batch.column(0).null_count or batch.column(1).null_count or not np.all(batch.column(1).to_numpy(zero_copy_only=False)):
            raise ValueError('Malformed official site identity')
        ids=batch.column(0).to_numpy(zero_copy_only=False)
        pos=np.searchsorted(keys,ids)
        if np.any(pos>=n) or not np.array_equal(keys[pos],ids):raise ValueError('Official source/membership identity mismatch')
        if np.any(seen[pos]) or len(np.unique(pos))!=len(pos):raise ValueError('Duplicate official source site')
        seen[pos]=True;xy[pos,0]=batch.column(2).to_numpy(zero_copy_only=False);xy[pos,1]=batch.column(3).to_numpy(zero_copy_only=False)
        count+=len(batch)
    if count!=n or not np.all(seen):raise ValueError('Missing official source site, denominator not filtered')
    xy.flush();progress(dict(phase='all_current_site_coordinates_joined',sites=n,seconds=time.monotonic()-began));return xy
