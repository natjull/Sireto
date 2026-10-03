"""Exact V131 CSR packing, with unsigned/vectorized audit lookups.

The source and all semantics are unchanged. V131's Python-int search into a
uint64 array can promote/copy the WHOLE array on EACH vocabulary run. This
adapter uses one native uint64 vector search per batch. Every run is still
reconstructed against an independent, complete source re-read.
"""
from pathlib import Path

import numpy as np
import pyarrow.compute as pc

from src.xgb_matcher.retrieval_name_access_v131 import write_array
from src.xgb_matcher.retrieval_variant_factor_v119 import literal


def run_positions(offsets,pos,ends):
    begins=np.r_[np.uint64(0),np.asarray(ends[:-1],dtype=np.uint64)]
    return np.searchsorted(offsets,np.uint64(pos)+begins,side='right')-1


def pack(c,files,output,channel,bucket,buckets,expected,names):
    output=Path(output)
    if files:
        source='read_parquet(['+','.join(literal(p) for p in files)+'],hive_partitioning=false)'
        c.execute('CREATE TEMP VIEW input134 AS SELECT * FROM '+source+' WHERE channel='+str(channel))
    else:c.execute('CREATE TEMP VIEW input134 AS SELECT NULL::UINTEGER alias_id,NULL::VARCHAR token,NULL::UTINYINT channel,NULL::UTINYINT term_bucket WHERE FALSE')
    count,bad=c.execute('SELECT count(*),count(*) FILTER(WHERE alias_id IS NULL OR alias_id>='+str(names)+
        ' OR token IS NULL OR term_bucket IS NULL OR term_bucket<>'+str(bucket)+' OR term_bucket<>hash(token)%'+str(buckets)+') FROM input134').fetchone()
    if count!=expected or bad:raise ValueError('Whole channel/storage bucket/range required')
    nt,nb=c.execute('SELECT count(*),coalesce(sum(length(token)),0) FROM (SELECT DISTINCT token FROM input134)').fetchone()
    ids=write_array(output,'post_ids',np.uint32,(count,));data=write_array(output,'token_data',np.uint8,(nb,))
    offsets=write_array(output,'token_offsets',np.uint64,(nt+1,));posts=write_array(output,'post_offsets',np.uint64,(nt+1,))
    offsets[0]=posts[0]=0;pos=0;term=-1;byte=0;last=None;last_id=None
    cursor=c.execute('SELECT token,alias_id FROM input134 ORDER BY token,alias_id').fetch_record_batch(rows_per_batch=131072)
    for batch in cursor:
        values=batch.column(1).to_numpy(zero_copy_only=False);tokens=batch.column(0)
        encoded=pc.run_end_encode(tokens);ends=encoded.run_ends.to_numpy(zero_copy_only=False);labels=encoded.values.to_pylist();begin=0
        ids[pos:pos+len(batch)]=values
        for token,end in zip(labels,ends):
            chunk=values[begin:end]
            if token==last:
                if len(chunk) and last_id>=int(chunk[0]):raise ValueError('Posting duplicates/order drift across chunks')
            else:
                if last is not None and token<=last:raise ValueError('Vocabulary must be strictly ordered')
                term+=1;raw=token.encode('ascii');data[byte:byte+len(raw)]=np.frombuffer(raw,dtype=np.uint8)
                byte+=len(raw);offsets[term+1]=byte;posts[term]=pos+begin;last=token
            if len(chunk)>1 and np.any(chunk[1:]<=chunk[:-1]):raise ValueError('Every per-term alias must be unique/sorted')
            last_id=int(chunk[-1]);begin=int(end)
        pos+=len(batch)
    posts[-1]=pos
    if pos!=count or term+1!=nt or byte!=nb:raise ValueError('Every vocabulary/posting position required')
    for a in (ids,data,offsets,posts):a.flush()
    pos=0;cursor=c.execute('SELECT token,alias_id FROM input134 ORDER BY token,alias_id').fetch_record_batch(rows_per_batch=131072)
    for batch in cursor:
        values=batch.column(1).to_numpy(zero_copy_only=False)
        if not np.array_equal(ids[pos:pos+len(batch)],values):raise ValueError('All posting IDs must reconstruct exactly')
        runs=pc.run_end_encode(batch.column(0));ends=runs.run_ends.to_numpy(zero_copy_only=False);labels=runs.values.to_pylist()
        keys=run_positions(posts,pos,ends)
        for token,end,key in zip(labels,ends,keys):
            lo,hi=map(int,offsets[key:key+2])
            if data[lo:hi].tobytes()!=token.encode('ascii') or pos+int(end)>int(posts[key+1]):raise ValueError('Every posting token/CSR boundary must reconstruct exactly')
        pos+=len(batch)
    if pos!=count:raise ValueError('Whole posting audit incomplete')
    return dict(channel=channel,bucket=bucket,postings=count,terms=nt,token_bytes=nb,
        every_original_posting_and_token_reconstructed_exactly=True,all_postings_sorted_unique=True,
        storage_hash_only=True,raw_alias_ids_not_features=True,no_posting_or_owner_cap=True,
        native_unsigned_batched_csr_audit=True,source_semantics_unchanged=True)
