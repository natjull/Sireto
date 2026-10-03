"""Exact V121 whole-variant channels, guarded feature text and packed postings.

All raw V130 names remain. Storage hashes/positions are NEVER score features.
"""
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_name_postings_v86 import masked_sql
from src.xgb_matcher.retrieval_variant_factor_v119 import literal

ASCII=re.compile(r'[A-Z0-9]+( [A-Z0-9]+)*\Z')


def stage(c,table,output,name_start,row_start,buckets):
    """Every raw name in one physical group, exact scalar set certificates.

    Per-name cardinality + uniqueness + every token is a member of the scalar
    word/substr/exact set => equality of EVERY set, not a checksum sample.
    """
    output=Path(output);n=len(table);local=table['name_id'].to_numpy(zero_copy_only=False)
    if n<=0 or not np.array_equal(local,np.arange(row_start,row_start+n,dtype=np.uint32)) or name_start+row_start+n>=2**32:
        raise ValueError('Every raw name ID/physical position required')
    names=table['name_text'].to_pylist()
    if any(not isinstance(s,str) or not ASCII.fullmatch(s) for s in names):raise ValueError('Original canonical ASCII source names required')
    expected_text=[];counts=np.empty((n,4),dtype=np.uint32)
    for i,text in enumerate(names):
        feature=normalized(text);sets=channels(text);expected_text.append(feature)
        counts[i]=[*(len(s) for s in sets),int(bool(feature))]
    ids=(local.astype(np.uint64)+name_start).astype(np.uint32)
    expected=pa.table(dict(alias_id=pa.array(ids),raw_text=table['name_text'],expected_text=pa.array(expected_text),
        **{f'count{j}':pa.array(counts[:,j]) for j in range(4)}))
    c.register('reference131',expected)
    c.execute('CREATE TEMP TABLE features131 AS SELECT alias_id,'+masked_sql('raw_text')+
        ' feature_text FROM reference131')
    if c.execute('SELECT count(*) FROM features131 f FULL OUTER JOIN reference131 r USING(alias_id) WHERE '
        'f.alias_id IS NULL OR r.alias_id IS NULL OR f.feature_text IS DISTINCT FROM r.expected_text').fetchone()[0]:
        raise ValueError('Every feature text must equal the unchanged scalar guard')
    c.execute("CREATE TEMP VIEW compact131 AS SELECT *,replace(feature_text,' ','_') compact FROM features131")
    clauses=["SELECT alias_id,CAST(0 AS UTINYINT) channel,unnest(list_distinct(string_split(feature_text,' '))) token FROM compact131 WHERE feature_text<>''"]
    for channel,k in ((1,3),(2,4)):
        clauses.append('SELECT alias_id,CAST('+str(channel)+' AS UTINYINT) channel,unnest(list_distinct(list_transform('
            'range(1,greatest(1,length(compact)-'+str(k)+'+2)),i->substr(compact,i,'+str(k)+')))) token FROM compact131')
    clauses.append("SELECT alias_id,CAST(3 AS UTINYINT) channel,feature_text token FROM compact131 WHERE feature_text<>''")
    c.execute('CREATE TEMP TABLE posts131 AS SELECT *,CAST(hash(token)%'+str(buckets)+' AS UTINYINT) term_bucket FROM ('+
        ' UNION ALL '.join(clauses)+") WHERE token<>''")
    total,unique,bad=c.execute("""SELECT count(*),count(DISTINCT(p.alias_id,channel,token)),count(*) FILTER(WHERE
        f.alias_id IS NULL OR CASE channel WHEN 0 THEN NOT list_contains(string_split(feature_text,' '),token)
        WHEN 1 THEN length(token)<>3 OR position(token IN compact)=0
        WHEN 2 THEN length(token)<>4 OR position(token IN compact)=0
        WHEN 3 THEN token<>feature_text OR feature_text='' ELSE TRUE END)
        FROM posts131 p LEFT JOIN compact131 f USING(alias_id)""").fetchone()
    if total!=unique or bad:raise ValueError('Every channel token must be unique and belong to its whole variant')
    c.execute('CREATE TEMP TABLE counts131 AS SELECT alias_id,channel,count(*) n FROM posts131 GROUP BY alias_id,channel')
    for channel in range(4):
        bad=c.execute('SELECT count(*) FROM reference131 r LEFT JOIN counts131 a ON r.alias_id=a.alias_id AND a.channel='+str(channel)+
            ' WHERE r.count'+str(channel)+'<>coalesce(a.n,0)').fetchone()[0]
        if bad:raise ValueError('Every scalar whole-variant channel set must be complete')
    c.execute('COPY (SELECT * FROM features131 ORDER BY alias_id) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)',[str(output/'features.parquet')])
    if total:
        c.execute('COPY posts131 TO '+literal(output/'staging')+' (FORMAT PARQUET,PARTITION_BY(term_bucket),WRITE_PARTITION_COLUMNS true,COMPRESSION ZSTD)')
        files=sorted((output/'staging').glob('term_bucket=*/*.parquet'))
        c.execute('CREATE TEMP VIEW exported131 AS SELECT * FROM read_parquet(['+','.join(literal(p) for p in files)+'],hive_partitioning=false)')
        count,unique=c.execute('SELECT count(*),count(DISTINCT(alias_id,channel,token)) FROM exported131').fetchone()
        bad=c.execute('SELECT count(*) FROM posts131 p FULL OUTER JOIN exported131 e USING(alias_id,channel,token) WHERE '
            'p.alias_id IS NULL OR e.alias_id IS NULL OR p.term_bucket IS DISTINCT FROM e.term_bucket').fetchone()[0]
        if count!=total or unique!=total or bad:raise ValueError('Every exported posting must reconstruct exactly')
    else:files=[]
    feature=pq.read_table(output/'features.parquet',use_threads=False)
    if not feature.equals(pa.table(dict(alias_id=pa.array(ids),feature_text=pa.array(expected_text)))):raise ValueError('Every feature projection must survive export exactly')
    return dict(names=n,feature_empty=int(np.sum(counts[:,3]==0)),postings=total,
        channel_postings={str(j):int(np.sum(counts[:,j],dtype=np.uint64)) for j in range(4)},
        all_scalar_feature_texts_exact=True,all_scalar_channel_sets_exact=True,all_exported_postings_exact=True,
        all_raw_names_retained=True,files=[str(p) for p in files],no_variant_token_bags_mixed=True)


def write_array(output,name,dtype,shape):
    return np.lib.format.open_memmap(Path(output)/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)


def pack(c,files,output,channel,bucket,buckets,expected,names):
    """Sorted streaming compact vocab/CSR. ALL postings audited again exactly."""
    output=Path(output)
    if files:
        source='read_parquet(['+','.join(literal(p) for p in files)+'],hive_partitioning=false)'
        c.execute('CREATE TEMP VIEW input131 AS SELECT * FROM '+source+' WHERE channel='+str(channel))
    else:c.execute('CREATE TEMP VIEW input131 AS SELECT NULL::UINTEGER alias_id,NULL::VARCHAR token,NULL::UTINYINT channel,NULL::UTINYINT term_bucket WHERE FALSE')
    count,bad=c.execute('SELECT count(*),count(*) FILTER(WHERE alias_id IS NULL OR alias_id>='+str(names)+
        ' OR term_bucket<>'+str(bucket)+' OR term_bucket<>hash(token)%'+str(buckets)+') FROM input131').fetchone()
    if count!=expected or bad:raise ValueError('Whole channel/storage bucket/range required')
    nt,nb=c.execute('SELECT count(*),coalesce(sum(length(token)),0) FROM (SELECT DISTINCT token FROM input131)').fetchone()
    ids=write_array(output,'post_ids',np.uint32,(count,));data=write_array(output,'token_data',np.uint8,(nb,))
    offsets=write_array(output,'token_offsets',np.uint64,(nt+1,));posts=write_array(output,'post_offsets',np.uint64,(nt+1,))
    offsets[0]=posts[0]=0;pos=0;term=-1;byte=0;last=None;last_id=None
    cursor=c.execute('SELECT token,alias_id FROM input131 ORDER BY token,alias_id').fetch_record_batch(rows_per_batch=131072)
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
    # Re-read original sorted stream, reconstruct every token through CSR and
    # compare every alias. Not a sample/checksum nor a scalar reused output.
    pos=0;cursor=c.execute('SELECT token,alias_id FROM input131 ORDER BY token,alias_id').fetch_record_batch(rows_per_batch=131072)
    for batch in cursor:
        values=batch.column(1).to_numpy(zero_copy_only=False)
        if not np.array_equal(ids[pos:pos+len(batch)],values):raise ValueError('All posting IDs must reconstruct exactly')
        runs=pc.run_end_encode(batch.column(0));ends=runs.run_ends.to_numpy(zero_copy_only=False);labels=runs.values.to_pylist();begin=0
        for token,end in zip(labels,ends):
            key=int(np.searchsorted(posts,pos+begin,side='right')-1)
            lo,hi=map(int,offsets[key:key+2])
            if data[lo:hi].tobytes()!=token.encode('ascii') or pos+int(end)>int(posts[key+1]):raise ValueError('Every posting token/CSR boundary must reconstruct exactly')
            begin=int(end)
        pos+=len(batch)
    if pos!=count:raise ValueError('Whole posting audit incomplete')
    return dict(channel=channel,bucket=bucket,postings=count,terms=nt,token_bytes=nb,
        every_original_posting_and_token_reconstructed_exactly=True,all_postings_sorted_unique=True,
        storage_hash_only=True,raw_alias_ids_not_features=True,no_posting_or_owner_cap=True)
