"""Complete exact name tuples and all original observation ordinals, including empty context."""
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_name_facts_v85 import literal,MAX_ID

FIELDS=('subject_code','owner_key','official_parent','scope_code','name_text')


def partition(c,projection_sql,output,n,progress=lambda row:None):
    output=Path(output)
    if not 0<n<MAX_ID: raise ValueError('Observation ordinal range drift')
    c.execute('CREATE TEMP VIEW complete_names AS '+projection_sql)
    c.execute("""CREATE TEMP VIEW observations AS SELECT source_observation_index observation_index,
        CAST(CASE WHEN subject_kind='SIREN' THEN siren ELSE siret END AS UBIGINT) owner_key,
        CAST(siren AS UINTEGER) official_parent,
        CAST(CASE subject_kind WHEN 'SIREN' THEN 1 WHEN 'SIRET' THEN 2 END AS UTINYINT) subject_code,
        CAST(CASE WHEN NOT lexical_eligible THEN 0 WHEN lexical_scope='LEGAL_PERSON_NAME' THEN 1
            WHEN lexical_scope='LEGAL_PERSON_COMMERCIAL_NAME' THEN 2 WHEN lexical_scope='EXPLICIT_ENDPOINT_NAME' THEN 3 END
            AS UTINYINT) scope_code,normalized_value name_text,name_grain_valid,
        alias_of_related_target,name_relation_co_validity_proven FROM complete_names""")
    count,bad=c.execute("""SELECT count(*),count(*) FILTER(WHERE name_grain_valid IS NOT TRUE
        OR owner_key IS NULL OR official_parent IS NULL OR subject_code IS NULL OR scope_code IS NULL
        OR observation_index IS NULL OR observation_index>=?
        OR alias_of_related_target IS NOT FALSE OR name_relation_co_validity_proven IS NOT FALSE
        OR (subject_code=1 AND owner_key<>official_parent) OR (subject_code=2 AND owner_key//100000<>official_parent)
        OR (subject_code=1 AND scope_code=3) OR (subject_code=2 AND scope_code NOT IN (0,3))) FROM observations""",[n]).fetchone()
    if count!=n or bad: raise ValueError('Complete observation grain/scope/denominator drift')
    began=time.monotonic()
    c.execute('COPY (SELECT observation_index,'+','.join(FIELDS)+',owner_key%16 owner_bucket FROM observations) TO '+
        literal(output/'staging')+' (FORMAT PARQUET,PARTITION_BY(owner_bucket),COMPRESSION ZSTD)')
    mapping=np.lib.format.open_memmap(output/'observation_facts.npy',mode='w+',dtype=np.uint32,shape=(n,))
    mapping[:]=MAX_ID; mapping.flush()
    counts={}
    for bucket in range(16):
        folder=output/'staging'/f'owner_bucket={bucket}'
        counts[str(bucket)]=sum(pq.ParquetFile(p).metadata.num_rows for p in folder.glob('*.parquet'))
    if sum(counts.values())!=n: raise ValueError('Complete partition denominator drift')
    progress(dict(phase='all_observations_partitioned',seconds=time.monotonic()-began,observations=n))
    return dict(observations=n,bucket_observations=counts,storage_buckets=16,query_label_selection=False,
        source_ordinals_preserved=True,empty_names_retained_as_nonlexical_context=True)


def factor_bucket(c,output,bucket,fact_offset,n,expected_observations):
    output=Path(output); folder=output/'staging'/f'owner_bucket={bucket}'
    if not expected_observations: return dict(bucket=bucket,fact_start=fact_offset,facts=0,observations=0)
    source=literal(folder/'*.parquet'); target=output/f'facts_part{bucket:02d}.parquet'; fields=','.join(FIELDS)
    c.execute('COPY (SELECT DISTINCT '+fields+' FROM read_parquet('+source+',hive_partitioning=false) ORDER BY '+fields+
        ') TO '+literal(target)+' (FORMAT PARQUET,COMPRESSION ZSTD)')
    count=pq.ParquetFile(target).metadata.num_rows
    if not count or fact_offset+count>=MAX_ID: raise ValueError('Fact identity range drift')
    mapping=np.load(output/'observation_facts.npy',mmap_mode='r+',allow_pickle=False)
    if len(mapping)!=n: raise ValueError('Original ordinal mapping range drift')
    # NULL and empty normalized strings remain distinct full tuples.
    join=' AND '.join('s.'+k+' IS NOT DISTINCT FROM f.'+k for k in FIELDS)
    sql=('SELECT s.observation_index,CAST(f.file_row_number+'+str(fact_offset)+''' AS UINTEGER) fact_index
        FROM read_parquet('''+source+''',hive_partitioning=false) s JOIN read_parquet('''+
        literal(target)+''',file_row_number=true) f ON '''+join)
    seen=0
    for batch in c.execute(sql).fetch_record_batch(131072):
        indices=batch.column(0).to_numpy(zero_copy_only=False); ids=batch.column(1).to_numpy(zero_copy_only=False)
        if (np.any(indices>=n) or len(np.unique(indices))!=len(indices) or np.any(mapping[indices]!=MAX_ID)
                or np.any(ids<fact_offset) or np.any(ids>=fact_offset+count)):
            raise ValueError('Nonunique full tuple/source ordinal assignment')
        mapping[indices]=ids; seen+=len(indices)
    mapping.flush()
    if seen!=expected_observations: raise ValueError('Bucket observation denominator drift')
    return dict(bucket=bucket,path=str(target),fact_start=fact_offset,facts=count,observations=seen)


def invert(output,parts,n,progress=lambda row:None):
    """Same complete inverse, smaller scratch than one argsort over162m rows."""
    output=Path(output); mapping=np.load(output/'observation_facts.npy',mmap_mode='r',allow_pickle=False)
    nf=sum(p['facts'] for p in parts)
    if len(mapping)!=n or nf<=0 or np.any(mapping==MAX_ID) or np.any(mapping>=nf):
        raise ValueError('Complete source ordinal mapping required')
    counts=np.bincount(mapping,minlength=nf)
    if len(counts)!=nf or np.any(counts==0) or int(np.sum(counts))!=n: raise ValueError('Fact/observation denominator drift')
    offsets=np.r_[0,np.cumsum(counts,dtype=np.uint64)].astype(np.uint32)
    maximum=int(np.max(counts)); del counts
    np.save(output/'fact_offsets.npy',offsets,allow_pickle=False)
    inverse=np.lib.format.open_memmap(output/'fact_observations.npy',mode='w+',dtype=np.uint32,shape=(n,))
    seen=0; expected_start=0
    for part in parts:
        lo=part['fact_start']; hi=lo+part['facts']
        if lo!=expected_start: raise ValueError('Noncontiguous full tuple catalog')
        expected_start=hi
        chosen=np.flatnonzero((mapping>=lo)&(mapping<hi))
        if len(chosen)!=part['observations']: raise ValueError('Inverse bucket observation drift')
        ordered=chosen[np.argsort(mapping[chosen],kind='stable')].astype(np.uint32)
        left,right=int(offsets[lo]),int(offsets[hi]); inverse[left:right]=ordered
        if right-left!=len(ordered): raise ValueError('Inverse fact offset drift')
        seen+=len(ordered); progress(dict(phase='inverse_original_ordinals',bucket=part['bucket'],observations=len(ordered)))
    inverse.flush()
    if seen!=n: raise ValueError('Complete inverse denominator drift')
    return dict(observations_indexed=n,max_observations_per_fact=maximum,alias_observation_cap=None,
        all_source_ordinals_mapped_once=True,all_full_tuple_facts_referenced=True)
