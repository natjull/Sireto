"""Factor exact scoped names while retaining every authoritative observation."""
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

SCOPE={'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED':0,'LEGAL_PERSON_NAME':1,
       'LEGAL_PERSON_COMMERCIAL_NAME':2,'EXPLICIT_ENDPOINT_NAME':3}
MAX_ID=np.iinfo(np.uint32).max


def literal(path): return "'"+str(Path(path)).replace("'","''")+"'"


def factorize(c,source,output,progress=lambda value:None):
    """Full-key DISTINCT is physical factorization, not dropping evidence."""
    output=Path(output); n=pq.ParquetFile(source).metadata.num_rows
    if not 0<n<MAX_ID: raise ValueError('Observation count outside physical uint32 range')
    source_view='''CREATE TEMP VIEW scoped_observations AS SELECT
        CAST(file_row_number AS UINTEGER) observation_index,
        CAST(CASE WHEN subject_kind='SIRET' THEN siret WHEN subject_kind='SIREN' THEN siren END AS UBIGINT) owner_key,
        CAST(siren AS UINTEGER) official_parent,
        CAST(CASE subject_kind WHEN 'SIREN' THEN 1 WHEN 'SIRET' THEN 2 END AS UTINYINT) subject_code,
        CAST(CASE lexical_scope WHEN 'CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED' THEN 0 WHEN 'LEGAL_PERSON_NAME' THEN 1
            WHEN 'LEGAL_PERSON_COMMERCIAL_NAME' THEN 2 WHEN 'EXPLICIT_ENDPOINT_NAME' THEN 3 END AS UTINYINT) scope_code,
        normalized_value name_text,alias_of_related_target,name_relation_co_validity_proven,
        (regexp_full_match(siren,'[0-9]{9}') AND (subject_kind='SIREN'
            OR regexp_full_match(siret,'[0-9]{14}'))) valid_identity
        FROM read_parquet('''+literal(source)+''',file_row_number=true)'''
    c.execute(source_view)
    bad=c.execute('''SELECT count(*) FROM scoped_observations WHERE
        owner_key IS NULL OR official_parent IS NULL OR subject_code IS NULL OR scope_code IS NULL
        OR coalesce(name_text,'')='' OR alias_of_related_target IS NOT FALSE OR name_relation_co_validity_proven IS NOT FALSE
        OR valid_identity IS NOT TRUE
        OR (subject_code=1 AND owner_key<>official_parent)
        OR (subject_code=2 AND owner_key//100000<>official_parent)
        OR (subject_code=1 AND scope_code=3) OR (subject_code=2 AND scope_code<>3)''').fetchone()[0]
    if bad: raise ValueError('Official name grain/scope/membership drift')
    began=time.monotonic()
    c.execute('''COPY (SELECT observation_index,owner_key,official_parent,subject_code,scope_code,name_text,
        owner_key%16 owner_bucket FROM scoped_observations) TO '''+literal(output/'staging')+
        ''' (FORMAT PARQUET,PARTITION_BY(owner_bucket),COMPRESSION ZSTD)''')
    progress(dict(phase='partition_all_name_observations',seconds=time.monotonic()-began,observations=n))
    mapping=np.lib.format.open_memmap(output/'observation_facts.npy',mode='w+',dtype=np.uint32,shape=(n,))
    mapping[:]=MAX_ID; parts=[]; fact_offset=0; seen=0
    keys=['subject_code','owner_key','official_parent','scope_code','name_text']
    fields=','.join(keys)
    for bucket in range(16):
        directory=output/'staging'/f'owner_bucket={bucket}'
        if not directory.exists(): continue
        began=time.monotonic(); glob=literal(directory/'*.parquet'); target=output/f'facts_part{bucket:02d}.parquet'
        c.execute('COPY (SELECT DISTINCT '+fields+' FROM read_parquet('+glob+',hive_partitioning=false) ORDER BY '+fields+
                  ') TO '+literal(target)+' (FORMAT PARQUET,COMPRESSION ZSTD)')
        count=pq.ParquetFile(target).metadata.num_rows
        if fact_offset+count>=MAX_ID: raise ValueError('Fact catalog outside physical uint32 range')
        join=' AND '.join('s.'+k+'=f.'+k for k in keys)
        sql='''SELECT s.observation_index,CAST(f.file_row_number+'''+str(fact_offset)+''' AS UINTEGER) fact_index
            FROM read_parquet('''+glob+''',hive_partitioning=false) s JOIN
            read_parquet('''+literal(target)+''',file_row_number=true) f ON '''+join
        reader=c.execute(sql).fetch_record_batch(131072); observations=0
        for batch in reader:
            indices=batch.column(0).to_numpy(zero_copy_only=False)
            facts=batch.column(1).to_numpy(zero_copy_only=False)
            if (np.any(indices>=n) or len(np.unique(indices))!=len(indices) or np.any(mapping[indices]!=MAX_ID)
                or np.any(facts<fact_offset) or np.any(facts>=fact_offset+count)):
                raise ValueError('Nonunique or invalid full-key observation assignment')
            mapping[indices]=facts; observations+=len(indices)
        seen+=observations
        parts.append(dict(bucket=bucket,path=str(target),fact_start=fact_offset,facts=count,observations=observations))
        fact_offset+=count
        progress(dict(phase='factor_complete_owner_bucket',bucket=bucket,observations=observations,facts=count,
                      seconds=time.monotonic()-began))
    mapping.flush()
    if seen!=n or np.any(mapping==MAX_ID): raise ValueError('Source observation denominator drift')
    c.execute('DROP VIEW scoped_observations')
    return dict(observations=n,lexical_facts_including_context=fact_offset,parts=parts,
        exact_full_tuple_grouping=True,all_observations_mapped_once=True,source_observation_order_preserved=True,
        identifiers_as_features=False,original_name_observations_not_removed=True,
        distinct_names_not_bags=True,lexical_identifier_mask_not_yet_applied=True)


def invert(output,stats,progress=lambda value:None):
    output=Path(output); mapping=np.load(output/'observation_facts.npy',mmap_mode='r',allow_pickle=False)
    nf=stats['lexical_facts_including_context']; n=stats['observations']; began=time.monotonic()
    counts=np.bincount(mapping,minlength=nf)
    if len(counts)!=nf or np.any(counts==0) or int(np.sum(counts))!=n:
        raise ValueError('Unreferenced fact or observation count drift')
    offsets=np.r_[0,np.cumsum(counts,dtype=np.uint64)].astype(np.uint32)
    order=np.argsort(mapping,kind='stable')
    np.save(output/'fact_offsets.npy',offsets,allow_pickle=False)
    np.save(output/'fact_observations.npy',order.astype(np.uint32),allow_pickle=False)
    progress(dict(phase='invert_every_observation',seconds=time.monotonic()-began))
    return dict(max_observations_per_fact=int(np.max(counts)),observations_indexed=n,
                every_fact_has_observations=True,alias_observation_cap=None)


class FactObservations:
    """Read-only ALL observation ordinals for a lexical fact, not metadata."""
    def __init__(self,output):
        output=Path(output)
        self.offsets=np.load(output/'fact_offsets.npy',mmap_mode='r',allow_pickle=False)
        self.observations=np.load(output/'fact_observations.npy',mmap_mode='r',allow_pickle=False)

    def all(self,fact_index):
        if not isinstance(fact_index,int) or not 0<=fact_index<len(self.offsets)-1:
            raise ValueError('Invalid fact join key')
        lo,hi=self.offsets[fact_index:fact_index+2]
        return self.observations[int(lo):int(hi)]
