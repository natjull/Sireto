"""Complete same-alias national word access, physical stages only."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_name_facts_v85 import literal
from src.xgb_matcher.retrieval_name_postings_v86 import masked_sql


def prepare(c,parts,output):
    output=Path(output); nf=sum(p['facts'] for p in parts)
    clauses=['SELECT CAST(file_row_number+'+str(p['fact_start'])+' AS UINTEGER) fact_id,* EXCLUDE(file_row_number) FROM read_parquet('
             +literal(p['path'])+',file_row_number=true)' for p in parts]
    c.execute('CREATE TEMP VIEW name_facts AS '+' UNION ALL '.join(clauses))
    # Empty/NULL context preserved as facts, never made lexical. All other
    # text must obey the same authoritative ASCII normalizer/mask as V86.
    bad=c.execute("""SELECT count(*) FROM name_facts WHERE NOT (
        (scope_code=0 AND coalesce(name_text,'')='') OR
        coalesce(regexp_full_match(name_text,'[A-Z0-9]+( [A-Z0-9]+)*'),FALSE))""").fetchone()[0]
    if bad: raise ValueError('Noncanonical source text: authoritative normalization required')
    masked='SELECT *, '+masked_sql("coalesce(name_text,'')")+' masked_text FROM name_facts'
    counts=c.execute("SELECT count(*),count(*) FILTER(WHERE scope_code=0),count(*) FILTER(WHERE scope_code<>0 AND masked_text='') FROM ("+masked+')').fetchone()
    if counts[0]!=nf: raise ValueError('Whole fact denominator drift')
    c.execute("""COPY (SELECT fact_id,token,hash(token)%32 token_bucket FROM (
        SELECT fact_id,unnest(list_distinct(string_split(masked_text,' '))) token FROM ("""+masked+
        ") WHERE scope_code<>0) WHERE token<>'') TO "+literal(output/'word_staging')+
        ' (FORMAT PARQUET,PARTITION_BY(token_bucket),COMPRESSION ZSTD)')
    return dict(facts=nf,context_only_facts=counts[1],identifier_only_admissible_facts=counts[2],
        identifier_mask_applied=True,per_fact_word_dedup=True,same_alias_word_bags_only=True,
        empty_context_retained=True,whole_source_ngrams_not_indexed=True)


def word_bucket(c,output,bucket,nf):
    output=Path(output); folder=output/'word_staging'/f'token_bucket={bucket}'
    if not folder.exists(): return None
    sorted_path=output/f'words_sorted{bucket:02d}.parquet'; vocabulary=output/f'word_vocabulary{bucket:02d}.parquet'
    c.execute('COPY (SELECT token,fact_id FROM read_parquet('+literal(folder/'*.parquet')+
        ',hive_partitioning=false) ORDER BY token,fact_id) TO '+literal(sorted_path)+' (FORMAT PARQUET,COMPRESSION ZSTD)')
    c.execute('''COPY (SELECT token,CAST(min(file_row_number) AS UINTEGER) posting_start,
        CAST(max(file_row_number)+1 AS UINTEGER) posting_stop,CAST(count(*) AS UINTEGER) postings
        FROM read_parquet('''+literal(sorted_path)+''',file_row_number=true) GROUP BY token ORDER BY token)
        TO '''+literal(vocabulary)+' (FORMAT PARQUET,COMPRESSION ZSTD)')
    values=pq.read_table(sorted_path,columns=['fact_id'],use_threads=False)['fact_id'].to_numpy(zero_copy_only=False)
    if len(values)>=2**32 or len(values) and np.max(values)>=nf: raise ValueError('Word fact identity range drift')
    np.save(output/f'word_postings{bucket:02d}.npy',values,allow_pickle=False)
    return dict(bucket=bucket,vocabulary=str(vocabulary),postings_path=str(output/f'word_postings{bucket:02d}.npy'),
        words=pq.ParquetFile(vocabulary).metadata.num_rows,postings=len(values))
