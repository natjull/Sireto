"""V164 exact whole-address channels with bounded physical COPY fan-out.

No token/alias/channel cap or normalization change. Every posting audited.
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


def stage_names(c,table,output,name_start,row_start,buckets,writer_group=8):
    """Every raw name in one physical group, exact scalar set certificates.

    Per-name cardinality + uniqueness + every token is a member of the scalar
    word/substr/exact set => equality of EVERY set, not a checksum sample.
    """
    if (type(writer_group) is not int or not 1<=writer_group<=8
        or type(buckets) is not int or not 1<=buckets<=254):
        raise ValueError('Explicit bounded physical writers required')
    output=Path(output)
    if (output/'features.parquet').exists() or (output/'staging').exists():
        raise ValueError('Immutable lexical physical part, no retry')
    copies=0;n=len(table);local=table['name_id'].to_numpy(zero_copy_only=False)
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
        for lower in range(0,buckets,writer_group):
            upper=min(lower+writer_group,buckets)
            c.execute('COPY (SELECT * FROM posts131 WHERE term_bucket>='+str(lower)+' AND term_bucket<'+str(upper)+
                ') TO '+literal(output/'staging')+' (FORMAT PARQUET,PARTITION_BY(term_bucket),WRITE_PARTITION_COLUMNS true,APPEND true,COMPRESSION ZSTD)')
            copies+=1
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
        all_raw_names_retained=True,files=[str(p) for p in files],no_variant_token_bags_mixed=True,
        maximum_physical_writers_per_copy=min(writer_group,buckets),copy_operations=copies,
        physical_writer_group_is_not_a_candidate_cap=True,all_original_channels_and_tokens_unchanged=True)


def stage_addresses(c,table,output,alias_start,row_start,buckets,writer_group=8):
    from .retrieval_address_access_v152 import SCHEMA
    if table.schema!=SCHEMA:
        raise ValueError('Exact original whole-address dictionary schema required')
    if (any(type(v) is not int or v<0 for v in (alias_start,row_start,buckets))
        or not 1<=buckets<=254 or alias_start+row_start+len(table)>=2**32):
        raise ValueError('Complete typed address/storage namespace required')
    result=stage_names(c,table.rename_columns(['name_id','name_text']),output,alias_start,row_start,buckets,writer_group)
    result['addresses']=result.pop('names')
    result['all_raw_addresses_retained']=result.pop('all_raw_names_retained')
    result.update(address_fact_attributes_or_scope_changed=False,alias_or_proof_cap=None,
        source_address_normalization_unchanged=True)
    return result
