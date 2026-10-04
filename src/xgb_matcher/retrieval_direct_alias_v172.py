"""Exact V138 features from ALL already guarded whole-alias feature rows.

Alternative physical execution, not a different score or candidate pool.
Requires independently sealed V166 feature projections/source authorities.
This kernel alone does not certify a national corpus, latency or recall.
"""
import time

import numpy as np
import pyarrow as pa

from .retrieval_coupled_blocks_v121 import channels
from .retrieval_variants_v118 import normalized

SCHEMA=pa.schema([('alias_id',pa.uint32()),('feature_text',pa.string())])


def evaluate(c,table,text):
    """Count DISTINCT query tokens present in one WHOLE guarded alias.

    Word boundaries are spaces; char3/char4 use underscores for spaces, as
    V121/V131. Presence, not occurrence counts, makes the feature exactly
    equal to sealed posting counters. No identifiers are extracted/scored.
    """
    if (table.schema!=SCHEMA or not isinstance(text,str) or table['feature_text'].null_count
        or table['alias_id'].null_count):
        raise ValueError('Exact sealed whole-alias feature projection required')
    source_ids=table['alias_id'].to_numpy(zero_copy_only=False)
    if np.any(source_ids[1:]<=source_ids[:-1]):
        raise ValueError('Every unique source alias in canonical order required')
    query=channels(text);terms={};scores=[];began=time.monotonic()
    for ch,tokens in enumerate(query):
        terms['q'+str(ch)]=sorted(tokens)
        if not tokens:scores.append('0.0::DOUBLE');continue
        subject="' '||feature_text||' '" if ch==0 else "replace(feature_text,' ','_')"
        token="' '||t||' '" if ch==0 else 't'
        scores.append('list_sum(list_transform($q'+str(ch)+'::VARCHAR[],t->CAST(contains('+subject+','+token+') AS DOUBLE)))/'+str(len(tokens))+'.0')
    terms['exact']=normalized(text)
    c.register('direct172',table)
    try:
        # Remove unused parameters (DuckDB rejects bound-but-unreferenced keys).
        params={k:v for k,v in terms.items() if k=='exact' or v}
        result=c.execute('SELECT alias_id,greatest('+','.join(scores)+')::DOUBLE coverage,'
            "($exact<>'' AND feature_text=$exact) exact FROM direct172 ORDER BY alias_id",params).fetch_arrow_table()
    finally:c.unregister('direct172')
    ids=result['alias_id'].to_numpy(zero_copy_only=False)
    cov=result['coverage'].to_numpy(zero_copy_only=False)
    exact=result['exact'].to_numpy(zero_copy_only=False)
    if (ids.dtype!=np.uint32 or cov.dtype!=np.float64 or exact.dtype!=np.bool_
        or len(ids)!=len(table) or not np.isfinite(cov).all() or np.any((cov<0)|(cov>1))
        or np.any(exact&(cov!=1.))):raise ValueError('Every exact typed whole-alias score required')
    return ids,cov,exact,dict(rows=len(ids),seconds=time.monotonic()-began,
        same_whole_alias_word_char3_char4_features=True,no_alias_or_candidate_cap=True,
        corpus_scope='CALLER_SEALED_ALL_FEATURE_ROWS_REQUIRED',recall_measured=False)
