"""Exact whole-alias presence counts without row-wise token-list allocation."""
import time

import numpy as np
import pyarrow.compute as pc

from .retrieval_direct_alias_v172 import SCHEMA
from .retrieval_coupled_blocks_v121 import channels
from .retrieval_variants_v118 import normalized

TOKEN_QUANTUM=32  # Physical expression width, never a token/candidate quota.


def evaluate(c,table,text):
    began=time.monotonic()
    if (table.schema!=SCHEMA or not isinstance(text,str) or table['feature_text'].null_count
        or table['alias_id'].null_count):raise ValueError('Exact sealed whole-alias projection required')
    ids=table['alias_id'].to_numpy(zero_copy_only=False)
    if np.any(ids[1:]<=ids[:-1]):raise ValueError('Every unique canonical source alias required')
    cov=np.zeros(len(ids),dtype=np.float64);query=channels(text);groups=0
    c.register('direct174',table);created=False
    try:
        c.execute("CREATE TEMP TABLE direct174_base AS SELECT alias_id,' '||feature_text||' ' padded,"
            "replace(feature_text,' ','_') compact FROM direct174")
        created=True
        for ch,tokens in enumerate(query):
            ordered=sorted(tokens);counts=np.zeros(len(ids),dtype=np.float64)
            for begin in range(0,len(ordered),TOKEN_QUANTUM):
                selected=ordered[begin:begin+TOKEN_QUANTUM]
                params={'t'+str(i):(' '+t+' ' if ch==0 else t) for i,t in enumerate(selected)}
                subject='padded' if ch==0 else 'compact'
                expr='+'.join('CAST(contains('+subject+',$'+key+') AS DOUBLE)' for key in params)
                result=c.execute('SELECT alias_id,'+expr+' n FROM direct174_base ORDER BY alias_id',params).fetch_arrow_table()
                actual=result['alias_id'].to_numpy(zero_copy_only=False);values=result['n'].to_numpy(zero_copy_only=False)
                if (not np.array_equal(actual,ids) or values.dtype!=np.float64 or not np.isfinite(values).all()
                    or np.any((values<0)|(values>len(selected)))):raise ValueError('Every alias and exact group count required')
                counts+=values;groups+=1
            if ordered:np.maximum(cov,counts/len(ordered),out=cov)
    finally:
        if created:c.execute('DROP TABLE direct174_base')
        c.unregister('direct174')
    normalized_query=normalized(text)
    exact=pc.equal(table['feature_text'],normalized_query).to_numpy(zero_copy_only=False)
    if not normalized_query:exact=np.zeros(len(ids),dtype=np.bool_)
    if (exact.dtype!=np.bool_ or not np.isfinite(cov).all() or np.any((cov<0)|(cov>1))
        or np.any(exact&(cov!=1.))):raise ValueError('Every exact typed whole-alias score required')
    return ids,cov,exact,dict(rows=len(ids),seconds=time.monotonic()-began,
        physical_expression_quantum=TOKEN_QUANTUM,groups=groups,all_query_tokens_consumed=True,
        same_whole_alias_word_char3_char4_features=True,no_alias_or_candidate_cap=True,
        corpus_scope='CALLER_SEALED_ALL_FEATURE_ROWS_REQUIRED',recall_measured=False)
