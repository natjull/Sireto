"""Exact V138 coverage with query-bounded unsigned counters, not quantization.

Each sealed token posting contains an alias at most once. A channel's count
is therefore bounded by its number of DISTINCT query tokens. Select a dtype
that can represent that bound, never truncate tokens/postings/aliases. Scores
and divisions stay float64; outputs retain their original whole namespace.
This is an unmeasured optimization, not a recall/SLO or national certificate.
"""
import time

import numpy as np

from .retrieval_coupled_blocks_v121 import channels
from .retrieval_name_pages_v151 import ScopedNamePages
from .retrieval_variants_v118 import normalized


def counter_dtype(tokens):
    if not isinstance(tokens,int) or isinstance(tokens,bool) or not 0<=tokens<2**32:
        raise ValueError('Exact unsigned distinct query-token count required')
    return np.dtype(np.uint8 if tokens<=255 else np.uint16 if tokens<=65535 else np.uint32)


def coverage(index,text,max_seconds=20.,chunk=131072):
    """Identical V138 features, given the same sealed sorted/unique postings."""
    if (not isinstance(text,str) or not np.isfinite(max_seconds) or max_seconds<=0
        or not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0):
        raise ValueError('Explicit text and positive query cost guard required')
    began=time.monotonic();query=channels(text)
    dtype=counter_dtype(max(map(len,query)))
    cov=np.zeros(index.names);exact=np.zeros(index.names,dtype=np.bool_)
    hits=np.zeros(index.names,dtype=dtype);statistics=[]
    def guard():
        if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_OFFICIAL_NAME_QUERY_COST')
    for channel,tokens in enumerate(query):
        hits.fill(0);postings=0
        for token in sorted(tokens):
            values=index.posting(channel,token);postings+=len(values)
            for start in range(0,len(values),chunk):
                guard();ids=values[start:start+chunk]
                if np.any(ids>=index.names):raise ValueError('Every posting alias position required')
                hits[ids]+=1
        if tokens:
            for start in range(0,index.names,chunk):
                guard();sl=slice(start,start+chunk)
                cov[sl]=np.maximum(cov[sl],hits[sl]/len(tokens))
        statistics.append(dict(channel=channel,tokens=len(tokens),postings=postings,
                               positive_aliases=int(np.count_nonzero(hits))))
    value=normalized(text)
    if value:
        ids=index.posting(3,value)
        for start in range(0,len(ids),chunk):
            guard();selected=ids[start:start+chunk]
            if np.any(selected>=index.names):raise ValueError('Every exact-feature alias position required')
            exact[selected]=True
    guard()
    return cov,exact,dict(seconds=time.monotonic()-began,channels=statistics,
        positive_aliases=int(np.count_nonzero(cov)),exact_aliases=int(np.count_nonzero(exact)),
        whole_alias_max=True,postings_or_alias_cap=None,source_hashes_checked_at_open_not_per_query=True,
        counter_dtype=dtype.name,counter_bytes=hits.nbytes,
        counter_bound=max(map(len,query)),float_score_precision_unchanged=True)


class CompactNamePages(ScopedNamePages):
    """V151 map lifetime and authority checks, same exact V138 result vectors."""
    def coverage(self,text,max_seconds=20.,chunk=131072):
        if not self.valid or self.in_coverage:raise ValueError('Single live name-page query required')
        self.in_coverage=True
        try:cov,exact,stats=coverage(self,text,max_seconds,chunk)
        finally:self.in_coverage=False
        if not cov.flags.owndata or not exact.flags.owndata:
            raise ValueError('Whole alias feature outputs must own storage before source maps close')
        return cov,exact,dict(stats,source_page_rotation=self.rotate())
