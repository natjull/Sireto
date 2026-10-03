"""Full lexical source plus conditional lexical+initialism source; no pool.

If the initialism branch does not match, L alone covers the site. If it
does match, its nonnegative score A makes L+A >= L. The corpus is unchanged.
"""
from src.xgb_matcher.retrieval_split_native_v107 import queries as split_queries
from src.xgb_matcher.retrieval_factored_native_v105 import BoundedNativeStream


def queries(backend,q,weights,terms):
    parts=split_queries(backend,q,weights,terms); tv=backend._tantivy
    conditional=tv.Query.boolean_query([(tv.Occur.Must,parts['initialism']),
                                        (tv.Occur.Should,parts['lexical'])])
    return {'lexical':parts['lexical'],'initialism_conditional':conditional}


def streams(backend,q,weights,terms):
    return {k:BoundedNativeStream(backend,p) for k,p in queries(backend,q,weights,terms).items()}
