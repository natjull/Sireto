"""Whole short nominal abbreviation excluding pre-existing context words.

New language after V248 STOP: ordinary context nouns from the older222
grammar are not interpreted as multiword initialisms. Their original text
paths remain open. All eligible partitions/fanouts remain uncapped.
"""
from .retrieval_single_abbreviation_v246 import tokens as whole_tokens
from .retrieval_compound_initialisms_v222 import STOP,partitions,SKIP,BETWEEN,TAIL
from .retrieval_corroborated_initial_v217 import EXACT_FIELDS
from .retrieval_variants_v118 import normalized
import re


def tokens(name):
    return [t for t in whole_tokens(name) if t not in STOP]


def patterns(name):
    return [SKIP+BETWEEN.join(chunk+'[A-Z0-9]*' for chunk in split)+TAIL
            for token in tokens(name) for split in partitions(token)]


def membership(name,aliases):
    return any(re.fullmatch(p,normalized(a)) is not None for p in patterns(name) for a in aliases)


def component(native,q,base):
    ps=patterns(q.name)
    if not ps or base[4] is None:return None
    tv=native._tantivy
    regex=tv.Query.disjunction_max_query([tv.Query.regex_query(native.index.schema,f,p)
        for f in EXACT_FIELDS for p in ps],tie_breaker=0.)
    return tv.Query.const_score_query(tv.Query.boolean_query([
        (tv.Occur.Must,regex),(tv.Occur.Must,tv.Query.const_score_query(base[4],0.))]),1.)
