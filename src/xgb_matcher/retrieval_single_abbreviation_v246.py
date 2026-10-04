"""Whole-query short-name abbreviation, not arbitrary embedded word expansion.

New semantic hypothesis: only a sole content token (3..6 letters) may name
an entire current official alias by compound word prefixes. Full unchanged
global channels remain open. No candidate/partition/fanout budget truncation.
"""
import re

from .retrieval_compound_initialisms_v222 import partitions, SKIP, BETWEEN, TAIL
from .retrieval_corroborated_initial_v217 import EXACT_FIELDS, PARTICLES, LEGAL
from .retrieval_variants_v118 import normalized


def tokens(name):
    content=[w for w in normalized(name).split() if w not in PARTICLES | LEGAL]
    # Two-letter fragments and embedded ordinary words do not express this
    # WHOLE-query hypothesis. This is a language change, not a fanout cap.
    if len(content)!=1 or not re.fullmatch('[A-Z]{3,6}',content[0]):return []
    return content


def patterns(name):
    return [SKIP+BETWEEN.join(chunk+'[A-Z0-9]*' for chunk in split)+TAIL
            for token in tokens(name) for split in partitions(token)]


def membership(name,aliases):
    return any(re.fullmatch(p,normalized(a)) is not None for p in patterns(name) for a in aliases)


def component(native,q,base):
    patterns_=patterns(q.name)
    if not patterns_ or base[4] is None:return None
    tv=native._tantivy
    regex=tv.Query.disjunction_max_query([tv.Query.regex_query(native.index.schema,f,p)
        for f in EXACT_FIELDS for p in patterns_],tie_breaker=0.)
    gate=tv.Query.const_score_query(base[4],0.)
    return tv.Query.const_score_query(tv.Query.boolean_query([
        (tv.Occur.Must,regex),(tv.Occur.Must,gate)]),1.)
