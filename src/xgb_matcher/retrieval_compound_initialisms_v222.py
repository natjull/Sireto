"""Whole current official aliases, compound initialisms; no CRM dictionary.

All partitions of a short token into >=2 word prefixes. Grammar is independent
of labels and respects alias boundaries. Extra arbitrary trailing words are
NOT accepted. IDs are removed before grammar compilation.
"""
import re
from functools import lru_cache

from .retrieval_corroborated_initial_v217 import EXACT_FIELDS, PARTICLES, LEGAL
from .retrieval_variants_v118 import normalized

STOP = PARTICLES | LEGAL | frozenset('ECOLE VILLE HOTEL LYCEE GROUPE MAIRIE RUE AV BD'.split())
SKIP = '('+'|'.join(v+' ' for v in sorted(PARTICLES | LEGAL))+')*'
BETWEEN = ' '+ '('+'|'.join(v+' ' for v in sorted(PARTICLES))+')*'
TAIL = '( ('+'|'.join(sorted(LEGAL))+'))*'


def tokens(name):
    return sorted({t for t in normalized(name).split()
        if re.fullmatch('[A-Z]{2,6}', t) and t not in STOP})


@lru_cache(maxsize=256)
def partitions(token):
    """ALL flat nonempty contiguous splits into >=2 chunks, no semantic cap."""
    return tuple(tuple(token[a:b] for a,b in zip([0,*cuts],[*cuts,len(token)]))
        for mask in range(1,1 << (len(token)-1))
        for cuts in [[i for i in range(1,len(token)) if mask & (1 << (i-1))]])


def patterns(name):
    return [SKIP+BETWEEN.join(chunk+'[A-Z0-9]*' for chunk in split)+TAIL
        for t in tokens(name) for split in partitions(t)]


def membership(name, aliases):
    return any(re.fullmatch(p, normalized(a)) is not None for p in patterns(name) for a in aliases)


def component(native, q, base):
    """Soft location corroboration applies ONLY to the new acronym proof.

    Other original global lexical/owner/spatial paths remain unchanged. No
    alias can borrow tokens from another alias or sibling establishment.
    """
    ps=patterns(q.name)
    if not ps or base[4] is None: return None
    tv=native._tantivy
    regex=tv.Query.disjunction_max_query([tv.Query.regex_query(native.index.schema,f,p)
        for f in EXACT_FIELDS for p in ps], tie_breaker=0.)
    gate=tv.Query.const_score_query(base[4],0.)
    return tv.Query.const_score_query(tv.Query.boolean_query([
        (tv.Occur.Must,regex),(tv.Occur.Must,gate)]),1.)
