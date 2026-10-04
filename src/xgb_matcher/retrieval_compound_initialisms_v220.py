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
def partition_tail(token):
    """Remaining acronym, one OR more words, not a candidate top-k."""
    alternatives = [token+'[A-Z0-9]*']
    for i in range(1,len(token)):
        alternatives.append(token[:i]+'[A-Z0-9]*'+BETWEEN+partition_tail(token[i:]))
    return '('+'|'.join(alternatives)+')'


def patterns(name):
    result=[]
    for t in tokens(name):
        body='('+'|'.join(t[:i]+'[A-Z0-9]*'+BETWEEN+partition_tail(t[i:])
                          for i in range(1,len(t)))+')'
        result.append(SKIP+body+TAIL)
    return result


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
