"""Official exact historical-LEGAL alias join, scored in initial site search.

The parent predicate is exhaustive equality, never a ranked/capped parent pool.
All SIRET descendants remain searchable; current place is a bonus, not a gate.
No discovered leaf is rescored and no historical address is manufactured.
"""
import math

import numpy as np

from scripts.retrieval_v40_initial_top100 import NativeStream, collect
from src.xgb_matcher.retrieval_history_site_v75 import query as site_query
from src.xgb_matcher.retrieval_proof_v56 import components, index_query, signal


def alias_pairs(rows):
    pairs = set()
    for r in rows:
        parent = r['siren']
        if len(parent) != 9 or not parent.isdecimal():
            raise ValueError('Invalid official LEGAL membership')
        name = signal(r['normalized_name'])
        if name:
            pairs.add((name, parent))
    return sorted(pairs)


def matching_parents(connection, name):
    normalized = signal(name)
    if not normalized:
        return []
    # No LIMIT, geography, state, teacher, query ID or label in this lookup.
    return [r[0] for r in connection.execute(
        'SELECT siren FROM aliases WHERE name = ? ORDER BY siren', [normalized]).fetchall()]


def parent_query(backend, q, parents, weights):
    w = np.asarray(weights, dtype=float)
    if w.shape != (11,) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('Historical transferred native metric drift')
    if not signal(q.name) or not parents:
        return backend._tantivy.Query.empty_query()
    if len(parents) != len(set(parents)) or any(len(p) != 9 or not p.isdecimal() for p in parents):
        raise ValueError('Exhaustive official parent predicate drift')
    tv = backend._tantivy
    identity = tv.Query.term_set_query(backend.index.schema, 'siren', parents)
    name = tv.Query.const_score_query(identity, 1.)
    zero_identity = tv.Query.const_score_query(identity, 0.)
    def const(p, score=1.):
        return tv.Query.const_score_query(p, score) if p is not None else None
    def all_of(*ps):
        return tv.Query.boolean_query([(tv.Occur.Must, p) for p in ps]) if all(p is not None for p in ps) else None
    current = components(backend, q)
    address, ae, place, number, siege = (current[i] for i in (1, 3, 4, 5, 9))
    place_gate = const(place, 0.)
    # Exact alias implies historical-name coverage=1 and exact-name=1.
    # The name+address conjunction contributes 1+a only when address matches.
    parts = [name, address, name, ae, place, number, all_of(name, place_gate),
             all_of(address, place_gate), all_of(name, address, place_gate),
             siege, const(all_of(name, ae, place_gate))]
    scored = [tv.Query.boost_query(p, float(v)) for p, v in zip(parts, w) if p is not None and v > 0]
    if not scored:
        return tv.Query.empty_query()
    site = const(tv.Query.term_query(backend.index.schema, 'document_type', 'siret', 'basic'), 0.)
    return tv.Query.boolean_query([(tv.Occur.Must, zero_identity), (tv.Occur.Must, site),
        (tv.Occur.Must, tv.Query.boolean_query([(tv.Occur.Should, p) for p in scored]))])


def search(base, history, aliases, q, weights):
    parents = matching_parents(aliases, q.name)
    decode = lambda d: {k:d[k][0] for k in ('siret', 'siren')}
    streams = dict(current=NativeStream(base.searcher, index_query(base, q, weights), decode),
        historical_site=NativeStream(history.searcher(), site_query(history, q, weights), decode, probe_first=True))
    if parents:
        streams['historical_legal_exact'] = NativeStream(base.searcher,
            parent_query(base, q, parents, weights), decode, probe_first=True)
    rows, trace = collect(streams)
    if len(rows) > 100 or any(not math.isfinite(r['score']) for r in rows):
        raise ValueError('Initial cap/score drift')
    trace.update(certified=True, fixed_reservoir=None, parent_limit=None,
        historical_parent_predicate='EXHAUSTIVE_EXACT_NORMALIZED_OFFICIAL_LEGAL_ALIAS',
        matching_parents=len(parents), historical_addresses_claimed=False,
        historical_alias_current_site_coexistence_proven=False,
        score='MAX_NATIVE_CURRENT_SITE_HISTORY_OR_EXACT_LEGAL_HISTORY_JOIN',
        sources={k:dict(calls=s.calls, decoded=s.decoded, native_hits_fetched=s.offset) for k,s in streams.items()},
        tie_scope='NATIVE_SOURCE_TIES_NOT_GLOBAL_CANONICAL_ID_TIES')
    return rows, trace
