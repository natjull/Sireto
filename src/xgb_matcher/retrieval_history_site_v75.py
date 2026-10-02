"""Initial historical-name/current-scaffold search, no temporal-address claim.

Source score uses unchanged V57 weights as a preregistered transfer hypothesis.
The complete initial score is MAX(current source, historical source), merged
from sorted native score frontiers; no fixed candidate pool is rescored.
"""
from collections import defaultdict
import json

import numpy as np
import tantivy

from scripts import retrieval_v38_initial_top100 as v38
from scripts.retrieval_v40_initial_top100 import NativeStream, collect
from src.xgb_matcher.retrieval_proof_v56 import signal, index_query

RAW = ('siret', 'siren', 'names_exact', 'scaffold_exact', 'insee', 'postcode', 'number', 'siege')
TEXT = ('names', 'scaffold')
GRAM = ('namegrams', 'scaffoldgrams')


def schema():
    b = tantivy.SchemaBuilder()
    for f in RAW:
        b.add_text_field(f, stored=f in ('siret', 'siren'), tokenizer_name='raw', index_option='basic')
    for f in TEXT:
        b.add_text_field(f, stored=True, index_option='basic')
    for f in GRAM:
        b.add_text_field(f, stored=False, tokenizer_name='raw', index_option='basic')
    b.add_text_field('provenance', stored=True, tokenizer_name='raw', index_option='basic')
    return b.build()


def grouped(rows, prefix):
    groups = defaultdict(list)
    for r in rows:
        sid, parent = r['siret'], r['siren']
        if (len(sid) != 14 or len(parent) != 9 or not sid.isdecimal() or not parent.isdecimal()
            or not sid.startswith(parent) or not parent.startswith(prefix)):
            raise ValueError('Official partition/site identity drift')
        if r['geo_scaffold_source'] != 'SIRENE_CURRENT_ESTABLISHMENT' or not r['evidence_id']:
            raise ValueError('Missing official source/scaffold provenance')
        groups[sid].append(r)
    return groups


def document(rows):
    first = rows[0]
    fields = ('siren', 'siret', 'insee', 'postcode', 'number', 'number_suffix',
              'is_headquarters', 'current_state', 'geo_scaffold_address', 'geo_scaffold_source')
    if any(any(r[f] != first[f] for f in fields) for r in rows):
        raise ValueError('Conflicting current scaffold; do not fabricate one')
    doc = tantivy.Document()
    for f, value in [('siret', first['siret']), ('siren', first['siren']),
        ('insee', first['insee']), ('postcode', first['postcode']),
        ('number', signal(first['number'])), ('siege', '1' if first['is_headquarters'] else '0')]:
        if value:
            doc.add_text(f, value)
    names = sorted({signal(r['normalized_name']) for r in rows if signal(r['normalized_name'])})
    scaffold = signal(first['geo_scaffold_address'])
    for f, values, exact, grams in [('names', names, 'names_exact', 'namegrams'),
                                  ('scaffold', [scaffold] if scaffold else [], 'scaffold_exact', 'scaffoldgrams')]:
        for value in values:
            doc.add_text(f, value); doc.add_text(exact, value)
        for gram in sorted({g for value in values for g in v38.tokens(value, True)}):
            doc.add_text(grams, gram)
    doc.add_text('provenance', json.dumps(dict(historical_addresses_claimed=False,
        source='SIRENE_HISTORY_SITE', current_scaffold_source=first['geo_scaffold_source'],
        observations=[{k:r[k] for k in ('evidence_id', 'normalized_name', 'name_kind',
            'valid_from', 'valid_to', 'historical_state')} for r in rows]), default=str, sort_keys=True))
    return doc


def query(index, q, weights):
    w = np.asarray(weights, dtype=float)
    if w.shape != (11,) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('Historical transferred native metric drift')
    def const(p, score=1.):
        return tantivy.Query.const_score_query(p, score) if p is not None else None
    def term(f, value):
        return tantivy.Query.term_query(index.schema, f, value, 'basic') if value else None
    def maximum(parts):
        parts = [p for p in parts if p is not None]
        return tantivy.Query.disjunction_max_query(parts, tie_breaker=0.) if parts else None
    def all_of(*parts):
        return tantivy.Query.boolean_query([(tantivy.Occur.Must, p) for p in parts]) if all(p is not None for p in parts) else None
    def coverage(f, grams, value):
        normal = signal(value); values = []
        for field, tokens in ((f, v38.tokens(normal)), (grams, v38.tokens(normal, True))):
            if tokens:
                values.append(tantivy.Query.boolean_query([(tantivy.Occur.Should,
                    const(term(field, t), 1./len(tokens))) for t in tokens]))
        return maximum(values)
    name = coverage('names', 'namegrams', q.name)
    address = coverage('scaffold', 'scaffoldgrams', q.address)
    ne, ae = const(term('names_exact', signal(q.name))), const(term('scaffold_exact', signal(q.address)))
    place = maximum([const(term('insee', q.insee)), const(term('postcode', q.postcode))])
    gate = const(place, 0.)
    parts = [name, address, ne, ae, place, const(all_of(term('number', signal(q.number)), gate)),
             all_of(name, gate), all_of(address, gate), all_of(name, address, gate),
             const(all_of(term('siege', '1'), gate)), const(all_of(ne, ae, gate))]
    scored = [tantivy.Query.boost_query(p, float(value)) for p, value in zip(parts, w) if p is not None and value > 0]
    return tantivy.Query.boolean_query([(tantivy.Occur.Should, p) for p in scored]) if scored else tantivy.Query.empty_query()


def search(base, history, q, weights):
    def decode(d):
        return {k:d[k][0] for k in ('siret', 'siren')}
    streams = dict(current=NativeStream(base.searcher, index_query(base, q, weights), decode),
                   historical_site=NativeStream(history.searcher(), query(history, q, weights), decode, probe_first=True))
    rows, trace = collect(streams)
    if len(rows) > 100 or len({r['siret'] for r in rows}) != len(rows):
        raise ValueError('Initial cap/uniqueness drift')
    trace.update(certified=True, unique_documents_visited=sum(s.decoded for s in streams.values()),
        fixed_reservoir=None, historical_addresses_claimed=False,
        sources={k:dict(calls=s.calls, decoded=s.decoded, native_hits_fetched=s.offset) for k,s in streams.items()},
        score='MAX_NATIVE_CURRENT_OR_HISTORICAL_SITE_CURRENT_SCAFFOLD',
        tie_scope='NATIVE_SOURCE_TIES_NOT_GLOBAL_CANONICAL_ID_TIES')
    return rows, trace
