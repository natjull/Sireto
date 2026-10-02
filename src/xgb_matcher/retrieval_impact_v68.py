"""Direct sparse top100 with conservative block bounds, not a candidate reranker.

Nonnegative document impacts and query weights are part of ONE retrieval
score. A block is skipped only when it cannot beat the current top-k heap.
This is a small Python reference executor, NOT production Block-Max WAND.
Official initialisms are compiled at indexing time from individual aliases.
"""
from __future__ import annotations

import heapq
import re
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix

from src.xgb_matcher.hierarchical_retrieval import normalize_text
from src.xgb_matcher.retrieval_initialisms_v67 import LEGAL, PARTICLES, STOP
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_sparse_query_v59 import words


def official_initialisms(alias, max_variants=4096):
    """All searchable length2..6 initialisms of ONE normalized current alias.

    This implements the V67 regexp language, including optional leading
    legal forms/particles and optional interior particles. Numerical words
    cannot be skipped in the middle. A complexity guard STOPS, never truncates.
    """
    tokens = normalize_text(alias).split()
    starts = [0]
    for i, token in enumerate(tokens):
        if token not in (*PARTICLES, *LEGAL):
            break
        starts.append(i + 1)
    result = set()
    explored = 0

    def visit(pos, code):
        nonlocal explored
        explored += 1
        if explored > max_variants:
            raise ValueError('STOP_INITIALISM_COMPLEXITY_NOT_TRUNCATION')
        if pos >= len(tokens) or re.fullmatch(r'[A-Z][A-Z0-9]*', tokens[pos]) is None:
            return
        code += tokens[pos][0]
        if 2 <= len(code) <= 6 and code not in STOP:
            result.add(code)
        if len(code) >= 6:
            return
        following = pos + 1
        visit(following, code)
        while following < len(tokens) and tokens[following] in PARTICLES:
            following += 1
            visit(following, code)

    for start in starts:
        visit(start, '')
    return result


def query_initialisms(name):
    return [t for t in dict.fromkeys(re.findall(r'\b([A-Z]{2,6})[0-9]*\b', signal(name)))
            if t not in STOP][:2]


def document_impacts(document):
    """Unlearned COST-ONLY representation. No IDs, CRM aliases or label access.

    Current aliases remain individual for initialisms. Other word views are
    still bags: this prototype does NOT yet preserve all source/time pairs,
    serve XY, typed graph proofs, or implement a learned document encoder.
    """
    out = {}
    current = ('legal_names', 'trade_names', 'site_names')
    name_fields = (*current, 'historical_names', 'supporting_names')
    address_fields = ('addresses', 'historical_addresses', 'supporting_addresses')
    names = set().union(*(words(v) for f in name_fields for v in document.get(f, [])))
    addresses = set().union(*(words(v) for f in address_fields for v in document.get(f, [])))
    initials = set().union(*(official_initialisms(v) for f in current for v in document.get(f, [])))
    places = [('C', v) for v in document.get('insee', []) if v]
    places += [('Z', v) for v in document.get('postcode', []) if v]
    for prefix, values in (('N', names), ('A', addresses), ('I', initials)):
        for value in values:
            out[prefix + ':' + value] = 1.
            if prefix in ('N', 'I'):
                for kind, place in places:
                    out[f'{prefix}{kind}:{place}:{value}'] = 1.
    for kind, place in places:
        out[f'{kind}:{place}'] = 1.
    return out


def query_impacts(row):
    """Preregistered fixed-cost weights, NOT a quality-trained recipe."""
    out = {}
    places = [(kind, row.get(field, '')) for kind, field in (('C', 'crm_insee'), ('Z', 'crm_postcode'))
              if row.get(field)]
    views = [('N', words(row['crm_name']), 1.), ('I', query_initialisms(row['crm_name']), 1.),
             ('A', words(' '.join(row.get(f, '') for f in ('crm_number', 'crm_suffix', 'crm_street'))), 1.)]
    for prefix, values, weight in views:
        for value in values:
            out[prefix + ':' + value] = weight
            if prefix in ('N', 'I'):
                for kind, place in places:
                    out[f'{prefix}{kind}:{place}:{value}'] = weight
    for kind, place in places:
        out[f'{kind}:{place}'] = 1.
    return out


@dataclass
class ImpactIndex:
    keys: tuple[str, ...]
    terms: dict[str, int]
    postings: csc_matrix
    maxima: csc_matrix
    block_size: int

    @classmethod
    def build(cls, documents, block_size=256):
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError('Invalid block size')
        keys = tuple(sorted(documents))
        vocab = sorted({t for d in documents.values() for t in d})
        terms = {t: i for i, t in enumerate(vocab)}
        rows, cols, impacts = [], [], []
        for i, key in enumerate(keys):
            for term, impact in documents[key].items():
                if not np.isfinite(impact) or impact < 0:
                    raise ValueError('Impacts must be finite and nonnegative')
                if impact:
                    rows.append(i); cols.append(terms[term]); impacts.append(float(impact))
        postings = csc_matrix((impacts, (rows, cols)), shape=(len(keys), len(vocab)), dtype=np.float64)
        postings.sort_indices()
        block_count = (len(keys) + block_size - 1) // block_size
        br, bc, bv = [], [], []
        for term in range(len(vocab)):
            lo, hi = postings.indptr[term:term+2]
            blocks = postings.indices[lo:hi] // block_size
            if not len(blocks):
                continue
            starts = np.r_[0, np.flatnonzero(np.diff(blocks)) + 1]
            maxima = np.maximum.reduceat(postings.data[lo:hi], starts)
            br.extend(blocks[starts]); bc.extend([term] * len(starts)); bv.extend(maxima)
        maxima = csc_matrix((bv, (br, bc)), shape=(block_count, len(vocab)), dtype=np.float64)
        maxima.sort_indices()
        return cls(keys, terms, postings, maxima, block_size)

    def active(self, query):
        if any(not np.isfinite(w) or w < 0 for w in query.values()):
            raise ValueError('Query weights must be finite and nonnegative')
        return sorted((self.terms[t], float(w)) for t, w in query.items() if t in self.terms and w > 0)

    def exhaustive(self, query, k=100):
        """Oracle only: whole-corpus score, same ordered float64 additions."""
        self._cap(k)
        scores = np.zeros(len(self.keys), dtype=np.float64)
        for term, weight in self.active(query):
            lo, hi = self.postings.indptr[term:term+2]
            scores[self.postings.indices[lo:hi]] += self.postings.data[lo:hi] * weight
        ids = np.flatnonzero(scores > 0)
        ids = ids[np.lexsort((ids, -scores[ids]))][:k]
        return [(self.keys[i], float(scores[i])) for i in ids]

    @staticmethod
    def _cap(k):
        if not isinstance(k, int) or not 1 <= k <= 100:
            raise ValueError('Initial output cap is 100, not a reranking reservoir')

    def search(self, query, k=100):
        self._cap(k)
        active = self.active(query)
        bounds = np.zeros(self.maxima.shape[0], dtype=np.float64)
        for term, weight in active:
            lo, hi = self.maxima.indptr[term:term+2]
            blocks = self.maxima.indices[lo:hi]
            # Round UP after every positive addition. Equal-bound ties are
            # visited too, so deterministic ID ties cannot be pruned away.
            values = self.maxima.data[lo:hi] * weight
            bounds[blocks] = np.nextafter(bounds[blocks] + values, np.inf)
        if not np.isfinite(bounds).all():
            raise ValueError('STOP_BOUND_OVERFLOW')
        blocks = np.flatnonzero(bounds > 0)
        blocks = blocks[np.lexsort((blocks, -bounds[blocks]))]
        heap = []; visited = 0; scored = 0; postings_read = 0
        for block in blocks:
            if len(heap) == k and bounds[block] < heap[0][0]:
                break
            start = int(block) * self.block_size
            end = min(start + self.block_size, len(self.keys))
            scores = np.zeros(end - start, dtype=np.float64)
            for term, weight in active:
                lo, hi = self.postings.indptr[term:term+2]
                indices = self.postings.indices[lo:hi]
                a, b = np.searchsorted(indices, [start, end])
                scores[indices[a:b] - start] += self.postings.data[lo+a:lo+b] * weight
                postings_read += int(b - a)
            visited += 1; scored += end - start
            for local in np.flatnonzero(scores > 0):
                i = start + int(local)
                candidate = (float(scores[local]), -i)
                if len(heap) < k:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    heapq.heapreplace(heap, candidate)
        result = [(self.keys[-i], score) for score, i in sorted(heap, reverse=True)]
        return result, dict(corpus_documents=len(self.keys), active_terms=len(active),
            active_blocks=len(blocks), visited_blocks=visited, scored_documents=scored,
            postings_read=postings_read, heap_capacity=k, fixed_candidate_reservoir=None,
            pruned_by_conservative_score_bound=True, downstream_reranker_used=False)
