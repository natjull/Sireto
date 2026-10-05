"""Additional address-decontaminated nominal proof, never an invented alias.

Query-only generic hypothesis. ALL shared contiguous spans of at least two
content words may be removed from a COPY of name. Original views remain open.
The residual must retain at least two distinct content words. Exact-name,
owner, semantic and parent/site proofs always use the original query only.
"""
from dataclasses import replace

from .retrieval_corroborated_initial_v217 import PARTICLES, LEGAL
from .retrieval_proof_v56 import components, signal
from .retrieval_variants_v118 import normalized


def residual_name(q):
    name = normalized(signal(q.name)).split()
    address = normalized(signal(q.address)).split()
    def content(words):
        return [w for w in words if w.isalpha() and w not in PARTICLES | LEGAL]
    remove = set()
    for lo in range(len(name)):
        for hi in range(lo+1, len(name)+1):
            span = name[lo:hi]
            if len(set(content(span))) < 2:
                continue
            if any(address[j:j+len(span)] == span for j in range(len(address)-len(span)+1)):
                remove.update(range(lo, hi))
    remaining = [w for i,w in enumerate(name) if i not in remove]
    if not remove or len(set(content(remaining))) < 2:
        return None
    return ' '.join(remaining)


def augmented_parts(backend, q):
    original = components(backend, q)
    residual = residual_name(q)
    if residual is None:
        return original
    other = components(backend, replace(q, name=residual))
    tv = backend._tantivy
    # Alternate nominal coverage is corroborated by place ONLY. It cannot
    # create exact-string proof2/10, foreign owner or a legal identity claim.
    for target, alternate in ((0, 6), (6, 6), (8, 8)):
        values = [p for p in (original[target], other[alternate]) if p is not None]
        original[target] = tv.Query.disjunction_max_query(values, tie_breaker=0.) if len(values)>1 else (values[0] if values else None)
    return original
