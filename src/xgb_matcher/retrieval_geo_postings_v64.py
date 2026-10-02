"""Coordinate-to-postings join, scored by the native initial100 collector.

Hard radial basis is NOT the continuous V63 geometry kernel. SIRETs below
are join keys resolved from the complete official spatial predicate, never
learned identity features, GT lookups or a fixed/reranked candidate pool.
"""
import numpy as np

from src.xgb_matcher.retrieval_proof_v56 import components, LEXICAL, SCALES
from src.xgb_matcher.retrieval_joint_v46 import geo_parameters


def radial_memberships(tree, keys, geo, max_joined_sites=200000):
    point, _, reliable = geo_parameters(geo)
    if point is None:
        return [[] for _ in SCALES], dict(compatible=False, reliable=False, joined_sites=0)
    if len(keys) != tree.n:
        raise ValueError('Spatial join cardinality drift')
    # Complete predicate. A computational guard STOPS rather than truncates it.
    positions = np.asarray(tree.query_ball_point(point, SCALES[-1]+1e-8, workers=2), dtype=np.int64)
    if len(positions) > max_joined_sites:
        raise RuntimeError('STOP_COMPLETE_GEO_JOIN_TOO_LARGE')
    delta = tree.data[positions] - point
    distances = np.sqrt(np.sum(delta*delta, axis=1))
    bands = [[f'{int(s):014d}' for s in sorted(keys[positions[distances <= scale]])] for scale in SCALES]
    return bands, dict(compatible=True, reliable=bool(reliable), joined_sites=len(bands[-1]),
                       band_sizes=[len(b) for b in bands], radii_m=list(SCALES), truncated=False)


def query(backend, q, weights, bands, reliable):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(LEXICAL)+8,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Radial native weights drift')
    if len(bands) != len(SCALES) or any(len(set(b)) != len(b) for b in bands):
        raise ValueError('Radial join shape/duplicate drift')
    if any(not set(a).issubset(b) for a,b in zip(bands,bands[1:])):
        raise ValueError('Radial membership nesting drift')
    tv = backend._tantivy
    parts = [tv.Query.boost_query(p,float(w)) for p,w in zip(components(backend,q),weights[:len(LEXICAL)]) if p is not None and w>0]
    for i, ids in enumerate(bands):
        if any(len(s)!=14 or not s.isdigit() for s in ids):
            raise ValueError('Official join key drift')
        w = weights[len(LEXICAL)+i] + bool(reliable)*weights[len(LEXICAL)+4+i]
        if ids and w>0:
            parts.append(tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',ids),float(w)))
    if not parts:
        return tv.Query.empty_query()
    guard = tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,guard),
        (tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))])
