"""Exact spatial first-stage retrieval, with deterministic ties and cap100.

Only source coordinates affect distance. IDs bind/emit official records and
canonicalize equal-distance ties; they are never similarity features. This
module does not interpret absent coordinates as a negative matching signal.
"""
import numpy as np
from scipy.spatial import cKDTree


def validate_arrays(xy, ids, parents):
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)!=len(ids) or len(ids)!=len(parents):
        raise ValueError('Spatial index alignment drift')
    if xy.dtype!=np.float64 or not np.isfinite(xy).all():
        raise ValueError('Expected finite float64 source coordinates')
    if not len(ids) or np.any(ids[1:]<=ids[:-1]):
        raise ValueError('Spatial SIRET keys must be unique and sorted')
    if np.any(ids<10**8) or np.any(ids>=10**14) or np.any(ids//100000!=parents):
        raise ValueError('Official membership/identity drift')


def nearest_sites(tree, xy, ids, parents, point, limit=100):
    if not 1<=limit<=100: raise ValueError('Initial spatial cap')
    point=np.asarray(point,dtype=np.float64)
    if point.shape!=(2,) or not np.isfinite(point).all(): raise ValueError('Invalid query coordinate')
    if tree.n!=len(xy) or len(ids)!=len(xy) or len(parents)!=len(xy):
        raise ValueError('Tree/identity alignment drift')
    count=min(limit,len(xy))
    distances,_=tree.query(point,k=count,workers=2)
    cutoff=float(np.max(distances))
    # Include EVERY boundary tie, not an arbitrary 100 from a large building.
    # The tiny guard absorbs tree-vs-vector float64 boundary roundoff; final
    # order is recomputed under the declared Euclidean metric for these leaves.
    radius=cutoff+max(1e-8,abs(cutoff)*1e-12)
    positions=np.asarray(tree.query_ball_point(point,radius,workers=2),dtype=np.int64)
    delta=np.asarray(xy[positions])-point
    exact=np.sqrt(np.sum(delta*delta,axis=1))
    order=np.lexsort((ids[positions],exact))[:count]
    result=[dict(siret=f'{int(ids[positions[i]]):014d}',siren=f'{int(parents[positions[i]]):09d}',
                 score=-float(exact[i]),distance_m=float(exact[i])) for i in order]
    if len(result)!=count or len({r['siret'] for r in result})!=count:
        raise ValueError('Spatial collector cardinality drift')
    return result,dict(boundary_radius_m=cutoff,leaves_within_boundary=len(positions),
                       boundary_tie_overflow=max(0,len(positions)-count),fixed_reservoir=None,
                       approximate_search=False)


def build_tree(xy):
    return cKDTree(xy,leafsize=64,compact_nodes=True,balanced_tree=True,copy_data=False)
