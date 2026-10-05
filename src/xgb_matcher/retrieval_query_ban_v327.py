"""Prepared same V43 serving BAN preprocessing; no run/admission authority.

Caller must pin/full-hash source pre/post, supervise limits and seal ALL rows.
Never use raw-number legacy BAN or label-side geocoding to repair queries.
"""
from pathlib import Path

import pyarrow as pa

from scripts.run_retrieval_geo_v16_workbench import _resolve_ban
from .retrieval_geo_v16 import normalize_street
from .retrieval_query_population_v192 import FIELDS

BAN = dict(path='/Volumes/CATNAT_DATA/oubusiness/silver/ban_addresses_2026-08-14_semantics-v1.parquet',
           sha256='48f9e3d4f9bbb47dd5e8bd55bc5f76a75e2e6a8ff1c4821026ba858f6e40247f')


def surface(queries):
    if (not queries or len({q['query_id'] for q in queries}) != len(queries)
            or any(set(q) != FIELDS or type(q['fold']) is not int or q['fold'] not in (2,3,4)
                or not q['query_id'] or any(not isinstance(q[k],str) for k in FIELDS-{'fold'}) for q in queries)):
        raise ValueError('Complete typed query-only safe-fold surface required')
    return pa.Table.from_pylist([dict(query_id=q['query_id'],fold=q['fold'],crm_number=q['crm_number'],
        crm_street_norm=normalize_street(q['crm_street']),crm_postcode=q['crm_postcode'],crm_insee=q['crm_insee']) for q in queries])


def compatible(q,g):
    if (q['query_id'],q['fold']) != (g['query_id'],g['fold']):
        raise ValueError('Exact ordered original BAN controls required')
    return dict(g,spatial_crs_compatible=bool(q['crm_insee'] and not q['crm_insee'].startswith(('97','98'))
        and g['ban_resolved'] and 0<=g['ban_x_l93']<=1300000 and 6000000<=g['ban_y_l93']<=7300000))


def resolve(connection,queries,files):
    """Use ONLY actual V43 query surface. Caller owns readonly hashing/budget."""
    inputs=surface(queries)
    source=files.file(BAN)
    if not isinstance(source,Path):raise ValueError('Pinned official readonly file authority required')
    result=_resolve_ban(connection,inputs,source).to_pylist()
    if len(result)!=len(queries):raise ValueError('No unresolved query may be dropped')
    return [compatible(q,g) for q,g in zip(queries,result)]
