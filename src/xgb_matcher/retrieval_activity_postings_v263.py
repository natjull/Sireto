"""Exact weak activity-at-place proof over ALL current site arrays.

NEW proof semantics, not an optimization/retry of the STOP V115 metric.
No learned head here; no labels, candidate pool, nearest-k or fanout cap.
Continuous activity relevance is not a calibrated probability or identity.
"""
import numpy as np

from .retrieval_site_attributes_v139 import encode_code, MISSING


def block_groups(keys, postcode, insee, nomenclature, activity, q, grades, mapping):
    """IDs serve membership JOIN only. Same official versioned activity/site."""
    arrays=(keys,postcode,insee,nomenclature,activity)
    dtypes=(np.uint64,np.uint32,np.uint32,np.uint8,np.uint16)
    n=len(keys)
    if any(a.shape!=(n,) or a.dtype!=np.dtype(d) for a,d in zip(arrays,dtypes)):
        raise ValueError('Exact current site attributes and common block shape required')
    grades=np.asarray(grades,dtype=np.float64)
    mapping=np.asarray(mapping)
    if (grades.ndim!=1 or not len(grades) or not np.isfinite(grades).all()
            or np.any(grades<0) or np.any(grades>1) or mapping.ndim!=2
            or mapping.dtype.kind not in 'iu' or np.any(mapping < -1) or np.any(mapping>=len(grades))
            or np.any(nomenclature>=mapping.shape[0]) or np.any(activity>=mapping.shape[1])):
        raise ValueError('Versioned official definitions and finite bounded relevance required')
    pc,ic=encode_code(q.postcode),encode_code(q.insee)
    # UNKNOWN does NOT match UNKNOWN. No geography/code magnitude as feature.
    local=np.zeros(n,dtype=bool)
    if pc!=MISSING:local|=postcode==pc
    if ic!=MISSING:local|=insee==ic
    pos=np.flatnonzero(local)
    classes=mapping[nomenclature[pos],activity[pos]]
    known=classes>=0
    pos,classes=pos[known],classes[known]
    relevant=grades[classes]>0
    pos,classes=pos[relevant],classes[relevant]
    if np.any(keys[pos]>=10**14):raise ValueError('Official SIRET membership range drift')
    result={}
    for c in np.unique(classes):
        grade=float(grades[c])
        result.setdefault(grade,set()).update(f'{int(v):014d}' for v in keys[pos[classes==c]])
    return result,dict(scanned=n,at_place=int(local.sum()),known_positive_relevance=len(pos))


def collect_groups(evidence,q,grades,mapping,chunk=131072):
    """FULL exhaustive current-site scan; pages are physical, never rank cuts."""
    if type(chunk) is not int or chunk<1:raise ValueError('Physical chunk size required')
    if encode_code(q.postcode)==MISSING and encode_code(q.insee)==MISSING:
        return {},dict(scanned=0,at_place=0,known_positive_relevance=0,no_place_proof=True)
    refs=[evidence.keys_ref,evidence.structured['artifacts']['postcode_codes.npy'],
        evidence.structured['artifacts']['insee_codes.npy'],
        evidence.structured['reused_arrays']['nomenclature_codes'],
        evidence.structured['reused_arrays']['original_activity_codes']]
    dtypes=(np.uint64,np.uint32,np.uint32,np.uint8,np.uint16)
    # Every used source is fully hashed before first read, then epoch bound.
    paths=[evidence.files.file(r) for r in refs]
    result={};stats=dict(scanned=0,at_place=0,known_positive_relevance=0,no_place_proof=False)
    for lo in range(0,evidence.sites,chunk):
        evidence.files.guard();blocks=[]
        for path,dtype in zip(paths,dtypes):
            a=np.load(path,mmap_mode='r',allow_pickle=False)
            try:
                if a.shape!=(evidence.sites,) or a.dtype!=np.dtype(dtype) or a.flags.writeable:
                    raise ValueError('Whole readonly current-site arrays required')
                blocks.append(a[lo:min(lo+chunk,evidence.sites)].copy())
            finally:a._mmap.close()
        groups,counts=block_groups(*blocks,q,grades,mapping)
        for grade,ids in groups.items():result.setdefault(grade,set()).update(ids)
        for key,value in counts.items():stats[key]+=value
    evidence.files.guard();evidence.files.check_epochs()
    if stats['scanned']!=evidence.sites:raise ValueError('ALL current sites must be scanned')
    return result,stats


def native_proof(backend,groups):
    """Compile exact continuous grades BEFORE any global native collector."""
    tv=backend._tantivy;clauses=[];seen=set()
    for grade,ids in sorted(groups.items()):
        if (isinstance(grade,bool) or not np.isfinite(grade) or not 0<grade<=1
                or not isinstance(ids,set) or not ids or seen.intersection(ids)
                or any(not isinstance(s,str) or not s.isascii() or not s.isdigit() or len(s)!=14 for s in ids)):
            raise ValueError('Unique versioned-site membership and bounded grade required')
        seen.update(ids)
        predicate=tv.Query.term_set_query(backend.index.schema,'siret',sorted(ids))
        clauses.append(tv.Query.const_score_query(predicate,float(grade)))
    return tv.Query.disjunction_max_query(clauses,tie_breaker=0.) if clauses else None
