"""Source-only inventory of official SIRET absent from current membership.

An absence here is NOT a label loss or license to discard a human query.
Identifiers are joins/output identities only, never similarity features.
"""
from pathlib import Path

import numpy as np


def present(keys,values):
    if keys.dtype!=np.uint64 or values.dtype!=np.uint64 or keys.ndim!=1 or values.ndim!=1:
        raise ValueError('Typed unsigned official key vectors required')
    positions=np.searchsorted(keys,values);valid=positions<len(keys);found=np.zeros(len(values),dtype=np.bool_)
    found[valid]=keys[positions[valid]]==values[valid];return found


def inventory(keys,streams,output):
    if keys.dtype!=np.uint64 or keys.ndim!=1 or np.any(keys[1:]<=keys[:-1]):raise ValueError('Whole sorted unique official membership required')
    output=Path(output);counts={};missing={}
    for role,values in streams:
        if not role.replace('_','').isalnum():raise ValueError('Controlled official source role required')
        found=present(keys,values);c=counts.setdefault(role,dict(references=0,present_references=0,absent_references=0))
        c['references']+=len(values);c['present_references']+=int(np.sum(found));c['absent_references']+=int(np.sum(~found))
        if np.any(~found):missing.setdefault(role,[]).append(values[~found].copy())
    unique=[]
    for role,c in counts.items():
        values=np.unique(np.concatenate(missing[role])) if missing.get(role) else np.empty(0,dtype=np.uint64)
        if np.any(present(keys,values)):raise ValueError('Every absent official identity must remain absent from current catalog')
        c['unique_absent_siret']=len(values);np.save(output/(role+'_absent.npy'),values,allow_pickle=False);unique.append(values)
    union=np.unique(np.concatenate(unique)) if unique else np.empty(0,dtype=np.uint64)
    np.save(output/'all_absent_siret.npy',union,allow_pickle=False)
    return dict(current_sites=len(keys),sources=counts,union_unique_absent_siret=len(union),
        every_official_reference_tested=True,candidate_catalog_changed=False,identifiers_as_features=False,
        truth_opened=False,recall_measured=False,human_denominator_filtered=False)
