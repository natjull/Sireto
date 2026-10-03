"""All registry sites and exhaustive parent membership; no graph/query/label filter."""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_scoped_paths_v90 import typed_position

ARRAYS=('endpoint_keys','endpoint_parents','endpoint_source_observations','endpoint_states',
        'parent_keys','parent_offsets','parent_endpoints')


def build(source,output,progress=lambda row:None):
    output=Path(output); file=pq.ParquetFile(source); n=file.metadata.num_rows
    if not 0<n<2**32: raise ValueError('Registry position range drift')
    if file.schema_arrow.field('site_key').type!=pa.uint64() or file.schema_arrow.field('parent_key').type!=pa.uint32():
        raise ValueError('Official registry key dtype drift')
    keys=np.empty(n,dtype=np.uint64); parents=np.empty(n,dtype=np.uint32); states=np.empty(n,dtype=np.uint8); offset=0
    for batch in file.iter_batches(batch_size=131072,columns=['site_key','parent_key','administrative_state'],use_threads=False):
        stop=offset+len(batch)
        if batch.column(0).null_count or batch.column(1).null_count: raise ValueError('Unknown official membership')
        keys[offset:stop]=batch.column(0).to_numpy(zero_copy_only=False)
        parents[offset:stop]=batch.column(1).to_numpy(zero_copy_only=False)
        states[offset:stop]=[0 if x in (None,'') else 1 if x=='A' else 2 if x=='F' else 3 for x in batch.column(2).to_pylist()]
        offset=stop
    if offset!=n or np.any(keys>=10**14) or np.any(parents>=10**9) or np.any(keys//100000!=parents):
        raise ValueError('National registry membership/grain drift')
    progress(dict(phase='all_registry_rows_read',sites=n))
    order=np.argsort(keys,kind='stable'); keys=keys[order]; parents=parents[order]
    if np.any(keys[1:]<=keys[:-1]): raise ValueError('Nonunique official registry site')
    np.save(output/'endpoint_source_observations.npy',order.astype(np.uint32),allow_pickle=False)
    np.save(output/'endpoint_states.npy',states[order],allow_pickle=False); del states,order
    np.save(output/'endpoint_keys.npy',keys,allow_pickle=False)
    np.save(output/'endpoint_parents.npy',parents,allow_pickle=False)
    progress(dict(phase='all_sites_sorted_with_original_ordinals',sites=n))
    order=np.argsort(parents,kind='stable'); sorted_parents=parents[order]
    starts=np.r_[0,np.flatnonzero(sorted_parents[1:]!=sorted_parents[:-1])+1]
    parent_keys=sorted_parents[starts]; offsets=np.r_[starts,n].astype(np.uint32)
    members=order.astype(np.uint32)
    if len(members)!=n or int(offsets[-1])!=n: raise ValueError('Parent membership denominator drift')
    np.save(output/'parent_keys.npy',parent_keys,allow_pickle=False)
    np.save(output/'parent_offsets.npy',offsets,allow_pickle=False)
    np.save(output/'parent_endpoints.npy',members,allow_pickle=False)
    progress(dict(phase='all_parent_membership_complete',parents=len(parent_keys),sites=n))
    return dict(sites=n,parents=len(parent_keys),parent_site_references=n,
        max_sites_per_parent=int(np.max(np.diff(offsets))),parent_or_site_cap=None,
        closed_sites_removed=False,graph_membership_required=False,original_site_ordinals_retained=True,
        original_state_strings_in_source=True,identifiers_as_features=False)


class NationalMembership:
    def __init__(self,directory):
        self.a={k:np.load(Path(directory)/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ARRAYS}

    def endpoint_position(self,siret):
        if not (isinstance(siret,str) and len(siret)==14 and siret.isascii() and siret.isdigit()):
            raise ValueError('Invalid site join key')
        keys=self.a['endpoint_keys']; pos=typed_position(keys,int(siret))
        return pos if pos<len(keys) and keys[pos]==int(siret) else None

    def parent_endpoint_positions(self,siren):
        if not (isinstance(siren,str) and len(siren)==9 and siren.isascii() and siren.isdigit()):
            raise ValueError('Invalid parent join key')
        keys=self.a['parent_keys']; pos=typed_position(keys,int(siren))
        if pos==len(keys) or keys[pos]!=int(siren): return self.a['parent_endpoints'][:0]
        lo,hi=self.a['parent_offsets'][pos:pos+2]
        return self.a['parent_endpoints'][int(lo):int(hi)]
