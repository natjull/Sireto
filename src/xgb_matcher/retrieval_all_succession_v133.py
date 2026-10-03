"""Every raw official succession observation, typed bidirectional incidence.

Self links remain source context, never an expansion. No continuity filter,
observation deduplication, endpoint/member cap, or inferred identity merge.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE=('siretEtablissementPredecesseur','siretEtablissementSuccesseur','dateLienSuccession',
    'transfertSiege','continuiteEconomique','dateDernierTraitementLienSuccession')
FIELDS={'predecessor_keys':np.uint64,'successor_keys':np.uint64,'effective_days':np.int32,
    'effective_valid':np.bool_,'updated_micros':np.int64,'updated_valid':np.bool_,
    'headquarters_codes':np.uint8,'continuity_codes':np.uint8}


def array(output,name,dtype,shape):
    return np.lib.format.open_memmap(Path(output)/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)


def boolean_codes(value):
    return pc.fill_null(pc.if_else(value,2,1),0).cast(pa.uint8()).to_numpy(zero_copy_only=False)


def decode_codes(values):
    return pa.array([None if x==0 else x==2 for x in values],type=pa.bool_())


def build(source,output,expected,batch_rows=131072,progress=lambda value:None):
    output=Path(output);f=pq.ParquetFile(source);n=f.metadata.num_rows
    if (not isinstance(expected,int) or isinstance(expected,bool) or expected!=n or not 0<n<2**31
        or tuple(f.schema_arrow.names)!=SOURCE or f.schema_arrow.types!=[pa.string(),pa.string(),pa.date32(),pa.bool_(),pa.bool_(),pa.timestamp('us')]
        or not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0):raise ValueError('Every original typed official succession row required')
    a={key:array(output,key,dtype,(n,)) for key,dtype in FIELDS.items()};pos=0
    for batch in f.iter_batches(batch_size=batch_rows,use_threads=False):
        d=dict(zip(SOURCE,batch.columns));stop=pos+len(batch)
        for key,field in (('predecessor_keys',SOURCE[0]),('successor_keys',SOURCE[1])):
            value=d[field]
            if value.null_count or not pc.all(pc.match_substring_regex(value,'^[0-9]{14}$')).as_py():raise ValueError('Explicit ASCII14 endpoint required')
            a[key][pos:stop]=pc.cast(value,pa.uint64()).to_numpy(zero_copy_only=False)
        for prefix,field,dtype in (('effective',SOURCE[2],pa.int32()),('updated',SOURCE[5],pa.int64())):
            a[prefix+'_valid'][pos:stop]=pc.is_valid(d[field]).to_numpy(zero_copy_only=False)
            key='effective_days' if prefix=='effective' else 'updated_micros'
            a[key][pos:stop]=pc.fill_null(pc.cast(d[field],dtype),0).to_numpy(zero_copy_only=False)
        a['headquarters_codes'][pos:stop]=boolean_codes(d[SOURCE[3]])
        a['continuity_codes'][pos:stop]=boolean_codes(d[SOURCE[4]]);pos=stop
    if pos!=n:raise ValueError('Whole succession source coverage required')
    for value in a.values():value.flush()
    progress(dict(phase='every_raw_observation_read',observations=n))
    anchors=np.concatenate((a['predecessor_keys'],a['successor_keys']))
    order=np.argsort(anchors,kind='stable');sorted_keys=anchors[order]
    starts=np.r_[0,np.flatnonzero(sorted_keys[1:]!=sorted_keys[:-1])+1]
    endpoints=array(output,'endpoint_keys',np.uint64,(len(starts),));endpoints[:]=sorted_keys[starts]
    offsets=array(output,'endpoint_offsets',np.uint64,(len(starts)+1,));offsets[:-1]=starts;offsets[-1]=2*n
    incidences=array(output,'endpoint_incidences',np.uint32,(2*n,));incidences[:]=order
    # Incidence identity is its original physical observation orientation,
    # predecessor r or successor n+r. Self links retain BOTH raw incidences.
    del order,sorted_keys,starts,anchors
    for value in (endpoints,offsets,incidences):value.flush()
    parents=(endpoints//100000).astype(np.uint32)
    starts=np.r_[0,np.flatnonzero(parents[1:]!=parents[:-1])+1]
    parent_keys=array(output,'parent_keys',np.uint32,(len(starts),));parent_keys[:]=parents[starts]
    parent_offsets=array(output,'parent_endpoint_offsets',np.uint64,(len(starts)+1,));parent_offsets[:-1]=starts;parent_offsets[-1]=len(endpoints)
    del parents,starts
    for value in (parent_keys,parent_offsets):value.flush()
    progress(dict(phase='all_bidirectional_incidence_and_parent_ranges',endpoints=len(endpoints),parents=len(parent_keys)))
    # Exact all-row source reconstruction of EVERY field, including nulls,
    # leading zeros, continuity/transfer UNKNOWN vs FALSE vs TRUE and dates.
    pos=0
    for batch in f.iter_batches(batch_size=batch_rows,use_threads=False):
        d=dict(zip(SOURCE,batch.columns));stop=pos+len(batch)
        for key,field in (('predecessor_keys',SOURCE[0]),('successor_keys',SOURCE[1])):
            strings=pc.utf8_lpad(pc.cast(pa.array(a[key][pos:stop]),pa.string()),14,padding='0')
            if not strings.equals(d[field]):raise ValueError('Every original endpoint string must reconstruct exactly')
        for prefix,field,key in (('effective',SOURCE[2],'effective_days'),('updated',SOURCE[5],'updated_micros')):
            raw=pa.array(a[key][pos:stop],mask=~a[prefix+'_valid'][pos:stop]).cast(f.schema_arrow.field(field).type)
            if not raw.equals(d[field]):raise ValueError('Every original date/null must reconstruct exactly')
        for key,field in (('headquarters_codes',SOURCE[3]),('continuity_codes',SOURCE[4])):
            if not decode_codes(a[key][pos:stop]).equals(d[field]):raise ValueError('Every original UNKNOWN/FALSE/TRUE must reconstruct exactly')
        pos=stop
    # Full permutation certificate + sorted source-key reconstruction ensures
    # EVERY endpoint incidence once. No hash/sampled/fanout-limited check.
    seen=np.zeros(2*n,dtype=np.bool_);last=None
    for begin in range(0,2*n,batch_rows):
        end=min(2*n,begin+batch_rows);raw=incidences[begin:end]
        if np.any(raw>=2*n) or np.any(seen[raw]) or len(np.unique(raw))!=len(raw):raise ValueError('Every oriented source incidence once required')
        seen[raw]=True;observation=(raw%n).astype(np.int64);pred=raw<n
        original=np.where(pred,a['predecessor_keys'][observation],a['successor_keys'][observation])
        position=np.searchsorted(offsets,np.arange(begin,end,dtype=np.uint64),side='right')-1
        if not np.array_equal(endpoints[position],original):raise ValueError('Every incidence must reconstruct its original endpoint')
        if np.any(original[1:]<original[:-1]) or last is not None and int(original[0])<last:raise ValueError('Every original incidence endpoint order required')
        last=int(original[-1])
    if not np.all(seen):raise ValueError('Every source orientation required')
    del seen
    for begin in range(0,len(endpoints),batch_rows):
        end=min(len(endpoints),begin+batch_rows);position=np.searchsorted(parent_offsets,np.arange(begin,end,dtype=np.uint64),side='right')-1
        if not np.array_equal(parent_keys[position],endpoints[begin:end]//100000):raise ValueError('Every endpoint parent range must reconstruct exactly')
    self_links=int(np.sum(a['predecessor_keys']==a['successor_keys']))
    return dict(observations=n,endpoints=len(endpoints),parents=len(parent_keys),oriented_incidences=2*n,
        self_context_links=self_links,continuity_counts={str(k):int(np.sum(a['continuity_codes']==k)) for k in range(3)},
        headquarters_counts={str(k):int(np.sum(a['headquarters_codes']==k)) for k in range(3)},
        max_endpoint_incidences=int(np.max(np.diff(offsets))),every_original_field_reconstructed_exactly=True,
        every_oriented_incidence_once_and_exact=True,every_endpoint_parent_exact=True,
        observations_deduplicated=False,continuity_filtered=False,self_links_expanded=False,
        economic_continuity_not_identity_merge=True,identifiers_as_features=False,endpoint_or_relation_cap=None)
