"""All V130 name/owner facts in array-only CSR, without site expansion.

Every original observation and its27 fields remain in sealed V129/V130.
Names and IDs are joins/array positions, never retrieval score features.
"""
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA=pa.schema([('fact_id',pa.uint32()),('name_id',pa.uint32()),('parent_key',pa.uint32()),
    ('owner_key',pa.uint64()),('subject_code',pa.uint8()),('scope_code',pa.uint8()),('current_exact',pa.bool_())])
FIELDS={'parent_key':np.uint32,'owner_key':np.uint64,'subject_code':np.uint8,
    'scope_code':np.uint8,'current_exact':np.bool_}


def build(source,output,names,facts,batch_rows=131072):
    """Stream every fact, then reconstruct every source tuple through CSR.

    Physical V130 fact order is already (name,parent,owner,subject,scope).
    No owner→site Cartesian product, source/date pruning, or fanout cap.
    Owners outside the candidate membership remain (needed for relations).
    """
    output=Path(output);f=pq.ParquetFile(source)
    if (not isinstance(names,int) or isinstance(names,bool) or not 0<=names<2**32
        or not isinstance(facts,int) or isinstance(facts,bool) or not 0<=facts<2**32
        or not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0
        or f.metadata.num_rows!=facts or not f.schema_arrow.equals(SCHEMA)):
        raise ValueError('Whole typed V130 name/owner source required')
    arrays={key:np.lib.format.open_memmap(output/(key+'.npy'),mode='w+',dtype=dtype,shape=(facts,)) for key,dtype in FIELDS.items()}
    offsets=np.lib.format.open_memmap(output/'name_fact_offsets.npy',mode='w+',dtype=np.uint64,shape=(names+1,))
    counts=np.zeros(names,dtype=np.uint64);pos=0;last=None;subject_counts={1:0,2:0};scope_counts={1:0,2:0,3:0};current_count=0
    for batch in f.iter_batches(batch_size=batch_rows,use_threads=False):
        if any(column.null_count for column in batch.columns):raise ValueError('Every original fact field must be present')
        columns={name:batch.column(i).to_numpy(zero_copy_only=False) for i,name in enumerate(SCHEMA.names)}
        ids=columns['fact_id'];name=columns['name_id'];parent=columns['parent_key'];owner=columns['owner_key']
        subject=columns['subject_code'];scope=columns['scope_code'];n=len(batch)
        if not np.array_equal(ids,np.arange(pos,pos+n,dtype=np.uint32)) or np.any(name>=names):
            raise ValueError('Every original contiguous fact/name position required')
        if (np.any(parent>999999999) or np.any(owner>99999999999999)
            or np.any(~np.isin(subject,[1,2])) or np.any(~np.isin(scope,[1,2,3]))
            or np.any((subject==1)&((owner!=parent)|~np.isin(scope,[1,2])))
            or np.any((subject==2)&((owner//100000!=parent)|(scope!=3)))):
            raise ValueError('Explicit typed official owner/scope required')
        # Structured comparison of full tuples (not only name order) checks
        # uniqueness and every source-order boundary, including batch edges.
        dtype=np.dtype([(key,columns[key].dtype) for key in ('name_id','parent_key','owner_key','subject_code','scope_code')])
        keys=np.empty(n,dtype=dtype)
        for key in keys.dtype.names:keys[key]=columns[key]
        order=np.argsort(keys,kind='stable')
        if not np.array_equal(order,np.arange(n)) or (n>1 and np.any(keys[1:]==keys[:-1])):
            raise ValueError('All unique owner facts must retain their original sorted order')
        first=tuple(int(columns[key][0]) for key in keys.dtype.names)
        if last is not None and first<=last:raise ValueError('Whole owner ordering across batches required')
        last=tuple(int(columns[key][-1]) for key in keys.dtype.names)
        np.add.at(counts,name,1)
        for key,value in arrays.items():value[pos:pos+n]=columns[key]
        for key in subject_counts:subject_counts[key]+=int(np.sum(subject==key))
        for key in scope_counts:scope_counts[key]+=int(np.sum(scope==key))
        current_count+=int(np.sum(columns['current_exact']));pos+=n
    if pos!=facts or (names and np.any(counts==0)):raise ValueError('Every raw name and whole owner fact required')
    offsets[0]=0;np.cumsum(counts,dtype=np.uint64,out=offsets[1:])
    if int(offsets[-1])!=facts:raise ValueError('Whole name-owner CSR coverage required')
    for a in (*arrays.values(),offsets):a.flush()
    # Every row is audited against an independent source re-read, including
    # reconstruction of name_id via its CSR range. No checksum/sample oracle.
    pos=0
    for batch in f.iter_batches(batch_size=batch_rows,use_threads=False):
        n=len(batch);name=batch.column(SCHEMA.get_field_index('name_id')).to_numpy(zero_copy_only=False)
        positions=np.arange(pos,pos+n,dtype=np.uint64)
        if not np.array_equal(np.searchsorted(offsets,positions,side='right')-1,name):
            raise ValueError('Every name must reconstruct through owner CSR exactly')
        for key,value in arrays.items():
            expected=batch.column(SCHEMA.get_field_index(key)).to_numpy(zero_copy_only=False)
            if not np.array_equal(value[pos:pos+n],expected):raise ValueError('Every original owner/scope/current field must reconstruct exactly')
        pos+=n
    if pos!=facts:raise ValueError('Whole independent fact audit required')
    return dict(names=names,facts=facts,subject_counts={str(k):v for k,v in subject_counts.items()},
        scope_counts={str(k):v for k,v in scope_counts.items()},current_exact_true=current_count,
        every_original_fact_reconstructed_exactly=True,every_raw_name_has_original_owner=True,
        all_facts_sorted_unique=True,original_observations_unchanged=True,
        owners_outside_candidate_catalog_preserved=True,site_cartesian_product_materialized=False,
        owner_or_name_or_period_cap=None,identifiers_as_features=False)
