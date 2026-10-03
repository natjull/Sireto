"""Whole-address lexical access for V145/V147, not a candidate pool.

Reuse the unchanged scalar channels and exact packed postings from V131/134.
Aliases identify whole source strings; CP/INSEE/number/current flags remain
on their OWN site facts in V147. Empty guarded feature strings retain their
original alias positions. No owner join, alias cap or learned CRM dictionary.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_address_reverse_v147 import AliasSnapshot
from .retrieval_name_access_v131 import stage
from .retrieval_national_names_v138 import NationalNames,PackedTerms
from .retrieval_proof_bridge_v137 import PROTECTED
from .retrieval_v20_workbench import verify

SCHEMA=pa.schema([('address_id',pa.uint32()),('address_text',pa.string())])
STATUS='SEALED_V152_ALL_OFFICIAL_ADDRESS_CHANNEL_ACCESS'


def stage_addresses(c,table,output,alias_start,row_start,buckets):
    if table.schema!=SCHEMA:
        raise ValueError('Exact original whole-address dictionary schema required')
    if (any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (alias_start,row_start,buckets))
        or not 1<=buckets<=254 or alias_start+row_start+len(table)>=2**32):
        raise ValueError('Complete typed address/storage namespace required')
    # Column rename ONLY, not a new name normalization or owner propagation.
    result=stage(c,table.rename_columns(['name_id','name_text']),output,alias_start,row_start,buckets)
    result['addresses']=result.pop('names')
    result['all_raw_addresses_retained']=result.pop('all_raw_names_retained')
    result.update(address_fact_attributes_or_scope_changed=False,alias_or_proof_cap=None,
        source_address_normalization_unchanged=True)
    return result


def physical_plan(buckets,aliases,batch_rows=65536):
    """All dictionary rowgroups/slices, incl empty buckets, no query selection."""
    if (not isinstance(aliases,int) or isinstance(aliases,bool) or not 0<=aliases<2**32
        or not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0):
        raise ValueError('Whole typed address namespace and physical batch required')
    cursor=0;parts=[]
    for index,b in enumerate(buckets):
        if b['bucket']!=index or b['alias_start']!=cursor:
            raise ValueError('Every original dictionary bucket/alias interval exactly once required')
        reference=b['artifacts']['addresses.parquet'];f=pq.ParquetFile(verify(reference))
        if f.schema_arrow!=SCHEMA or f.metadata.num_rows!=b['aliases']:
            raise ValueError('Whole original address dictionary required')
        row=0
        for group in range(f.metadata.num_row_groups):
            length=f.metadata.row_group(group).num_rows
            for begin in range(0,length,batch_rows):
                size=min(batch_rows,length-begin)
                parts.append(dict(part_id=len(parts),source_bucket=index,source=reference,
                    alias_start=cursor,row_group=group,row_start=row+begin,row_stop=row+begin+size,
                    group_slice_start=begin,group_slice_stop=begin+size))
            row+=length
        if row!=b['aliases']:raise ValueError('Every original rowgroup required')
        cursor+=row
    if cursor!=aliases:raise ValueError('Whole address alias denominator required')
    return parts


class NationalAddresses:
    """Complete access authority -> readonly V147 whole-alias query snapshot.

    Opening validates the V152 build and its original V148 authority. This
    reader is not a national retrieval certificate: no initial100 or recall
    is produced here. Public postings are copies, internal query uses slices.
    """
    def __init__(self,reference):
        from scripts.compile_retrieval_v152_address_access import load,complete_stage,validate_access
        self.valid=False;self.in_coverage=False;self.access={};self.c=None;self.maps=[]
        path=verify(reference);root=path.parent
        if path.resolve()!=(root/'manifest.json').resolve():raise ValueError('Exact sealed address access manifest required')
        launch,plan=load(root);staged=complete_stage(root,launch,plan)
        self.metadata=json.loads(path.read_text());validate_access(root,launch,plan,staged,self.metadata)
        self.aliases=plan['aliases'];self.names=self.aliases  # V138's raw-alias vector length, not names as evidence.
        self.source=plan['source_store'];self.buckets=launch['config']['term_buckets'];self.references=[]
        try:
            self.c=duckdb.connect();self.c.execute('SET threads=1');self.c.execute("SET memory_limit='32MiB'")
            for item in self.metadata['access']:
                key=item['channel'],item['bucket'];arrays={}
                for name,ref in item['artifacts'].items():
                    arrays[name.removesuffix('.npy')]=np.load(verify(ref),mmap_mode='r',allow_pickle=False)
                    self.maps.append(arrays[name.removesuffix('.npy')])
                    self.references.append(ref)
                self.access[key]=PackedTerms(arrays)
            self.valid=True
        except BaseException:
            self.close();raise

    def posting(self,channel,token):
        if not self.valid:raise ValueError('Closed address access reader')
        values=NationalNames.posting(self,channel,token)
        return values if self.in_coverage else values.copy()

    def snapshot(self,text,max_seconds=20.,chunk=131072):
        if not self.valid or self.in_coverage:raise ValueError('Single live whole-address query required')
        self.in_coverage=True
        try:cov,exact,stats=NationalNames.coverage(self,text,max_seconds,chunk)
        finally:self.in_coverage=False
        cov.setflags(write=False);exact.setflags(write=False)
        return AliasSnapshot(cov,exact),dict(stats,scope='WHOLE_ADDRESS_ALIAS_ONLY',
            site_or_parent_proof_propagated=False,candidates_emitted=False)

    def verify_after(self):
        if not self.valid:raise ValueError('Live address access required')
        for ref in self.references:verify(ref)
        verify(self.source)

    def close(self):
        self.valid=False
        if self.c is not None:self.c.close();self.c=None
        for a in self.maps:a._mmap.close()
        self.maps=[]
        for access in self.access.values():access.a={}
