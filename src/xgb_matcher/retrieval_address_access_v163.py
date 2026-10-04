"""V153-policy reader; unchanged V152 whole-address features and postings."""
import json
from pathlib import Path

import duckdb
import numpy as np

from .retrieval_address_access_v152 import SCHEMA,physical_plan,stage_addresses
from .retrieval_address_reverse_v147 import AliasSnapshot
from .retrieval_national_names_v138 import NationalNames,PackedTerms
from .retrieval_v20_workbench import verify

STATUS='SEALED_V163_ALL_OFFICIAL_ADDRESS_CHANNEL_ACCESS'



class NationalAddresses:
    """Complete access authority -> readonly V147 whole-alias query snapshot.

    Opening validates the V163 build and its complete V160 authority. This
    reader is not a national retrieval certificate: no initial100 or recall
    is produced here. Public postings are copies, internal query uses slices.
    """
    def __init__(self,reference):
        from scripts.compile_retrieval_v163_address_access import load,complete_stage,validate_access
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
