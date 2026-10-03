"""Release only this reader's file-backed pages after each complete query.

Same V138 lexical vectors and complete corpus; no approximation, posting cap
or global cache purge. Returned feature vectors own their storage.
"""
from pathlib import Path
import time

import numpy as np
import psutil

from .retrieval_national_names_v138 import NationalNames


def fingerprint(path):
    s=Path(path).stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns


class ScopedNamePages(NationalNames):
    """Single-query reader: source mmap residency cannot accumulate by query.

Public postings are copies; only internal coverage borrows mapped slices.
Inputs still receive full hashes before/after the entire probe. Remapping
checks inode/size/mtime and every dtype/shape, without repeated full hashing.
"""
    def __init__(self,reference):
        super().__init__(reference);self.epoch=0;self.valid=True;self.in_coverage=False;self.files={}
        for item in self.metadata['access']:
            key=item['channel'],item['bucket'];self.files[key]={}
            for name,ref in item['artifacts'].items():
                name=name.removesuffix('.npy');a=self.access[key].a[name];path=Path(ref['path'])
                if not isinstance(a,np.memmap) or a.flags.writeable:
                    raise ValueError('Only readonly official file-backed maps may rotate')
                self.files[key][name]=dict(path=path,fingerprint=fingerprint(path),shape=a.shape,dtype=a.dtype)

    def posting(self,channel,token):
        if not self.valid:raise ValueError('Closed or failed source page reader')
        values=super().posting(channel,token)
        return values if self.in_coverage else values.copy()

    def rotate(self):
        if not self.valid:raise ValueError('Closed or failed source page reader')
        began=time.monotonic();before=psutil.Process().memory_info().rss;maps=0
        try:
            # No source slice remains in a successful coverage return. Close
            # ALL old maps before reopening, so none can retain old residency.
            for access in self.access.values():
                for a in access.a.values():a._mmap.close()
                access.a={}
            for key,files in self.files.items():
                for name,s in files.items():
                    if fingerprint(s['path'])!=s['fingerprint']:raise ValueError('Official name file epoch drift')
                    a=np.load(s['path'],mmap_mode='r',allow_pickle=False)
                    if (not isinstance(a,np.memmap) or a.flags.writeable or a.dtype!=s['dtype'] or a.shape!=s['shape']
                        or fingerprint(s['path'])!=s['fingerprint']):
                        if isinstance(a,np.memmap):a._mmap.close()
                        raise ValueError('Exact unchanged readonly alias access required')
                    self.access[key].a[name]=a;maps+=1
            self.epoch+=1
            return dict(epoch=self.epoch,maps_reopened=maps,seconds=time.monotonic()-began,
                rss_before_bytes=before,rss_after_bytes=psutil.Process().memory_info().rss,
                full_alias_and_posting_namespace_unchanged=True,global_cache_purged=False)
        except BaseException:
            self.valid=False;raise

    def coverage(self,text,max_seconds=20.,chunk=131072):
        if not self.valid or self.in_coverage:raise ValueError('Single live name-page query required')
        self.in_coverage=True
        try:cov,exact,stats=super().coverage(text,max_seconds=max_seconds,chunk=chunk)
        finally:self.in_coverage=False
        if not cov.flags.owndata or not exact.flags.owndata:
            raise ValueError('Whole alias feature outputs must own storage before source maps close')
        rotation=self.rotate();stats=dict(stats,source_page_rotation=rotation)
        return cov,exact,stats

    def close(self):
        if getattr(self,'valid',False):
            super().close();self.valid=False
        elif hasattr(self,'access'):
            # A failed rotation may have reopened only a prefix. Never query
            # it or silently reinterpret that prefix as a complete index.
            self.c.close()
            for access in self.access.values():
                for a in access.a.values():a._mmap.close()
