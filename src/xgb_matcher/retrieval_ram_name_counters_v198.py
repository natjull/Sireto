"""Exact V170 name proofs: small integral counters in RAM, no semantic change.

The float64 coverage and bool flags remain file-backed. ALL tokens/postings
are processed; larger counters use the original disk implementation, never
a token/source cap. Caller still supervises and FULL hashes every source.
"""
from pathlib import Path
import time

import numpy as np

from . import retrieval_paged_names_v170 as paged
from .retrieval_paged_names_v170 import (DiskVector, PagedSnapshot, quantum, scatter, one_hop,
                                       close_maps, counter_dtype, rotate_posting, channels, normalized)

MAX_COUNTER_RAM_BYTES = 64 * 1024**2


def coverage(index, text, directory, guard, chunk=131072):
    quantum(chunk)
    if not isinstance(text, str) or type(index.names) is not int or index.names < 0:
        raise ValueError('Whole identical typed alias namespace required')
    terms = channels(text); bound = max(map(len, terms)); dtype = counter_dtype(bound)
    if dtype.kind != 'u' or np.iinfo(dtype).max < bound:
        raise ValueError('Exact integral counter dtype must represent the FULL query bound')
    if index.names*dtype.itemsize > MAX_COUNTER_RAM_BYTES:
        cov, exact, stats = paged.coverage(index, text, directory, guard, chunk)
        return cov, exact, dict(stats, counter_backing='ORIGINAL_DISK_EXACT_FALLBACK', counter_ram_bytes=0)
    directory = Path(directory); began = time.monotonic()
    cov = DiskVector(directory / 'alias_coverage.npy', index.names, np.float64)
    exact = DiskVector(directory / 'alias_exact.npy', index.names, np.bool_)
    hits = np.zeros(index.names, dtype=dtype); statistics = []
    try:
        for channel, tokens in enumerate(terms):
            guard(); hits.fill(0); postings = 0
            for token in sorted(tokens):
                guard(); values = index.posting(channel, token); postings += len(values); last = -1
                for start in range(0, len(values), chunk):
                    guard(); ids = np.array(values[start:start+chunk], copy=True)
                    if (ids.dtype != np.uint32 or ids.ndim != 1 or np.any(ids >= index.names)
                            or (len(ids) and (int(ids[0]) <= last or np.any(ids[1:] <= ids[:-1])))):
                        raise ValueError('Every sealed sorted unique token posting required')
                    if len(ids): last = int(ids[-1])
                    if np.any(hits[ids] >= len(tokens)):
                        raise ValueError('No exact token-counter overflow allowed')
                    hits[ids] += 1
                del values
                rotate_posting(index, channel, token)
            positive = 0
            for start in range(0, index.names, chunk):
                guard(); stop = min(index.names, start+chunk); counts = hits[start:stop]
                positive += int(np.count_nonzero(counts))
                if tokens: cov.assign(start, np.maximum(cov.read(start, stop), counts/len(tokens)))
            statistics.append(dict(channel=channel, tokens=len(tokens), postings=postings, positive_aliases=positive))
        value = normalized(text)
        if value:
            ids = index.posting(3, value)
            for start in range(0, len(ids), chunk):
                guard(); selected = np.array(ids[start:start+chunk], copy=True)
                if selected.dtype != np.uint32 or selected.ndim != 1 or np.any(selected >= index.names):
                    raise ValueError('Every exact-feature alias position required')
                exact.flags(selected, np.ones(len(selected), dtype=np.bool_))
            del ids
            rotate_posting(index, 3, value)
        guard()
        return cov, exact, dict(seconds=time.monotonic()-began, channels=statistics,
            positive_aliases=cov.count(chunk, guard), exact_aliases=exact.count(chunk, guard),
            whole_alias_max=True, postings_or_alias_cap=None, source_hashes_checked_at_open_not_per_query=True,
            counter_dtype=dtype.name, counter_bound=bound, counter_bytes=index.names*dtype.itemsize,
            counter_backing='RAM_INTEGRAL_ONLY_FULL_NAMESPACE', counter_ram_bytes=hits.nbytes,
            float_score_precision_unchanged=True, query_vectors_file_backed=True)
    except BaseException:
        cov.close(); exact.close()  # Own failed evidence retained; no snapshot returned.
        raise


class RamCountersNames(paged.PagedReboundNames):
    """Same complete certified owner/one-hop reader; only coverage workspace differs."""
    def query(self,index,text,directory,max_seconds=20.,chunk=131072,progress=None):
        if (self.keys is None or index.names!=self.names or index.source!=self.source
            or not isinstance(text,str) or isinstance(max_seconds,bool) or not np.isfinite(max_seconds) or max_seconds<=0):
            raise ValueError('Whole identical alias source, text and physical cost guard required')
        quantum(chunk);directory=Path(directory)
        if directory.exists():raise ValueError('Fresh immutable query scratch required')
        directory.mkdir(parents=True,exist_ok=False);began=time.monotonic();vectors=[]
        def checkpoint(stage,**stats):
            if progress is not None:progress(dict(stage=stage,seconds=time.monotonic()-began,**stats))
        def guard():
            if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_PAGED_NAME_QUERY_COST')
        try:
            if hasattr(index,'in_coverage'):
                if not index.valid or index.in_coverage:raise ValueError('Single live official alias query required')
                index.in_coverage=True
            try:cov,exact,lexical=coverage(index,text,directory,guard,chunk)
            finally:
                if hasattr(index,'in_coverage'):index.in_coverage=False
            vectors.extend((cov,exact));lex_done=time.monotonic()
            checkpoint('LEXICAL_COMPLETE',positive_aliases=lexical.get('positive_aliases'))
            pc=DiskVector(directory/'parent_coverage.npy',self.parents,np.float64)
            pe=DiskVector(directory/'parent_exact.npy',self.parents,np.bool_)
            sc=DiskVector(directory/'site_coverage.npy',self.old_sites,np.float64)
            se=DiskVector(directory/'site_exact.npy',self.old_sites,np.bool_)
            vectors.extend((pc,pe,sc,se));processed=0
            for item in self.items:
                guard();b=item['source_bucket'];src={};old={};patch={}
                try:
                    src={k:np.load(b['artifacts'][k+'.npy']['path'],mmap_mode='r',allow_pickle=False) for k in ('name_fact_offsets','subject_code','current_exact')}
                    old={k.removesuffix('.npy'):np.load(ref['path'],mmap_mode='r',allow_pickle=False) for k,ref in item['previous_artifacts'].items()}
                    patch={k.removesuffix('.npy'):np.load(ref['path'],mmap_mode='r',allow_pickle=False) for k,ref in item['artifacts'].items()}
                    trace=scatter(cov,exact,b['name_start'],src,old,patch,pc,pe,sc,se,self.old_parents,guard,chunk)
                    processed+=trace['references_processed']
                finally:close_maps(src);close_maps(old);close_maps(patch)
                checkpoint('OWNER_BUCKET_COMPLETE',bucket=item['bucket'],references_processed=processed)
            scatter_done=time.monotonic();cov.close(remove=True);exact.close(remove=True)
            checkpoint('SCATTER_COMPLETE',references_processed=processed)
            relation=one_hop(pc,sc,self.read_graph,len(self.graph['graph_site_positions']),
                len(self.graph['predecessor_graph_positions']),directory,guard,chunk)
            vectors.append(relation);relation_done=time.monotonic()
            checkpoint('ONE_HOP_COMPLETE')
            snapshot=PagedSnapshot(self,pc,pe,sc,se,relation)
            ids=np.arange(self.old_sites,self.sites,dtype=np.uint32);nc,ne=snapshot.gather(ids)
            stats=dict(lexical_seconds=lex_done-began,scatter_seconds=scatter_done-lex_done,relation_seconds=relation_done-scatter_done,
                lexical=lexical,references_processed=processed,original_parent_proofs=pc.count(chunk,guard,stop=self.old_parents),
                appended_parent_proofs=pc.count(chunk,guard,start=self.old_parents),appended_parent_exact=pe.count(chunk,guard,start=self.old_parents),
                direct_site_proofs=sc.count(chunk,guard),related_endpoint_proofs=relation.count(chunk,guard),
                appended_sites_with_name_proofs=int(np.count_nonzero(nc)),appended_sites_with_direct_exact=int(np.count_nonzero(ne)),
                all_sealed_parent_patches_applied=True,no_alias_or_owner_or_member_cap=True,site_alias_lifted_to_parent=False,
                aliases_released_before_relations=True,query_vectors_file_backed=True,float_score_precision_unchanged=True,
                recall_measured=False,initial100_emitted=False)
            guard();stats['seconds']=time.monotonic()-began
            return snapshot,stats
        except BaseException:
            for a in vectors:a.close()
            raise
