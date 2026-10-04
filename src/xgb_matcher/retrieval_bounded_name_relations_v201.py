"""Same V198 proofs; bounded physical blocks in complete frozen one-hop only."""
from pathlib import Path
import time

import numpy as np

from . import retrieval_ram_name_counters_v198 as ram
from .retrieval_ram_name_counters_v198 import coverage, MAX_COUNTER_RAM_BYTES
from . import retrieval_paged_names_v170 as paged
from .retrieval_paged_names_v170 import DiskVector, PagedSnapshot, quantum, scatter, close_maps

MAX_RELATION_CHUNK = 131072


def one_hop(pc, sc, read_graph, endpoints, observations, directory, guard, chunk):
    quantum(chunk)
    return paged.one_hop(pc, sc, read_graph, endpoints, observations, directory, guard,
                         min(chunk, MAX_RELATION_CHUNK))


class BoundedRelationNames(ram.RamCountersNames):
    """Same complete certified owner/one-hop reader; only relation workspace quantum differs."""
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
                relation_physical_chunk=min(chunk, MAX_RELATION_CHUNK),
                aliases_released_before_relations=True,query_vectors_file_backed=True,float_score_precision_unchanged=True,
                recall_measured=False,initial100_emitted=False)
            guard();stats['seconds']=time.monotonic()-began
            return snapshot,stats
        except BaseException:
            for a in vectors:a.close()
            raise

