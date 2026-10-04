"""Exact full official name proofs in query-owned file-backed vectors.

No candidate reservoir, approximation or source/token/owner cap. V138/V149/
V137 arithmetic is unchanged; physical pages are closed only after copies or
completed updates. This is not a national recall or latency certificate.
"""
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np

from .retrieval_compact_coverage_v156 import counter_dtype
from .retrieval_coupled_blocks_v121 import channels
from .retrieval_name_pages_v151 import fingerprint
from .retrieval_national_names_v138 import references
from .retrieval_proof_bridge_v137 import MISSING
from .retrieval_rebound_names_v150 import ReboundNames,close_maps
from .retrieval_variants_v118 import normalized


def quantum(value):
    if type(value) is not int or value<=0:raise ValueError('Positive physical chunk required')
    return value


def rotate_posting(index,channel,token):
    """Release just the four source maps touched by this complete posting.

    All other buckets remain reachable. Closing/reopening 1024 untouched
    maps per token would add overhead but no extra semantic protection.
    """
    if not hasattr(index,'files'):
        if hasattr(index,'rotate'):index.rotate()
        return
    bucket=int(index.c.execute('SELECT hash(?) % ?',[token,index.buckets]).fetchone()[0])
    key=channel,bucket;access=index.access[key]
    try:
        for a in access.a.values():
            if not isinstance(a,np.memmap) or a.flags.writeable:raise ValueError('Only own readonly source posting maps may rotate')
            a._mmap.close()
        access.a={}
        for name,spec in index.files[key].items():
            if fingerprint(spec['path'])!=spec['fingerprint']:raise ValueError('Official name file epoch drift')
            a=np.load(spec['path'],mmap_mode='r',allow_pickle=False)
            if a.dtype!=spec['dtype'] or a.shape!=spec['shape'] or fingerprint(spec['path'])!=spec['fingerprint']:
                a._mmap.close();raise ValueError('Every unchanged readonly posting map required')
            access.a[name]=a
    except BaseException:
        index.valid=False;raise


class DiskVector:
    """Own a fresh .npy, never expose a borrowed view beyond its map lifetime."""
    def __init__(self,path,size,dtype):
        self.path=Path(path);self.size=size;self.dtype=np.dtype(dtype);self.valid=True
        if type(size) is not int or size<0 or self.path.exists():
            raise ValueError('Fresh whole query vector required')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        # A fresh extended file is zero-filled by the filesystem. No large
        # zero heap array or full resident memset is allocated.
        a=np.lib.format.open_memmap(self.path,mode='w+',dtype=self.dtype,shape=(size,))
        a._mmap.close()
        self.epoch=fingerprint(self.path);self.live_maps=0

    def __len__(self):return self.size

    @contextmanager
    def mapped(self,write=False):
        if not self.valid or self.live_maps or fingerprint(self.path)!=self.epoch:
            raise ValueError('Live unchanged singly owned query vector required')
        a=np.load(self.path,mmap_mode='r+' if write else 'r',allow_pickle=False)
        self.live_maps=1
        try:
            if a.dtype!=self.dtype or a.shape!=(self.size,):raise ValueError('Exact whole query vector shape required')
            yield a
        finally:
            a._mmap.close();self.live_maps=0
            if write:self.epoch=fingerprint(self.path)

    def take(self,ids):
        if (not isinstance(ids,np.ndarray) or ids.ndim!=1 or ids.dtype.kind not in 'iu'
            or np.any(ids<0) or np.any(ids>=self.size)):
            raise ValueError('Every actual query vector position required')
        with self.mapped() as a:return np.array(a[ids],copy=True)

    def read(self,start,stop):
        if type(start) is not int or type(stop) is not int or not 0<=start<=stop<=self.size:
            raise ValueError('Actual whole query vector interval required')
        with self.mapped() as a:return np.array(a[start:stop],copy=True)

    def assign(self,start,values):
        if values.dtype!=self.dtype or values.ndim!=1 or type(start) is not int or not 0<=start<=start+len(values)<=self.size:
            raise ValueError('Exact typed bounded query vector write required')
        with self.mapped(True) as a:a[start:start+len(values)]=values

    def reset(self,chunk,guard):
        for start in range(0,self.size,quantum(chunk)):
            guard();stop=min(self.size,start+chunk)
            with self.mapped(True) as a:a[start:stop]=0

    def count(self,chunk,guard,start=0,stop=None):
        stop=self.size if stop is None else stop;total=0
        for lo in range(start,stop,quantum(chunk)):
            guard();total+=int(np.count_nonzero(self.read(lo,min(stop,lo+chunk))))
        return total

    def maximum(self,ids,values):
        if (self.dtype!=np.float64 or values.dtype!=np.float64 or values.shape!=ids.shape
            or not np.isfinite(values).all() or np.any((values<0)|(values>1))):
            raise ValueError('Exact finite float64 proof maximum required')
        self.check_ids(ids)
        with self.mapped(True) as a:np.maximum.at(a,ids,values)

    def flags(self,ids,values):
        if self.dtype!=np.bool_ or values.dtype!=np.bool_ or values.shape!=ids.shape:
            raise ValueError('Exact original proof flags required')
        self.check_ids(ids)
        with self.mapped(True) as a:np.logical_or.at(a,ids,values)

    def check_ids(self,ids):
        if ids.dtype.kind not in 'iu' or ids.ndim!=1 or np.any(ids<0) or np.any(ids>=self.size):
            raise ValueError('Every actual proof vector position required')

    def close(self,remove=False):
        if self.live_maps:raise ValueError('No close with borrowed query pages')
        if self.valid and remove:
            if fingerprint(self.path)!=self.epoch:raise ValueError('Do not remove changed query vector')
            self.path.unlink()  # ONLY this explicitly owned fresh file.
        self.valid=False


def coverage(index,text,directory,guard,chunk=131072):
    """V138 max-over-whole-alias channels, exact counters with no saturation."""
    quantum(chunk)
    if not isinstance(text,str) or type(index.names) is not int or index.names<0:
        raise ValueError('Whole identical typed alias namespace required')
    directory=Path(directory);query=channels(text);bound=max(map(len,query));dtype=counter_dtype(bound)
    cov=DiskVector(directory/'alias_coverage.npy',index.names,np.float64)
    exact=DiskVector(directory/'alias_exact.npy',index.names,np.bool_)
    hits=DiskVector(directory/'token_counts.npy',index.names,dtype);statistics=[];began=time.monotonic()
    try:
        for channel,tokens in enumerate(query):
            hits.reset(chunk,guard);postings=0
            for token in sorted(tokens):
                guard();values=index.posting(channel,token);postings+=len(values);last=-1
                for start in range(0,len(values),chunk):
                    guard();ids=np.array(values[start:start+chunk],copy=True)
                    if (ids.dtype!=np.uint32 or np.any(ids>=index.names)
                        or (len(ids) and (int(ids[0])<=last or np.any(ids[1:]<=ids[:-1])))):
                        raise ValueError('Every sealed sorted unique token posting required')
                    if len(ids):last=int(ids[-1])
                    with hits.mapped(True) as a:
                        if np.any(a[ids]>=len(tokens)):raise ValueError('No exact token-counter overflow allowed')
                        a[ids]+=1
                del values
                rotate_posting(index,channel,token)
            positive=0
            for start in range(0,index.names,chunk):
                guard();stop=min(index.names,start+chunk);counts=hits.read(start,stop)
                positive+=int(np.count_nonzero(counts))
                if tokens:cov.assign(start,np.maximum(cov.read(start,stop),counts/len(tokens)))
            statistics.append(dict(channel=channel,tokens=len(tokens),postings=postings,positive_aliases=positive))
        value=normalized(text)
        if value:
            ids=index.posting(3,value)
            for start in range(0,len(ids),chunk):
                guard();selected=np.array(ids[start:start+chunk],copy=True)
                if selected.dtype!=np.uint32 or np.any(selected>=index.names):raise ValueError('Every exact-feature alias position required')
                exact.flags(selected,np.ones(len(selected),dtype=np.bool_))
            del ids
            rotate_posting(index,3,value)
        guard()
        stats=dict(seconds=time.monotonic()-began,channels=statistics,positive_aliases=cov.count(chunk,guard),
            exact_aliases=exact.count(chunk,guard),whole_alias_max=True,postings_or_alias_cap=None,
            source_hashes_checked_at_open_not_per_query=True,counter_dtype=dtype.name,counter_bound=bound,
            counter_bytes=index.names*dtype.itemsize,float_score_precision_unchanged=True,query_vectors_file_backed=True)
        hits.close(remove=True)
        return cov,exact,stats
    except BaseException:
        for a in (cov,exact,hits):a.close()  # Keep failed-query evidence, never publish partial.
        raise


def overlay(previous,rows,positions,selected,old_parents,total_parents):
    """Sparse V149 patches on selected original fact rows, no bucket copy."""
    if (previous.dtype!=np.uint32 or previous.ndim!=1 or rows.dtype!=np.uint32
        or rows.ndim!=1 or positions.dtype!=np.uint32 or positions.shape!=rows.shape
        or selected.dtype.kind not in 'iu' or selected.ndim!=1 or np.any(selected<0) or np.any(selected>=len(previous))
        or np.any(rows[1:]<=rows[:-1]) or np.any(rows>=len(previous))
        or np.any(positions<old_parents) or np.any(positions>=total_parents)
        or np.any(previous[rows]!=MISSING)):
        raise ValueError('Exact sorted unique sparse append-only patches required')
    result=np.array(previous[selected],copy=True)
    if np.any((result!=MISSING)&(result>=old_parents)):
        raise ValueError('Every unchanged previous parent position required')
    p=np.searchsorted(rows,selected)
    hit=p<len(rows);where=np.flatnonzero(hit);where=where[rows[p[where]]==selected[where]]
    result[where]=positions[p[where]]
    return result


def scatter(cov,exact,start,source,old,patch,pc,pe,sc,se,old_parents,guard,chunk):
    """All positive alias -> all original facts, zero proofs alone are skipped."""
    offsets=source['name_fact_offsets'];subject=source['subject_code'];current=source['current_exact'];processed=positive=0
    if (offsets.dtype!=np.uint64 or offsets.ndim!=1 or offsets[0]!=0 or offsets[-1]!=len(subject)
        or subject.dtype!=np.uint8 or current.dtype!=np.bool_ or current.shape!=subject.shape
        or old['fact_parent_positions'].dtype!=np.uint32 or old['fact_site_positions'].dtype!=np.uint32
        or any(v.shape!=subject.shape for v in old.values()) or start+len(offsets)-1>len(cov)):
        raise ValueError('Whole typed original alias-owner bucket required')
    for lo in range(0,len(offsets)-1,quantum(chunk)):
        guard();hi=min(len(offsets)-1,lo+chunk)
        if np.any(offsets[lo+1:hi+1]<=offsets[lo:hi]):raise ValueError('Every nonempty original alias range required')
        values=cov.read(start+lo,start+hi);flags=exact.read(start+lo,start+hi)
        if not np.isfinite(values).all() or np.any((values<0)|(values>1)) or np.any(flags&(values!=1.)):
            raise ValueError('Exact coherent whole-alias coverage required')
        selected=lo+np.flatnonzero((values>0)|flags);positive+=len(selected)
        for rows,aliases in references(offsets,selected,chunk):
            guard();s=np.array(subject[rows],copy=True);e=flags[aliases-lo]&current[rows];v=values[aliases-lo]
            if np.any(~np.isin(s,[1,2])):raise ValueError('Explicit official owner grain required')
            pp=overlay(old['fact_parent_positions'],patch['fact_rows'],patch['parent_positions'],rows,old_parents,len(pc))
            sp=np.array(old['fact_site_positions'][rows],copy=True)
            for code,pos,out,ex in ((1,pp,pc,pe),(2,sp,sc,se)):
                mask=s==code
                if code==2 and np.any(pos[mask]==MISSING):raise ValueError('Every official site proof must be admitted')
                mask&=pos!=MISSING
                out.maximum(pos[mask],v[mask]);ex.flags(pos[mask],e[mask])
            processed+=len(rows)
    return dict(positive_aliases=positive,references_processed=processed,no_positive_reference_cap=True,
        zero_proofs_skipped_only=True,site_alias_lifted_to_parent=False)


def one_hop(pc,sc,read_graph,endpoints,observations,directory,guard,chunk):
    """V137 frozen endpoint snapshot, two directions, no exact or second hop."""
    snapshot=DiskVector(Path(directory)/'endpoint_snapshot.npy',endpoints,np.float64)
    relation=DiskVector(Path(directory)/'relation_coverage.npy',len(sc),np.float64)
    try:
        for lo in range(0,endpoints,quantum(chunk)):
            guard();ids=np.arange(lo,min(endpoints,lo+chunk),dtype=np.uint32)
            site=read_graph('graph_site_positions',ids);parent=read_graph('graph_parent_positions',ids)
            snapshot.assign(lo,np.maximum(sc.take(site),pc.take(parent)))
        for lo in range(0,observations,chunk):
            guard();ids=np.arange(lo,min(observations,lo+chunk),dtype=np.uint32)
            p=read_graph('predecessor_graph_positions',ids);s=read_graph('successor_graph_positions',ids)
            c=read_graph('continuity_codes',ids)
            if p.dtype!=np.uint32 or s.dtype!=np.uint32 or c.dtype!=np.uint8 or np.any(c>2):
                raise ValueError('Every original endpoint and continuity code required')
            mask=p!=s;p=p[mask];s=s[mask];grade=np.where(c[mask]==2,.5,.25)
            relation.maximum(read_graph('graph_site_positions',p),snapshot.take(s)*grade)
            relation.maximum(read_graph('graph_site_positions',s),snapshot.take(p)*grade)
        guard();snapshot.close(remove=True);return relation
    except BaseException:
        snapshot.close();relation.close();raise


class PagedSnapshot:
    def __init__(self,reader,pc,pe,sc,se,relation):
        self.reader=reader;self.sites=reader.sites;self.vectors=(pc,pe,sc,se,relation);self.valid=True

    def gather(self,ids):
        r=self.reader
        if not self.valid or r.keys is None:raise ValueError('Closed official paged name snapshot')
        if ids.dtype!=np.uint32 or ids.ndim!=1 or np.any(ids>=self.sites):raise ValueError('Every actual site position required')
        pc,pe,sc,se,relation=self.vectors;old=ids<r.old_sites;parents=np.empty(len(ids),dtype=np.uint32)
        parents[old]=r.read_graph('site_parent_positions',ids[old])
        parents[~old]=r.new['new_site_parent_positions'][ids[~old]-r.old_sites]
        coverage=pc.take(parents);exact=pe.take(parents).astype(np.float64)
        coverage[old]=np.maximum(coverage[old],np.maximum(sc.take(ids[old]),relation.take(ids[old])))
        exact[old]=np.maximum(exact[old],se.take(ids[old]))
        return coverage,exact

    def close(self,remove=False):
        for a in self.vectors:a.close(remove=remove)
        self.valid=False


class PagedReboundNames(ReboundNames):
    """Same strict complete V150 source audit, bounded query-owned features."""
    def rotate_keys(self):
        if self.keys is None:return
        for owner in (self.keys,self.keys.previous):
            for name in ('site','parent','extra_site','extra_parent','new_site','new_parent'):
                a=getattr(owner,name,None)
                if isinstance(a,np.memmap):setattr(owner,name,self.remap(a))

    @staticmethod
    def remap(a):
        path=Path(a.filename);shape=a.shape;dtype=a.dtype;epoch=fingerprint(path)
        if a.flags.writeable:raise ValueError('Only own readonly official source maps may rotate')
        a._mmap.close();b=np.load(path,mmap_mode='r',allow_pickle=False)
        if b.shape!=shape or b.dtype!=dtype or fingerprint(path)!=epoch:
            b._mmap.close();raise ValueError('Official source epoch/shape drift during page rotation')
        return b

    def map(self,ref):
        # AppendKeys may touch many identity pages during its own full audit;
        # release only those owned pages before subsequent proof validations.
        self.rotate_keys()
        return super().map(ref)

    def __init__(self,reference):
        super().__init__(reference)
        self.rotate_keys()
        for key,a in self.graph.items():self.graph[key]=self.remap(a)

    def read_graph(self,name,ids):
        a=self.graph[name]
        if ids.dtype!=np.uint32 or ids.ndim!=1 or np.any(ids>=len(a)):
            raise ValueError('Every actual original graph position required')
        result=np.array(a[ids],copy=True)
        self.graph[name]=self.remap(a)
        return result

    def query(self,index,text,directory,max_seconds=20.,chunk=131072):
        if (self.keys is None or index.names!=self.names or index.source!=self.source
            or not isinstance(text,str) or isinstance(max_seconds,bool) or not np.isfinite(max_seconds) or max_seconds<=0):
            raise ValueError('Whole identical alias source, text and physical cost guard required')
        quantum(chunk);directory=Path(directory)
        if directory.exists():raise ValueError('Fresh immutable query scratch required')
        directory.mkdir(parents=True,exist_ok=False);began=time.monotonic();vectors=[]
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
            scatter_done=time.monotonic();cov.close(remove=True);exact.close(remove=True)
            relation=one_hop(pc,sc,self.read_graph,len(self.graph['graph_site_positions']),
                len(self.graph['predecessor_graph_positions']),directory,guard,chunk)
            vectors.append(relation);relation_done=time.monotonic()
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
