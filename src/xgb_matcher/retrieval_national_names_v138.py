"""Query all V134 official aliases and attach coherent proofs without fanout.

This is the name/one-hop evidence stage, not a complete retrieval scorer.
No pool, no source/candidate cap, no CRM-derived dictionary or GT access.
"""
import json
import time

import duckdb
import numpy as np

from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED,MISSING,one_hop_names
from src.xgb_matcher.retrieval_v20_workbench import verify


class PackedTerms:
    def __init__(self,arrays):
        self.a=arrays;self.terms=len(arrays['token_offsets'])-1
    def key(self,i):
        lo,hi=map(int,self.a['token_offsets'][i:i+2]);return self.a['token_data'][lo:hi].tobytes()
    def lookup(self,value):
        raw=value.encode('ascii');lo=0;hi=self.terms
        while lo<hi:
            mid=(lo+hi)//2
            if self.key(mid)<raw:lo=mid+1
            else:hi=mid
        if lo==self.terms or self.key(lo)!=raw:return self.a['post_ids'][:0]
        begin,end=map(int,self.a['post_offsets'][lo:lo+2]);return self.a['post_ids'][begin:end]


class NationalNames:
    """Verify once at opening, not every query. Every posting remains reachable."""
    def __init__(self,reference):
        self.metadata=json.loads(verify(reference).read_text());m=self.metadata
        if (m['status']!='SEALED_V134_COMPLETE_ALL_OFFICIAL_NAME_CHANNEL_ACCESS'
            or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index'] or m['ready_for_fold1']):
            raise ValueError('Protected complete official name access required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
        stage=json.loads(verify(m['source_stage']).read_text())
        if stage['names']!=m['names'] or any(stage.get(k) is not False for k in PROTECTED):
            raise ValueError('Unchanged official raw-alias namespace required')
        verify(stage['source']);self.source=stage['source'];self.names=m['names']
        cfg=launch['config'];self.buckets=cfg['term_buckets']
        if duckdb.__version__!=cfg['duckdb_version'] or np.__version__!=cfg['numpy_version']:
            raise ValueError('Pinned storage hash/unsigned library drift')
        self.c=duckdb.connect();self.c.execute('SET threads=1');self.c.execute("SET memory_limit='32MiB'")
        self.access={};posts=terms=0
        for b in m['access']:
            key=b['channel'],b['bucket']
            if key in self.access or not 0<=key[0]<4 or not 0<=key[1]<self.buckets:raise ValueError('Every physical access bucket once required')
            a={k.removesuffix('.npy'):np.load(verify(ref),mmap_mode='r',allow_pickle=False) for k,ref in b['artifacts'].items()}
            specs={'post_ids':(np.uint32,b['postings']),'post_offsets':(np.uint64,b['terms']+1),
                'token_offsets':(np.uint64,b['terms']+1),'token_data':(np.uint8,len(a['token_data']))}
            if set(a)!=set(specs) or any(a[k].dtype!=dtype or a[k].shape!=(n,) for k,(dtype,n) in specs.items()):
                raise ValueError('Whole typed official access arrays required')
            if (a['post_offsets'][0]!=0 or a['post_offsets'][-1]!=len(a['post_ids'])
                or a['token_offsets'][0]!=0 or a['token_offsets'][-1]!=len(a['token_data'])):
                raise ValueError('Whole vocabulary/posting CSR required')
            self.access[key]=PackedTerms(a);posts+=b['postings'];terms+=b['terms']
        if len(self.access)!=4*self.buckets or posts!=m['postings'] or terms!=m['terms']:
            raise ValueError('Complete all-channel denominator required')

    def close(self):
        self.c.close()
        for access in self.access.values():
            for a in access.a.values():a._mmap.close()

    def posting(self,channel,token):
        if not isinstance(channel,int) or isinstance(channel,bool) or not 0<=channel<4 or not isinstance(token,str) or not token.isascii():
            raise ValueError('Explicit official ASCII lexical channel required')
        b=int(self.c.execute('SELECT hash(?) % ?',[token,self.buckets]).fetchone()[0])
        return self.access[channel,b].lookup(token)

    def coverage(self,text,max_seconds=20.,chunk=131072):
        if not isinstance(text,str) or not np.isfinite(max_seconds) or max_seconds<=0 or not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0:
            raise ValueError('Explicit text and positive query cost guard required')
        began=time.monotonic();cov=np.zeros(self.names);exact=np.zeros(self.names,dtype=np.bool_)
        hits=np.zeros(self.names,dtype=np.uint32);statistics=[]
        def guard():
            if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_OFFICIAL_NAME_QUERY_COST')
        for channel,tokens in enumerate(channels(text)):
            hits.fill(0);postings=0
            if len(tokens)>=2**32:raise ValueError('Exact unsigned token count required')
            for token in sorted(tokens):
                values=self.posting(channel,token);postings+=len(values)
                for start in range(0,len(values),chunk):
                    guard();ids=values[start:start+chunk]
                    if np.any(ids>=self.names):raise ValueError('Every posting alias position required')
                    # Each term's alias IDs are globally sorted/unique in the
                    # sealed V134 source, so native indexed += is exact.
                    hits[ids]+=1
            if tokens:
                for start in range(0,self.names,chunk):
                    guard();sl=slice(start,start+chunk);cov[sl]=np.maximum(cov[sl],hits[sl]/len(tokens))
            statistics.append(dict(channel=channel,tokens=len(tokens),postings=postings,positive_aliases=int(np.count_nonzero(hits))))
        value=normalized(text)
        if value:
            ids=self.posting(3,value)
            for start in range(0,len(ids),chunk):
                guard();selected=ids[start:start+chunk]
                if np.any(selected>=self.names):raise ValueError('Every exact-feature alias position required')
                exact[selected]=True
        guard()
        return cov,exact,dict(seconds=time.monotonic()-began,channels=statistics,
            positive_aliases=int(np.count_nonzero(cov)),exact_aliases=int(np.count_nonzero(exact)),
            whole_alias_max=True,postings_or_alias_cap=None,source_hashes_checked_at_open_not_per_query=True)


def references(offsets,selected,chunk=131072):
    """Every positive alias→fact reference, bounded gather RAM, no fanout cap."""
    for start in range(0,len(selected),chunk):
        names=selected[start:start+chunk];begins=offsets[names].astype(np.int64)
        counts=(offsets[names+1]-offsets[names]).astype(np.int64);prefix=np.r_[0,np.cumsum(counts)];pos=0
        while pos<len(names):
            if counts[pos]>chunk:
                for begin in range(int(begins[pos]),int(begins[pos]+counts[pos]),chunk):
                    end=min(begin+chunk,int(begins[pos]+counts[pos]))
                    yield np.arange(begin,end,dtype=np.int64),np.full(end-begin,names[pos],dtype=np.int64)
                pos+=1;continue
            stop=min(len(names),max(pos+1,int(np.searchsorted(prefix,np.int64(prefix[pos]+chunk),side='right')-1)))
            lengths=counts[pos:stop];total=int(np.sum(lengths));relative=np.r_[0,np.cumsum(lengths[:-1])]
            if total:
                rows=np.repeat(begins[pos:stop]-relative,lengths)+np.arange(total,dtype=np.int64)
                yield rows,np.repeat(names[pos:stop],lengths)
            pos=stop


def scatter_positive(coverage,exact,source,bridge,parent_cov,parent_exact,site_cov,site_exact,chunk=131072):
    """Same bit-exact max/flags as V137, skipping only provably zero proofs."""
    offsets=source['name_fact_offsets'];subject=source['subject_code'];current=source['current_exact']
    pp=bridge['fact_parent_positions'];sp=bridge['fact_site_positions']
    if (coverage.dtype!=np.float64 or exact.dtype!=np.bool_ or coverage.ndim!=1 or exact.shape!=coverage.shape
        or offsets.dtype!=np.uint64 or offsets.shape!=(len(coverage)+1,) or offsets[0]!=0 or offsets[-1]!=len(subject)
        or np.any(offsets[1:]<=offsets[:-1])
        or subject.dtype!=np.uint8 or current.dtype!=np.bool_ or pp.dtype!=np.uint32 or sp.dtype!=np.uint32
        or any(a.shape!=subject.shape for a in (current,pp,sp)) or not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0
        or any(a.ndim!=1 for a in (subject,parent_cov,parent_exact,site_cov,site_exact))
        or parent_cov.dtype!=np.float64 or site_cov.dtype!=np.float64 or parent_exact.dtype!=np.bool_ or site_exact.dtype!=np.bool_
        or parent_cov.shape!=parent_exact.shape or site_cov.shape!=site_exact.shape
        or not np.isfinite(coverage).all() or np.any((coverage<0)|(coverage>1))):
        raise ValueError('Exact typed coherent alias/owner proofs required')
    selected=np.flatnonzero((coverage>0)|exact);processed=0
    for rows,aliases in references(offsets,selected,chunk):
        subjects=subject[rows];flags=exact[aliases]&current[rows]
        if np.any(~np.isin(subjects,[1,2])):raise ValueError('Explicit official owner grain required')
        for code,pos,out,ex in ((1,pp[rows],parent_cov,parent_exact),(2,sp[rows],site_cov,site_exact)):
            mask=subjects==code
            if code==2 and np.any(pos[mask]==MISSING):raise ValueError('Every official site proof must be admitted')
            mask&=pos!=MISSING
            if np.any(pos[mask]>=len(out)):raise ValueError('Every official owner position required')
            np.maximum.at(out,pos[mask],coverage[aliases[mask]]);np.logical_or.at(ex,pos[mask],flags[mask])
        processed+=len(rows)
    return dict(positive_aliases=len(selected),references_processed=processed,no_positive_reference_cap=True,
        zero_proofs_skipped_only=True,site_alias_lifted_to_parent=False)
