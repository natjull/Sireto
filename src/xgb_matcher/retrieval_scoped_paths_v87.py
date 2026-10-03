"""Initial ranked search adapters: same-alias words, official scope, one hop.

The graph branch is explicitly limited to V84 endpoints. Random access never
gives a name score to an unexpanded non-graph sibling. Native corpus is intact.
"""
from pathlib import Path
import math
import time

import numpy as np

from src.xgb_matcher.retrieval_joint_v46 import SpatialStream,geo_features,geo_parameters
from src.xgb_matcher.retrieval_proof_v56 import signal
from src.xgb_matcher.retrieval_ranked_path_v83 import collect
from src.xgb_matcher.retrieval_sparse_native_v60 import query as native_query

GRADES={1:1.,2:.5,3:.25}  # hypothesis, not identity probabilities


class OwnerFacts:
    def __init__(self,directory):
        root=Path(directory)
        self.a={k:np.load(root/(k+'.npy'),mmap_mode='r',allow_pickle=False)
                for k in ('fact_owners','fact_parents','fact_subjects','fact_scopes')}
        self.owner={label:{k:np.load(root/(label+'_owner_'+k+'.npy'),mmap_mode='r',allow_pickle=False)
                           for k in ('keys','offsets','facts')} for label in ('parent','site')}

    def references(self,label,key):
        a=self.owner[label]; p=int(np.searchsorted(a['keys'],int(key)))
        if p==len(a['keys']) or a['keys'][p]!=int(key): return a['facts'][:0]
        lo,hi=a['offsets'][p:p+2]
        return a['facts'][int(lo):int(hi)]


def literal_scores(index,name):
    """Squared CRM-word coverage, IDF from full official scoped fact corpus."""
    terms=sorted(set(signal(name).split())); weights={}
    for term in terms:
        df=0
        if term in index.terms:
            _,lo,hi=index.terms[term]; df=hi-lo
        weights[term]=math.log1p((index.facts-df+.5)/(df+.5))
    total=sum(weights.values())
    if not total: return np.zeros(index.facts),dict(postings_read=0,total_facts=index.facts,fixed_name_fact_pool=None)
    raw,trace=index.scores(weights)
    # Floating accumulation can exceed the mathematically exact upper bound
    # by an ulp; identical clamp/square is used for sorted and random access.
    raw/=total; np.minimum(raw,1.,out=raw); np.square(raw,out=raw)
    trace.update(query_words=len(terms),idf='OFFICIAL_FACT_DF_NO_CRM_LABELS',same_alias_only=True)
    return raw,trace


class NameStream:
    def __init__(self,scores):
        self.scores=scores; ids=np.flatnonzero(scores>0)
        self.order=ids[np.lexsort((ids,-scores[ids]))]; self.offset=0; self.exhausted=not len(ids)

    def fetch(self,n):
        ids=self.order[self.offset:self.offset+n]; self.offset+=len(ids)
        self.exhausted=self.offset==len(self.order)
        return [dict(fact_id=int(i),score=float(self.scores[i])) for i in ids]


class ScopedPaths:
    def __init__(self,owners,graph,scores):
        self.owners=owners; self.graph=graph; self.scores=scores
        self.owner_cache={}; self.anchor_cache={}; self.target_cache={}

    def owner_best(self,label,key):
        cache_key=(label,int(key))
        if cache_key not in self.owner_cache:
            ids=self.owners.references(label,key)
            positive=ids[self.scores[ids]>0]
            if len(positive):
                values=self.scores[positive]; pos=int(np.argmax(values))
                result=(float(values[pos]),int(positive[pos]))
            else: result=(0.,None)
            self.owner_cache[cache_key]=result
        return self.owner_cache[cache_key]

    def anchor_best(self,pos):
        pos=int(pos)
        if pos not in self.anchor_cache:
            a=self.graph.a
            choices=[self.owner_best('site',a['endpoint_keys'][pos]),self.owner_best('parent',a['endpoint_parents'][pos])]
            self.anchor_cache[pos]=max(choices,key=lambda x:(x[0],-(x[1] if x[1] is not None else 2**32)))
        return self.anchor_cache[pos]

    def targets(self,row):
        fact=row['fact_id']; value=float(row['score']); a=self.owners.a
        subject=int(a['fact_subjects'][fact]); scope=int(a['fact_scopes'][fact])
        owner=int(a['fact_owners'][fact]); parent=int(a['fact_parents'][fact])
        if value!=self.scores[fact]: raise ValueError('Sorted fact score drift')
        if subject==1 and scope in (1,2):
            if owner!=parent: raise ValueError('Legal owner scope drift')
            anchors=self.graph.parent_endpoint_positions(f'{parent:09d}')
        elif subject==2 and scope==3:
            pos=self.graph.endpoint_position(f'{owner:014d}')
            if pos is None: raise ValueError('Scoped source endpoint absent from graph')
            if self.graph.a['endpoint_parents'][pos]!=parent: raise ValueError('Site owner scope drift')
            anchors=(pos,)
        else: raise ValueError('Context-only or unknown lexical scope')
        for raw in anchors:
            pos=int(raw); g=self.graph.a
            yield dict(siret=f'{int(g["endpoint_keys"][pos]):014d}',siren=f'{int(g["endpoint_parents"][pos]):09d}',path_name_score=value)
            for edge in self.graph.neighbors_position(pos):
                yield dict(siret=edge['siret'],siren=edge['siren'],path_name_score=value*GRADES[edge['site_binding_code']])

    def best(self,siret):
        if siret in self.target_cache: return self.target_cache[siret]
        pos=self.graph.endpoint_position(siret)
        if pos is None:
            result=dict(score=0.,fact_id=None,relation_observation=None,outside_graph_scope=True)
        else:
            score,fact=self.anchor_best(pos)
            result=dict(score=score,fact_id=fact,relation_observation=None,outside_graph_scope=False)
            # Every admissible incoming proof is direct-own or one-hop from
            # the opposite anchor; never propagate an endpoint name to siblings.
            for edge in self.graph.neighbors_position(pos):
                anchor=self.graph.endpoint_position(edge['siret'])
                incoming,other_fact=self.anchor_best(anchor)
                value=incoming*GRADES[edge['site_binding_code']]
                if value>result['score']:
                    result=dict(score=value,fact_id=other_fact,relation_observation=edge['observation_index'],
                                site_binding_code=edge['site_binding_code'],outside_graph_scope=False,
                                historical_site_coexistence_proven=False,name_relation_co_validity_proven=False)
        self.target_cache[siret]=result
        return result


class NativeStream:
    def __init__(self,backend,parsed):
        self.backend=backend; self.parsed=parsed; self.offset=0; self.exhausted=False; self.cache={}; self.seen=set()

    def fetch(self,n):
        hits=self.backend.searcher.search(self.parsed,limit=n,offset=self.offset,count=False).hits
        self.offset+=len(hits); self.exhausted=len(hits)<n; rows=[]
        for score,loc in hits:
            doc=self.backend.searcher.doc(loc).to_dict(); s,p=(self.backend._first(doc,k) for k in ('siret','siren'))
            if s in self.seen: raise ValueError('Unstable native pagination')
            self.seen.add(s)
            if s in self.cache and self.cache[s]!=float(score): raise ValueError('Native sorted/random score drift')
            self.cache[s]=float(score); rows.append(dict(siret=s,siren=p,score=float(score)))
        return rows

    def complete(self,ids):
        missing=[s for s in ids if s not in self.cache]; self.cache.update({s:0. for s in missing})
        if missing:
            tv=self.backend._tantivy
            gate=tv.Query.const_score_query(tv.Query.term_set_query(self.backend.index.schema,'siret',missing),0.)
            parsed=tv.Query.boolean_query([(tv.Occur.Must,self.parsed),(tv.Occur.Must,gate)])
            found=set()
            for score,loc in self.backend.searcher.search(parsed,limit=len(missing),count=False).hits:
                s=self.backend._first(self.backend.searcher.doc(loc).to_dict(),'siret')
                if s not in missing or s in found: raise ValueError('Native random identity drift')
                found.add(s); self.cache[s]=float(score)
        return {s:self.cache[s] for s in ids}


def search(backend,tree,keys,parents,word_index,owners,graph,q,geo,weights,terms,alpha,*,path_weight=64.,max_seconds=20.):
    began=time.monotonic(); scores,word_trace=literal_scores(word_index,q.name)
    names=NameStream(scores); paths=ScopedPaths(owners,graph,scores)
    native=NativeStream(backend,native_query(backend,q,weights,terms,alpha))
    _,_,reliable=geo_parameters(geo); reliability=1. if reliable else .5
    spatial=SpatialStream(tree,keys,parents,geo,reliability)
    def complete(rows):
        ids=[r['siret'] for r in rows]; base=native.complete(ids)
        values=geo_features(keys,tree.data,ids,geo)[:,0]*reliability
        return {s:dict(native_score=base[s],path_name_score=paths.best(s)['score'],site_score=float(values[i]))
                for i,s in enumerate(ids)}
    remaining=max_seconds-(time.monotonic()-began)
    if remaining<=0: raise TimeoutError('STOP_WORD_INDEX_PREPARATION_COST')
    rows,trace=collect(native,names,spatial,paths.targets,complete,path_weight=path_weight,
                       name_score_max=1.,site_score_max=reliability,max_seconds=remaining)
    for row in rows: row['scoped_name_proof']=paths.best(row['siret'])
    trace.update(word_index=word_trace,seconds=time.monotonic()-began,unique_documents_visited=trace['unique_targets_completed'],
                 native_corpus_unchanged=True,graph_scope='V84_ENDPOINTS_ONLY',geo_query_available=spatial.point is not None,
                 ban_reliability_factor=reliability,native_postings_visits_not_instrumented=True,
                 proof_co_validity_unknown=True,identifier_features=False)
    return rows,trace
