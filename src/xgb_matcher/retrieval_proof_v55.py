"""Initial native proof metric: coverage/exact identity/site, not BM25.

All lexical components are const-score membership/coverage, compiled into
the national index. No fixed reservoir or downstream pair scorer. Four
monotone physical scales distinguish site extent from BAN point accuracy.
"""
import math
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from scripts import retrieval_v38_initial_top100 as v38
from src.xgb_matcher.hierarchical_retrieval import normalize_text
from src.xgb_matcher.retrieval_joint_v46 import geo_parameters,SpatialStream as OldSpatial
from src.xgb_matcher.retrieval_joint_v47 import collect

LEXICAL=('name_coverage','address_coverage','name_alias_exact','address_alias_exact','place',
    'number_at_place','name_at_place','address_at_place','name_and_address_at_place','siege_at_place','exact_name_address_at_place')
SCALES=(25.,100.,500.,2000.)
FEATURES=(*LEXICAL,*(f'geo_{int(s)}m' for s in SCALES),*(f'geo_reliable_{int(s)}m' for s in SCALES))
BASE=np.array([1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.])
GLOBAL=(0,1,2,3)


def signal(value):
    return normalize_text(re.sub(r'(?<!\d)(?:\d{9}|\d{14})(?!\d)',' ',str(value or '')))


def components(backend,q):
    tv=backend._tantivy
    def all_of(*parts):
        if any(p is None for p in parts): return None
        return tv.Query.boolean_query([(tv.Occur.Must,p) for p in parts])
    def const(p,value=1.): return tv.Query.const_score_query(p,float(value)) if p is not None else None
    def maximum(parts):
        parts=[p for p in parts if p is not None]
        return tv.Query.disjunction_max_query(parts,tie_breaker=0.) if parts else None
    def coverage(fields,charfield,text):
        normal=signal(text); words=v38.tokens(normal); chars=v38.tokens(normal,True); views=[]
        for field,terms in [*((f,words) for f in fields),(charfield,chars)]:
            if terms:
                views.append(tv.Query.boolean_query([(tv.Occur.Should,const(tv.Query.term_query(backend.index.schema,field,t,'basic'),1./len(terms))) for t in terms]))
        return maximum(views)
    name=coverage(v38.NAMES,'current_name_ngrams',q.name)
    address=coverage(v38.ADDRESSES,'address_ngrams',q.address)
    def exact(field,value):
        value=signal(value)
        return const(tv.Query.term_query(backend.index.schema,field,value,'basic')) if value else None
    ne=exact('names_exact',q.name); ae=exact('addresses_exact',q.address)
    place=maximum([const(tv.Query.term_query(backend.index.schema,f,v,'basic')) for f,v in (('insee',q.insee),('postcode',q.postcode)) if v])
    gate=const(place,0.)
    number=exact('number',q.number)
    siege=exact('is_siege','1')
    # A conjunction sums its text scores; gate contributes exactly0.
    return [name,address,ne,ae,place,const(all_of(number,gate)),all_of(name,gate),all_of(address,gate),
        all_of(name,address,gate),const(all_of(siege,gate)),const(all_of(ne,ae,gate))]


def index_query(backend,q,weights):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(len(LEXICAL),) or not np.isfinite(w).all() or np.any(w<0): raise ValueError('Proof metric drift')
    parts=[backend._tantivy.Query.boost_query(p,float(value)) for p,value in zip(components(backend,q),w) if p is not None and value>0]
    tv=backend._tantivy
    if not parts: return tv.Query.empty_query()
    doc=tv.Query.const_score_query(tv.Query.term_query(backend.index.schema,'document_type','siret','basic'),0.)
    return tv.Query.boolean_query([(tv.Occur.Must,doc),(tv.Occur.Must,tv.Query.boolean_query([(tv.Occur.Should,p) for p in parts]))])


def point_features(backend,q,ids):
    tv=backend._tantivy; ids=sorted(set(ids)); byid={s:i for i,s in enumerate(ids)}
    identity=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',ids),0.)
    result=np.zeros((len(ids),len(LEXICAL)))
    for j,part in enumerate(components(backend,q)):
        if part is None: continue
        parsed=tv.Query.boolean_query([(tv.Occur.Must,identity),(tv.Occur.Must,part)])
        for score,loc in backend.searcher.search(parsed,limit=len(ids),count=False).hits:
            s=backend._first(backend.searcher.doc(loc).to_dict(),'siret'); result[byid[s],j]=float(score)
    return ids,result


def geo_features(keys,xy,ids,g):
    values=np.zeros((len(ids),8)); point,_,reliable=geo_parameters(g)
    if point is None: return values
    numeric=np.array([int(s) for s in ids],dtype=np.uint64); pos=np.searchsorted(keys,numeric); safe=np.minimum(pos,len(keys)-1)
    mask=(pos<len(keys))&(keys[safe]==numeric)
    delta=xy[pos[mask]]-point; distance=np.sqrt(np.sum(delta*delta,axis=1))
    for j,scale in enumerate(SCALES): values[mask,j]=1./(1.+distance/scale)
    if reliable: values[:,4:]=values[:,:4]
    return values


def geo_score(distance,weights,reliable):
    # Identical ordered float64 arithmetic in sorted and random access.
    result=np.zeros_like(np.asarray(distance,dtype=np.float64))
    for i,s in enumerate(SCALES): result+=float(weights[i]+weights[i+4]*reliable)*(1./(1.+distance/s))
    return result


class LexicalStream:
    def __init__(self,backend,q,w):
        self.backend=backend; self.offset=0; self.cache={}; self.seen=set(); self.exhausted=not np.any(w>0)
        self.parsed=index_query(backend,q,w); self.pruned=False
        if not self.exhausted and not np.any(w[list(GLOBAL)]):
            place=components(backend,q)[4]
            if place is None: self.exhausted=True
            else:
                tv=backend._tantivy
                self.parsed=tv.Query.boolean_query([(tv.Occur.Must,self.parsed),(tv.Occur.Must,tv.Query.const_score_query(place,0.))]); self.pruned=True
    def fetch(self,n):
        if self.exhausted: return []
        hits=self.backend.searcher.search(self.parsed,limit=n,offset=self.offset,count=False).hits
        self.offset+=len(hits); self.exhausted=len(hits)<n; rows=[]
        for score,loc in hits:
            doc=self.backend.searcher.doc(loc).to_dict(); s=self.backend._first(doc,'siret'); p=self.backend._first(doc,'siren')
            if s in self.seen: raise ValueError('Native pagination drift')
            self.seen.add(s); self.cache[s]=float(score); rows.append(dict(siret=s,siren=p,score=float(score)))
        return rows


class SpatialStream(OldSpatial):
    def __init__(self,tree,keys,parents,g,w):
        super().__init__(tree,keys,parents,g,1.)
        self.weights=w; _,_,self.reliable=geo_parameters(g)
        self.exhausted=self.point is None or not np.any(np.asarray(w[:4])+np.asarray(w[4:])*self.reliable)
    def fetch(self,n):
        if self.exhausted: return []
        count=min(self.tree.n,self.offset+n); distances,_=self.tree.query(self.point,k=count,workers=2); radius=float(np.max(distances))
        pos=np.asarray(self.tree.query_ball_point(self.point,radius+max(1e-8,radius*1e-12),workers=2),dtype=np.int64)
        delta=self.tree.data[pos]-self.point; d=np.sqrt(np.sum(delta*delta,axis=1)); scores=geo_score(d,self.weights,self.reliable)
        order=np.lexsort((self.keys[pos],d))[self.offset:count]
        rows=[dict(siret=f'{int(self.keys[pos[i]]):014d}',siren=f'{int(self.parents[pos[i]]):09d}',score=float(scores[i])) for i in order]
        if len(rows)!=count-self.offset: raise ValueError('Spatial pagination drift')
        self.offset=count; self.exhausted=count==self.tree.n; return rows


def search(backend,tree,keys,parents,q,g,weights,max_seconds=20.):
    w=np.array(weights)
    if w.shape!=(len(FEATURES),) or not np.isfinite(w).all() or np.any(w<0) or not np.any(w>0): raise ValueError('Invalid initial proof score')
    lexical=LexicalStream(backend,q,w[:len(LEXICAL)]); spatial=SpatialStream(tree,keys,parents,g,w[len(LEXICAL):]); tv=backend._tantivy
    def complete(rows):
        ids=[r['siret'] for r in rows]; unknown=[s for s in ids if s not in lexical.cache]
        lexical.cache.update({s:0. for s in unknown})
        if unknown and not lexical.exhausted:
            gate=tv.Query.const_score_query(tv.Query.term_set_query(backend.index.schema,'siret',unknown),0.)
            parsed=tv.Query.boolean_query([(tv.Occur.Must,lexical.parsed),(tv.Occur.Must,gate)])
            for score,loc in backend.searcher.search(parsed,limit=len(unknown),count=False).hits:
                s=backend._first(backend.searcher.doc(loc).to_dict(),'siret'); lexical.cache[s]=float(score)
        values=geo_features(keys,tree.data,ids,g); _,_,reliable=geo_parameters(g)
        # Use the sorted-stream kernel order, avoiding a BLAS reduction drift.
        geometry=np.zeros(len(ids))
        for i in range(4): geometry+=float(w[len(LEXICAL)+i]+w[len(LEXICAL)+i+4]*reliable)*values[:,i]
        return {s:float(lexical.cache[s]+geometry[i]) for i,s in enumerate(ids)}
    channels={k:s for k,s in (('lexical',lexical),('spatial',spatial)) if not s.exhausted}
    rows,trace=collect(channels,complete,max_seconds=max_seconds)
    trace.update(metric='OFFICIAL_PROOF_COVERAGE_EXACT_SITE_MULTISCALE_NO_BM25',zero_nonlocal_lexical_pruning=lexical.pruned)
    return rows,trace


def fit(scenes,held,registry,ridge=.005):
    if held not in (2,3,4) or set(registry)!={'2','3','4'} or not registry[str(held)]: raise ValueError('Protected fold drift')
    protected=set(registry[str(held)]); diffs=[]; sw=[]; kept=[]
    for r in scenes:
        if r['fold'] not in (2,3,4) or r['positive_siren'] not in registry[str(r['fold'])]: raise ValueError('OOF membership drift')
        if r['fold']==held or r['positive_siren'] in protected: continue
        neg=[n for n in r['negatives'] if n['siren'] not in protected]
        if not neg: continue
        diffs.extend(np.array(r['positive_features'])-np.array([n['features'] for n in neg])); sw.extend([1./len(neg)]*len(neg)); kept.append(r)
    if len(kept)<16: raise ValueError('Insufficient safe proof scenes')
    x=np.array(diffs); sw=np.array(sw)/len(kept)
    if x.shape[1]!=len(FEATURES) or not np.isfinite(x).all() or ridge<=0: raise ValueError('Feature shape/finite drift')
    def objective(w):
        margin=x@w; delta=w-BASE
        return float(sw@np.logaddexp(0.,-margin)+ridge*np.sum(delta*delta)),x.T@(-sw*expit(-margin))+2*ridge*delta
    result=minimize(objective,BASE,method='L-BFGS-B',jac=True,bounds=[(0.,64.)]*len(FEATURES),options=dict(maxiter=500,ftol=1e-10,gtol=1e-7))
    if not result.success or not np.isfinite(result.x).all(): raise ValueError('Proof fit failed')
    return dict(weights=result.x.tolist(),features=list(FEATURES),heldout_fold=held,training_folds=sorted({r['fold'] for r in kept}),
        query_ids=[r['query_id'] for r in kept],training_queries=len(kept),training_pairs=len(x),heldout_entity_overlap=0,
        identifiers_as_features=False,ridge=ridge,objective_before=objective(BASE)[0],objective_after=float(result.fun))
