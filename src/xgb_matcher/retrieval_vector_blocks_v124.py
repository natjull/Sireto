"""V123 bounds and V121 exact leaf features, computed once per physical block.

Source/provenance objects remain untouched. Arrays collapse only identical
scoring text within a site; lexical and current-exact grades stay separate.
Query state lives in the query terms/caches, not in the shared index.
"""
import numpy as np

from src.xgb_matcher.retrieval_coupled_blocks_v121 import native,point,SCALES
from src.xgb_matcher.retrieval_tight_blocks_v123 import TightCoupledBlocks
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_activity_joint_v114 import INTERACTIONS,vector


class ScoredTerms(tuple):
    def __new__(cls,values,scores):
        obj=super().__new__(cls,values);obj.scores=scores;return obj


class VectorCoupledBlocks(TightCoupledBlocks):
    def __init__(self,sites,block_size=128):
        super().__init__(sites,block_size)
        self.site_blocks=np.empty(len(self.sites),dtype=np.int32)
        self.site_positions=np.empty(len(self.sites),dtype=np.int32)
        self.by_alias=[{text:i for i,text in enumerate(values)} for values in self.aliases]
        self.block_proofs=[]
        for b,ids in enumerate(self.groups):
            self.site_blocks[ids]=b;self.site_positions[ids]=np.arange(len(ids))
            sides=[]
            for side in range(2):
                values={}
                for local,i in enumerate(ids):
                    for proof in self.sites[i].names if side==0 else self.sites[i].addresses:
                        text=normalized(proof.text)
                        if not text or proof.grade==0.:
                            continue
                        key=(local,self.by_alias[side][text]);old=values.get(key,(0.,0.))
                        values[key]=(max(old[0],proof.grade),max(old[1],proof.grade if proof.current_exact else 0.))
                keys=sorted(values)
                sides.append((np.array([k[0] for k in keys],dtype=np.int32),
                    np.array([k[1] for k in keys],dtype=np.int32),
                    np.array([values[k][0] for k in keys],dtype=np.float64),
                    np.array([values[k][1] for k in keys],dtype=np.float64)))
            self.block_proofs.append(sides)
        self.postcodes=np.array([s.postcode for s in self.sites])
        self.insees=np.array([s.insee for s in self.sites])
        self.site_numbers=np.array([normalized(s.number) for s in self.sites])
        self.site_sieges=np.array([float(s.headquarters is True) for s in self.sites])
        self.site_classes=np.array([s.activity_class for s in self.sites],dtype=np.int32)

    def _prepare(self,query,w,activity):
        terms,caches=super()._prepare(query,w,activity);scores=[]
        for side in range(2):
            alias_score=np.zeros(len(self.aliases[side]))
            for j,tokens in enumerate(terms[side]):
                hits=np.zeros(len(alias_score),dtype=np.int32)
                for token in tokens:
                    ids=self.alias_postings[side][j].get(token)
                    if ids is not None:
                        hits[ids]+=1
                if tokens:
                    alias_score=np.maximum(alias_score,hits/len(tokens))
            scores.append(alias_score)
            caches[side]['_exact_alias']=self.by_alias[side].get(caches[side]['_query_text'],-1)
        caches[0]['_block_features']={}
        return ScoredTerms(terms,scores),caches

    def bounds(self,query,w,activity,terms):
        count=len(self.groups);covers=np.zeros((count,2));exact=np.zeros((count,2))
        for side,value in enumerate((query.name,query.address)):
            aliases,blocks,grades=self.attachments[side]
            np.maximum.at(covers[:,side],blocks,terms.scores[side][aliases]*grades)
            for b,grade in self.exact[side].get(normalized(value),[]):
                exact[b,side]=grade
        places={prefix+value for prefix,value in (('CP:',query.postcode),('INSEE:',query.insee)) if value}
        p=np.array([float(bool(places&values)) for values in self.places]);number=normalized(query.number)
        num=np.array([float(bool(number) and number in values) for values in self.numbers])
        geometry=np.zeros((count,4));known=np.isfinite(self.lo[:,0])
        if query.xy is not None and np.any(known):
            xy=point(query.xy)
            delta=np.maximum(self.lo[known]-xy,0.)+np.maximum(xy-self.hi[known],0.)
            distance=np.sqrt(np.sum(delta*delta,axis=1))
            guard=64*np.finfo(float).eps*(1.+np.linalg.norm(xy)+np.linalg.norm(self.lo[known],axis=1)+np.linalg.norm(self.hi[known],axis=1))
            geometry[known]=1./(1.+np.maximum(0.,distance-guard)[:,None]/SCALES)
        result=np.zeros(count)
        for b in range(count):
            f=native(*covers[b],*exact[b],p[b],num[b],self.sieges[b],geometry[b],query.reliable_geo)
            grade=max((activity.get(k,0.) for k in self.class_sets[b]),default=0.)
            value=float(np.sum(vector(f,grade)*w))
            result[b]=np.nextafter(value+128*np.finfo(float).eps*(1.+value+float(np.sum(w))),np.inf) if value else 0.
        return result

    def features(self,i,query,activity,terms,caches):
        b=int(self.site_blocks[i]);cache=caches[0]['_block_features']
        if b not in cache:
            # Search completes one block at a time; keep query memory bounded
            # by one physical block, not by the number of visited candidates.
            cache.clear()
            ids=self.groups[b];e=np.zeros((len(ids),4))
            for side in range(2):
                rows,aliases,grades,current_grades=self.block_proofs[b][side]
                np.maximum.at(e[:,side],rows,terms.scores[side][aliases]*grades)
                np.maximum.at(e[:,side+2],rows,(aliases==caches[side]['_exact_alias'])*current_grades)
            n,a,ne,ae=e.T
            p=((bool(query.postcode)&(self.postcodes[ids]==query.postcode))|
               (bool(query.insee)&(self.insees[ids]==query.insee))).astype(np.float64)
            number=normalized(query.number);num=(bool(number)&(self.site_numbers[ids]==number)).astype(np.float64)
            geometry=np.zeros((len(ids),4));known=np.isfinite(self.xy[ids,0])
            if query.xy is not None and np.any(known):
                distance=np.sqrt(np.sum((self.xy[ids[known]]-point(query.xy))**2,axis=1))
                geometry[known]=1./(1.+distance[:,None]/SCALES)
            base=np.column_stack((n,a,ne,ae,p,num*p,n*p,a*p,
                np.where((n>0)&(a>0),(n+a)*p,0.),self.site_sieges[ids]*p,ne*ae*p,
                geometry,geometry if query.reliable_geo else np.zeros_like(geometry)))
            grade=np.array([activity.get(int(k),0.) for k in self.site_classes[ids]])
            cache[b]=np.column_stack((base,grade[:,None]*base[:,INTERACTIONS]))
            # Parent _prepare and constructor validate proofs/geometry/activity.
            # Same ranges as vector(); vector fixtures below prove every column.
            if not np.isfinite(cache[b]).all():
                raise ValueError('Nonfinite vector block proof')
        return cache[b][self.site_positions[i]]

    def exhaustive(self,query,value,activity=None,limit=100):
        # Independent scalar oracle, not the vector implementation under test.
        from src.xgb_matcher.retrieval_coupled_blocks_v121 import CoupledBlocks,weights
        w=weights(value);activity={} if activity is None else activity
        terms,caches=CoupledBlocks._prepare(self,query,w,activity)
        rows=[dict(siret=s.siret,siren=s.siren,score=float(np.sum(CoupledBlocks.features(self,i,query,activity,terms,caches)*w)))
            for i,s in enumerate(self.sites)]
        return sorted((r for r in rows if r['score']>0.),key=lambda r:(-r['score'],r['siret']))[:limit]
