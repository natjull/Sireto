"""Same V121 leaf score, exact coherent-alias maxima tighten block bounds.

Every unique text and every block attachment is represented; no top-alias
shortlist, name/site cap or candidate reranking. Sparse query posting counts
score whole variants before bounding the blocks that reference them.
"""
import numpy as np

from src.xgb_matcher.retrieval_coupled_blocks_v121 import (
    CoupledBlocks, channels, native, point, SCALES,
)
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_activity_joint_v114 import vector


class TightCoupledBlocks(CoupledBlocks):
    def __init__(self,sites,block_size=128):
        super().__init__(sites,block_size)
        self.aliases=[];self.alias_postings=[];self.attachments=[]
        for side in range(2):
            texts=[];by_text={};attached={}
            for b,ids in enumerate(self.groups):
                for i in ids:
                    for proof in self.sites[i].names if side==0 else self.sites[i].addresses:
                        text=normalized(proof.text)
                        if not text or proof.grade==0.:
                            continue
                        if text not in by_text:
                            by_text[text]=len(texts);texts.append(text)
                        key=(by_text[text],b);attached[key]=max(attached.get(key,0.),proof.grade)
            posting=[{} for _ in range(3)]
            for i,text in enumerate(texts):
                for j,terms in enumerate(channels(text)):
                    for term in terms:
                        posting[j].setdefault(term,[]).append(i)
            posting=[{t:np.array(ids,dtype=np.int32) for t,ids in values.items()} for values in posting]
            pairs=sorted(attached)
            self.aliases.append(tuple(texts));self.alias_postings.append(posting)
            self.attachments.append((np.array([key[0] for key in pairs],dtype=np.int32),
                np.array([key[1] for key in pairs],dtype=np.int32),np.array([attached[key] for key in pairs],dtype=np.float64)))

    def bounds(self,query,w,activity,terms):
        count=len(self.groups);covers=np.zeros((count,2));exact=np.zeros((count,2))
        for side,value in enumerate((query.name,query.address)):
            alias_score=np.zeros(len(self.aliases[side]))
            for j,tokens in enumerate(terms[side]):
                hits=np.zeros(len(alias_score),dtype=np.int32)
                for token in tokens:
                    if token in self.alias_postings[side][j]:
                        hits[self.alias_postings[side][j][token]]+=1
                if tokens:
                    alias_score=np.maximum(alias_score,hits/len(tokens))
            aliases,blocks,grades=self.attachments[side]
            np.maximum.at(covers[:,side],blocks,alias_score[aliases]*grades)
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
