"""Persistent array-only query engine for the V121/V123 exact initial100 score.

This exporter accepts a bounded V124 fixture; national streaming compilation
is a separate requirement. Every scoring attachment survives, with original
source/provenance authorities kept outside these derived score arrays.
"""
from pathlib import Path
import heapq
import json
import time

import numpy as np

from src.xgb_matcher.retrieval_coupled_blocks_v121 import channels,point,weights,SCALES
from src.xgb_matcher.retrieval_activity_joint_v114 import INTERACTIONS
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import record,verify,write_json


def save_array(root,name,value,dtype=None):
    np.save(Path(root)/(name+'.npy'),np.asarray(value,dtype=dtype),allow_pickle=False)


def save_texts(root,prefix,texts):
    values=[s.encode('ascii') for s in texts]
    if len(set(values))!=len(values):raise ValueError('Unique normalized dictionary required')
    save_array(root,prefix+'_offsets',np.r_[0,np.cumsum([len(s) for s in values],dtype=np.uint64)],np.uint64)
    save_array(root,prefix+'_data',np.frombuffer(b''.join(values),dtype=np.uint8))
    save_array(root,prefix+'_order',sorted(range(len(values)),key=values.__getitem__),np.uint32)


def save_postings(root,prefix,postings):
    terms=sorted(postings);save_texts(root,prefix,terms)
    values=[np.asarray(postings[t],dtype=np.uint32) for t in terms]
    save_array(root,prefix+'_post_offsets',np.r_[0,np.cumsum([len(s) for s in values],dtype=np.uint64)],np.uint64)
    save_array(root,prefix+'_post_ids',np.concatenate(values) if values else [],np.uint32)


def block_postings(values):
    result={}
    for b,terms in enumerate(values):
        for term in terms:result.setdefault(term,[]).append(b)
    return result


def export_fixture(index,root,source_authorities):
    """Lossless score projection only; not a full-source streaming compiler."""
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    if not source_authorities:raise ValueError('Original source authorities required')
    for ref in source_authorities:verify(ref)
    n=len(index.sites);nb=len(index.groups)
    if n>=2**32 or nb>=2**32:raise ValueError('Mapped site/block position range')
    save_array(root,'site_keys',[int(s.siret) for s in index.sites],np.uint64)
    save_array(root,'site_parents',[int(s.siren) for s in index.sites],np.uint32)
    save_array(root,'site_xy',index.xy,np.float64)
    save_array(root,'site_activity',index.site_classes,np.int32)
    save_array(root,'site_hq',[0 if s.headquarters is None else 2 if s.headquarters else 1 for s in index.sites],np.uint8)
    save_array(root,'site_state',[0 if s.state is None else 1 if s.state=='A' else 2 if s.state=='F' else 3 for s in index.sites],np.uint8)
    save_array(root,'block_offsets',np.r_[0,np.cumsum([len(ids) for ids in index.groups],dtype=np.uint64)],np.uint64)
    save_array(root,'block_sites',np.concatenate(index.groups),np.uint32)
    save_array(root,'block_lo',index.lo,np.float64);save_array(root,'block_hi',index.hi,np.float64)
    save_array(root,'block_hq',index.sieges,np.float64)
    save_postings(root,'block_place',block_postings(index.places))
    save_postings(root,'block_number',block_postings(index.numbers))
    save_array(root,'block_class_offsets',np.r_[0,np.cumsum([len(v) for v in index.class_sets],dtype=np.uint64)],np.uint64)
    save_array(root,'block_class_ids',[c for values in index.class_sets for c in sorted(values)],np.int32)
    for prefix,values in (('site_cp',index.postcodes),('site_insee',index.insees),('site_number',index.site_numbers)):
        texts=sorted(set(values));by_text={s:i for i,s in enumerate(texts)}
        save_texts(root,prefix,texts);save_array(root,prefix+'_codes',[by_text[s] for s in values],np.uint32)
    for side in range(2):
        prefix=f'side{side}';aliases=index.aliases[side];na=len(aliases)
        if na>=2**32:raise ValueError('Mapped alias position range')
        save_texts(root,prefix+'_alias',aliases)
        for j in range(3):save_postings(root,f'{prefix}_channel{j}',index.alias_postings[side][j])
        # CSR by alias; attachments are already sorted(alias,block) in V123.
        aids,bids,grades=index.attachments[side]
        save_array(root,prefix+'_attach_offsets',np.r_[0,np.cumsum(np.bincount(aids,minlength=na),dtype=np.uint64)],np.uint64)
        save_array(root,prefix+'_attach_blocks',bids,np.uint32);save_array(root,prefix+'_attach_grades',grades,np.float64)
        exact_offsets=[0];exact_blocks=[];exact_grades=[]
        for text in aliases:
            values=index.exact[side].get(text,[])
            exact_blocks.extend(b for b,g in values);exact_grades.extend(g for b,g in values);exact_offsets.append(len(exact_blocks))
        save_array(root,prefix+'_exact_offsets',exact_offsets,np.uint64)
        save_array(root,prefix+'_exact_blocks',exact_blocks,np.uint32);save_array(root,prefix+'_exact_grades',exact_grades,np.float64)
        proofs=[values[side] for values in index.block_proofs]
        save_array(root,prefix+'_proof_offsets',np.r_[0,np.cumsum([len(v[0]) for v in proofs],dtype=np.uint64)],np.uint64)
        for k,dtype in enumerate((np.uint32,np.uint32,np.float64,np.float64)):
            save_array(root,f'{prefix}_proof{k}',np.concatenate([v[k] for v in proofs]),dtype)
    files={p.name:record(p) for p in sorted(root.glob('*.npy'))}
    metadata=dict(schema='V125_ARRAY_ONLY_EXACT_COHERENT_INITIAL100',sites=n,blocks=nb,
        aliases=[len(v) for v in index.aliases],files=files,source_authorities=source_authorities,
        array_bytes=sum(r['size_bytes'] for r in files.values()),compiler_scope='BOUNDED_V124_FIXTURE_EXPORT_NOT_NATIONAL_COMPILER',
        national_corpus=False,recall_measured=False,truth_opened=False,identifier_features=False,
        score='V121_LEAF_V123_UPPER',site_or_alias_caps=None,source_provenance_preserved_in_authorities=True,
        original_state_strings_in_authorities=True)
    write_json(root/'index.json',metadata);return metadata


class Texts:
    def __init__(self,arrays,prefix):
        self.offsets=arrays[prefix+'_offsets'];self.data=arrays[prefix+'_data'];self.order=arrays[prefix+'_order']
    def __len__(self):return len(self.offsets)-1
    def at(self,i):
        lo,hi=self.offsets[int(i):int(i)+2];return self.data[int(lo):int(hi)].tobytes()
    def lookup(self,value):
        try:value=value.encode('ascii')
        except UnicodeEncodeError:return -1
        lo=0;hi=len(self)
        while lo<hi:
            mid=(lo+hi)//2;key=self.at(self.order[mid])
            if key<value:lo=mid+1
            else:hi=mid
        return int(self.order[lo]) if lo<len(self) and self.at(self.order[lo])==value else -1


def scatter_alias_max(offsets,targets,grades,scores,out,chunk=65536):
    """All positive alias references, bounded gather memory; no alias/fanout cap."""
    selected=np.flatnonzero(scores>0.);start=0
    while start<len(selected):
        # A bounded alias-count chunk and bounded reference chunks even if a
        # single official alias attaches to more than65536 physical blocks.
        keys=selected[start:start+chunk];begins=offsets[keys];ends=offsets[keys+1]
        counts=(ends-begins).astype(np.int64);prefix=np.r_[0,np.cumsum(counts,dtype=np.int64)];pos=0
        while pos<len(keys):
            if counts[pos]>chunk:
                for lo in range(int(begins[pos]),int(ends[pos]),chunk):
                    hi=min(lo+chunk,int(ends[pos]));np.maximum.at(out,targets[lo:hi],scores[keys[pos]]*grades[lo:hi])
                pos+=1;continue
            stop=int(np.searchsorted(prefix,prefix[pos]+chunk,side='right')-1)
            stop=min(len(keys),max(pos+1,stop));lengths=counts[pos:stop];total=int(np.sum(lengths))
            if total:
                relative=np.r_[0,np.cumsum(lengths[:-1],dtype=np.int64)]
                positions=np.repeat(begins[pos:stop].astype(np.int64)-relative,lengths)+np.arange(total,dtype=np.int64)
                np.maximum.at(out,targets[positions],np.repeat(scores[keys[pos:stop]],lengths)*grades[positions])
            pos=stop
        start+=len(keys)


def matrix(n,a,ne,ae,p,num,hq,geometry,reliable,grade):
    base=np.column_stack((n,a,ne,ae,p,num*p,n*p,a*p,np.where((n>0)&(a>0),(n+a)*p,0.),
        hq*p,ne*ae*p,geometry,geometry if reliable else np.zeros_like(geometry)))
    return np.column_stack((base,grade[:,None]*base[:,INTERACTIONS]))


class MappedBlocks:
    def __init__(self,root):
        self.root=Path(root);self.metadata=json.loads((self.root/'index.json').read_text())
        if self.metadata['schema']!='V125_ARRAY_ONLY_EXACT_COHERENT_INITIAL100':raise ValueError('Mapped schema drift')
        self.a={}
        for name,ref in self.metadata['files'].items():
            if Path(ref['path']).resolve()!=(self.root/name).resolve():raise ValueError('Mapped array authority path drift')
            verify(ref);self.a[name[:-4]]=np.load(self.root/name,mmap_mode='r',allow_pickle=False)
        for ref in self.metadata['source_authorities']:verify(ref)
        self.text={prefix:Texts(self.a,prefix) for prefix in ('block_place','block_number','site_cp','site_insee','site_number',
            *[f'side{s}_alias' for s in range(2)],*[f'side{s}_channel{j}' for s in range(2) for j in range(3)])}
        self.sites=self.metadata['sites'];self.blocks=self.metadata['blocks']
        if (len(self.a['site_keys'])!=self.sites or len(self.a['block_offsets'])!=self.blocks+1
            or self.a['block_offsets'][-1]!=self.sites):raise ValueError('Mapped site denominator drift')

    def posting(self,prefix,value):
        i=self.text[prefix].lookup(value)
        if i<0:return self.a[prefix+'_post_ids'][:0]
        lo,hi=self.a[prefix+'_post_offsets'][i:i+2];return self.a[prefix+'_post_ids'][int(lo):int(hi)]

    def prepare(self,q,activity):
        point(q.xy)
        if not isinstance(q.reliable_geo,bool):raise ValueError('Explicit BAN reliability required')
        if any(not isinstance(k,int) or k<0 or not np.isfinite(v) or not 0<=v<=1 for k,v in activity.items()):
            raise ValueError('Explicit activity class grades required')
        scores=[];exact=[]
        for side,text in enumerate((q.name,q.address)):
            score=np.zeros(len(self.text[f'side{side}_alias']))
            for j,tokens in enumerate(channels(text)):
                hits=np.zeros(len(score),dtype=np.int32)
                for token in tokens:hits[self.posting(f'side{side}_channel{j}',token)]+=1
                if tokens:score=np.maximum(score,hits/len(tokens))
            scores.append(score);exact.append(self.text[f'side{side}_alias'].lookup(normalized(text)))
        return scores,exact

    def bounds(self,q,w,activity,scores,exact):
        covers=np.zeros((self.blocks,2));exacts=np.zeros_like(covers)
        for side in range(2):
            prefix=f'side{side}'
            scatter_alias_max(self.a[prefix+'_attach_offsets'],self.a[prefix+'_attach_blocks'],
                self.a[prefix+'_attach_grades'],scores[side],covers[:,side])
            if exact[side]>=0:
                lo,hi=self.a[prefix+'_exact_offsets'][exact[side]:exact[side]+2]
                np.maximum.at(exacts[:,side],self.a[prefix+'_exact_blocks'][int(lo):int(hi)],self.a[prefix+'_exact_grades'][int(lo):int(hi)])
        p=np.zeros(self.blocks);num=np.zeros(self.blocks)
        for prefix,value in (('CP:',q.postcode),('INSEE:',q.insee)):
            if value:p[self.posting('block_place',prefix+value)]=1.
        number=normalized(q.number)
        if number:num[self.posting('block_number',number)]=1.
        geometry=np.zeros((self.blocks,4));known=np.isfinite(self.a['block_lo'][:,0])
        if q.xy is not None and np.any(known):
            xy=point(q.xy);lo=self.a['block_lo'][known];hi=self.a['block_hi'][known]
            delta=np.maximum(lo-xy,0.)+np.maximum(xy-hi,0.);distance=np.sqrt(np.sum(delta*delta,axis=1))
            guard=64*np.finfo(float).eps*(1.+np.linalg.norm(xy)+np.linalg.norm(lo,axis=1)+np.linalg.norm(hi,axis=1))
            geometry[known]=1./(1.+np.maximum(0.,distance-guard)[:,None]/SCALES)
        grade=np.zeros(self.blocks)
        if activity:
            offsets=self.a['block_class_offsets'];ids=self.a['block_class_ids']
            values=np.zeros(len(ids))
            for key,value in activity.items():values[ids==key]=value
            grade=np.maximum.reduceat(values,offsets[:-1].astype(np.int64))
        f=matrix(*covers.T,*exacts.T,p,num,self.a['block_hq'],geometry,q.reliable_geo,grade)
        value=np.sum(f*w,axis=1);pad=128*np.finfo(float).eps*(1.+value+float(np.sum(w)))
        return np.where(value>0.,np.nextafter(value+pad,np.inf),0.)

    def block_features(self,b,q,activity,scores,exact):
        lo,hi=self.a['block_offsets'][b:b+2];ids=self.a['block_sites'][int(lo):int(hi)];e=np.zeros((len(ids),4))
        for side in range(2):
            prefix=f'side{side}';lo,hi=self.a[prefix+'_proof_offsets'][b:b+2];sl=slice(int(lo),int(hi))
            rows=self.a[prefix+'_proof0'][sl];aliases=self.a[prefix+'_proof1'][sl]
            np.maximum.at(e[:,side],rows,scores[side][aliases]*self.a[prefix+'_proof2'][sl])
            np.maximum.at(e[:,side+2],rows,(aliases==exact[side])*self.a[prefix+'_proof3'][sl])
        p=np.zeros(len(ids));num=np.zeros(len(ids))
        for prefix,value in (('site_cp',q.postcode),('site_insee',q.insee)):
            if value:
                code=self.text[prefix].lookup(value)
                if code>=0:p=np.maximum(p,(self.a[prefix+'_codes'][ids]==code).astype(np.float64))
        number=normalized(q.number)
        if number:
            code=self.text['site_number'].lookup(number)
            if code>=0:num=(self.a['site_number_codes'][ids]==code).astype(np.float64)
        geometry=np.zeros((len(ids),4));xy=self.a['site_xy'][ids];known=np.isfinite(xy[:,0])
        if q.xy is not None and np.any(known):
            distance=np.sqrt(np.sum((xy[known]-point(q.xy))**2,axis=1));geometry[known]=1./(1.+distance[:,None]/SCALES)
        grade=np.array([activity.get(int(c),0.) for c in self.a['site_activity'][ids]])
        return ids,matrix(*e.T,p,num,(self.a['site_hq'][ids]==2).astype(np.float64),geometry,q.reliable_geo,grade)

    def search(self,q,value,activity=None,limit=100,max_seconds=20.):
        if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=100:raise ValueError('Absolute initial output cap100')
        if not np.isfinite(max_seconds) or max_seconds<=0:raise ValueError('Positive query guard')
        began=time.monotonic();w=weights(value);activity={} if activity is None else activity
        scores,exact=self.prepare(q,activity);prepared=time.monotonic()
        upper=self.bounds(q,w,activity,scores,exact);bounded=time.monotonic()
        order=np.argsort(-upper,kind='stable');heap=[];visited=[];sites=0;cutoff=0.
        for b in order:
            if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_MAPPED_BLOCK_QUERY_COST')
            if upper[b]<=0. or (len(heap)==limit and upper[b]<cutoff):break
            ids,features=self.block_features(int(b),q,activity,scores,exact);values=np.sum(features*w,axis=1)
            if np.any(values>upper[b]):raise ValueError('Unsound mapped bound')
            visited.append(int(b));sites+=len(ids)
            for i,score in zip(ids,values):
                if score<=0.:continue
                item=(float(score),-int(self.a['site_keys'][i]),int(i))
                if len(heap)<limit:heapq.heappush(heap,item)
                elif item>heap[0]:heapq.heapreplace(heap,item)
                cutoff=heap[0][0] if len(heap)==limit else 0.
        if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_MAPPED_BLOCK_QUERY_COST')
        rows=[dict(siret=f'{int(self.a["site_keys"][i]):014d}',siren=f'{int(self.a["site_parents"][i]):09d}',score=score)
            for score,_,i in sorted(heap,reverse=True)]
        return rows,dict(certified=True,sites=self.sites,blocks=self.blocks,blocks_completed=len(visited),sites_completed=sites,
            blocks_pruned=self.blocks-len(visited),seconds=time.monotonic()-began,initial_cap=limit,fixed_reservoir=None,
            score_after_fixed_pool=False,global_boundary_ties=True,locality_or_closed_state_filter=False,
            source_schema='COHERENT_WORD_CHAR3_CHAR4_NATIVE19_ACTIVITY30',array_only_runtime=True,
            prepare_seconds=prepared-began,bounds_seconds=bounded-prepared,completion_seconds=time.monotonic()-bounded,
            corpus_scope='BOUNDED_SOURCE_FIXTURE_NOT_NATIONAL_COVERAGE')
