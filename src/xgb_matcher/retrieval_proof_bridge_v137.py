"""Whole official name/graph proofs joined to the V136 virtual namespace.

Positions are opaque joins, not score features. No alias×member expansion,
seed pool, truncation, learned CRM aliases or label-side data.
"""
from pathlib import Path
import json

import numpy as np

from src.xgb_matcher.retrieval_extended_catalog_v136 import positions,MISSING
from src.xgb_matcher.retrieval_v20_workbench import verify

PROTECTED=('truth_opened','crm_opened','model_fit','recall_measured','fold0_opened','fold1_opened')


class CatalogKeys:
    """Load just immutable typed join dictionaries; hash once per worker."""
    def __init__(self,reference):
        m=json.loads(verify(reference).read_text());self.metadata=m
        if (m['status']!='SEALED_V136_CURRENT_PLUS_ALL_V135_OFFICIAL_SITE_NAMESPACE'
            or any(m.get(k) is not False for k in PROTECTED) or m['complete_retrieval_index']
            or not m['stats']['every_current_site_position_unchanged']
            or not m['stats']['every_extra_site_admitted_once']):
            raise ValueError('Protected exact official namespace required')
        launch=json.loads(verify(m['launch']).read_text())
        for ref in launch['implementation'].values():verify(ref)
        self.references=(m['input_arrays']['endpoint_keys'],m['input_arrays']['parent_keys'],m['extra_sites'],m['artifacts']['extra_parent_keys.npy'])
        self.site,self.parent,self.extra_site,self.extra_parent=(np.load(verify(ref),mmap_mode='r',allow_pickle=False) for ref in self.references)
        self.sites=len(self.site)+len(self.extra_site);self.parents=len(self.parent)+len(self.extra_parent)
        if (self.sites!=m['stats']['sites'] or self.parents!=m['stats']['parents']
            or any(a.ndim!=1 for a in (self.site,self.parent,self.extra_site,self.extra_parent))
            or self.site.dtype!=np.uint64 or self.extra_site.dtype!=np.uint64
            or self.parent.dtype!=np.uint32 or self.extra_parent.dtype!=np.uint32):
            raise ValueError('Whole official virtual namespace required')

    def site_positions(self,keys):return positions(self.site,self.extra_site,keys)
    def parent_positions(self,keys):return positions(self.parent,self.extra_parent,keys)

    def reconstruct(self,pos,grain):
        if grain not in ('site','parent'):raise ValueError('Explicit position grain required')
        current,extra=(self.site,self.extra_site) if grain=='site' else (self.parent,self.extra_parent)
        if pos.dtype!=np.uint32 or np.any(pos==MISSING) or np.any(pos>=len(current)+len(extra)):
            raise ValueError('Every admitted typed virtual position required')
        out=np.empty(len(pos),dtype=current.dtype);native=pos<len(current)
        out[native]=current[pos[native]];out[~native]=extra[pos[~native]-len(current)];return out


def facts_bridge(catalog,source,output,batch_rows=131072):
    """Every V132 fact retains its row and scope, including unbound ULs."""
    if not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0:raise ValueError('Positive batch size required')
    parent,owner,subject,scope=(source[k] for k in ('parent_key','owner_key','subject_code','scope_code'))
    n=len(parent)
    if (parent.dtype!=np.uint32 or owner.dtype!=np.uint64 or subject.dtype!=np.uint8 or scope.dtype!=np.uint8
        or any(a.ndim!=1 or len(a)!=n for a in (parent,owner,subject,scope))):
        raise ValueError('Whole exact typed official facts required')
    dest=Path(output)
    pp=np.lib.format.open_memmap(dest/'fact_parent_positions.npy',mode='w+',dtype=np.uint32,shape=(n,))
    sp=np.lib.format.open_memmap(dest/'fact_site_positions.npy',mode='w+',dtype=np.uint32,shape=(n,))
    missing_ul=0;site_facts=0
    for begin in range(0,n,batch_rows):
        end=min(n,begin+batch_rows);p=parent[begin:end];o=owner[begin:end];s=subject[begin:end];sc=scope[begin:end]
        if (np.any(p>=10**9) or np.any(o>=10**14) or np.any(~np.isin(s,[1,2])) or np.any(~np.isin(sc,[1,2,3]))
            or np.any((s==1)&((o!=p)|~np.isin(sc,[1,2])))
            or np.any((s==2)&((o//100000!=p)|(sc!=3)))):
            raise ValueError('Explicit official parent/site proof grain and scope required')
        pp[begin:end]=catalog.parent_positions(p);sp[begin:end]=MISSING
        sites=s==2;sp[begin:end][sites]=catalog.site_positions(o[sites]);site_facts+=int(np.sum(sites))
        if np.any(sp[begin:end][sites]==MISSING) or np.any(pp[begin:end][sites]==MISSING):
            raise ValueError('Every official site proof must be admitted in V136')
        missing_ul+=int(np.sum((s==1)&(pp[begin:end]==MISSING)))
    pp.flush();sp.flush()
    # Independent full reconstruction through exported positions. ULs absent
    # from membership stay UNKNOWN plus the original V132 source, not dropped.
    for begin in range(0,n,batch_rows):
        end=min(n,begin+batch_rows);p=parent[begin:end];o=owner[begin:end];s=subject[begin:end]
        rp=pp[begin:end];rs=sp[begin:end];bound=rp!=MISSING;sites=s==2
        if (not np.array_equal(catalog.reconstruct(rp[bound],'parent'),p[bound])
            or not np.array_equal(catalog.reconstruct(rs[sites],'site'),o[sites])
            or np.any(rs[~sites]!=MISSING)
            or not np.array_equal(catalog.parent_positions(p),rp)):
            raise ValueError('Every original fact/owner/scope must reconstruct exactly')
    return dict(facts=n,site_facts=site_facts,unbound_parent_facts=missing_ul,
        every_original_fact_reconstructed_exactly=True,every_site_fact_admitted=True,
        unbound_parent_observations_preserved=True,site_cartesian_product_materialized=False,
        name_or_owner_or_member_cap=None,identifiers_as_features=False)


def graph_bridge(catalog,source,output,batch_rows=131072):
    if not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0:raise ValueError('Positive batch size required')
    keys=source['endpoint_keys'];pred=source['predecessor_keys'];succ=source['successor_keys']
    if (any(a.dtype!=np.uint64 or a.ndim!=1 for a in (keys,pred,succ))
        or len(pred)!=len(succ) or np.any(keys[1:]<=keys[:-1])):
        raise ValueError('Whole sorted raw official graph required')
    dest=Path(output)
    specs={'site_parent_positions':catalog.sites,'graph_site_positions':len(keys),
        'graph_parent_positions':len(keys),'predecessor_graph_positions':len(pred),'successor_graph_positions':len(succ)}
    arrays={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype=np.uint32,shape=(n,)) for k,n in specs.items()}
    site_parent=arrays['site_parent_positions']
    for raw,base in ((catalog.site,0),(catalog.extra_site,len(catalog.site))):
        for begin in range(0,len(raw),batch_rows):
            end=min(len(raw),begin+batch_rows);p=(raw[begin:end]//100000).astype(np.uint32)
            got=catalog.parent_positions(p)
            if np.any(got==MISSING) or not np.array_equal(catalog.reconstruct(got,'parent'),p):
                raise ValueError('Every candidate site parent must reconstruct exactly')
            site_parent[base+begin:base+end]=got
    for begin in range(0,len(keys),batch_rows):
        end=min(len(keys),begin+batch_rows);raw=keys[begin:end];got=catalog.site_positions(raw)
        if np.any(got==MISSING) or not np.array_equal(catalog.reconstruct(got,'site'),raw):
            raise ValueError('Every raw official graph endpoint must be admitted')
        arrays['graph_site_positions'][begin:end]=got
        arrays['graph_parent_positions'][begin:end]=site_parent[got]
        if not np.array_equal(catalog.reconstruct(site_parent[got],'parent'),(raw//100000).astype(np.uint32)):
            raise ValueError('Every graph endpoint parent must reconstruct exactly')
    for name,raw in (('predecessor_graph_positions',pred),('successor_graph_positions',succ)):
        for begin in range(0,len(raw),batch_rows):
            end=min(len(raw),begin+batch_rows);values=raw[begin:end];got=np.searchsorted(keys,values)
            if np.any(got>=len(keys)) or not np.array_equal(keys[got],values):
                raise ValueError('Every raw relation endpoint must reconstruct exactly')
            arrays[name][begin:end]=got.astype(np.uint32)
    for a in arrays.values():a.flush()
    # Independent source reconstruction through every EXPORTED position.
    exported={k:np.load(dest/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in specs}
    for raw,base in ((catalog.site,0),(catalog.extra_site,len(catalog.site))):
        for begin in range(0,len(raw),batch_rows):
            end=min(len(raw),begin+batch_rows);got=exported['site_parent_positions'][base+begin:base+end]
            if not np.array_equal(catalog.reconstruct(got,'parent'),(raw[begin:end]//100000).astype(np.uint32)):
                raise ValueError('Every exported site parent must reconstruct exactly')
    for begin in range(0,len(keys),batch_rows):
        end=min(len(keys),begin+batch_rows)
        if (not np.array_equal(catalog.reconstruct(exported['graph_site_positions'][begin:end],'site'),keys[begin:end])
            or not np.array_equal(catalog.reconstruct(exported['graph_parent_positions'][begin:end],'parent'),(keys[begin:end]//100000).astype(np.uint32))):
            raise ValueError('Every exported graph endpoint must reconstruct exactly')
    for name,raw in (('predecessor_graph_positions',pred),('successor_graph_positions',succ)):
        for begin in range(0,len(raw),batch_rows):
            end=min(len(raw),begin+batch_rows);got=exported[name][begin:end]
            if np.any(got>=len(keys)) or not np.array_equal(keys[got],raw[begin:end]):
                raise ValueError('Every exported relation must reconstruct exactly')
    return dict(sites=catalog.sites,endpoints=len(keys),observations=len(pred),
        every_site_parent_and_graph_endpoint_admitted=True,every_original_relation_reconstructed_exactly=True,
        all_observations_and_provenance_unchanged=True,relation_or_endpoint_or_parent_cap=None,
        identifiers_as_features=False,attributes_inferred_from_relation=False)


def scatter_name_facts(coverage,exact,offsets,subject,current,parent_pos,site_pos,parent_cov,parent_exact,site_cov,site_exact,batch_rows=131072):
    """Maximum WHOLE alias proof per owner; site alias never lifted to parent.

    Inputs coverage/exact are per raw alias, after independent lexical lookup.
    This is not a candidate pool or a scorer; every original owner is processed.
    """
    if not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0:raise ValueError('Positive batch size required')
    n=len(subject)
    if (coverage.dtype!=np.float64 or exact.dtype!=np.bool_ or offsets.dtype!=np.uint64
        or offsets.shape!=(len(coverage)+1,) or exact.shape!=coverage.shape
        or offsets[0]!=0 or offsets[-1]!=n or np.any(offsets[1:]<=offsets[:-1])
        or not np.isfinite(coverage).all() or np.any((coverage<0)|(coverage>1))
        or subject.dtype!=np.uint8 or current.dtype!=np.bool_ or parent_pos.dtype!=np.uint32 or site_pos.dtype!=np.uint32
        or any(a.shape!=(n,) for a in (subject,current,parent_pos,site_pos))
        or parent_cov.dtype!=np.float64 or site_cov.dtype!=np.float64
        or parent_exact.dtype!=np.bool_ or site_exact.dtype!=np.bool_
        or any(a.ndim!=1 for a in (coverage,exact,offsets,parent_cov,parent_exact,site_cov,site_exact))
        or parent_exact.shape!=parent_cov.shape or site_exact.shape!=site_cov.shape):
        raise ValueError('Whole typed alias/owner feature arrays required')
    for begin in range(0,n,batch_rows):
        end=min(n,begin+batch_rows);r=np.arange(begin,end,dtype=np.uint64)
        aliases=np.searchsorted(offsets,r,side='right')-1;s=subject[begin:end]
        if np.any(~np.isin(s,[1,2])):raise ValueError('Explicit owner proof scope required')
        cp=parent_pos[begin:end];sp=site_pos[begin:end];ne=exact[aliases]&current[begin:end]
        for mask,pos,out,flags in ((s==1,cp,parent_cov,parent_exact),(s==2,sp,site_cov,site_exact)):
            missing=mask&(pos==MISSING)
            if np.any(missing) and out is site_cov:raise ValueError('Every official site proof must be admitted')
            mask=mask&(pos!=MISSING)
            if np.any(pos[mask]>=len(out)):raise ValueError('Every bound owner position required')
            np.maximum.at(out,pos[mask],coverage[aliases[mask]])
            np.logical_or.at(flags,pos[mask],ne[mask])


def one_hop_names(parent_cov,site_cov,graph_site,graph_parent,pred,succ,continuity,batch_rows=131072):
    """Both directions, endpoint-only, snapshot-based, duplicate max, no exact.

    Grade .5 only for explicit continuity TRUE (V133 code2), else .25.
    Self-links are context, not expansion. A→B→C never spreads two hops.
    """
    if not isinstance(batch_rows,int) or isinstance(batch_rows,bool) or batch_rows<=0:raise ValueError('Positive batch size required')
    if (parent_cov.dtype!=np.float64 or site_cov.dtype!=np.float64 or parent_cov.ndim!=1 or site_cov.ndim!=1
        or any(a.dtype!=np.uint32 or a.ndim!=1 for a in (graph_site,graph_parent,pred,succ))
        or graph_site.shape!=graph_parent.shape or pred.shape!=succ.shape or continuity.shape!=pred.shape
        or continuity.dtype!=np.uint8 or np.any(continuity>2)
        or np.any(graph_site>=len(site_cov)) or np.any(graph_parent>=len(parent_cov))
        or np.any(pred>=len(graph_site)) or np.any(succ>=len(graph_site))
        or any(not np.isfinite(a).all() or np.any((a<0)|(a>1)) for a in (parent_cov,site_cov))):
        raise ValueError('Whole exact direct-proof graph snapshot required')
    snapshot=np.maximum(site_cov[graph_site],parent_cov[graph_parent])
    result=np.zeros(len(site_cov),dtype=np.float64)
    for begin in range(0,len(pred),batch_rows):
        end=min(len(pred),begin+batch_rows);p=pred[begin:end];s=succ[begin:end];mask=p!=s
        grade=np.where(continuity[begin:end][mask]==2,.5,.25)
        np.maximum.at(result,graph_site[p[mask]],snapshot[s[mask]]*grade)
        np.maximum.at(result,graph_site[s[mask]],snapshot[p[mask]]*grade)
    return result
