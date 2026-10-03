"""Coherent whole-address scoring and exact streaming initial100 collection.

No fixed candidate pool: the collector requires every opaque site position
exactly once. This is an unmeasured integration kernel, NOT a sealed national
engine or evidence of recall/SLO. Source adapters remain responsible for
complete official aliases/proofs and source-bound namespace verification.
"""
from dataclasses import dataclass
import heapq
import time

import numpy as np

from src.xgb_matcher.retrieval_coupled_blocks_v121 import point,SCALES,weights
from src.xgb_matcher.retrieval_mapped_blocks_v125 import matrix
from src.xgb_matcher.retrieval_site_attributes_v139 import encode_code,MISSING
from src.xgb_matcher.retrieval_variants_v118 import normalized


class AppendedAttributes:
    """V139 current evidence plus V141 UNKNOWNs, no sibling inheritance."""
    def __init__(self,attributes,catalog):
        if (attributes.metadata['sources']['catalog']!=catalog.metadata['previous_catalog']
            or attributes.sites!=catalog.previous.sites or attributes.base>attributes.sites
            or not catalog.metadata['stats']['every_previous_site_and_parent_position_unchanged']):
            raise ValueError('Exact unchanged structured/append source namespace required')
        self.attributes=attributes;self.sites=catalog.sites

    def positions(self,ids):
        if ids.dtype!=np.uint32 or ids.ndim!=1 or np.any(ids>=self.sites):
            raise ValueError('Every actual appended site position required')
        return ids<self.attributes.sites

    def flags(self,ids,postcode='',insee='',number=''):
        old=self.positions(ids)
        out=[np.zeros(len(ids)) for _ in range(3)]+[np.full(len(ids),-1,dtype=np.int32)]
        if np.any(old):
            for dest,values in zip(out,self.attributes.flags(ids[old],postcode,insee,number)):
                dest[old]=values
        return tuple(out)

    def coordinates(self,ids):
        old=self.positions(ids);out=np.full((len(ids),2),np.nan)
        if np.any(old):out[old]=self.attributes.coordinates(ids[old])
        return out


@dataclass(frozen=True)
class AddressBatch:
    # Explicit site scope only. Coverage belongs to one complete address alias.
    site_positions: np.ndarray
    coverage: np.ndarray
    alias_exact: np.ndarray
    current_exact: np.ndarray
    postcode_codes: np.ndarray
    insee_codes: np.ndarray
    number_texts: tuple


class NameSnapshot:
    """Direct owner proofs plus one-hop names; exact flags are direct only.

    Typed arrays use the unchanged V137 namespace. V141 appended owners have
    zero direct name evidence unless a future explicit source supplies it;
    existing parent proofs still apply through their exact parent membership.
    No site-name lift to siblings, no iterative relation propagation.
    """
    def __init__(self,catalog,site_parents,parent_cov,parent_exact,site_cov,site_exact,relation_cov):
        old_sites=catalog.previous.sites;old_parents=catalog.previous.parents
        if (site_parents.dtype!=np.uint32 or site_parents.shape!=(old_sites,)
            or np.any(site_parents>=old_parents)
            or parent_exact.dtype!=np.bool_ or parent_exact.shape!=(old_parents,)
            or site_exact.dtype!=np.bool_ or site_exact.shape!=(old_sites,)
            or catalog.new['new_site_parent_positions'].dtype!=np.uint32
            or catalog.new['new_site_parent_positions'].shape!=(catalog.sites-old_sites,)
            or np.any(catalog.new['new_site_parent_positions']>=catalog.parents)):
            raise ValueError('Whole unchanged name/parent/append position snapshot required')
        for a,n,key in ((parent_cov,old_parents,'parent-name'),(site_cov,old_sites,'site-name'),
                        (relation_cov,old_sites,'one-hop-name')):
            bounded_vector(a,n,np.float64,key)
        if np.any(parent_exact&(parent_cov!=1.)) or np.any(site_exact&(site_cov!=1.)):
            raise ValueError('Direct exact-current names require whole direct alias coverage')
        self.catalog=catalog;self.sites=catalog.sites;self.site_parents=site_parents
        self.parent_cov=parent_cov;self.parent_exact=parent_exact
        self.site_cov=site_cov;self.site_exact=site_exact;self.relation_cov=relation_cov

    def gather(self,ids):
        if ids.dtype!=np.uint32 or ids.ndim!=1 or np.any(ids>=self.sites):
            raise ValueError('Every actual site position required')
        old=ids<self.catalog.previous.sites;parents=np.empty(len(ids),dtype=np.uint32)
        parents[old]=self.site_parents[ids[old]]
        parents[~old]=self.catalog.new['new_site_parent_positions'][ids[~old]-self.catalog.previous.sites]
        pc=np.zeros(len(ids));pe=pc.copy();known=parents<len(self.parent_cov)
        pc[known]=self.parent_cov[parents[known]];pe[known]=self.parent_exact[parents[known]]
        pc[old]=np.maximum(pc[old],np.maximum(self.site_cov[ids[old]],self.relation_cov[ids[old]]))
        pe[old]=np.maximum(pe[old],self.site_exact[ids[old]])
        return pc,pe


def bounded_vector(a,n,dtype,name,maximum=1.):
    if (not isinstance(a,np.ndarray) or a.dtype!=dtype or a.shape!=(n,)
        or not np.isfinite(a).all() or np.any(a<0) or np.any(a>maximum)):
        raise ValueError('Exact typed '+name+' evidence required')


def score_sites(ids,q,value,name_coverage,name_exact,attributes,address_batches,activity=None):
    """Score all sites in a physical chunk, then max over each complete proof.

    name_coverage/name_exact are query-side coherent official maxima, including
    parent/direct-site/one-hop proofs from the source stage. Relations cannot
    contribute exact-current flags. No alias tokens or address attributes are
    pooled across proofs here. All address batches must be supplied by a
    complete sealed source adapter; this function cannot certify that source.
    """
    w=weights(value);n=len(ids)
    if (ids.dtype!=np.uint32 or ids.ndim!=1 or not n or np.any(ids>=attributes.sites)
        or not np.array_equal(ids,np.arange(int(ids[0]),int(ids[0])+n,dtype=np.uint32))):
        raise ValueError('Complete contiguous physical site chunk required')
    bounded_vector(name_coverage,n,np.float64,'whole-name')
    bounded_vector(name_exact,n,np.float64,'exact-name')
    if np.any(name_exact>name_coverage):raise ValueError('Exact name cannot exceed its whole proof coverage')
    if not isinstance(q.reliable_geo,bool):raise ValueError('Explicit BAN reliability required')
    activity={} if activity is None else activity
    if any(not isinstance(k,int) or isinstance(k,bool) or k<0 or not np.isfinite(v) or not 0<=v<=1
           for k,v in activity.items()):raise ValueError('Explicit activity grades required')
    place,num,hq,classes=attributes.flags(ids,q.postcode,q.insee,q.number)
    for a,key in ((place,'place'),(num,'number'),(hq,'headquarters')):
        bounded_vector(a,n,np.float64,key)
    if classes.dtype!=np.int32 or classes.shape!=(n,) or np.any(classes< -1):
        raise ValueError('Explicit current activity or UNKNOWN required')
    xy=attributes.coordinates(ids)
    if (xy.dtype!=np.float64 or xy.shape!=(n,2) or np.isinf(xy).any()
        or np.any(np.isfinite(xy[:,0])!=np.isfinite(xy[:,1]))):
        raise ValueError('Original same-CRS XY pair or UNKNOWN required')
    query_xy=point(q.xy);geometry=np.zeros((n,4));known=np.isfinite(xy[:,0])
    if query_xy is not None and np.any(known):
        if np.any(np.abs(xy[known])>1e9):raise ValueError('Projected source XY range required')
        distance=np.sqrt(np.sum((xy[known]-query_xy)**2,axis=1))
        geometry[known]=1./(1.+distance[:,None]/SCALES)
    grades=np.array([activity.get(int(k),0.) for k in classes],dtype=np.float64)
    zero=np.zeros(n)
    # Current structured evidence is an independent fallback, not attached to
    # an arbitrary historical alias. UNKNOWN/closed sites remain reachable.
    f=matrix(name_coverage,zero,name_exact,zero,place,num,hq,geometry,q.reliable_geo,grades)
    scores=np.sum(f*w,axis=1);count=0;batch_count=0
    cp,ins=encode_code(q.postcode),encode_code(q.insee);number=normalized(q.number)
    for b in address_batches:
        size=len(b.site_positions)
        if (b.site_positions.dtype!=np.uint32 or b.site_positions.shape!=(size,)
            or np.any(b.site_positions<ids[0]) or np.any(b.site_positions>ids[-1])):
            raise ValueError('Every address explicitly bound to this site chunk required')
        bounded_vector(b.coverage,size,np.float64,'whole-address')
        for a in (b.alias_exact,b.current_exact):
            if a.dtype!=np.bool_ or a.shape!=(size,):raise ValueError('Exact original address flags required')
        if np.any(b.alias_exact&(b.coverage!=1.)):
            raise ValueError('Exact address requires whole alias coverage')
        for a in (b.postcode_codes,b.insee_codes):
            if (a.dtype!=np.uint32 or a.shape!=(size,)
                or np.any((a!=MISSING)&(a>=36**5))):raise ValueError('Exact proof-owned public codes required')
        if (len(b.number_texts)!=size or any(not isinstance(s,str) or not s.isascii()
            or s!=s.strip() or '  ' in s or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ' for c in s)
            for s in b.number_texts)):
            raise ValueError('Original once-normalized proof number strings required')
        local=(b.site_positions-ids[0]).astype(np.int64)
        p=np.zeros(size)
        if cp!=MISSING:p=np.maximum(p,b.postcode_codes==cp)
        if ins!=MISSING:p=np.maximum(p,b.insee_codes==ins)
        nm=np.fromiter((bool(number) and s==number for s in b.number_texts),dtype=np.float64,count=size)
        ae=(b.alias_exact&b.current_exact).astype(np.float64)
        f=matrix(name_coverage[local],b.coverage,name_exact[local],ae,p,nm,hq[local],
            geometry[local],q.reliable_geo,grades[local])
        # Max after the complete leaf score, not separate attribute maxima.
        # Even zero lexical coverage may carry explicit place/number evidence.
        np.maximum.at(scores,local,np.sum(f*w,axis=1))
        count+=size;batch_count+=1
    if not np.isfinite(scores).all():raise ValueError('Finite complete leaf scores required')
    return scores,dict(sites=n,address_facts=count,address_batches=batch_count,
        coherent_whole_address_score=True,proof_or_owner_cap=None,
        relation_address_or_attribute_inheritance=False,closed_or_unknown_sites_filtered=False)


def collect(catalog,scored_chunks,limit=100,max_seconds=20.):
    """Complete universe traversal, exact local collectors merged into top100.

    Local top-k is a collector optimization AFTER scoring every leaf, never a
    candidate pool that is later rescored. Equal scores use global SIRET order,
    not physical append positions. No partial result escapes a STOP.
    """
    if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=100:
        raise ValueError('Absolute initial output cap100 required')
    if not np.isfinite(max_seconds) or max_seconds<=0:raise ValueError('Positive query cost guard required')
    if not isinstance(catalog.sites,int) or not 0<catalog.sites<int(MISSING):
        raise ValueError('Complete official site namespace required')
    began=time.monotonic();seen=0;heap=[];chunks=0;positive=0
    def guard():
        if time.monotonic()-began>max_seconds:raise TimeoutError('STOP_INITIAL100_COLLECTION_COST')
    for ids,scores in scored_chunks:
        guard();n=len(ids)
        if (ids.dtype!=np.uint32 or ids.ndim!=1 or not n or seen+n>catalog.sites
            or not np.array_equal(ids,np.arange(seen,seen+n,dtype=np.uint32))):
            raise ValueError('Every official site exactly once, no candidate pool required')
        bounded_vector(scores,n,np.float64,'leaf-score',maximum=np.inf)
        keys=catalog.keys.reconstruct(ids,'site')
        if keys.dtype!=np.uint64 or keys.shape!=(n,) or np.any(keys>=10**14):
            raise ValueError('Every original site identity reconstructed exactly required')
        positive+=int(np.count_nonzero(scores>0))
        # Anything below the existing global cutoff cannot enter top-k.
        cutoff=heap[0][0] if len(heap)==limit else 0.
        rows=np.flatnonzero((scores>0)&(scores>=cutoff))
        if len(rows):
            # Exact local collector <=100, including ALL boundary ties before
            # global identity ordering. There is no subsequent different score.
            order=np.lexsort((keys[rows],-scores[rows]))[:limit]
            for row in rows[order]:
                item=(float(scores[row]),-int(keys[row]),int(ids[row]))
                if len(heap)<limit:heapq.heappush(heap,item)
                elif item>heap[0]:heapq.heapreplace(heap,item)
        seen+=n;chunks+=1;guard()
    guard()
    if seen!=catalog.sites:raise ValueError('Incomplete official universe, no partial initial100 result')
    result=[dict(siret=f'{-key:014d}',siren=f'{(-key)//100000:09d}',score=score)
        for score,key,pos in sorted(heap,reverse=True)]
    return result,dict(sites_completed=seen,positive_sites=positive,chunks=chunks,initial_cap=limit,
        every_site_scored_once=True,heap_peak=len(heap),global_identity_boundary_ties=True,
        fixed_reservoir=None,score_after_fixed_pool=False,identifiers_as_features=False,
        seconds=time.monotonic()-began,national_source_coverage_certified_here=False,
        recall_measured=False,slo_validated=False)


def search(q,value,catalog,names,attributes,addresses,activity=None,limit=100,chunk=65536,max_seconds=20.):
    """Assemble query-side proofs -> coherent full-universe score -> initial100.

    `addresses.batches(ids,q)` must yield every explicit site address fact,
    with query lexical coverage per whole alias and original proof attributes.
    This kernel does NOT certify a caller's sources as complete/official; a
    national launch must separately verify all source manifests and hashes.
    No GT, old hits or downstream model are accepted by this interface.
    """
    if (names.sites!=catalog.sites or attributes.sites!=catalog.sites or addresses.sites!=catalog.sites
        or not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0):
        raise ValueError('Identical complete site namespace and physical chunk required')
    w=weights(value);totals=dict(address_facts=0,address_batches=0,score_chunks=0)
    def scored():
        for start in range(0,catalog.sites,chunk):
            ids=np.arange(start,min(start+chunk,catalog.sites),dtype=np.uint32)
            n,ne=names.gather(ids)
            scores,stats=score_sites(ids,q,w,n,ne,attributes,addresses.batches(ids,q),activity)
            for key in ('address_facts','address_batches'):totals[key]+=stats[key]
            totals['score_chunks']+=1
            yield ids,scores
    rows,stats=collect(catalog,scored(),limit=limit,max_seconds=max_seconds)
    return rows,{**stats,**totals,'coherent_whole_address_score':True,'proof_or_owner_cap':None,
        'relation_address_or_attribute_inheritance':False,'closed_or_unknown_sites_filtered':False}
