"""Append-only official parent-name joins, retaining every V137 fact position.

Sparse patches are identity joins, never similarity features or CRM aliases.
No source proof, old position, direct-site scope or exact flag is rewritten.
"""
import numpy as np

from .retrieval_proof_bridge_v137 import MISSING
from .retrieval_initial_collector_v146 import NameSnapshot,bounded_vector


def rebind(keys,source,bridge,chunk=131072):
    """Audit ALL facts; resolve only old MISSING UL positions into V141."""
    if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<=0:
        raise ValueError('Positive physical audit chunk required')
    p,o,s,sc=(source[k] for k in ('parent_key','owner_key','subject_code','scope_code'))
    pp,sp=(bridge[k] for k in ('fact_parent_positions','fact_site_positions'))
    n=len(p)
    for a,dtype in ((p,np.uint32),(o,np.uint64),(s,np.uint8),(sc,np.uint8),(pp,np.uint32),(sp,np.uint32)):
        if a.dtype!=dtype or a.shape!=(n,):raise ValueError('Whole exact typed official source and old positions required')
    if n>=MISSING:raise ValueError('Whole unsigned fact namespace required')
    rows=[];positions=[];missing=sites=0
    for lo in range(0,n,chunk):
        hi=min(n,lo+chunk);parent,owner,subject,scope=(a[lo:hi] for a in (p,o,s,sc))
        old,site=pp[lo:hi],sp[lo:hi];ul=subject==1;leaf=subject==2;bound=old!=MISSING
        if (np.any(parent>=10**9) or np.any(owner>=10**14) or np.any(~np.isin(subject,[1,2]))
            or np.any(~np.isin(scope,[1,2,3])) or np.any(ul&((owner!=parent)|~np.isin(scope,[1,2])))
            or np.any(leaf&((owner//np.uint64(100000)!=parent)|(scope!=3)))
            or np.any(old[bound]>=keys.previous.parents) or np.any(site[leaf]>=keys.previous.sites)
            or np.any(site[ul]!=MISSING) or np.any(leaf&~bound)):
            raise ValueError('Unchanged parent/site grain, scope and old namespace required')
        if (not np.array_equal(keys.previous.reconstruct(old[bound],'parent'),parent[bound])
            or not np.array_equal(keys.previous.reconstruct(site[leaf],'site'),owner[leaf])):
            raise ValueError('Every original bound fact must reconstruct exactly')
        missing_rows=np.flatnonzero(~bound);values=parent[missing_rows];missing+=len(values);sites+=int(np.sum(leaf))
        if np.any(keys.previous.parent_positions(values)!=MISSING):
            raise ValueError('Old MISSING must be genuinely absent, never hide an old bound proof')
        new=keys.parent_positions(values);hit=new!=MISSING
        if (np.any(new[hit]<keys.previous.parents) or np.any(new[hit]>=keys.parents)
            or not np.array_equal(keys.reconstruct(new[hit],'parent'),values[hit])):
            raise ValueError('Every recovered parent must be an exact appended official identity')
        rows.append((np.uint64(lo)+missing_rows[hit]).astype(np.uint32));positions.append(new[hit])
    r=np.concatenate(rows) if rows else np.empty(0,dtype=np.uint32)
    pos=np.concatenate(positions) if positions else np.empty(0,dtype=np.uint32)
    return r,pos,dict(facts=n,site_facts=sites,previous_unbound_parent_facts=missing,
        recovered_parent_facts=len(r),remaining_unbound_parent_facts=missing-len(r),
        recovered_parent_positions=np.unique(pos).tolist(),every_original_fact_audited=True,
        every_previous_bound_position_unchanged=True,every_patch_reconstructed_exactly=True,
        every_unbound_fact_preserved=True,site_alias_lifted_to_parent=False,
        name_or_owner_or_member_cap=None,identifiers_as_features=False)


def patched_positions(previous,rows,positions,old_parents,total_parents):
    """Copy a physical bucket only; patch exactly listed old MISSING facts."""
    if (previous.dtype!=np.uint32 or previous.ndim!=1 or rows.dtype!=np.uint32 or rows.ndim!=1
        or positions.dtype!=np.uint32 or positions.shape!=rows.shape
        or any(not isinstance(v,int) or isinstance(v,bool) for v in (old_parents,total_parents))
        or not 0<=old_parents<=total_parents<MISSING or np.any(rows>=len(previous))
        or np.any(rows[1:]<=rows[:-1]) or np.any(positions<old_parents) or np.any(positions>=total_parents)
        or np.any(previous[rows]!=MISSING)
        or np.any((previous!=MISSING)&(previous>=old_parents))):
        raise ValueError('Exact sorted unique sparse append-only patches required')
    result=previous.copy();result[rows]=positions
    return result


class ReboundNameSnapshot:
    """V146 unchanged old-site semantics plus explicit names of new parents.

All parent vectors must come from the same complete source-side scatter,
using audited V149 patches. This adapter does not certify its caller's
source authority or compute name features by itself.
"""
    def __init__(self,catalog,site_parents,parent_cov,parent_exact,site_cov,site_exact,relation_cov):
        bounded_vector(parent_cov,catalog.parents,np.float64,'whole rebound parent-name')
        if (parent_exact.dtype!=np.bool_ or parent_exact.shape!=(catalog.parents,)
            or np.any(parent_exact&(parent_cov!=1.))):
            raise ValueError('Whole exact direct official parent-name evidence required')
        old=catalog.previous.parents
        self.previous=NameSnapshot(catalog,site_parents,parent_cov[:old],parent_exact[:old],site_cov,site_exact,relation_cov)
        self.catalog=catalog;self.sites=catalog.sites;self.parent_cov=parent_cov;self.parent_exact=parent_exact

    def gather(self,ids):
        cov,exact=self.previous.gather(ids);new_sites=ids>=self.catalog.previous.sites
        parents=self.catalog.new['new_site_parent_positions'][ids[new_sites]-self.catalog.previous.sites]
        new_parents=parents>=self.catalog.previous.parents;rows=np.flatnonzero(new_sites)[new_parents]
        cov[rows]=self.parent_cov[parents[new_parents]];exact[rows]=self.parent_exact[parents[new_parents]]
        return cov,exact
