from types import SimpleNamespace

import numpy as np
import pytest

from src.xgb_matcher.retrieval_initial_collector_v146 import AddressBatch,AppendedAttributes,NameSnapshot,score_sites,collect,search
from src.xgb_matcher.retrieval_coupled_blocks_v121 import Query,native
from src.xgb_matcher.retrieval_activity_joint_v114 import vector
from src.xgb_matcher.retrieval_site_attributes_v139 import encode_code


class Attributes:
    def __init__(self,n=3):self.sites=n
    def flags(self,ids,*unused):
        return np.zeros(len(ids)),np.zeros(len(ids)),np.zeros(len(ids)),np.full(len(ids),-1,dtype=np.int32)
    def coordinates(self,ids):return np.full((len(ids),2),np.nan)


def address(positions,covers,cps,numbers,current=None,exact=None):
    n=len(positions)
    return AddressBatch(np.array(positions,dtype=np.uint32),np.array(covers,dtype=np.float64),
        np.array([False]*n if exact is None else exact,dtype=np.bool_),
        np.array([False]*n if current is None else current,dtype=np.bool_),
        np.array([encode_code(v) for v in cps],dtype=np.uint32),
        np.array([encode_code('')]*n,dtype=np.uint32),tuple(numbers))


def test_no_frankenaddress_text_place_and_number_from_different_proofs():
    q=Query(address='RUE A',postcode='75001',number='7');w=np.zeros(30);w[7]=1.;w[5]=1.
    ids=np.arange(3,dtype=np.uint32);names=np.zeros(3)
    # Site0 has lexical evidence elsewhere + separate matching CP/number;
    # site1 has ALL evidence in the SAME proof. Site2 has no proofs at all.
    b=address([0,0,1],[1.,0.,1.],['69001','75001','75001'],['9','7','7'])
    scores,stats=score_sites(ids,q,w,names,names,Attributes(),[b])
    assert scores.tolist()==[1.,2.,0.]
    assert stats['address_facts']==3 and stats['coherent_whole_address_score']
    assert stats['proof_or_owner_cap'] is None


def test_historical_proof_exact_never_inherits_current_flag_from_another_address():
    w=np.zeros(30);w[3]=10.;w[1]=1.;ids=np.arange(3,dtype=np.uint32)
    b=address([0,0,1],[1.,0.,1.],['','',''],['','',''],current=[False,True,True],exact=[True,False,True])
    scores,s=score_sites(ids,Query(address='RUE A'),w,np.zeros(3),np.zeros(3),Attributes(),[b])
    assert scores.tolist()==[1.,11.,0.]


def test_proof_batches_are_storage_only_all_2501_observations_scored():
    ids=np.arange(3,dtype=np.uint32);w=np.ones(30);n=np.array([.2,.5,0.])
    b=address([0]*2500+[1],[0.]*2500+[1.],['']*2500+['75001'],['']*2500+['7'])
    q=Query(postcode='75001',number='7');full,s=score_sites(ids,q,w,n,np.zeros(3),Attributes(),[b])
    slices=[]
    for lo in range(0,2501,13):
        hi=min(lo+13,2501)
        slices.append(AddressBatch(*(a[lo:hi] for a in (b.site_positions,b.coverage,b.alias_exact,b.current_exact,
            b.postcode_codes,b.insee_codes,b.number_texts))))
    batched,t=score_sites(ids,q,w,n,np.zeros(3),Attributes(),iter(slices))
    np.testing.assert_array_equal(full,batched)
    assert s['address_facts']==t['address_facts']==2501 and t['address_batches']==len(slices)


def test_complete_proof_score_matches_scalar_native30_with_soft_geo_and_activity():
    class Evidence(Attributes):
        def flags(self,ids,*unused):
            return np.zeros(len(ids)),np.zeros(len(ids)),np.array([1.,0.,0.])[ids],np.array([5,-1,5],dtype=np.int32)[ids]
        def coordinates(self,ids):return np.array([[100.,0.],[np.nan,np.nan],[500.,0.]])[ids]
    ids=np.arange(3,dtype=np.uint32);w=np.arange(1,31,dtype=np.float64)/30
    n=np.array([.8,.2,.5]);ne=np.array([0.,.2,0.]);q=Query(postcode='75001',number='7',xy=(0.,0.),reliable_geo=True)
    b=address([0,1,2],[1.,.4,.7],['75001','69001','75001'],['7','7','9'],current=[True,False,False],exact=[True,False,False])
    scores,s=score_sites(ids,q,w,n,ne,Evidence(),[b],activity={5:.6})
    for i in range(3):
        geo=np.zeros(4) if i==1 else 1./(1.+(100. if i==0 else 500.)/np.array([25.,100.,500.,2000.]))
        grade=0. if i==1 else .6;hq=float(i==0);p=float(i!=1);num=float(i!=2)
        baseline=np.sum(vector(native(n[i],0.,ne[i],0.,0.,0.,hq,geo,True),grade)*w)
        proof=np.sum(vector(native(n[i],b.coverage[i],ne[i],float(i==0),p,num,hq,geo,True),grade)*w)
        assert scores[i]==max(baseline,proof)


def test_appended_attributes_unknown_never_inherit_from_same_parent():
    ref={'path':'catalog','sha256':'a'}
    a=Attributes(3);a.base=2;a.metadata={'sources':{'catalog':ref}}
    catalog=SimpleNamespace(sites=5,previous=SimpleNamespace(sites=3),
        metadata={'previous_catalog':ref,'stats':{'every_previous_site_and_parent_position_unchanged':True}})
    e=AppendedAttributes(a,catalog);ids=np.array([0,3,4],dtype=np.uint32)
    assert e.flags(ids)[3].tolist()==[-1,-1,-1] and np.isnan(e.coordinates(ids)).all()
    catalog.metadata['previous_catalog']={'path':'different','sha256':'b'}
    with pytest.raises(ValueError,match='namespace'):AppendedAttributes(a,catalog)


class Keys:
    def __init__(self,values):self.values=np.array(values,dtype=np.uint64)
    def reconstruct(self,ids,grain):assert grain=='site';return self.values[ids]


def catalog(keys):return SimpleNamespace(sites=len(keys),keys=Keys(keys))


def chunks(scores,size):
    scores=np.array(scores,dtype=np.float64)
    for start in range(0,len(scores),size):
        yield np.arange(start,min(len(scores),start+size),dtype=np.uint32),scores[start:start+size]


def test_collector_complete_all_3007_sites_not_pool_and_global_append_ties():
    # Physical namespace deliberately is NOT globally sorted by SIRET.
    keys=np.arange(1,3008,dtype=np.uint64)[::-1]+np.uint64(100000)
    scores=np.ones(3007);c=catalog(keys)
    expected=[f'{int(k):014d}' for k in sorted(keys)[:100]]
    for size in (11,128,3007):
        rows,s=collect(c,chunks(scores,size))
        assert [r['siret'] for r in rows]==expected
        assert s['sites_completed']==3007 and s['heap_peak']==100
        assert s['every_site_scored_once'] and s['fixed_reservoir'] is None
        assert not s['score_after_fixed_pool'] and not s['recall_measured']


def test_collector_matches_full_global_sort_random_scores_and_boundary_ties():
    rng=np.random.default_rng(56);keys=rng.permutation(np.arange(700001,702502,dtype=np.uint64))
    scores=rng.integers(0,9,len(keys)).astype(np.float64)/8
    expected=sorted(((float(s),int(k)) for s,k in zip(scores,keys) if s>0),key=lambda pair:(-pair[0],pair[1]))[:100]
    rows,s=collect(catalog(keys),chunks(scores,37))
    assert [(r['score'],int(r['siret'])) for r in rows]==expected


@pytest.mark.parametrize('kind',['missing_tail','skipped_site','duplicate_site','negative_score','nan_score'])
def test_incomplete_or_corrupt_universe_stops_without_partial_result(kind):
    c=catalog([100001,100002,100003]);parts=list(chunks([1.,2.,3.],2))
    if kind=='missing_tail':parts.pop()
    elif kind=='skipped_site':parts[0]=(np.array([0,2],dtype=np.uint32),np.ones(2))
    elif kind=='duplicate_site':parts[1]=(np.array([1],dtype=np.uint32),np.ones(1))
    elif kind=='negative_score':parts[0][1][0]=-1.
    else:parts[0][1][0]=np.nan
    with pytest.raises(ValueError):collect(c,parts)


def test_timeout_or_invalid_cap_never_returns_partial100(monkeypatch):
    for limit in (0,101,True):
        with pytest.raises(ValueError,match='cap100'):collect(catalog([100001]),chunks([1.],1),limit=limit)
    times=iter([0.,0.,2.]);monkeypatch.setattr('src.xgb_matcher.retrieval_initial_collector_v146.time.monotonic',lambda:next(times))
    with pytest.raises(TimeoutError,match='STOP_INITIAL100'):collect(catalog([100001]),chunks([1.],1),max_seconds=1.)


def test_bad_proof_site_code_exact_or_number_is_stop_not_ignored():
    ids=np.arange(3,dtype=np.uint32);w=np.ones(30)
    examples=[address([3],[1.],[''],['']),address([0],[.5],[''],[''],exact=[True]),address([0],[1.],[''],['[ND]'])]
    for b in examples:
        with pytest.raises(ValueError):score_sites(ids,Query(),w,np.zeros(3),np.zeros(3),Attributes(),[b])
    with pytest.raises(ValueError,match='Exact name'):
        score_sites(ids,Query(),w,np.zeros(3),np.ones(3),Attributes(),[])


def test_name_parent_direct_site_and_relation_scopes_survive_appended_sites():
    c=SimpleNamespace(sites=5,parents=3,previous=SimpleNamespace(sites=3,parents=2),
        new={'new_site_parent_positions':np.array([0,2],dtype=np.uint32)})
    names=NameSnapshot(c,np.array([0,0,1],dtype=np.uint32),np.array([.7,0.]),np.array([False,False]),
        np.array([0.,1.,0.]),np.array([False,True,False]),np.array([.25,0.,.5]))
    coverage,exact=names.gather(np.arange(5,dtype=np.uint32))
    assert coverage.tolist()==[.7,1.,.5,.7,0.]
    assert exact.tolist()==[0.,1.,0.,0.,0.]
    # Site1 exact alias does NOT become a parent alias or sibling exact proof.
    assert names.site_cov.tolist()==[0.,1.,0.] and names.parent_cov.tolist()==[.7,0.]


def test_end_to_end_score_and_collector_chunk_parity_without_any_pre_pool():
    c=catalog([100004,100003,100002,100001]);c.parents=1
    c.previous=SimpleNamespace(sites=4,parents=1)
    c.new={'new_site_parent_positions':np.array([],dtype=np.uint32)}
    names=NameSnapshot(c,np.zeros(4,dtype=np.uint32),np.array([.2]),np.array([False]),
        np.array([0.,1.,0.,0.]),np.array([False,True,False,False]),np.array([.5,0.,0.,0.]))
    proofs=address([0,0,2,3],[1.,0.,1.,.3],['69001','75001','75001',''],['9','7','7',''])
    class Addresses:
        sites=4
        def batches(self,ids,q):
            selected=np.flatnonzero((proofs.site_positions>=ids[0])&(proofs.site_positions<=ids[-1]))
            if len(selected):
                yield AddressBatch(*(v[selected] for v in (proofs.site_positions,proofs.coverage,proofs.alias_exact,
                    proofs.current_exact,proofs.postcode_codes,proofs.insee_codes)),
                    tuple(proofs.number_texts[int(i)] for i in selected))
    w=np.zeros(30);w[0]=1.;w[7]=1.;w[5]=1.
    q=Query(address='RUE A',postcode='75001',number='7');observed=[]
    for chunk in (1,2,4):
        rows,s=search(q,w,c,names,Attributes(4),Addresses(),chunk=chunk)
        observed.append(rows)
        assert s['sites_completed']==4 and s['address_facts']==4
        assert not s['national_source_coverage_certified_here'] and not s['recall_measured']
    assert observed[0]==observed[1]==observed[2]
    assert [r['siret'] for r in observed[0]]==['00000000100002','00000000100004','00000000100003','00000000100001']


def test_search_rejects_mismatched_source_namespaces():
    names=SimpleNamespace(sites=3);a=SimpleNamespace(sites=2);addresses=SimpleNamespace(sites=3)
    with pytest.raises(ValueError,match='namespace'):
        search(Query(),np.ones(30),catalog([100001,100002,100003]),names,a,addresses)
