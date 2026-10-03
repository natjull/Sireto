import numpy as np
import pytest

from src.xgb_matcher.retrieval_coupled_blocks_v121 import CoupledBlocks,Proof,Query,Site
from src.xgb_matcher.retrieval_tight_blocks_v123 import TightCoupledBlocks


def site(i,**values):
    return Site(f'123456789{i:05d}','123456789',**values)


def test_mixed_alias_words_upper_not_used_as_actual_alias_maximum():
    sites=[site(0,names=(Proof('ALPHA'),)),site(1,names=(Proof('BETA'),))]
    old=CoupledBlocks(sites,128);new=TightCoupledBlocks(sites,128);w=np.zeros(30);w[0]=1.
    q=Query(name='ALPHA BETA');terms,_=old._prepare(q,w,{})
    assert new.bounds(q,w,{},terms)[0]<old.bounds(q,w,{},terms)[0]
    assert new.search(q,w)[0]==old.search(q,w)[0]==new.exhaustive(q,w)


@pytest.mark.parametrize('reliable,xy',[(True,(0.,0.)),(False,(0.,0.)),(False,None)])
def test_tighter_safe_bound_and_identical30_score_order_over_random_corpus(reliable,xy):
    rng=np.random.default_rng(123);words=['ALPHA','BETA','GAMMA','DELTA'];rows=[]
    for i in range(400):
        rows.append(site(i,names=tuple(Proof(' '.join(rng.choice(words,size=2)),grade=float(rng.choice([.25,.5,1.])),
            current_exact=bool(rng.integers(2))) for _ in range(int(rng.integers(4)))),
            addresses=(Proof('RUE '+rng.choice(words),current_exact=True),),postcode=rng.choice(['75000','69000','']),
            number=str(rng.integers(3)),headquarters=bool(rng.integers(2)),xy=tuple(rng.normal(size=2)*500) if i%3 else None,
            activity_class=int(rng.integers(-1,3))))
    q=Query(name='ALPHA BETA',address='RUE GAMMA',number='1',postcode='75000',xy=xy,reliable_geo=reliable)
    for size in [1,17,128]:
        old=CoupledBlocks(rows,size);new=TightCoupledBlocks(rows,size)
        for _ in range(3):
            w=rng.random(30)*4.;activity={i:float(rng.random()) for i in range(3)}
            terms,caches=old._prepare(q,w,activity);a=old.bounds(q,w,activity,terms);b=new.bounds(q,w,activity,terms)
            assert np.all(b<=a)
            for j,ids in enumerate(new.groups):
                assert max(float(np.sum(new.features(i,q,activity,terms,caches)*w)) for i in ids)<=b[j]
            before,told=old.search(q,w,activity,limit=37);after,tnew=new.search(q,w,activity,limit=37)
            assert before==after==new.exhaustive(q,w,activity,limit=37)
            assert tnew['sites_completed']<=told['sites_completed']


def test_no_alias_cap_and_all_ties_kept():
    rows=[site(i,names=(Proof('ALPHA'),)) for i in reversed(range(325))]
    rows.append(site(999,names=tuple(Proof(f'OFFICIAL {i}') for i in range(250))))
    q=Query(name='ALPHA');w=np.zeros(30);w[0]=1.;index=TightCoupledBlocks(rows,16)
    result,trace=index.search(q,w)
    assert len(index.aliases[0])==251
    assert [r['siret'] for r in result]==[site(i).siret for i in range(100)]
    assert trace['sites_completed']>=325 and trace['fixed_reservoir'] is None
