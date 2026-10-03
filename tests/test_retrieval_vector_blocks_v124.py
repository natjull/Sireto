import numpy as np
import pytest

from src.xgb_matcher.retrieval_coupled_blocks_v121 import CoupledBlocks,Proof,Query,Site
from src.xgb_matcher.retrieval_tight_blocks_v123 import TightCoupledBlocks
from src.xgb_matcher.retrieval_vector_blocks_v124 import VectorCoupledBlocks


def site(i,**values):
    return Site(f'123456789{i:05d}','123456789',**values)


@pytest.mark.parametrize('reliable,xy',[(True,(.001,1e6)),(False,(.001,1e6)),(False,None)])
def test_every_feature_bound_and_output_bit_exact_to_scalar(reliable,xy):
    rng=np.random.default_rng(124);words=['ALPHA','BETA','GAMMA','DELTA'];rows=[]
    for i in range(300):
        rows.append(site(i,names=tuple(Proof(' '.join(rng.choice(words,size=2)),
            grade=float(rng.choice([0.,.25,.5,1.])),current_exact=bool(rng.integers(2))) for _ in range(int(rng.integers(4)))),
            addresses=(Proof('RUE '+rng.choice(words),current_exact=True),Proof('[ND]')),
            postcode=rng.choice(['75000','69000','']),insee=rng.choice(['75056','']),number=str(rng.integers(3)),
            headquarters=[True,False,None][i%3],xy=tuple(rng.normal(size=2)*500+(0.,1e6)) if i%3 else None,
            activity_class=int(rng.integers(-1,4)),state=['A','F',None][i%3]))
    q=Query(name='ALPHA BETA',address='RUE GAMMA',number='1',postcode='75000',insee='75056',xy=xy,reliable_geo=reliable)
    for block_size in (1,17,128):
        old=TightCoupledBlocks(rows,block_size);new=VectorCoupledBlocks(rows,block_size)
        for _ in range(2):
            w=rng.random(30)*64;activity={i:float(rng.random()) for i in range(4)}
            t,c=old._prepare(q,w,activity);u,d=new._prepare(q,w,activity)
            assert np.array_equal(old.bounds(q,w,activity,t),new.bounds(q,w,activity,u))
            for i in rng.permutation(len(rows)):
                assert np.array_equal(old.features(i,q,activity,t,c),new.features(i,q,activity,u,d))
                assert len(d[0]['_block_features'])==1
            a,x=old.search(q,w,activity,limit=37);b,y=new.search(q,w,activity,limit=37)
            assert a==b==new.exhaustive(q,w,activity,limit=37)
            assert (x['blocks_completed'],x['sites_completed'])==(y['blocks_completed'],y['sites_completed'])


def test_current_exact_grade_not_replaced_by_stronger_historical_grade():
    rows=[site(0,names=(Proof('ALPHA',grade=1.),Proof('ALPHA',grade=.25,current_exact=True))),
        site(1,names=(Proof('[ND]'),Proof('ALPHA',grade=0.,current_exact=True)))]
    index=VectorCoupledBlocks(rows,2);q=Query(name='ALPHA');w=np.ones(30)
    terms,caches=index._prepare(q,w,{})
    assert index.features(0,q,{},terms,caches)[0]==1.
    assert index.features(0,q,{},terms,caches)[2]==.25
    assert index.features(1,q,{},terms,caches)[0]==0.
    assert index.search(q,w)[0]==index.exhaustive(q,w)


def test_query_state_is_isolated_when_interleaved_and_oracle_is_scalar(monkeypatch):
    rows=[site(i,names=(Proof('ALPHA' if i%2 else 'BETA',current_exact=True),)) for i in range(225)]
    index=VectorCoupledBlocks(rows,16);w=np.ones(30);a=Query(name='ALPHA');b=Query(name='BETA')
    x,c=index._prepare(a,w,{});y,d=index._prepare(b,w,{})
    original=CoupledBlocks(rows,16)
    for i in range(len(rows)):
        for query,terms,caches in ((a,x,c),(b,y,d)):
            t,z=original._prepare(query,w,{})
            assert np.array_equal(index.features(i,query,{},terms,caches),original.features(i,query,{},t,z))
    expected=original.exhaustive(a,w)
    monkeypatch.setattr(index,'features',lambda *args: (_ for _ in ()).throw(AssertionError('vector oracle reused')))
    assert index.exhaustive(a,w)==expected


def test_all_variants_and_global_boundary_ties_without_candidate_pool():
    rows=[site(i,names=(Proof('ALPHA'),)) for i in reversed(range(325))]
    rows.append(site(999,names=tuple(Proof(f'OFFICIAL {i}') for i in range(250))))
    index=VectorCoupledBlocks(rows,16);w=np.zeros(30);w[0]=1.;q=Query(name='ALPHA')
    result,trace=index.search(q,w)
    assert len(index.aliases[0])==251
    assert [r['siret'] for r in result]==[site(i).siret for i in range(100)]
    assert result==index.exhaustive(q,w)
    assert trace['sites_completed']>=325 and trace['fixed_reservoir'] is None
