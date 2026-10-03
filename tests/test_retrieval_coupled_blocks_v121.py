from dataclasses import replace

import numpy as np
import pytest

from src.xgb_matcher.retrieval_coupled_blocks_v121 import CoupledBlocks, Proof, Query, Site


def site(i, **values):
    return Site(siret=f'123456789{i:05d}',siren='123456789',**values)


def weight(*indices):
    w=np.zeros(30);w[list(indices)]=1.;return w


def test_deep_joint_winner_not_in_top100_of_either_channel():
    rows=[site(i,names=(Proof('A B C D E'),)) for i in range(150)]
    rows += [site(i,xy=(0.,0.)) for i in range(150,300)]
    rows += [site(999,names=(Proof('A B C'),),xy=(50.,0.))]
    q=Query(name='A B C D E',xy=(0.,0.));index=CoupledBlocks(rows,16)
    result,trace=index.search(q,weight(0,12))
    assert result[0]['siret']==site(999).siret and result[0]['score']>1.
    assert result==index.exhaustive(q,weight(0,12))
    assert trace['fixed_reservoir'] is None and len(result)==100


def test_multiple_variants_never_combined_to_fabricate_name():
    index=CoupledBlocks([site(0,names=(Proof('ALPHA'),Proof('BETA'))),
                         site(1,names=(Proof('ALPHA BETA'),))],1)
    rows,_=index.search(Query(name='ALPHA BETA'),weight(0))
    assert rows[0]['siret']==site(1).siret and rows[0]['score']==1.
    assert rows[1]['score']<1.


def test_former_enseigne_searchable_but_not_claimed_current_exact():
    index=CoupledBlocks([site(0,names=(Proof('ENSEIGNE ANCIENNE',current_exact=False),),state='F')],1)
    assert index.search(Query(name='ENSEIGNE ANCIENNE'),weight(0))[0][0]['score']==1.
    assert index.search(Query(name='ENSEIGNE ANCIENNE'),weight(2))[0]==[]


def test_no_coordinates_and_wrong_locality_still_lexically_reachable():
    index=CoupledBlocks([site(0,names=(Proof('ALPHA'),),postcode='69000',state='F',headquarters=None)],1)
    result,trace=index.search(Query(name='ALPHA',postcode='75000',xy=(800000.,6200000.)),weight(0))
    assert result[0]['siret']==site(0).siret and not trace['locality_or_closed_state_filter']


def test_adjacent_street_target_has_distance_proof_without_address_match():
    index=CoupledBlocks([site(0,addresses=(Proof('RUE PERPENDICULAIRE'),),xy=(0.,50.)),
                         site(1,addresses=(Proof('RUE ADJACENTE'),),xy=(500.,0.))],1)
    q=Query(address='RUE PRINCIPALE',xy=(0.,0.))
    result,_=index.search(q,weight(12))
    assert result[0]['siret']==site(0).siret
    assert result[0]['score']==pytest.approx(1./1.5)


def test_all_boundary_ties_considered_and_canonical_global_order():
    index=CoupledBlocks([site(i,names=(Proof('ALPHA'),),xy=(0.,0.)) for i in reversed(range(325))],13)
    result,trace=index.search(Query(name='ALPHA'),weight(0))
    assert [r['siret'] for r in result]==[site(i).siret for i in range(100)]
    assert trace['sites_completed']==325 and trace['global_boundary_ties']


def test_unknown_activity_not_a_filter_and_activity_only_modulates_site_evidence():
    index=CoupledBlocks([site(0,names=(Proof('ALPHA'),),activity_class=-1),
        site(1,names=(Proof('ALPHA'),),activity_class=1),site(2,activity_class=1)],1)
    result,_=index.search(Query(name='ALPHA'),weight(0),{1:1.})
    assert [r['siret'] for r in result]==[site(0).siret,site(1).siret]
    assert index.search(Query(),weight(19),{1:1.})[0]==[]


def test_bbox_pruning_does_not_prune_exact_joint_winner():
    rows=[site(i,xy=(float(i*1000),0.)) for i in range(2000)]
    index=CoupledBlocks(rows,16);q=Query(xy=(0.,0.));w=weight(12)
    result,trace=index.search(q,w)
    assert result==index.exhaustive(q,w)
    assert trace['blocks_pruned']>0 and trace['sites_completed']<200


def test_empty_evidence_returns_no_arbitrary_id_candidates():
    index=CoupledBlocks([site(0,names=(Proof('[ND]'),))],1)
    assert index.search(Query(name='123456789'),weight(0))[0]==[]


def test_hypothesis_cannot_claim_current_exact_identity():
    with pytest.raises(ValueError,match='hypothesis'):
        CoupledBlocks([site(0,names=(Proof('ALPHA',current_exact=True,relation_hypothesis=True),))])


@pytest.mark.parametrize('reliable,point',[(True,(0.,0.)),(False,(0.,0.)),(False,None)])
def test_random_all30_nonnegative_bounds_and_exact_order(reliable,point):
    rng=np.random.default_rng(121);vocab=['ALPHA','BETA','GAMMA','DELTA','EPSILON']
    rows=[]
    for i in range(450):
        names=tuple(Proof(' '.join(rng.choice(vocab,size=int(rng.integers(1,4)))),
                    grade=float(rng.choice([.25,.5,1.])),current_exact=bool(rng.integers(2)))
                    for _ in range(int(rng.integers(4))))
        addresses=tuple(Proof('RUE '+rng.choice(vocab),current_exact=True) for _ in range(int(rng.integers(3))))
        rows.append(site(i,names=names,addresses=addresses,xy=tuple(rng.normal(size=2)*1000) if i%4 else None,
            insee=rng.choice(['75056','69383','']),postcode=rng.choice(['75000','69000','']),
            number=str(rng.integers(3)),headquarters=bool(rng.integers(2)),activity_class=int(rng.integers(-1,4))))
    q=Query(name='ALPHA BETA',address='RUE GAMMA',insee='75056',postcode='75000',number='1',xy=point,reliable_geo=reliable)
    for size in [1,17,128]:
        index=CoupledBlocks(rows,size)
        for _ in range(4):
            w=rng.random(30)*3;activity={i:float(rng.random()) for i in range(4)}
            terms,caches=index._prepare(q,w,activity);upper=index.bounds(q,w,activity,terms)
            for b,ids in enumerate(index.groups):
                assert max(float(np.sum(index.features(i,q,activity,terms,caches)*w)) for i in ids)<=upper[b]
            assert index.search(q,w,activity,limit=37)[0]==index.exhaustive(q,w,activity,limit=37)


def test_score_is_independently_declared30_formula_for_simple_complete_case():
    s=site(0,names=(Proof('ALPHA',grade=.5,current_exact=True),),addresses=(Proof('RUE A',current_exact=True),),
           insee='75056',number='1',headquarters=True,xy=(100.,0.),activity_class=1)
    q=Query(name='ALPHA',address='RUE A',insee='75056',number='1',xy=(0.,0.),reliable_geo=True)
    w=np.arange(1,31,dtype=float)/3.;index=CoupledBlocks([s],1)
    geo=[1/(1+100/s) for s in (25,100,500,2000)]
    declared=[.5,1.,.5,1.,1.,1.,.5,1.,1.5,1.,.5,*geo,*geo]
    declared += [.7*declared[i] for i in (4,5,7,11,12,13,14,15,16,17,18)]
    result,_=index.search(q,w,{1:.7})
    assert result[0]['score']==pytest.approx(sum(a*b for a,b in zip(declared,w)),abs=1e-12)


@pytest.mark.parametrize('limit',[0,101,True,1.1])
def test_invalid_initial_caps_rejected(limit):
    with pytest.raises(ValueError):
        CoupledBlocks([site(0)]).search(Query(),weight(0),limit=limit)


def test_query_budget_stops_without_returning_uncertified_outputs():
    with pytest.raises(TimeoutError):
        CoupledBlocks([site(0,names=(Proof('ALPHA'),))]).search(Query(name='ALPHA'),weight(0),max_seconds=1e-12)
