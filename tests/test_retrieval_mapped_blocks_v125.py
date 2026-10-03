import numpy as np
import pytest

from src.xgb_matcher.retrieval_coupled_blocks_v121 import Proof,Query,Site
from src.xgb_matcher.retrieval_vector_blocks_v124 import VectorCoupledBlocks
from src.xgb_matcher.retrieval_mapped_blocks_v125 import MappedBlocks,export_fixture,scatter_alias_max
from src.xgb_matcher.retrieval_v20_workbench import record,write_json


def site(i,**values):return Site(f'123456789{i:05d}','123456789',**values)


def fixture(index,path):
    write_json(path/'source.json',dict(scope='UNIT_TEST_SOURCE_ONLY_FIXTURE'))
    export_fixture(index,path/'index',[record(path/'source.json')])
    return MappedBlocks(path/'index')


@pytest.mark.parametrize('reliable,xy',[(True,(0.,1e6)),(False,(0.,1e6)),(False,None)])
def test_mapped_every_feature_bound_and_top100_match_independent_scalar(tmp_path,reliable,xy):
    rng=np.random.default_rng(125);words=['ALPHA','BETA','GAMMA','DELTA'];rows=[]
    for i in range(300):
        rows.append(site(i,names=tuple(Proof(' '.join(rng.choice(words,size=2)),
            grade=float(rng.choice([0.,.123,.25,.5,1.])),current_exact=bool(rng.integers(2))) for _ in range(int(rng.integers(4)))),
            addresses=(Proof('RUE '+rng.choice(words),current_exact=True),Proof('[ND]')),
            postcode=rng.choice(['75000','69000','']),insee=rng.choice(['75056','']),number=str(rng.integers(3)),
            headquarters=[True,False,None][i%3],xy=tuple(rng.normal(size=2)*500+(0.,1e6)) if i%3 else None,
            activity_class=int(rng.integers(-1,4)),state=['A','F',None][i%3]))
    old=VectorCoupledBlocks(rows,17);new=fixture(old,tmp_path)
    q=Query(name='ALPHA BETA',address='RUE GAMMA',number='1',postcode='75000',insee='75056',xy=xy,reliable_geo=reliable)
    for _ in range(3):
        w=rng.random(30)*64;activity={i:float(rng.random()) for i in range(4)}
        terms,caches=old._prepare(q,w,activity);scores,exact=new.prepare(q,activity)
        assert all(np.array_equal(a,b) for a,b in zip(terms.scores,scores))
        assert np.array_equal(old.bounds(q,w,activity,terms),new.bounds(q,w,activity,scores,exact))
        for b,ids in enumerate(old.groups):
            got,f=new.block_features(b,q,activity,scores,exact)
            assert np.array_equal(got,ids)
            assert np.array_equal(f,np.array([old.features(i,q,activity,terms,caches) for i in ids]))
        before,x=old.search(q,w,activity,limit=37);after,y=new.search(q,w,activity,limit=37)
        assert before==after==old.exhaustive(q,w,activity,limit=37)
        assert (x['sites_completed'],x['blocks_completed'])==(y['sites_completed'],y['blocks_completed'])
    assert all(isinstance(a,np.memmap) and not a.flags.writeable for a in new.a.values())
    assert not hasattr(new,'aliases') and not hasattr(new,'groups') and isinstance(new.sites,int)


def test_all_masked_unknowns_keep_geo_search_and_empty_dictionaries(tmp_path):
    rows=[site(i,names=(Proof('[ND]'),),addresses=(),xy=(float(i),0.) if i%2 else None,
        headquarters=None,state=None) for i in range(225)]
    old=VectorCoupledBlocks(rows,16);new=fixture(old,tmp_path)
    q=Query(name='INCONNU',postcode='é',xy=(0.,0.));w=np.ones(30)
    a,x=old.search(q,w);b,y=new.search(q,w)
    assert a==b==old.exhaustive(q,w)
    assert len(new.text['side0_alias'])==0
    assert len(b)==100 and y['locality_or_closed_state_filter'] is False
    assert np.all(new.a['site_hq']==0) and np.all(new.a['site_state']==0)


def test_current_grade_different_from_history_and_no_alias_cap_or_tie_loss(tmp_path):
    rows=[site(i,names=(Proof('ALPHA',grade=1.),Proof('ALPHA',grade=.25,current_exact=True))) for i in reversed(range(325))]
    rows.append(site(999,names=tuple(Proof(f'OFFICIAL {i}') for i in range(250))))
    old=VectorCoupledBlocks(rows,16);new=fixture(old,tmp_path);w=np.ones(30);q=Query(name='ALPHA')
    a,x=old.search(q,w);b,y=new.search(q,w)
    assert a==b==old.exhaustive(q,w)
    assert [r['siret'] for r in b]==[site(i).siret for i in range(100)]
    assert len(new.text['side0_alias'])==251 and y['sites_completed']>=325


@pytest.mark.parametrize('chunk',[1,2,3,17,65536])
def test_sparse_scatter_all_references_with_empty_and_very_large_fanout(chunk):
    counts=np.array([0,3,0,100,1,20,0]);offsets=np.r_[0,np.cumsum(counts)].astype(np.uint64)
    rng=np.random.default_rng(12);targets=rng.integers(0,37,size=offsets[-1],dtype=np.uint32)
    grades=rng.random(len(targets));scores=np.array([0.,1.,.2,.5,0.,.3,1.]);out=np.zeros(37);expected=np.zeros(37)
    for i in range(len(scores)):
        lo,hi=offsets[i:i+2];np.maximum.at(expected,targets[int(lo):int(hi)],scores[i]*grades[int(lo):int(hi)])
    scatter_alias_max(offsets,targets,grades,scores,out,chunk=chunk)
    assert np.array_equal(out,expected)


def test_authority_corruption_and_output_cap_stop(tmp_path):
    old=VectorCoupledBlocks([site(0,names=(Proof('ALPHA'),))]);new=fixture(old,tmp_path)
    for cap in (0,101,True):
        with pytest.raises(ValueError):new.search(Query(name='ALPHA'),np.ones(30),limit=cap)
    with pytest.raises(TimeoutError):new.search(Query(name='ALPHA'),np.ones(30),max_seconds=1e-20)
    path=tmp_path/'index'/'site_keys.npy'
    with path.open('r+b') as f:f.seek(-1,2);f.write(b'\xff')
    with pytest.raises(ValueError,match='Authority drift'):MappedBlocks(tmp_path/'index')


def test_missing_source_authority_cannot_publish(tmp_path):
    old=VectorCoupledBlocks([site(0,names=(Proof('ALPHA'),))])
    with pytest.raises(ValueError,match='source'):export_fixture(old,tmp_path/'index',[])
