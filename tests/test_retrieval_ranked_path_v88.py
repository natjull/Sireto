from collections import defaultdict
import itertools

import numpy as np
import pytest

from src.xgb_matcher import retrieval_ranked_path_v88 as kernel


class Stream:
    def __init__(self,rows):
        self.rows=rows; self.offset=0; self.exhausted=not rows

    def fetch(self,n):
        rows=self.rows[self.offset:self.offset+n]; self.offset+=len(rows)
        self.exhausted=self.offset==len(self.rows)
        return rows


def graph_fixture(native,site,name,edges,weight=2.,duplicate_names=0):
    ids=[f'{100000000+i:09d}00001' for i in range(len(native))]
    outgoing=defaultdict(list); incoming=defaultdict(list)
    for owner,target,grade in edges:
        outgoing[owner].append((target,grade)); incoming[target].append((owner,grade))
    name_max=[max((name[o]*g for o,g in incoming[t]),default=0.) for t in range(len(native))]
    full={s:dict(native_score=float(native[i]),site_score=float(site[i]),path_name_score=float(name_max[i]))
          for i,s in enumerate(ids)}
    def target(i,score): return dict(siret=ids[i],siren=ids[i][:9],score=float(score))
    native_stream=Stream([target(i,native[i]) for i in np.argsort(-np.asarray(native),kind='stable')])
    site_stream=Stream([target(i,site[i]) for i in np.argsort(-np.asarray(site),kind='stable')])
    name_rows=[dict(owner=int(i),score=float(name[i])) for i in np.argsort(-np.asarray(name),kind='stable')]
    name_rows=sorted(name_rows+name_rows[:duplicate_names],key=lambda r:-r['score'])
    name_stream=Stream(name_rows)
    def targets(row):
        for target_id,grade in outgoing[row['owner']]:
            yield dict(siret=ids[target_id],siren=ids[target_id][:9],path_name_score=float(row['score']*grade))
    def complete(rows): return {r['siret']:full[r['siret']] for r in rows}
    scores={s:max(v['native_score'],weight*v['path_name_score']*v['site_score']) for s,v in full.items()}
    expected=sorted((s for s in ids if scores[s]>0),key=lambda s:(-scores[s],s))
    return (native_stream,name_stream,site_stream,targets,complete),expected,scores


@pytest.mark.parametrize('seed',[0,1,2,3,4,5])
def test_many_to_many_exactly_matches_exhaustive_distinct_siret_oracle(seed):
    rng=np.random.default_rng(seed); n=1300; m=500
    native=rng.random(n); site=rng.random(n); names=rng.random(m)
    edges=[(int(o),int(t),float(g)) for o,t,g in zip(rng.integers(m,size=9000),rng.integers(n,size=9000),rng.random(9000))]
    # Duplicate evidence rows must not add score or consume output slots.
    args,expected,scores=graph_fixture(native,site,names,edges+edges[:1000],duplicate_names=100)
    rows,trace=kernel.collect(*args,path_weight=2.,page_size=7)
    assert [r['siret'] for r in rows]==expected[:100]
    assert all(r['score']==scores[r['siret']] for r in rows)
    assert trace['certified'] and trace['fixed_reservoir'] is None and trace['live_ranked_groups']==100
    assert trace['unseen_score_upper_bound']<rows[-1]['score']


def test_full_name_fanout_beyond_2000_is_not_a_fixed_site_pool():
    n=2501; args,expected,_=graph_fixture(np.zeros(n),np.linspace(.01,1.,n),[1.],[(0,i,1.) for i in range(n)])
    rows,trace=kernel.collect(*args,path_weight=2.,page_size=1)
    assert [r['siret'] for r in rows]==expected[:100]
    assert rows[0]['siret']=='10000250000001'
    assert trace['path_endpoint_observations']==2501 and trace['unique_targets_completed']==2501


def test_repeated_proofs_do_not_fill_top100_and_graph_only_target_survives():
    n=300; native=np.full(n,.01); native[-1]=0.
    site=np.full(n,.05); site[-1]=1.
    edges=[(0,0,1.)]*1200+[(1,n-1,1.)]
    args,expected,_=graph_fixture(native,site,[1.,.9],edges,duplicate_names=2)
    rows,trace=kernel.collect(*args,path_weight=2.,page_size=1)
    assert [r['siret'] for r in rows]==expected[:100] and rows[0]['siret']=='10000029900001'
    assert len({r['siret'] for r in rows})==100 and not trace['proof_rows_cut_before_projection']


def test_equal_scores_require_global_canonical_ties_not_only_visited_order():
    n=300; args,expected,_=graph_fixture(np.ones(n),np.zeros(n),[],[])
    # Reverse native plateau; stopping at >=frontier would be wrong.
    args[0].rows.reverse()
    rows,trace=kernel.collect(*args,path_weight=0.,page_size=11)
    assert [r['siret'] for r in rows]==expected[:100]
    assert trace['source_rows']['native']==300 and trace['unseen_score_upper_bound']==0


def test_no_name_proof_does_not_pad_empty_output_with_site_only_zero_scores():
    args,_,_=graph_fixture(np.zeros(200),np.ones(200),[],[])
    rows,trace=kernel.collect(*args,path_weight=2.,page_size=3)
    assert rows==[] and trace['certified']


def test_timeout_during_fanout_never_returns_a_partial_uncertified_top100(monkeypatch):
    args,_,_=graph_fixture(np.zeros(3000),np.ones(3000),[1.],[(0,i,1.) for i in range(3000)])
    ticks=itertools.count(); monkeypatch.setattr(kernel.time,'monotonic',lambda:next(ticks)*.001)
    with pytest.raises(kernel.UncertifiedPathBudget) as caught:
        kernel.collect(*args,path_weight=2.,page_size=1,max_seconds=.2)
    trace=caught.value.trace
    assert not trace['certified'] and 0<trace['path_endpoint_observations']<3000
    assert trace['frontiers']['names']==1. # Not advanced past a partial fanout.


def test_incomplete_random_access_fails_closed():
    args,_,_=graph_fixture([1.],[1.],[1.],[(0,0,1.)])
    args=(*args[:4],lambda rows:{})
    with pytest.raises(ValueError,match='coverage drift'): kernel.collect(*args,path_weight=2.)


def test_scored_site_or_native_different_from_sorted_source_fails_closed():
    args,_,_=graph_fixture([1.],[1.],[1.],[(0,0,1.)])
    complete=args[4]
    def bad(rows):
        result=complete(rows)
        return {k:dict(v,native_score=.5) for k,v in result.items()}
    args=(*args[:4],bad)
    with pytest.raises(ValueError,match='metric drift'): kernel.collect(*args,path_weight=2.,page_size=1)


def test_factor_above_declared_bound_and_nonmonotone_stream_fail_closed():
    args,_,_=graph_fixture([0.],[1.],[1.2],[(0,0,1.)])
    # Native-first random access now catches the same invalid factor before
    # its names stream is fetched. Both are mandatory fail-closed guards.
    with pytest.raises(ValueError,match='(order|bound) drift'): kernel.collect(*args,path_weight=2.)
    args,_,_=graph_fixture([1.,.5],[1.,1.],[],[])
    args[0].rows.reverse()
    with pytest.raises(ValueError,match='order drift'): kernel.collect(*args,path_weight=0.)


def test_cannot_emit_more_than_100():
    args,_,_=graph_fixture([1.],[1.],[1.],[(0,0,1.)])
    with pytest.raises(ValueError,match='budget'): kernel.collect(*args,path_weight=2.,cap=101)


def test_short_name_fanouts_complete_across_page_not_one_point_query_per_fact():
    args,expected,_=graph_fixture(np.zeros(300),np.ones(300),np.ones(100),[(i,100+i,1.) for i in range(100)])
    calls=[]; original=args[4]
    def counted(rows):
        calls.append(len(rows)); return original(rows)
    args=(Stream([]),*args[1:4],counted)
    found,trace=kernel.collect(*args,path_weight=2.,page_size=100)
    assert [r['siret'] for r in found]==expected[:100]
    assert calls==[100,100] and trace['path_endpoint_observations']==100


def test_page_batching_still_checks_each_grade_against_its_own_fact_not_page_max():
    # Keep sites unexhausted: with only two sites the valid site-side bound
    # already completes all targets and legitimately never reads the bad page.
    args,_,_=graph_fixture(np.zeros(300),np.ones(300),[1.,.5],[(0,0,1.),(1,1,1.)])
    original=args[3]
    def broken(row):
        for target in original(row):
            if row['owner']==1: target=dict(target,path_name_score=.75)
            yield target
    args=(Stream([]),args[1],args[2],broken,args[4])
    with pytest.raises(ValueError,match='exceeds its own scoped name'):
        kernel.collect(*args,path_weight=2.,page_size=100)
