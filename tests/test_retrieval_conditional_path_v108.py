from collections import defaultdict
import itertools
import numpy as np
import pytest
from tests.test_retrieval_ranked_path_v105 import IntervalStream,Stream
from src.xgb_matcher import retrieval_conditional_path_v108 as kernel


def fixture(lexical,initialism,site,names,edges,weight=2.,duplicate_names=0):
    ids=[f'{100000000+i:09d}00001' for i in range(len(lexical))]
    outgoing=defaultdict(list); incoming=defaultdict(list)
    for owner,target,grade in edges:
        outgoing[owner].append((target,grade)); incoming[target].append((owner,grade))
    best=[max((names[o]*g for o,g in incoming[t]),default=0.) for t in range(len(ids))]
    def row(i,value): return dict(siret=ids[i],siren=ids[i][:9],score=float(value))
    components={k:IntervalStream([row(i,values[i]) for i in np.argsort(-np.asarray(values),kind='stable')])
                for k,values in [('lexical',lexical),('initialism_conditional',initialism)]}
    sites=Stream([row(i,site[i]) for i in np.argsort(-np.asarray(site),kind='stable')])
    name_rows=[dict(owner=int(i),score=float(names[i])) for i in np.argsort(-np.asarray(names),kind='stable')]
    name_rows=sorted(name_rows+name_rows[:duplicate_names],key=lambda r:-r['score'])
    name_stream=Stream(name_rows); calls=[]
    full={s:dict(component_scores={'lexical':float(lexical[i]),'initialism_conditional':float(initialism[i])},
                 path_name_score=float(best[i]),site_score=float(site[i])) for i,s in enumerate(ids)}
    def targets(row):
        for t,g in outgoing[row['owner']]:
            yield dict(siret=ids[t],siren=ids[t][:9],path_name_score=float(row['score']*g))
    def bound(rows):
        return {r['siret']:dict(components={k:dict(score=c.cache.get(r['siret']),upper=c.cache.get(r['siret'],c.upper))
            for k,c in components.items()},path_name_score=full[r['siret']]['path_name_score'],site_score=full[r['siret']]['site_score'])
            for r in rows}
    def complete(rows):
        calls.append(len(rows)); results={r['siret']:full[r['siret']] for r in rows}
        for k,c in components.items(): c.cache.update({s:v['component_scores'][k] for s,v in results.items()})
        return results
    scores={s:max(max(v['component_scores'].values()),
                  weight*v['path_name_score']*v['site_score']) for s,v in full.items()}
    expected=sorted((s for s in ids if scores[s]>0),key=lambda s:(-scores[s],s))
    return (components,name_stream,sites,targets,complete,bound),expected,scores,calls


@pytest.mark.parametrize('seed',range(10))
def test_exhaustive_sum_max_oracle_many_to_many_plateaus_and_duplicates(seed):
    rng=np.random.default_rng(seed); n=1500; m=350
    edges=[(int(o),int(t),float(g)) for o,t,g in zip(rng.integers(m,size=7000),rng.integers(n,size=7000),rng.integers(0,5,7000)/4.)]
    args,expected,scores,_=fixture(rng.integers(0,15,n)/10.,rng.integers(0,15,n)/10.,rng.integers(0,10,n)/10.,
        rng.integers(0,10,m)/10.,edges+edges[:1000],duplicate_names=50)
    for source in args[0].values(): source.rows=sorted(source.rows,key=lambda r:(-r['score'],-int(r['siret'])))
    rows,trace=kernel.collect(*args,path_weight=2.,page_size=7)
    assert [r['siret'] for r in rows]==expected[:100]
    assert all(r['score']==scores[r['siret']] for r in rows)
    assert trace['certified'] and trace['fixed_reservoir'] is None and trace['live_ranked_groups']==100


def test_graph_target_not_native_top100_is_still_retrieved():
    n=350; l=np.r_[np.ones(150),np.zeros(150),np.full(50,.75)]
    a=np.r_[np.zeros(150),np.ones(150),np.full(50,.75)]
    args,expected,scores,_=fixture(l,a,np.zeros(n),[],[])
    # Replace this native-only setup with a positive graph target whose
    # native score is below both individual top100 cuts.
    l[-1]=a[-1]=.01; site=np.zeros(n); site[-1]=1.
    args,expected,scores,_=fixture(l,a,site,[1.],[(0,n-1,1.)])
    rows,trace=kernel.collect(*args,path_weight=2.,page_size=11)
    assert [r['siret'] for r in rows]==expected[:100]
    assert rows[0]['siret']=='10000034900001' and rows[0]['score']==2.
    assert trace['certified']


def test_entire_fanout_beyond_8192_buffer_and_no_zero_imputation():
    n=20001
    args,expected,scores,_=fixture(np.full(n,.1),np.full(n,.1),np.linspace(.11,1.,n),[1.],[(0,i,1.) for i in range(n)])
    rows,trace=kernel.collect(*args,path_weight=2.,page_size=1)
    assert [r['siret'] for r in rows]==expected[:100]
    assert all(r['score']==scores[r['siret']] for r in rows)
    assert trace['path_endpoint_observations']==n and trace['targets_bound_checked']==n
    assert trace['component_interval_shortcuts']['exact_path']>10000 and trace['certified']


def test_proven_exhausted_component_zero_is_not_unknown_zero():
    n=300; args,expected,_,_=fixture(np.ones(n),np.zeros(n),np.zeros(n),[],[])
    args[0]['initialism_conditional']=IntervalStream([])
    # Callback refers to the same components object: zero bound is justified.
    rows,trace=kernel.collect(*args,path_weight=0.,page_size=100)
    assert [r['siret'] for r in rows]==expected[:100]
    assert trace['component_interval_shortcuts']['proven_zero_components']>0


def test_global_canonical_ties_on_split_sum_require_exhaustive_plateaus():
    n=350; args,expected,_,_=fixture(np.ones(n),np.ones(n),np.zeros(n),[],[])
    for source in args[0].values(): source.rows.reverse()
    rows,trace=kernel.collect(*args,path_weight=0.,page_size=11)
    assert [r['siret'] for r in rows]==expected[:100]
    assert trace['unseen_score_upper_bound']<rows[-1]['score']
    assert max(trace['source_rows']['lexical'],trace['source_rows']['initialism_conditional'])==n


@pytest.mark.parametrize('corruption',['coverage','underbound','known_bound','nan','factor','component_scope'])
def test_bad_bounds_fail_closed(corruption):
    args,_,_,_=fixture([.7,.6],[.5,.4],[1.,1.],[1.],[(0,0,1.),(0,1,1.)])
    bound=args[5]
    def broken(rows):
        result=bound(rows)
        if corruption=='coverage': return {}
        for r in result.values():
            if corruption=='underbound': r['components']['lexical']=dict(score=None,upper=.01)
            elif corruption=='known_bound': r['components']['lexical']['upper']=.01
            elif corruption=='nan': r['components']['initialism_conditional']['upper']=float('nan')
            elif corruption=='factor': r['path_name_score']=1.1
            elif corruption=='component_scope': r['components'].pop('initialism_conditional')
        return result
    with pytest.raises(ValueError,match='(coverage|metric|bound|scope|completion)'):
        kernel.collect(*args[:5],broken,path_weight=2.,page_size=1)


def test_full_path_timeout_never_returns_partial_top100(monkeypatch):
    n=3000; args,_,_,_=fixture(np.zeros(n),np.zeros(n),np.ones(n),[1.],[(0,i,1.) for i in range(n)])
    ticks=itertools.count(); monkeypatch.setattr(kernel.time,'monotonic',lambda:next(ticks)*.001)
    with pytest.raises(kernel.UncertifiedPathBudget) as caught:
        kernel.collect(*args,path_weight=2.,page_size=1,max_seconds=.2)
    assert not caught.value.trace['certified'] and caught.value.trace['frontiers']['names']==1.


def test_cap_and_component_scope_fail_closed():
    args,_,_,_=fixture([1.],[1.],[1.],[1.],[(0,0,1.)])
    with pytest.raises(ValueError,match='budget'): kernel.collect(*args,path_weight=2.,cap=101)
    with pytest.raises(ValueError,match='scope'): kernel.collect({'lexical':args[0]['lexical']},*args[1:],path_weight=2.)


def test_each_fact_grade_validated_against_its_own_name_not_page_max():
    n=300; args,_,_,_=fixture(np.zeros(n),np.zeros(n),np.ones(n),[1.,.5],[(0,0,1.),(1,1,1.)])
    original=args[3]
    def broken(row):
        for target in original(row):
            if row['owner']==1: target=dict(target,path_name_score=.75)
            yield target
    with pytest.raises(ValueError,match='own scoped name'):
        kernel.collect(args[0],args[1],args[2],broken,*args[4:],path_weight=2.,page_size=100)
