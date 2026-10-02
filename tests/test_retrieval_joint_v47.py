import numpy as np
import pytest

from src.xgb_matcher.retrieval_joint_v47 import collect,UncertifiedSearchBudget,search
from src.xgb_matcher.retrieval_joint_v46 import collect as old_collect
from tests.test_retrieval_joint_v46 import FixtureStream


def oracle(x,collector=collect):
    ids=[f'{100000000+i}00001' for i in range(len(x))]
    streams={str(j):FixtureStream([dict(siret=ids[i],siren=ids[i][:9],score=float(x[i,j]))
        for i in np.argsort(-x[:,j],kind='stable')]) for j in range(x.shape[1])}
    scores=dict(zip(ids,np.sum(x,axis=1)))
    rows,trace=collector(streams,lambda rs:{r['siret']:scores[r['siret']] for r in rs})
    expected=sorted(ids,key=lambda s:(-scores[s],s))[:100]
    return rows,trace,expected


def test_fairness_avoids_starving_spatial_source_on_lexical_plateau():
    x=np.zeros((10000,2)); x[:100]=[0,.7]; x[100:9000]=[1,.001]; x[9000:9100]=[1,.5]
    rows,trace,expected=oracle(x); old,oldtrace,_=oracle(x,old_collect)
    assert [r['siret'] for r in rows]==expected==[r['siret'] for r in old]
    assert sum(trace['source_calls'].values())<sum(oldtrace['source_calls'].values())/10
    assert trace['certified'] and trace['fixed_reservoir'] is None


@pytest.mark.parametrize('seed',[47,48,49,50])
def test_same_exact_metric_as_exhaustive_and_old_plan(seed):
    x=np.random.default_rng(seed).uniform(0,10,(2000,2))
    rows,trace,expected=oracle(x); old,_,_=oracle(x,old_collect)
    assert [r['siret'] for r in rows]==expected==[r['siret'] for r in old]
    assert rows==old and trace['certified']


def test_budget_preserves_trace_and_does_not_emit_uncertified_top100():
    stream=FixtureStream([dict(siret='10000000000001',siren='100000000',score=1.)])
    with pytest.raises(UncertifiedSearchBudget) as caught:
        collect({'fixture':stream},lambda rs:{r['siret']:1. for r in rs},max_seconds=1e-100)
    assert caught.value.trace['visited']==0
