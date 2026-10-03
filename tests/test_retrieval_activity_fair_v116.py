import numpy as np
import pytest
from tests.test_retrieval_activity_initial_v115 import sources
from src.xgb_matcher.retrieval_activity_initial_v115 import collect as old_collect
from src.xgb_matcher.retrieval_activity_fair_v116 import collect


@pytest.mark.parametrize('cap,page',[(100,128),(100,7),(7,16),(1,1),(25,256)])
def test_same_entire_corpus_score_and_canonical_order_as_v115(cap,page):
    rng=np.random.default_rng(116);n=rng.random(450);g=rng.random(450);a=rng.random(450)
    rows,channels=sources(n,g);exact={r['siret']:float(n[i]*(.5+.5*a[i])+g[i]*(.2+.8*a[i])) for i,r in enumerate(rows)}
    result,trace=collect(channels,lambda rs:{r['siret']:exact[r['siret']] for r in rs},cap=cap,page_size=page)
    _,other=sources(n,g);prior,_=old_collect(other,lambda rs:{r['siret']:exact[r['siret']] for r in rs},cap=cap,page_size=page)
    assert result==prior
    assert [r['siret'] for r in result]==sorted(exact,key=lambda s:(-exact[s],s))[:cap]
    assert trace['certified'] and trace['fixed_reservoir'] is None


def test_small_unread_geometry_frontier_is_not_starved_by_large_native_plateau():
    n=np.full(20000,2.);g=np.r_[np.full(100,.5),np.full(19900,.001)]
    rows,channels=sources(n,g);exact={r['siret']:2.1 if i<100 else .1 for i,r in enumerate(rows)}
    result,trace=collect(channels,lambda rs:{r['siret']:exact[r['siret']] for r in rs})
    assert len(result)==100 and all(r['score']==2.1 for r in result)
    assert trace['source_calls']=={'native':1,'geometry':1}
    assert trace['source_rows']=={'native':128,'geometry':128}
    assert trace['unseen_score_upper_bound']<trace['score_cutoff']


def test_global_ties_and_winner_below_both_first100_are_preserved():
    for n,g in [(np.ones(325),np.zeros(325)),(np.r_[np.ones(200),np.zeros(200),.8],np.r_[np.zeros(200),np.ones(200),.8])]:
        rows,channels=sources(n,g);exact={r['siret']:float(n[i]+g[i]) for i,r in enumerate(rows)}
        result,trace=collect(channels,lambda rs:{r['siret']:exact[r['siret']] for r in rs},page_size=16)
        assert [r['siret'] for r in result]==sorted(exact,key=lambda s:(-exact[s],s))[:100]
        assert trace['certified']


@pytest.mark.parametrize('mode',['reliable','weak','missing'])
def test_real_tantivy_activity_xy_oracle_unchanged(tmp_path,monkeypatch,mode):
    # Reuse the independent full-corpus fixture, replacing only its called
    # search function inside this test. No historical source file is edited.
    from tests.test_retrieval_activity_initial_v115 import test_real_tantivy_and_xy_activity_search_matches_entire_fixture_corpus
    from src.xgb_matcher import retrieval_activity_initial_v115 as old
    from src.xgb_matcher.retrieval_activity_fair_v116 import search
    monkeypatch.setattr(old,'search',search)
    test_real_tantivy_and_xy_activity_search_matches_entire_fixture_corpus(tmp_path,mode)


def test_only_physical_revision_changes_recipe_not_budgets_or_score():
    import json
    from scripts.retrieval_v115_activity_initial import CONFIG as old
    from scripts.retrieval_v116_activity_fair import CONFIG as new
    before=json.loads(old.read_text());after=json.loads(new.read_text())
    assert after.pop('physical_revision')=='FAIR_SOURCE_READS_V116_SAME_V114_METRIC_CAP_AND_BUDGET'
    assert after==before
