import pytest
from src.xgb_matcher.retrieval_score_diagnostic_v219 import check_scores, summarize
from scripts.audit_retrieval_v219_scores import misses_for_batch


def test_all10_physical_batches():
    assert [p for b in range(3) for p in misses_for_batch(b)] == list(range(10))


def test_frontier_reproduction():
    check_scores({'x': 3.0000001}, [dict(siret='x',source_score=3.)])
    with pytest.raises(ValueError):
        check_scores({'x': 3.01}, [dict(siret='x',source_score=3.)])


def test_partial_diagnostics_not_admitted():
    with pytest.raises(ValueError): summarize([])


def test_complete_diagnostics_never_claim_recall():
    rows=[dict(query_id=str(i),position=i,score_diagnostic=dict(
        native_score_reproduction_pass=True,deficit=.5)) for i in range(10)]
    r=summarize(rows)
    assert r['summary']['strictly_below_frontier'] == 10
    assert not r['recall_measured'] and not r['outputs_promoted_to_features']
