from src.xgb_matcher.blocking import (
    build_tfidf_index,
    prefilter_candidates_tfidf_scored,
)


def test_v9_sparse_retrieval_preserves_scores_and_order():
    candidates = [
        {"denomination": "ALPHA CONSEIL"},
        {"denomination": "BETA SERVICES"},
        {"denomination": "ALPHA BETA"},
    ]
    vectorizer, matrix, _names = build_tfidf_index(candidates)
    hits = prefilter_candidates_tfidf_scored(
        "ALPHA CONSEIL",
        vectorizer,
        matrix,
        3,
    )
    assert hits[0][0] == 0
    assert hits[0][1] >= hits[1][1] > 0
