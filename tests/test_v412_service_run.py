from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.xgb_matcher.v412_service_bundle import (
    load_frozen_v412_service_bundle,
)
from src.xgb_matcher.v412_service_run import collect_persistent_run
from src.xgb_matcher.v412_service_worker import PersistentV412Worker


REFERENCE = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/references/"
    "v4_12_service_parity/b4b7fef24c5e7036"
)


def test_collector_excludes_warmup_and_reproduces_one_real_query() -> None:
    queries = pd.read_parquet(REFERENCE / "queries.parquet").head(1)
    query_id = str(queries.iloc[0]["query_id"])
    with load_frozen_v412_service_bundle(include_evidence=True) as bundle:
        worker = PersistentV412Worker(bundle=bundle, mode="v412g")
        run = collect_persistent_run(
            worker=worker,
            queries=queries,
            model_load_count=1,
            store_load_count=1,
        )

    assert run.manifest["warmup_excluded"] is True
    assert run.manifest["warmup_query_id"] == query_id
    assert run.manifest["query_count"] == 1
    assert run.manifest["counters"]["query_count"] == 1
    assert run.manifest["counters"]["lookup_missing_count"] == 0
    assert run.manifest["counters"]["cache_rebuild_api_absent"] is True
    assert run.manifest["counters"]["cache_write_api_absent"] is True

    comparisons = (
        (run.candidates, "candidates_features.parquet"),
        (run.ranker, "ranker_reference.parquet"),
        (run.scenes, "scenes_reference.parquet"),
        (run.query_evidence, "query_evidence.parquet"),
        (run.candidate_evidence, "candidate_evidence.parquet"),
        (run.guard, "guard_reference.parquet"),
    )
    for observed, name in comparisons:
        expected = pd.read_parquet(REFERENCE / name)
        expected = expected.loc[
            expected["query_id"].astype(str).eq(query_id),
            observed.columns,
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected,
            check_dtype=False,
            check_exact=True,
        )
    expected_acceptor = (
        pd.read_parquet(REFERENCE / "acceptor_reference.parquet")
        .query("query_id == @query_id")
        .reset_index(drop=True)
    )
    assert run.acceptor.drop(columns=["score"]).equals(
        expected_acceptor.drop(columns=["score"])
    )
    assert np.allclose(
        run.acceptor["score"],
        expected_acceptor["score"],
        rtol=0.0,
        atol=1e-15,
    )


def test_collector_rejects_noncanonical_query_schema() -> None:
    queries = pd.read_parquet(REFERENCE / "queries.parquet").head(1)
    with load_frozen_v412_service_bundle(include_evidence=False) as bundle:
        worker = PersistentV412Worker(bundle=bundle, mode="v411")
        with pytest.raises(ValueError, match="query schema changed"):
            collect_persistent_run(
                worker=worker,
                queries=queries.assign(extra="forbidden"),
                model_load_count=1,
                store_load_count=1,
            )
