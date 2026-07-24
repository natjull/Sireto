from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_v9_retrieval_experiment import (
    _artifact_contract,
    _budget_compliant,
    retrieval_config,
    summarize_mode,
    wilson_interval,
)


def test_retrieval_experiment_configs_keep_fixed_budget() -> None:
    for mode in (
        "sparse",
        "hybrid_local",
        "dense_only",
        "hybrid_global_siren",
    ):
        config = retrieval_config(mode, per_channel_k=500, budget=50)
        assert config.fusion_mode == "rrf"
        assert config.retrieval_budget == 50
        assert config.prefilter_k == 500


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(90, 100, confidence=0.95)
    assert lower < 0.9 < upper
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_artifact_contract_hashes_directory_manifest(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    contract = _artifact_contract(tmp_path)

    assert contract["path"] == str(tmp_path)
    assert contract["contract_type"] == "manifest_or_file"
    assert len(contract["contract_sha256"]) == 64


def test_budget_compliance_allows_a_new_channel_to_fill_short_local_pool() -> None:
    assert _budget_compliant(50, expected_minimum=18, budget=50)
    assert _budget_compliant(18, expected_minimum=18, budget=50)
    assert not _budget_compliant(17, expected_minimum=18, budget=50)
    assert not _budget_compliant(51, expected_minimum=18, budget=50)


def test_summary_counts_misses_budget_and_segments() -> None:
    raw = pd.DataFrame(
        {
            "hit_at_budget_siret": [True, False],
            "hit_at_budget_siren": [True, True],
            "hit_at_1_siret": [True, False],
            "hit_at_1_siren": [True, False],
            "ground_truth_in_base": [True, False],
            "budget_compliant": [True, False],
            "latency_ms": [10.0, 30.0],
            "loss_reason": ["", "NOT_IN_PARTITION"],
            "ground_truth_state": ["A", "F"],
            "location_match_type": ["insee", "postcode"],
            "missing_insee": [False, True],
            "mega_base_pool": [False, False],
            "multi_site_siren": [False, True],
        }
    )

    summary = summarize_mode(raw, budget=50)

    assert summary["recall_at_budget_siret"]["rate"] == pytest.approx(0.5)
    assert summary["recall_at_budget_siren"]["rate"] == pytest.approx(1.0)
    assert summary["hit_at_1_siret"]["rate"] == pytest.approx(0.5)
    assert summary["hit_at_1_siren"]["rate"] == pytest.approx(0.5)
    assert summary["budget_violations"] == 1
    assert summary["loss_reasons"] == {"NOT_IN_PARTITION": 1, "": 1}
    assert summary["segments"]["gt_closed"]["recall_at_budget_siret"]["rate"] == 0.0
