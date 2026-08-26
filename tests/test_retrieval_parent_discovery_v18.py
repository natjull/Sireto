from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.evaluate_retrieval_parent_discovery_v18 import _report
from src.xgb_matcher.retrieval_parent_discovery_v18 import (
    SIGNAL_SCHEMA,
    SiretChannelSpec,
    collapse_siret_hits_to_parent_rows,
    load_query_selection,
    sha256_file,
    validate_signal_rows,
)


ROOT = Path(__file__).resolve().parents[1]
GEN_CONFIG = ROOT / "config/retrieval_all_human_v18_parent_discovery_generation.json"
EVAL_CONFIG = ROOT / "config/retrieval_all_human_v18_parent_discovery_evaluation.json"


def _hit(siret: str, siren: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(record=SimpleNamespace(siret=siret, siren=siren), score=score)


def test_siret_hits_use_stored_official_siren_and_collapse_siblings() -> None:
    rows = collapse_siret_hits_to_parent_rows(
        query_id="q",
        fold=2,
        channel="site",
        hits=[
            _hit("99999999900001", "123456789", 9.0),
            _hit("12345678900002", "123456789", 8.0),
            _hit("11111111100003", "987654321", 7.0),
        ],
    )
    assert [row["siren"] for row in rows] == ["123456789", "987654321"]
    assert rows[0]["source_siret"] == "99999999900001"
    assert rows[0]["siren"] != rows[0]["source_siret"][:9]
    assert pa.Table.from_pylist(rows, schema=SIGNAL_SCHEMA).num_rows == 2


def test_signal_validation_rejects_duplicate_or_non_contiguous_rank() -> None:
    base = {
        "query_id": "q", "fold": 2, "channel": "c", "siren": "123456789",
        "source_siret": "12345678900001", "score": 1.0, "seed_rank": None,
        "relation_direction": "", "relation_type": "SIRET_NATIVE",
    }
    validate_signal_rows([{**base, "channel_rank": 1}])
    with pytest.raises(ValueError, match="duplicate"):
        validate_signal_rows([{**base, "channel_rank": 1}, {**base, "channel_rank": 2}])
    with pytest.raises(ValueError, match="contiguous"):
        validate_signal_rows([{**base, "channel_rank": 2}])


def test_query_surface_rejects_truth_columns(tmp_path: Path) -> None:
    good = tmp_path / "good.parquet"
    pq.write_table(pa.table({"query_id": ["q"], "fold": pa.array([2], type=pa.int8())}), good)
    assert load_query_selection(good, expected_count=1) == (("q", 2),)
    bad = tmp_path / "bad.parquet"
    pq.write_table(
        pa.table(
            {
                "query_id": ["q"],
                "fold": pa.array([2], type=pa.int8()),
                "human_ground_truth_siret": ["12345678900001"],
            }
        ),
        bad,
    )
    with pytest.raises(ValueError, match="only query_id and fold"):
        load_query_selection(bad)


def test_channel_contract_is_bounded_and_label_free() -> None:
    config = json.loads(GEN_CONFIG.read_text())
    specs = [SiretChannelSpec.from_mapping(row) for row in config["siret_channels"]]
    assert len(specs) == 10
    assert {spec.limit for spec in specs} == {512}
    serialized = json.dumps(config).lower()
    assert "human_ground_truth" not in serialized
    assert "cohort_labels" not in serialized
    assert config["recipe"]["parent_selection_cap"] is None
    assert config["recipe"]["site_expansion"] is False
    assert config["recipe"]["model"] is None


def _rows(primary_hits: int, *, recent_hits: int | None = None) -> list[dict]:
    recent_hits = primary_hits if recent_hits is None else recent_hits
    rows = []
    for variant, hits in (("V17_BASE", 329), ("V18_ALL", primary_hits)):
        for index in range(522):
            origin = "REAL_CRM_20260817" if index < 389 else "REAL_CRM_HISTORICAL"
            if variant == "V18_ALL" and origin == "REAL_CRM_20260817":
                hit = index < min(recent_hits, 389)
            elif variant == "V18_ALL":
                hit = index - 389 < max(0, hits - recent_hits)
            else:
                hit = index < hits
            rows.append(
                {
                    "query_id": f"m{index}", "fold": 2 + index % 3,
                    "cohort_kind": "BASE_TOP1750_MISS", "data_origin": origin,
                    "variant": variant, "raw_parent_hit": hit,
                }
            )
        for index in range(512):
            rows.append(
                {
                    "query_id": f"c{index}", "fold": 2 + index % 3,
                    "cohort_kind": "BASE_TOP1750_CONTROL",
                    "data_origin": "REAL_CRM_20260817" if index < 283 else "REAL_CRM_HISTORICAL",
                    "variant": variant, "raw_parent_hit": True,
                }
            )
    return rows


def test_truth_side_decision_thresholds_are_exact() -> None:
    config = json.loads(EVAL_CONFIG.read_text())
    report, decision = _report(
        _rows(405, recent_hits=300),
        primary_variant="V18_ALL",
        gates=config["gates"],
        channels=config["variants"]["V18_ALL"],
    )
    assert decision == "PASS_MASTER_99_5_RAW_FEASIBILITY"
    primary = report["variant_reports"]["V18_ALL"]
    assert primary["projected_master_exact_hits"] == 23454
    assert primary["projected_master_exact_recall"] >= 0.995
    _, below = _report(
        _rows(333, recent_hits=250),
        primary_variant="V18_ALL",
        gates=config["gates"],
        channels=config["variants"]["V18_ALL"],
    )
    assert below == "PIVOT_PARENT_SOURCE_DISCOVERY"


def test_evaluation_config_pins_generation_config() -> None:
    config = json.loads(EVAL_CONFIG.read_text())
    assert config["evaluation_authorities"]["generation_config"]["sha256"] == sha256_file(GEN_CONFIG)
    assert config["claims"]["folds_2_3_4_are_development"] is True
    assert config["claims"]["cross_crm_generalization"] is False
