from __future__ import annotations

from pathlib import Path
import json
import logging

import pandas as pd
import pytest

from pipe_v6.exporter import export_results, compute_stats, save_stats_json, EXPORT_COLUMNS
from pipe_v6.pipeline import RESULT_COLUMNS
from pipe_v6.config import PipelineConfig


def _make_config(tmp_path: Path) -> PipelineConfig:
    """Minimal config helper reused across tests."""

    dummy = {
        "crm_path": tmp_path / "crm.csv",
        "output_path": tmp_path / "out.csv",
        "sqlite_path": tmp_path / "cache.sqlite",
        "log_path": tmp_path / "logs.log",
        "sirene_api_url": "https://api.insee.fr/entreprises/sirene/V3",
        "sirene_token": "token",
        "rne_api_url": "https://api.inpi.fr/api/rne/",
        "rne_token_url": "https://api.inpi.fr/api/sso/oauth2/token",
        "rne_client_id": "client",
        "rne_client_secret": "secret",
        "qwant_base_url": "https://api.qwant.com/api/search/web",
        "datagouv_api_url": "https://recherche-entreprises.api.gouv.fr",
        "model_name": "gpt-oss:20b",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 256,
        "ollama_base_url": "http://localhost:11434",
        "llm_timeout_sec": 120.0,
        "llm_connect_timeout_sec": 5.0,
        "llm_max_retries": 3,
        "llm_retry_backoff_base_sec": 1.0,
        "llm_max_concurrency": 2,
        "llm_log_prompts": False,
        "llm_log_responses": False,
        "llm_json_mode_default": False,
        "llm_seed": 42,
        "max_candidates_per_source": 10,
        "confidence_auto_match": 0.85,
        "confidence_review_min": 0.6,
        "category_filter_enabled": True,
        "category_filter_fallback": True,
        "max_candidates_llm_matcher": 15,
        "log_level": "INFO",
        "store_source_raw": True,
        "extra": {},
    }
    return PipelineConfig(**dummy)


def _write_sample_crm(path: Path) -> None:
    content = (
        "N° Service;Client final;Adresse;Commune;Code Postal;Code INSEE\n"
        "ID001;TEST COMPANY;12 RUE TEST;VILLEURBANNE;69100;69266\n"
        "ID002;OTHER CORP;5 AV EXAMPLE;LYON;69003;69123\n"
        "ID003;THIRD SA;8 BD DEMO;LYON;69007;69387\n"
    )
    path.write_text(content, encoding="utf-8")


def test_export_results_merges_crm_and_results_csv(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_sample_crm(config.crm_path)

    df_results = pd.DataFrame(
        [
            {
                "crm_id": "ID001",
                "status": "MATCH",
                "chosen_siret": "12345678901234",
                "confidence": 0.9234,
                "reason": "Good match",
                "crm_category": "PRIVE",
                "normalized_name": "TEST COMPANY",
                "normalized_address": "12 RUE TEST",
                "candidate_count_total": 5,
                "candidate_count_used": 3,
                "sources": ["DATAGOUV", "RNE"],
            },
            {
                "crm_id": "ID002",
                "status": "NO_MATCH",
                "chosen_siret": None,
                "confidence": 0.1,
                "reason": "Low confidence",
                "crm_category": "PUBLIC",
                "normalized_name": "OTHER CORP",
                "normalized_address": "5 AV EXAMPLE",
                "candidate_count_total": 2,
                "candidate_count_used": 0,
                "sources": [],
            },
        ]
    )

    output_path = tmp_path / "results.csv"
    export_results(df_results, config, output_path)

    assert output_path.exists()
    df_export = pd.read_csv(output_path, sep=";", encoding="utf-8")

    assert list(df_export.columns) == EXPORT_COLUMNS
    # All CRM rows preserved (left join)
    assert len(df_export) == 3

    # Sources normalized (sorted and pipe-joined)
    assert df_export.loc[0, "sources"] == "DATAGOUV|RNE"
    assert df_export.loc[1, "sources"] == ""

    # Confidence rounded to 4 decimals
    assert df_export.loc[0, "confidence"] == pytest.approx(0.9234, rel=0, abs=1e-4)

    # Rows without pipeline result keep NaN
    assert pd.isna(df_export.loc[2, "status"])


def test_export_results_empty_results_keeps_crm_rows(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_sample_crm(config.crm_path)

    df_results = pd.DataFrame(columns=RESULT_COLUMNS)
    output_path = tmp_path / "empty.csv"

    export_results(df_results, config, output_path)

    df_export = pd.read_csv(output_path, sep=";", encoding="utf-8")
    assert len(df_export) == 3  # CRM rows kept
    assert list(df_export.columns) == EXPORT_COLUMNS
    assert df_export["sources"].fillna("").tolist() == ["", "", ""]


def test_export_results_warns_on_unknown_crm_ids(tmp_path: Path, caplog) -> None:
    config = _make_config(tmp_path)
    _write_sample_crm(config.crm_path)

    df_results = pd.DataFrame(
        [
            {k: None for k in RESULT_COLUMNS}
        ]
    )
    df_results.loc[0, "crm_id"] = "UNKNOWN"

    output_path = tmp_path / "warn.csv"

    with caplog.at_level(logging.WARNING, logger="pipe_v6.exporter"):
        export_results(df_results, config, output_path)

    assert any("crm_id mismatch" in rec.message for rec in caplog.records)


def test_export_results_json_export(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_sample_crm(config.crm_path)

    df_results = pd.DataFrame(
        [
            {
                "crm_id": "ID001",
                "status": "MATCH",
                "chosen_siret": "123",
                "confidence": 0.5,
                "reason": "",
                "crm_category": "PRIVE",
                "normalized_name": "A",
                "normalized_address": "ADDR",
                "candidate_count_total": 1,
                "candidate_count_used": 1,
                "sources": ["RNE"],
            }
        ]
    )

    out_json = tmp_path / "out.json"
    export_results(df_results, config, out_json)

    loaded = pd.read_json(out_json)
    assert list(loaded.columns) == EXPORT_COLUMNS
    assert len(loaded) == 3


def test_compute_stats_structure_and_values() -> None:
    df_results = pd.DataFrame(
        [
            {
                "status": "MATCH",
                "crm_category": "PUBLIC",
                "confidence": 0.92,
                "candidate_count_total": 8,
                "candidate_count_used": 5,
            },
            {
                "status": "MATCH",
                "crm_category": "PRIVE",
                "confidence": 0.88,
                "candidate_count_total": 12,
                "candidate_count_used": 10,
            },
            {
                "status": "NO_MATCH",
                "crm_category": "EQUIPEMENT_URBAIN",
                "confidence": 0.3,
                "candidate_count_total": 2,
                "candidate_count_used": 0,
            },
        ]
    )

    stats = compute_stats(df_results)

    assert stats["total_rows"] == 3
    assert stats["status"]["counts"]["MATCH"] == 2
    assert stats["status"]["counts"]["NO_MATCH"] == 1
    assert stats["status"]["percentages"]["MATCH"] == pytest.approx(0.6667, rel=0, abs=1e-3)

    assert stats["crm_category"]["counts"]["PUBLIC"] == 1
    assert stats["crm_category"]["counts"]["PRIVE"] == 1
    assert stats["crm_category"]["counts"]["EQUIPEMENT_URBAIN"] == 1

    assert stats["candidates"]["avg_total"] == pytest.approx(7.3333, rel=0, abs=1e-3)
    assert stats["candidates"]["avg_used"] == pytest.approx(5.0, rel=0, abs=1e-3)

    assert stats["confidence"]["mean"] == pytest.approx(0.9, rel=0, abs=1e-3)
    assert stats["confidence"]["min"] == pytest.approx(0.88, rel=0, abs=1e-3)
    assert stats["confidence"]["max"] == pytest.approx(0.92, rel=0, abs=1e-3)


def test_compute_stats_empty_dataframe() -> None:
    df_empty = pd.DataFrame(columns=RESULT_COLUMNS)
    stats = compute_stats(df_empty)

    assert stats["total_rows"] == 0
    assert all(value == 0.0 for value in stats["status"]["percentages"].values())
    assert stats["candidates"]["avg_total"] == 0.0
    assert stats["confidence"]["mean"] == 0.0


def test_save_stats_json(tmp_path: Path) -> None:
    stats = {
        "total_rows": 42,
        "status": {"counts": {"MATCH": 30}},
    }

    target = tmp_path / "stats.json"
    save_stats_json(stats, target)

    assert target.exists()
    with target.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)

    assert loaded == stats
