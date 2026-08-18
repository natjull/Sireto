from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_retrieval_ltr_admission import (
    TEST_AUTH_SCHEMA_VERSION,
    _consume_test_authorization_once,
    _validate_test_authorization,
    build_union_command,
    evaluate_command,
    file_sha256,
    parser,
    train_command,
)
from src.xgb_matcher.retrieval_ltr_admission import DEFAULT_CONFIG_PATH


def _candidate_frame(fold: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": [f"q-{fold}"],
            "siret": ["12345678900001"],
            "siren": ["123456789"],
            "fold": [fold],
            "gt_siret": ["12345678900001"],
            "crm_name": ["ALPHA"],
            "crm_address": ["1 RUE TEST"],
            "crm_number": ["1"],
            "crm_insee": ["75056"],
            "crm_postcode": ["75001"],
            "names": [["ALPHA"]],
            "addresses": [["1 RUE TEST"]],
            "insee": ["75056"],
            "postcode": ["75001"],
            "number": ["1"],
            "retrieval_source": ["name_exact+rne_name"],
            "retrieval_rank": [1],
            "retrieval_score": [1.0],
            "source_kind": ["HUMAN_CRM"],
        }
    )


def _development_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in (0, 2, 3, 4):
        truth = f"{123450000 + fold:09d}00001"
        for rank in (1, 2):
            candidate = truth if rank == 1 else f"{223450000 + fold:09d}00001"
            rows.append(
                {
                    **_candidate_frame(fold).iloc[0].to_dict(),
                    "query_id": f"q-{fold}",
                    "siret": candidate,
                    "siren": candidate[:9],
                    "gt_siret": truth,
                    "names": ["ALPHA" if rank == 1 else "BETA"],
                    "retrieval_source": "name_exact" if rank == 1 else "name_char",
                    "retrieval_rank": rank,
                    "retrieval_score": 1.0 / rank,
                    "retrieval_latency_ms": 10.0,
                    "v2_exact": True,
                    "v3_exact": True,
                }
            )
    return pd.DataFrame(rows)


def test_build_union_cli_manifest_proves_test_was_not_opened(tmp_path: Path) -> None:
    source = tmp_path / "candidates.parquet"
    output = tmp_path / "union"
    _candidate_frame().to_parquet(source, index=False)
    args = argparse.Namespace(
        candidates=source,
        output_dir=output,
        scope="train",
        config=DEFAULT_CONFIG_PATH,
    )

    build_union_command(args)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["test_opened"] is False
    assert manifest["opened_folds"] == [2]
    assert manifest["contains_synthetic"] is False
    assert manifest["contains_dense"] is False


def test_build_union_parser_has_no_test_scope() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "build-union",
                "--candidates",
                "input.parquet",
                "--output-dir",
                "out",
                "--scope",
                "test",
            ]
        )


def test_test_authorization_is_bound_to_source_config_and_model(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.parquet"
    _candidate_frame(fold=1).to_parquet(candidates, index=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "ranker.json").write_text("model", encoding="utf-8")
    (model_dir / "metadata.json").write_text("metadata", encoding="utf-8")
    authorization_path = tmp_path / "authorization.json"
    authorization = {
        "schema_version": TEST_AUTH_SCHEMA_VERSION,
        "purpose": "retrieval_ltr_fold1_one_shot",
        "status": "FROZEN_AUTHORIZED",
        "authorization_id": "fixture-one-shot",
        "candidates_sha256": file_sha256(candidates),
        "config_sha256": file_sha256(DEFAULT_CONFIG_PATH),
        "model_sha256": file_sha256(model_dir / "ranker.json"),
        "model_metadata_sha256": file_sha256(model_dir / "metadata.json"),
    }
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")

    loaded = _validate_test_authorization(
        path=authorization_path,
        candidates=candidates,
        config_path=DEFAULT_CONFIG_PATH,
        model_dir=model_dir,
    )

    assert loaded["authorization_id"] == "fixture-one-shot"
    authorization["candidates_sha256"] = "0" * 64
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(ValueError, match="candidates_sha256"):
        _validate_test_authorization(
            path=authorization_path,
            candidates=candidates,
            config_path=DEFAULT_CONFIG_PATH,
            model_dir=model_dir,
        )

    authorization["candidates_sha256"] = file_sha256(candidates)
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    marker = _consume_test_authorization_once(authorization_path, authorization)
    assert marker.exists()
    with pytest.raises(FileExistsError, match="already consumed"):
        _consume_test_authorization_once(authorization_path, authorization)


def test_fixture_cli_development_path_never_opens_test(tmp_path: Path) -> None:
    candidates = tmp_path / "development_candidates.parquet"
    union_dir = tmp_path / "union"
    model_dir = tmp_path / "model"
    evaluation_dir = tmp_path / "dev_evaluation"
    _development_frame().to_parquet(candidates, index=False)

    build_union_command(
        argparse.Namespace(
            candidates=candidates,
            output_dir=union_dir,
            scope="development",
            config=DEFAULT_CONFIG_PATH,
        )
    )
    train_command(
        argparse.Namespace(
            union=union_dir / "union.parquet",
            output_dir=model_dir,
            config=DEFAULT_CONFIG_PATH,
        )
    )
    evaluate_command(
        argparse.Namespace(
            split="dev",
            union=union_dir / "union.parquet",
            candidates=None,
            model_dir=model_dir,
            output_dir=evaluation_dir,
            config=DEFAULT_CONFIG_PATH,
            open_test=False,
            authorization=None,
        )
    )

    manifest = json.loads(
        (evaluation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (evaluation_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert manifest["test_opened"] is False
    assert manifest["opened_folds"] == [0]
    assert summary["qualification_views"]["historical"]["status"] == "AVAILABLE"
    assert summary["qualification_views"]["v2"]["status"] == "AVAILABLE"
    assert summary["qualification_views"]["v3"]["status"] == "AVAILABLE"
    assert summary["gates"]["latency_measured"]["passed"] is True
    assert summary["verdict"] == "GO"
