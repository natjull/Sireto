import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_retrieval_all_human_v12_label_ledger import (
    ONLY_EXCLUSION,
    SCHEMA_VERSION,
    build,
    file_sha256,
)
from scripts.run_retrieval_hierarchical_admission_v12 import (
    _load_sealed_query_labels,
)


ROOT = Path(__file__).resolve().parents[1]


def _raw_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    rows = pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2", "q3", "q3-missing", "q4"],
            "oof_fold": [0, 1, 2, 3, 3, 4],
            "ground_truth_siret": [
                "00000000000000",
                "10000000000001",
                "11111111100001",
                "invalid-current",
                "also-invalid",
                "44444444400004",
            ],
            "historical_ground_truth_siret": [
                "",
                "",
                "99999999900009",
                "33333333300003",
                "",
                "",
            ],
            "acceptable_sirets_operational": [
                "[]",
                "[]",
                '["11111111100019"]',
                "[]",
                "[]",
                '["44444444400004"]',
            ],
            "ground_truth_state": ["A", "A", "A", "F", "", "A"],
            "label_kind": [
                "MATCH_EXACT",
                "MATCH_EXACT",
                "MATCH_EXACT",
                "AMBIGUOUS",
                "MATCH_EXACT",
                "UNRESOLVED",
            ],
            "qualification_v2": ["V2"] * 6,
            "qualification_v3": ["V3"] * 6,
            "source_kind": ["REAL_HUMAN_CRM"] * 6,
            "data_origin": ["CRM"] * 6,
            "split": ["diagnostic", "test", "train", "train", "train", "train"],
            "unseen_siren": [False, False, False, True, False, False],
            "new_site_known_siren": [False, False, True, False, False, True],
            "mega_base_pool": [0, 0, 0, 1, 1, 0],
            "pool_size": [10, 10, 25, 5000, 5000, 40],
            "insee": ["75101", "75101", "13055", "69123", "69123", "31555"],
            "is_synthetic": [False] * 6,
        }
    )
    extra_rows = []
    for index in range(509):
        fold = (2, 3, 4)[index % 3]
        siren = f"{500_000_000 + index:09d}"
        siret = f"{siren}00001"
        extra_rows.append(
            {
                "query_id": f"extra-{index:04d}",
                "oof_fold": fold,
                "ground_truth_siret": siret,
                "historical_ground_truth_siret": "",
                "acceptable_sirets_operational": json.dumps([siret]),
                "ground_truth_state": "A",
                "label_kind": "MATCH_EXACT",
                "qualification_v2": "V2",
                "qualification_v3": "V3",
                "source_kind": "REAL_HUMAN_CRM",
                "data_origin": "CRM",
                "split": "train",
                "unseen_siren": False,
                "new_site_known_siren": False,
                "mega_base_pool": 0,
                "pool_size": 10,
                "insee": f"{fold:02d}000",
                "is_synthetic": False,
            }
        )
    rows = pd.concat([rows, pd.DataFrame(extra_rows)], ignore_index=True)
    benchmark = source / "benchmark.parquet"
    development = source / "development.parquet"
    test_locked = source / "test_locked.parquet"
    rows.to_parquet(benchmark, index=False)
    rows[rows["oof_fold"].isin([0, 2, 3, 4])].to_parquet(development, index=False)
    rows[rows["oof_fold"].eq(1)].to_parquet(test_locked, index=False)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "sireto-crm-gt-v2-retrieval-input-3",
                "build_identity": {
                    "population_scope": "all_human",
                    "positive_injection": False,
                    "qualification_uses_retrieval_or_model_scores": False,
                    "builder_sha256": "1" * 64,
                    "population_manifest_sha256": "2" * 64,
                    "retrieval_contract_sha256": "3" * 64,
                    "partitions_sha256": "4" * 64,
                },
                "positive_injection": False,
                "qualification_uses_retrieval_or_model_scores": False,
                "development_opened_folds": [0, 2, 3, 4],
                "test_locked": True,
                "query_count": len(rows),
                "split_counts": {
                    "diagnostic": 1,
                    "crm_prospective_test": 1,
                    "crm_train": len(rows) - 2,
                },
                "outputs": {
                    "benchmark.parquet": file_sha256(benchmark),
                    "development.parquet": file_sha256(development),
                    "test_locked.parquet": file_sha256(test_locked),
                },
            }
        ),
        encoding="utf-8",
    )
    return source


def _args(tmp_path: Path, source: Path) -> Namespace:
    return Namespace(
        source=source,
        contract=ROOT / "docs/retrieval_all_human_v12_contract.md",
        config=ROOT / "config/retrieval_all_human_v12.json",
        output_root=tmp_path / "output",
    )


def _v11_source(tmp_path: Path) -> Path:
    source = tmp_path / "v11"
    source.mkdir()
    exact = pd.DataFrame(
        {
            "query_id": ["q0", "q2", "q3", "q4"],
            "oof_fold": [0, 2, 3, 4],
            "human_ground_truth_siret": [
                "00000000000000",
                "22222222200002",
                "33333333300003",
                "44444444400004",
            ],
            "human_ground_truth_source": ["CURRENT"] * 4,
            "ground_truth_state": ["A"] * 4,
            "label_kind": ["MATCH_EXACT", "AMBIGUOUS", "UNRESOLVED", "MATCH_EXACT"],
            "data_origin": ["CRM"] * 4,
        }
    )
    operational = pd.DataFrame(
        {
            "query_id": ["q0", "q2", "q3", "q4"],
            "oof_fold": [0, 2, 3, 4],
            "acceptable_sirets_operational_json": [
                '["00000000000000"]',
                '["22222222200002"]',
                '["33333333300003"]',
                '["44444444400004"]',
            ],
        }
    )
    extra_exact = []
    extra_operational = []
    for index in range(509):
        fold = (2, 3, 4)[index % 3]
        siren = f"{600_000_000 + index:09d}"
        siret = f"{siren}00001"
        query_id = f"v11-extra-{index:04d}"
        extra_exact.append(
            {
                "query_id": query_id,
                "oof_fold": fold,
                "human_ground_truth_siret": siret,
                "human_ground_truth_source": "CURRENT",
                "ground_truth_state": "A",
                "label_kind": "MATCH_EXACT",
                "data_origin": "CRM",
            }
        )
        extra_operational.append(
            {
                "query_id": query_id,
                "oof_fold": fold,
                "acceptable_sirets_operational_json": json.dumps([siret]),
            }
        )
    exact = pd.concat([exact, pd.DataFrame(extra_exact)], ignore_index=True)
    operational = pd.concat(
        [operational, pd.DataFrame(extra_operational)], ignore_index=True
    )
    excluded = pd.DataFrame(
        {
            "query_id": ["q0-missing", "q3-missing"],
            "oof_fold": [0, 3],
            "missing_reason": [ONLY_EXCLUSION, ONLY_EXCLUSION],
        }
    )
    development = source / "development.parquet"
    operational_dev = source / "operational_labels_development.parquet"
    missing_dev = source / "missing_human_labels_development.parquet"
    exact.to_parquet(development, index=False)
    operational.to_parquet(operational_dev, index=False)
    excluded.to_parquet(missing_dev, index=False)
    usable_by_fold = {"0": 1, "1": 0, "2": 171, "3": 171, "4": 170}
    missing_by_fold = {"0": 1, "1": 0, "2": 0, "3": 1, "4": 0}

    def metadata(path: Path, rows: int, folds: list[int], locked: bool) -> dict:
        return {
            "path": path.name,
            "sha256": file_sha256(path) if path.is_file() else "a" * 64,
            "rows": rows,
            "folds": folds,
            "locked": locked,
        }

    outputs = {
        "development": metadata(development, 513, [0, 2, 3, 4], False),
        "operational_labels_development": metadata(
            operational_dev, 513, [0, 2, 3, 4], False
        ),
        "missing_human_labels_development": metadata(
            missing_dev, 2, [0, 2, 3, 4], False
        ),
        "test_locked": metadata(source / "test_locked.parquet", 0, [1], True),
        "operational_labels_test_locked": metadata(
            source / "operational_labels_test_locked.parquet", 0, [1], True
        ),
        "missing_human_labels_test_locked": metadata(
            source / "missing_human_labels_test_locked.parquet", 0, [1], True
        ),
    }
    manifest = {
        "schema_version": "sireto-retrieval-all-human-v11-benchmark-1",
        "build_identity": {
            "only_exclusion": ONLY_EXCLUSION,
            "positive_injection": False,
            "development_folds": [0, 2, 3, 4],
            "test_locked_fold": 1,
            "label_precedence": [
                "ground_truth_siret",
                "historical_ground_truth_siret",
            ],
        },
        "population": {
            "source_rows": 515,
            "usable_human_labels": 513,
            "excluded_missing_human_labels": 2,
            "usable_by_fold": usable_by_fold,
            "missing_by_fold": missing_by_fold,
        },
        "fold_lock": {
            "development_folds": [0, 2, 3, 4],
            "test_fold": 1,
            "fold1_locked": True,
            "fold1_opened": False,
            "development_contains_fold1": False,
            "test_consumption_requires_dedicated_authorization": True,
        },
        "metrics": {"primary": "EXACT_SIRET_ALL_USABLE_HUMAN_LABELS"},
        "integrity": {
            "positive_injection": False,
            "retrieval_or_model_outputs_read": False,
            "synthetic_rows": 0,
            "operational_labels_physically_separate": True,
            "only_missing_labels_excluded": True,
        },
        "outputs": outputs,
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_current_precedes_historical_and_only_missing_is_excluded(tmp_path):
    destination = build(_args(tmp_path, _raw_source(tmp_path)))
    ledger = pd.read_parquet(destination / "label_ledger.parquet").set_index(
        "query_id"
    )
    excluded = pd.read_parquet(
        destination / "excluded_missing_human_labels.parquet"
    ).set_index("query_id")
    manifest = json.loads((destination / "manifest.json").read_text())

    assert {"q2", "q3", "q4"}.issubset(set(ledger.index))
    assert set(ledger["fold"]) == {2, 3, 4}
    assert ledger.at["q2", "human_ground_truth_siret"] == "11111111100001"
    assert ledger.at["q2", "human_ground_truth_source"] == "CURRENT"
    assert ledger.at["q3", "human_ground_truth_siret"] == "33333333300003"
    assert ledger.at["q3", "human_ground_truth_source"] == "HISTORICAL_FALLBACK"
    assert ledger.at["q3", "label_kind"] == "AMBIGUOUS"
    assert ledger.at["q4", "label_kind"] == "UNRESOLVED"
    assert bool(ledger.at["q3", "unseen_siren"])
    assert bool(ledger.at["q4", "new_site_known_siren"])
    assert int(ledger.at["q3", "mega_base_pool"]) == 1
    assert ledger.at["q3", "insee"] == "69123"
    assert list(excluded.index) == ["q3-missing"]
    assert excluded.at["q3-missing", "missing_reason"] == ONLY_EXCLUSION
    assert bool(excluded.at["q3-missing", "current_siret_present_but_invalid"])

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["population"] == "ALL_USABLE_HUMAN_LABELS"
    assert manifest["only_exclusion"] == ONLY_EXCLUSION
    assert manifest["folds"] == [2, 3, 4]
    assert manifest["query_count"] == 512
    assert manifest["excluded_count"] == 1
    assert manifest["counts_by_fold"] == {"2": 171, "3": 171, "4": 170}
    assert manifest["excluded_counts_by_fold"] == {"2": 0, "3": 1, "4": 0}
    assert manifest["fold0_opened"] is False
    assert manifest["fold1_opened"] is False
    assert manifest["official_catalog_presence_available"] is False
    assert ledger["official_catalog_present"].isna().all()
    runtime_sample = pd.read_parquet(
        destination / manifest["runtime_query_sample_path"]
    )
    assert len(runtime_sample) == 512
    assert list(runtime_sample.columns) == ["query_id", "fold"]
    assert runtime_sample["query_id"].is_unique
    assert manifest["runtime_query_sample_rule"] == "SHA256_QUERY_ID_ASC_FIRST_512"


def test_consumer_rejects_minimal_self_declared_manifest(tmp_path):
    build_dir = tmp_path / ("a" * 20)
    build_dir.mkdir()
    ledger_path = build_dir / "label_ledger.parquet"
    pd.DataFrame(
        {
            "query_id": ["q2"],
            "fold": [2],
            "human_ground_truth_siret": ["11111111100001"],
        }
    ).to_parquet(ledger_path, index=False)
    (build_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "population": "ALL_USABLE_HUMAN_LABELS",
                "only_exclusion": ONLY_EXCLUSION,
                "label_ledger_path": ledger_path.name,
                "label_ledger_sha256": file_sha256(ledger_path),
                "query_count": 1,
                "counts_by_fold": {"2": 1, "3": 0, "4": 0},
                "folds": [2, 3, 4],
                "positive_injection": False,
                "retrieval_outputs_read": False,
                "model_outputs_read": False,
                "fold0_opened": False,
                "fold1_opened": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="build_identity"):
        _load_sealed_query_labels(
            ledger_path,
            config_path=ROOT / "config/retrieval_all_human_v12.json",
        )


def test_consumer_rejects_subpopulation_artifact(tmp_path):
    destination = build(_args(tmp_path, _raw_source(tmp_path)))
    ledger_path = destination / "label_ledger.parquet"
    labels, _ = _load_sealed_query_labels(
        ledger_path,
        config_path=ROOT / "config/retrieval_all_human_v12.json",
    )
    assert len(labels) == 512
    subset = pd.read_parquet(ledger_path).iloc[1:].copy()
    subset.to_parquet(ledger_path, index=False)
    with pytest.raises(ValueError, match="metadata/hash/rows mismatch"):
        _load_sealed_query_labels(
            ledger_path,
            config_path=ROOT / "config/retrieval_all_human_v12.json",
        )


def test_consumer_rejects_reordered_runtime_query_sample(tmp_path):
    destination = build(_args(tmp_path, _raw_source(tmp_path)))
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    sample_path = destination / manifest["runtime_query_sample_path"]
    sample = pd.read_parquet(sample_path)
    sample.iloc[[0, 1]] = sample.iloc[[1, 0]].to_numpy()
    sample.to_parquet(sample_path, index=False, compression="zstd")
    observed_hash = file_sha256(sample_path)
    manifest["runtime_query_sample_sha256"] = observed_hash
    metadata = manifest["artifacts"]["runtime_query_sample"]
    metadata["sha256"] = observed_hash
    metadata["size_bytes"] = sample_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="deterministic selection"):
        _load_sealed_query_labels(
            destination / "label_ledger.parquet",
            config_path=ROOT / "config/retrieval_all_human_v12.json",
        )


def test_sidecars_hashes_and_zero_candidate_queries_are_ledger_driven(tmp_path):
    destination = build(_args(tmp_path, _raw_source(tmp_path)))
    manifest = json.loads((destination / "manifest.json").read_text())
    ledger = pd.read_parquet(destination / manifest["label_ledger_path"])
    operational = pd.read_parquet(destination / manifest["operational_sidecar_path"])

    assert manifest["positive_injection"] is False
    assert manifest["retrieval_outputs_read"] is False
    assert manifest["model_outputs_read"] is False
    assert manifest["candidate_group_dependency"] == "NONE"
    assert manifest["queries_with_zero_candidate_groups_remain_in_denominator"] is True
    assert len(ledger) == 512  # no candidate file was supplied at all
    assert operational["query_id"].is_unique
    assert set(operational["query_id"]) == set(ledger["query_id"])
    q2_operational = json.loads(
        operational.set_index("query_id").at[
            "q2", "acceptable_sirets_operational_json"
        ]
    )
    assert q2_operational == ["11111111100001"]
    assert (
        manifest["operational_equivalence_mode"]
        == "EXACT_ONLY_PENDING_SEALED_SAME_SITE_EVIDENCE"
    )
    assert manifest["unverified_operational_alternative_count"] == 1
    assert manifest["unverified_operational_alternative_count_by_fold"] == {
        "2": 1,
        "3": 0,
        "4": 0,
    }

    assert file_sha256(destination / manifest["label_ledger_path"]) == manifest[
        "label_ledger_sha256"
    ]
    assert file_sha256(destination / manifest["operational_sidecar_path"]) == manifest[
        "operational_sidecar_sha256"
    ]
    assert file_sha256(destination / manifest["excluded_sidecar_path"]) == manifest[
        "excluded_sidecar_sha256"
    ]
    for metadata in manifest["artifacts"].values():
        assert file_sha256(destination / metadata["path"]) == metadata["sha256"]
        assert metadata["schema"]
    assert manifest["ground_truth_siren_disjoint_across_folds"] is True
    assert manifest["siren_disjoint_proof"]["pairwise_overlap_counts"] == {
        "2:3": 0,
        "2:4": 0,
        "3:4": 0,
    }
    assert build(_args(tmp_path, tmp_path / "source")) == destination


def test_ground_truth_siren_overlap_across_folds_fails_closed(tmp_path):
    source = _raw_source(tmp_path)
    path = source / "development.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["query_id"].eq("q4"), "ground_truth_siret"] = "11111111199999"
    frame.loc[frame["query_id"].eq("q4"), "acceptable_sirets_operational"] = "[]"
    frame.to_parquet(path, index=False)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["outputs"]["development.parquet"] = file_sha256(path)
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="SIREN overlap"):
        build(_args(tmp_path, source))


def test_source_hash_and_retrieval_columns_fail_closed(tmp_path):
    source = _raw_source(tmp_path)
    path = source / "development.parquet"
    frame = pd.read_parquet(path)
    frame["union_rank"] = 1
    frame.to_parquet(path, index=False)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["outputs"]["development.parquet"] = file_sha256(path)
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Retrieval/model output"):
        build(_args(tmp_path, source))

    manifest["outputs"]["development.parquet"] = "0" * 64
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build(_args(tmp_path, source))


def test_raw_source_without_accepted_population_lineage_fails_closed(tmp_path):
    source = _raw_source(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "arbitrary-hashed-subset"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="ALL_USABLE population lineage"):
        build(_args(tmp_path, source))


def test_raw_operational_cross_siren_is_audited_but_not_trusted(tmp_path):
    source = _raw_source(tmp_path)
    path = source / "development.parquet"
    frame = pd.read_parquet(path)
    frame.loc[
        frame["query_id"].eq("q2"), "acceptable_sirets_operational"
    ] = '["88888888800008"]'
    frame.to_parquet(path, index=False)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["development.parquet"] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination = build(_args(tmp_path, source))
    operational = pd.read_parquet(
        destination / "operational_labels.parquet"
    ).set_index("query_id")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert json.loads(
        operational.at["q2", "acceptable_sirets_operational_json"]
    ) == ["11111111100001"]
    assert manifest["unverified_operational_alternative_count"] == 1


def test_sealed_v11_development_artifacts_are_reused_without_test_artifact(tmp_path):
    source = _v11_source(tmp_path)

    destination = build(_args(tmp_path, source))
    ledger = pd.read_parquet(destination / "label_ledger.parquet")
    manifest = json.loads((destination / "manifest.json").read_text())
    assert {"q2", "q3", "q4"}.issubset(set(ledger["query_id"]))
    assert manifest["source_scan"]["kind"] == "SEALED_V11_DEVELOPMENT"
    assert manifest["source_scan"]["locked_fold1_artifacts_opened"] is False
    assert not (source / "test_locked.parquet").exists()


def test_v11_missing_population_or_integrity_claim_fails_closed(tmp_path):
    source = _v11_source(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["integrity"]["only_missing_labels_excluded"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity claims"):
        build(_args(tmp_path, source))


def test_v11_operational_set_is_never_trusted_without_same_site_proof(tmp_path):
    source = _v11_source(tmp_path)
    path = source / "operational_labels_development.parquet"
    frame = pd.read_parquet(path)
    frame.loc[
        frame["query_id"].eq("q2"), "acceptable_sirets_operational_json"
    ] = "[]"
    frame.to_parquet(path, index=False)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["operational_labels_development"]["sha256"] = file_sha256(
        path
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination = build(_args(tmp_path, source))
    operational = pd.read_parquet(
        destination / "operational_labels.parquet"
    ).set_index("query_id")
    assert json.loads(
        operational.at["q2", "acceptable_sirets_operational_json"]
    ) == ["22222222200002"]


def test_v11_hashed_development_subset_fails_population_accounting(tmp_path):
    source = _v11_source(tmp_path)
    path = source / "development.parquet"
    frame = pd.read_parquet(path)
    frame[frame["query_id"].ne("q0")].to_parquet(path, index=False)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["development"]["sha256"] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="subset"):
        build(_args(tmp_path, source))


def test_sealed_official_establishments_attach_exact_catalog_presence(tmp_path):
    source = _raw_source(tmp_path)
    official = tmp_path / "official"
    official.mkdir()
    establishments = official / "establishments.parquet"
    pd.DataFrame(
        {
            "siret": [
                "11111111100001",
                "44444444400004",
                "77777777700007",
            ],
            "administrative_state": ["A", "F", "A"],
        }
    ).to_parquet(establishments, index=False)
    official_manifest = official / "manifest.json"
    official_manifest.write_text(
        json.dumps(
            {
                "schema_version": "sireto-siren-dossier-v3",
                "build_id": "d" * 64,
                "policy": {
                    "crm_labels_present": False,
                    "model_scores_present": False,
                },
                "files": {
                    "establishments.parquet": {
                        "size_bytes": establishments.stat().st_size,
                        "sha256": file_sha256(establishments),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = _args(tmp_path, source)
    args.official_establishments = establishments
    args.official_establishments_manifest = official_manifest
    destination = build(args)
    ledger = pd.read_parquet(destination / "label_ledger.parquet").set_index(
        "query_id"
    )
    manifest = json.loads((destination / "manifest.json").read_text())

    assert bool(ledger.at["q2", "official_catalog_present"])
    assert not bool(ledger.at["q3", "official_catalog_present"])
    assert bool(ledger.at["q4", "official_catalog_present"])
    assert manifest["official_catalog_presence_available"] is True
    assert manifest["official_catalog_present_count"] == 2
    assert manifest["official_catalog_absent_count"] == 510
    assert manifest["official_catalog_lineage"]["candidate_generation"] is False
    assert manifest["official_catalog_lineage"]["rank_or_score_read"] is False
    assert manifest["official_catalog_lineage"]["establishments_sha256"] == (
        file_sha256(establishments)
    )


def test_official_catalog_requires_both_paths_and_label_free_manifest(tmp_path):
    source = _raw_source(tmp_path)
    official = tmp_path / "official"
    official.mkdir()
    establishments = official / "establishments.parquet"
    pd.DataFrame({"siret": ["11111111100001"]}).to_parquet(
        establishments, index=False
    )
    args = _args(tmp_path, source)
    args.official_establishments = establishments
    with pytest.raises(ValueError, match="must be supplied together"):
        build(args)

    manifest_path = official / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "sireto-siren-dossier-v3",
                "policy": {
                    "crm_labels_present": False,
                    "model_scores_present": True,
                },
                "files": {
                    "establishments.parquet": {
                        "size_bytes": establishments.stat().st_size,
                        "sha256": file_sha256(establishments),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args.official_establishments_manifest = manifest_path
    with pytest.raises(ValueError, match="model scores absent"):
        build(args)
