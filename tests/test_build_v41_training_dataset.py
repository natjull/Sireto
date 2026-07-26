from pathlib import Path
import json

import pandas as pd
import pytest

from scripts.build_v41_training_dataset import (
    CANDIDATE_COLUMNS,
    FEATURE_ORDER,
    LEGACY_55_FEATURE_NAMES,
    LABEL_COLUMNS,
    QUERY_COLUMNS,
    assert_authorized_benchmark,
    build_dataset,
    load_denylist_ids,
    retrieval_signature,
)
from src.xgb_matcher.retrieval import CandidatePoolResult
from src.xgb_matcher.v41_retrieval import (
    InputSiretQualification,
    InputSiretState,
    V41CandidatePoolResult,
    V41RetrievalConfig,
    V41RetrievalVariant,
)
from src.xgb_matcher.v9_dataset import file_sha256


def _candidate(siret: str, *, rank: int = 1) -> dict:
    return {
        "siret": siret,
        "siren": siret[:9],
        "denomination": "ALPHA",
        "enseigne1": None,
        "enseigne2": None,
        "enseigne3": None,
        "numeroVoie": "1",
        "typeVoie": "RUE",
        "libelleVoie": "ALPHA",
        "complementAdresse": None,
        "postcode": "01000",
        "city": "BOURG",
        "insee": "01001",
        "etat_admin": "A",
        "is_siege": True,
        "rrf_score": 0.1,
        "retrieval_rank": rank,
        "retrieval_source": "sparse_active",
        "retrieval_channel_count": 1,
        "v41_channel_ranks": {"sparse_active": rank},
    }


class _FixtureRetriever:
    def build(
        self,
        *,
        crm_row,
        crm_pre,
        input_siret,
        gt_siret,
        persistent_cache,
    ):
        assert gt_siret is None
        if crm_row["query_id"] == "q-exact":
            # Deliberately omit the truth. The builder must retain the miss.
            candidates = [_candidate("22222222200022")]
            qualification = InputSiretQualification(
                raw_value=input_siret,
                normalized_siret=input_siret,
                siren=input_siret[:9],
                state=InputSiretState.ACTIVE,
                candidate=_candidate(input_siret),
            )
        else:
            candidates = [_candidate("33333333300033")]
            qualification = InputSiretQualification(
                raw_value=input_siret,
                normalized_siret=input_siret,
                siren=input_siret[:9],
                state=InputSiretState.CLOSED,
                candidate=None,
            )
        return V41CandidatePoolResult(
            candidates=candidates,
            input_siret=qualification,
            channels={"sparse_active": [candidates[0]["siret"]]},
            sparse_result=CandidatePoolResult(
                candidates=candidates,
                gt_was_injected=False,
            ),
        )


def _write_inputs(tmp_path: Path):
    fit = pd.DataFrame(
        {
            "query_id": ["q-exact"],
            "crm_record_id": ["crm-1"],
            "crm_name": ["ALPHA"],
            "crm_address": ["1 RUE ALPHA"],
            "crm_city": ["BOURG"],
            "postcode": ["01000"],
            "insee": ["01001"],
            "label_kind": ["MATCH_EXACT"],
            "ground_truth_siret": ["11111111100011"],
            "split": ["fit"],
        }
    )
    dev = pd.DataFrame(
        {
            "query_id": ["q-ambiguous"],
            "crm_record_id": ["crm-2"],
            "crm_name": ["BETA"],
            "crm_address": ["2 RUE BETA"],
            "crm_city": ["BOURG"],
            "postcode": ["01000"],
            "insee": ["01001"],
            "label_kind": ["AMBIGUOUS"],
            "ground_truth_siret": [None],
            "split": ["dev"],
        }
    )
    crm = pd.DataFrame(
        {
            "SERVICE ID": ["crm-1", "crm-2"],
            "SIRET": ["11111111100011", "99999999900099"],
        }
    )
    deny = pd.DataFrame({"crm_record_id": ["consumed"]})
    fit_path = tmp_path / "fit.parquet"
    dev_path = tmp_path / "dev.parquet"
    crm_path = tmp_path / "crm.csv"
    deny_path = tmp_path / "consumed.parquet"
    fit.to_parquet(fit_path, index=False)
    dev.to_parquet(dev_path, index=False)
    crm.to_csv(crm_path, sep=";", index=False)
    deny.to_parquet(deny_path, index=False)
    return fit_path, dev_path, crm_path, deny_path


def _write_gate(
    tmp_path: Path,
    *,
    crm_path: Path,
    config: V41RetrievalConfig,
    partitions_signature: str,
    global_store_signature: str,
) -> Path:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    raw_path = gate_dir / "raw_results.parquet"
    summary_path = gate_dir / "summary.json"
    pd.DataFrame({"query_id": ["dev-1"]}).to_parquet(raw_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "selection": {
                    "verdict": "GO",
                    "selected_variant": config.variant.value,
                }
            }
        ),
        encoding="utf-8",
    )
    manifest_path = gate_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "sireto-v4.1-retrieval-evaluation-1",
                "split": "dev",
                "positive_injection": False,
                "inputs": {
                    "crm_source": {"sha256": file_sha256(crm_path)},
                    "partitions": {
                        "runtime_signature": partitions_signature,
                    },
                    "global_store": {
                        "runtime_signature": global_store_signature,
                    },
                },
                "retrieval": {
                    config.variant.value: {
                        "v41_signature": config.signature(),
                        "dataset_retrieval_signature": retrieval_signature(
                            config,
                            partitions_signature=partitions_signature,
                            global_store_signature=global_store_signature,
                        ),
                    }
                },
                "outputs": {
                    "raw_results.parquet": file_sha256(raw_path),
                    "summary.json": file_sha256(summary_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_builder_keeps_misses_and_writes_canonical_hashes(tmp_path):
    fit, dev, crm, deny = _write_inputs(tmp_path)
    config = V41RetrievalConfig(
        variant=V41RetrievalVariant.B_INPUT_EVIDENCE,
        max_candidates=100,
    )
    gate = _write_gate(
        tmp_path,
        crm_path=crm,
        config=config,
        partitions_signature="partitions-fixture",
        global_store_signature="global-fixture",
    )
    target = build_dataset(
        benchmark_paths=[fit, dev],
        crm_source_path=crm,
        denylist_paths=[deny],
        retriever=_FixtureRetriever(),
        retrieval_config=config,
        persistent_cache=None,
        output_root=tmp_path / "out",
        partitions_signature="partitions-fixture",
        global_store_signature="global-fixture",
        retrieval_gate_manifest_path=gate,
    )
    queries = pd.read_parquet(target / "queries.parquet")
    labels = pd.read_parquet(target / "labels.parquet")
    candidates = pd.read_parquet(target / "candidates.parquet")
    manifest = __import__("json").loads((target / "manifest.json").read_text())

    assert set(labels["label_kind"]) == {"MATCH_EXACT", "AMBIGUOUS"}
    assert list(queries.columns) == QUERY_COLUMNS
    assert list(labels.columns) == LABEL_COLUMNS
    assert list(candidates.columns) == CANDIDATE_COLUMNS
    assert "split" not in queries and "split" not in labels
    assert queries.set_index("query_id").at[
        "q-exact", "input_siret_state"
    ] == "ACTIVE"
    assert candidates.groupby("query_id").size().max() <= 100
    assert candidates["is_ground_truth"].sum() == 0
    assert len(LEGACY_55_FEATURE_NAMES) == 55
    assert len(FEATURE_ORDER) == 64
    assert manifest["feature_order"] == FEATURE_ORDER
    assert manifest["positive_injection"] is False
    assert manifest["diagnostics"]["match_exact_retrieval_miss_count"] == 1
    assert manifest["invariants"]["ground_truth_miss_preserved"] is True
    assert manifest["output_hashes"]["candidates.parquet"]
    assert manifest["retrieval_gate"]["manifest_sha256"] == file_sha256(gate)


def test_authorization_rejects_consumed_id_and_holdout_markers():
    base = pd.DataFrame(
        {
            "query_id": ["q"],
            "crm_record_id": ["consumed"],
            "label_kind": ["AMBIGUOUS"],
        }
    )
    with pytest.raises(ValueError, match="consumed CRM IDs"):
        assert_authorized_benchmark(
            base,
            source_name="fit",
            denied_crm_ids={"consumed"},
        )
    forbidden = base.assign(crm_record_id="safe", fresh_role="holdout_sealed")
    with pytest.raises(ValueError, match="test/holdout values"):
        assert_authorized_benchmark(
            forbidden,
            source_name="fit",
            denied_crm_ids=set(),
        )
    forbidden_column = base.assign(crm_record_id="safe", test_flag=False)
    with pytest.raises(ValueError, match="columns are forbidden"):
        assert_authorized_benchmark(
            forbidden_column,
            source_name="fit",
            denied_crm_ids=set(),
        )


def test_full_historical_denylist_uses_only_consumed_test_rows(tmp_path):
    denylist = pd.DataFrame(
        {
            "crm_record_id": ["train-id", "dev-id", "test-id"],
            "split": ["train", "dev", "test"],
        }
    )
    path = tmp_path / "historical.parquet"
    denylist.to_parquet(path, index=False)
    ids, hashes = load_denylist_ids([path])
    assert ids == {"test-id"}
    assert hashes


def test_builder_rejects_gate_selected_for_another_variant(tmp_path):
    fit, dev, crm, deny = _write_inputs(tmp_path)
    config = V41RetrievalConfig(variant=V41RetrievalVariant.B_INPUT_EVIDENCE)
    gate = _write_gate(
        tmp_path,
        crm_path=crm,
        config=V41RetrievalConfig(
            variant=V41RetrievalVariant.A_SPARSE_ACTIVE
        ),
        partitions_signature="partitions-fixture",
        global_store_signature="global-fixture",
    )
    with pytest.raises(ValueError, match="selected variant"):
        build_dataset(
            benchmark_paths=[fit, dev],
            crm_source_path=crm,
            denylist_paths=[deny],
            retriever=_FixtureRetriever(),
            retrieval_config=config,
            persistent_cache=None,
            output_root=tmp_path / "out",
            partitions_signature="partitions-fixture",
            global_store_signature="global-fixture",
            retrieval_gate_manifest_path=gate,
        )


def test_unresolved_rows_are_not_written_to_training_dataset(tmp_path):
    fit, _dev, crm, deny = _write_inputs(tmp_path)
    frame = pd.read_parquet(fit)
    unresolved = frame.iloc[[0]].assign(
        query_id="q-unresolved",
        crm_record_id="crm-unresolved",
        label_kind="UNRESOLVED",
        ground_truth_siret=None,
    )
    pd.concat([frame, unresolved], ignore_index=True).to_parquet(fit, index=False)
    crm_frame = pd.read_csv(crm, sep=";", dtype=str)
    crm_frame.loc[len(crm_frame)] = ["crm-unresolved", "12345678900012"]
    crm_frame.to_csv(crm, sep=";", index=False)
    config = V41RetrievalConfig()
    gate = _write_gate(
        tmp_path,
        crm_path=crm,
        config=config,
        partitions_signature="partitions-fixture",
        global_store_signature="global-fixture",
    )
    target = build_dataset(
        benchmark_paths=[fit],
        crm_source_path=crm,
        denylist_paths=[deny],
        retriever=_FixtureRetriever(),
        retrieval_config=config,
        persistent_cache=None,
        output_root=tmp_path / "out",
        partitions_signature="partitions-fixture",
        global_store_signature="global-fixture",
        retrieval_gate_manifest_path=gate,
    )
    labels = pd.read_parquet(target / "labels.parquet")
    assert set(labels["query_id"]) == {"q-exact"}


def test_missing_crm_ids_are_excluded_before_training(tmp_path):
    fit, _dev, crm, deny = _write_inputs(tmp_path)
    frame = pd.read_parquet(fit)
    missing = frame.iloc[[0]].assign(
        query_id="q-missing-id",
        crm_record_id=None,
    )
    pd.concat([frame, missing], ignore_index=True).to_parquet(fit, index=False)
    config = V41RetrievalConfig()
    gate = _write_gate(
        tmp_path,
        crm_path=crm,
        config=config,
        partitions_signature="partitions-fixture",
        global_store_signature="global-fixture",
    )
    target = build_dataset(
        benchmark_paths=[fit],
        crm_source_path=crm,
        denylist_paths=[deny],
        retriever=_FixtureRetriever(),
        retrieval_config=config,
        persistent_cache=None,
        output_root=tmp_path / "out",
        partitions_signature="partitions-fixture",
        global_store_signature="global-fixture",
        retrieval_gate_manifest_path=gate,
    )
    manifest = __import__("json").loads((target / "manifest.json").read_text())
    assert manifest["diagnostics"]["excluded_missing_crm_record_id_count"] == 1
    assert manifest["row_counts"]["queries"] == 1


def test_builder_refuses_positive_injection(tmp_path):
    fit, _dev, crm, deny = _write_inputs(tmp_path)
    config = V41RetrievalConfig()
    gate = _write_gate(
        tmp_path,
        crm_path=crm,
        config=config,
        partitions_signature="partitions-fixture",
        global_store_signature="global-fixture",
    )

    class InjectingRetriever(_FixtureRetriever):
        def build(self, **kwargs):
            result = super().build(**kwargs)
            result.sparse_result.gt_was_injected = True
            return result

    with pytest.raises(ValueError, match="Positive injection"):
        build_dataset(
            benchmark_paths=[fit],
            crm_source_path=crm,
            denylist_paths=[deny],
            retriever=InjectingRetriever(),
            retrieval_config=config,
            persistent_cache=None,
            output_root=tmp_path / "out",
            partitions_signature="partitions-fixture",
            global_store_signature="global-fixture",
            retrieval_gate_manifest_path=gate,
        )
