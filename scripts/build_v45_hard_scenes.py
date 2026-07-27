#!/usr/bin/env python3
"""Build reproducible V4.5 hard scenes without training or test access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v41_training_dataset import (  # noqa: E402
    _path_signature,
    build_legacy_55_features,
)
from scripts.evaluate_v44_adjudication_gate import (  # noqa: E402
    SCHEMA_VERSION as GATE_SCHEMA_VERSION,
    load_adjudication_artifacts,
)
from scripts.run_v41_shadow import _annotate_candidate, _first_text  # noqa: E402
from src.xgb_matcher.features import (  # noqa: E402
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v41_features import build_v41_candidate_features  # noqa: E402
from src.xgb_matcher.v41_retrieval import (  # noqa: E402
    V41CandidateRetriever,
    V41CurrentStateStore,
    V41GlobalCandidateStore,
    V41RetrievalConfig,
    V41RetrievalVariant,
)
from src.xgb_matcher.v9_dataset import file_sha256, read_table  # noqa: E402
from src.xgb_matcher.v9_scene import (  # noqa: E402
    V9_SCENE_FEATURE_NAMES,
    build_inference_scene,
)


SCHEMA_VERSION = "sireto-v4.5-hard-scenes-1"
COMPATIBILITY_GATE_SCHEMA_VERSION = "sireto-v4.5-scene-compatibility-gate-1"
POLICY_VERSION = "v4.5-scene-binding-v1"
EXPECTED_CANONICAL_CASE_COUNT = 172
EXPECTED_ADJUDICATION_ARTIFACT_COUNT = 5
EXPECTED_RANKER_BUNDLE_ID = "a11b1356b8526165"
EXPECTED_V43_QUEUE_SHA256 = (
    "47af4887769a2edb11f1e629c38077edccd035dd96cb3a6d39620714361fdecc"
)
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "v4_5_exhaustive_pivot_contract.md"
)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _normalise_siret(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = "".join(char for char in str(value).strip() if char.isdigit())
    return text.zfill(14) if text else None


def load_canonical_gate(
    gate_dir: Path,
    *,
    enforce_contract_counts: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """Load only the five artifacts frozen by the canonical V4.4 gate."""

    gate_dir = Path(gate_dir).resolve()
    gate_manifest_path = gate_dir / "manifest.json"
    gate = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ValueError("Unsupported V4.4 gate schema")
    inputs = gate.get("inputs") or []
    if len(inputs) != EXPECTED_ADJUDICATION_ARTIFACT_COUNT:
        raise ValueError("V4.5 requires exactly five canonical V4.4 artifacts")
    if gate.get("verdict") != "STOP_AUTONOMOUS_LABELING":
        raise ValueError("V4.5 exhaustive pivot requires the frozen STOP verdict")
    if gate.get("stop_requested") is not True:
        raise ValueError("V4.5 exhaustive pivot requires explicit stop_requested")
    gate_report_path = gate_dir / "gate_report.json"
    declared_report_hash = (gate.get("outputs") or {}).get("gate_report.json")
    if (
        not declared_report_hash
        or file_sha256(gate_report_path) != declared_report_hash
    ):
        raise ValueError("V4.4 gate report hash mismatch")
    gate_report = json.loads(gate_report_path.read_text(encoding="utf-8"))
    if gate_report.get("verdict") != "STOP_AUTONOMOUS_LABELING":
        raise ValueError("V4.4 gate report does not preserve the STOP verdict")
    if gate_report.get("stop_requested") is not True:
        raise ValueError("V4.4 gate report stop was not explicitly requested")
    if gate_report.get("source_status") != "EXPLICITLY_EXHAUSTED":
        raise ValueError("V4.4 source population is not explicitly exhausted")
    metrics = gate_report.get("metrics") or {}
    expected_counts = {
        "unique_case_count": 172,
        "evidence_validated_count": 162,
        "acceptor_eligible_count": 162,
        "top1_correct_evidence_validated_count": 114,
        "top1_wrong_evidence_validated_count": 42,
        "unresolved_count": 10,
        "random_case_count": 57,
        "random_evidence_validated_count": 53,
    }
    for key, expected_count in expected_counts.items():
        if metrics.get(key) != expected_count:
            raise ValueError(
                f"Frozen V4.4 gate count mismatch for {key}: "
                f"{metrics.get(key)} != {expected_count}"
            )
    if (metrics.get("label_counts") or {}) != {
        "AMBIGUOUS": 6,
        "TOP1_CORRECT": 114,
        "TOP1_WRONG": 42,
        "UNRESOLVED": 10,
    }:
        raise ValueError("Frozen V4.4 label counts differ from the pivot contract")
    declared_hashes = sorted(
        str(item) for item in gate.get("adjudication_manifest_hashes") or []
    )
    input_hashes = sorted(str(item.get("manifest_sha256") or "") for item in inputs)
    if declared_hashes != input_hashes or any(not value for value in input_hashes):
        raise ValueError("V4.4 gate manifest input chain is inconsistent")

    adjudications, provenance = load_adjudication_artifacts(
        [Path(item["path"]) for item in inputs]
    )
    expected = {
        str(Path(item["path"]).resolve()): {
            "manifest_sha256": str(item["manifest_sha256"]),
            "adjudications_sha256": str(item["adjudications_sha256"]),
            "case_count": int(item["case_count"]),
        }
        for item in inputs
    }
    observed = {str(Path(item["path"]).resolve()): item for item in provenance}
    if set(observed) != set(expected):
        raise ValueError("V4.4 loaded artifacts differ from the gate chain")
    for path, declaration in expected.items():
        for key, value in declaration.items():
            if observed[path][key] != value:
                raise ValueError(f"V4.4 gate input mismatch for {path}: {key}")
    if enforce_contract_counts and len(adjudications) != EXPECTED_CANONICAL_CASE_COUNT:
        raise ValueError(
            f"V4.5 expects {EXPECTED_CANONICAL_CASE_COUNT} unique cases, "
            f"observed {len(adjudications)}"
        )
    labels = adjudications["adjudication_label"].astype(str)
    validated = adjudications["evidence_validated"].astype(bool)
    random = adjudications["sampling_stratum"].astype(str).eq(
        "RANDOM_POPULATION"
    )
    observed_counts = {
        "unique_case_count": int(len(adjudications)),
        "evidence_validated_count": int(validated.sum()),
        "acceptor_eligible_count": int(
            adjudications["acceptor_eligible"].astype(bool).sum()
        ),
        "top1_correct_evidence_validated_count": int(
            (labels.eq("TOP1_CORRECT") & validated).sum()
        ),
        "top1_wrong_evidence_validated_count": int(
            (labels.eq("TOP1_WRONG") & validated).sum()
        ),
        "unresolved_count": int(labels.eq("UNRESOLVED").sum()),
        "random_case_count": int(random.sum()),
        "random_evidence_validated_count": int((random & validated).sum()),
    }
    if enforce_contract_counts and observed_counts != expected_counts:
        raise ValueError(
            "Canonical V4.4 artifacts do not reproduce the exhaustive pivot "
            f"counts: {observed_counts}"
        )
    observed_labels = {
        label: int(labels.eq(label).sum())
        for label in ("AMBIGUOUS", "TOP1_CORRECT", "TOP1_WRONG", "UNRESOLVED")
    }
    if enforce_contract_counts and observed_labels != metrics["label_counts"]:
        raise ValueError("Canonical V4.4 artifacts do not reproduce label counts")
    gate_record = {
        "path": str(gate_dir),
        "manifest_sha256": file_sha256(gate_manifest_path),
        "build_id": str(gate.get("build_id") or ""),
        "verdict": str(gate.get("verdict") or ""),
        "stop_requested": True,
        "gate_report_path": str(gate_report_path),
        "gate_report_sha256": declared_report_hash,
        "source_status": "EXPLICITLY_EXHAUSTED",
        "frozen_counts": expected_counts,
    }
    return adjudications, gate_record, provenance


def link_v43_hard_label_queue(
    adjudications: pd.DataFrame,
    queue_path: Path,
) -> pd.DataFrame:
    """Verify every canonical V4.4 case against the frozen V4.3 queue."""

    queue_path = Path(queue_path)
    if file_sha256(queue_path) != EXPECTED_V43_QUEUE_SHA256:
        raise ValueError("V4.3 hard-label queue hash mismatch")
    queue = pd.read_parquet(queue_path)
    required = {
        "audit_case_id",
        "service_id",
        "top1_siret",
        "sampling_stratum",
        "source_row_number",
    }
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(f"V4.3 hard-label queue missing: {sorted(missing)}")
    if queue["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("V4.3 hard-label queue audit_case_id is not unique")
    scoped = queue.loc[
        queue["audit_case_id"].astype(str).isin(
            set(adjudications["audit_case_id"].astype(str))
        ),
        list(required),
    ].copy()
    scoped = scoped.rename(
        columns={
            "service_id": "v43_service_id",
            "top1_siret": "v43_top1_siret",
            "sampling_stratum": "v43_sampling_stratum",
            "source_row_number": "v43_source_row_number",
        }
    )
    linked = adjudications.merge(
        scoped,
        on="audit_case_id",
        how="left",
        validate="one_to_one",
    )
    if linked["v43_service_id"].isna().any():
        raise ValueError("Canonical V4.4 cases are missing from the V4.3 queue")
    if not linked["service_id"].astype(str).equals(
        linked["v43_service_id"].astype(str)
    ):
        raise ValueError("V4.3/V4.4 service_id mismatch")
    frozen_top1 = linked["frozen_top1_siret"].map(_normalise_siret)
    v43_top1 = linked["v43_top1_siret"].map(_normalise_siret)
    if not frozen_top1.equals(v43_top1):
        raise ValueError("V4.3/V4.4 frozen top-1 mismatch")
    if not linked["sampling_stratum"].astype(str).equals(
        linked["v43_sampling_stratum"].astype(str)
    ):
        raise ValueError("V4.3/V4.4 sampling stratum mismatch")
    return linked


def join_crm_rows(
    adjudications: pd.DataFrame,
    crm_source_path: Path,
) -> pd.DataFrame:
    """Join source CRM rows one-to-one on the adjudicated service IDs."""

    required = {
        "audit_case_id",
        "query_id",
        "service_id",
        "frozen_top1_siret",
        "adjudication_label",
    }
    missing = required - set(adjudications.columns)
    if missing:
        raise ValueError(f"V4.4 adjudications missing columns: {sorted(missing)}")
    crm = read_table(Path(crm_source_path)).fillna("")
    if "SERVICE ID" not in crm.columns:
        raise ValueError("CRM source is missing SERVICE ID")
    wanted = set(adjudications["service_id"].astype(str))
    scoped = crm.loc[crm["SERVICE ID"].astype(str).isin(wanted)].copy()
    scoped["SERVICE ID"] = scoped["SERVICE ID"].astype(str)
    # Exact duplicate lines carry no ambiguity and may safely collapse.
    scoped = scoped.drop_duplicates()
    if scoped["SERVICE ID"].duplicated().any():
        duplicates = sorted(
            scoped.loc[scoped["SERVICE ID"].duplicated(False), "SERVICE ID"].unique()
        )
        raise ValueError(f"Ambiguous CRM source rows: {duplicates[:5]}")
    joined = adjudications.merge(
        scoped,
        left_on="service_id",
        right_on="SERVICE ID",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing_ids = joined.loc[joined["_merge"].ne("both"), "service_id"].tolist()
        raise ValueError(f"CRM source rows missing for: {missing_ids[:5]}")
    return joined.drop(columns="_merge")


def load_frozen_ranker(
    model_dir: Path,
    *,
    expected_bundle_id: str = EXPECTED_RANKER_BUNDLE_ID,
) -> tuple[Any, list[str], dict[str, Any]]:
    """Load the frozen V4.1 ranker after checking its complete hash chain."""

    model_dir = Path(model_dir).resolve()
    release_path = model_dir / "release_manifest.json"
    metadata_path = model_dir / "ranker" / "metadata.json"
    model_path = model_dir / "ranker" / "ranker.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if release.get("ranker_bundle_id") != expected_bundle_id:
        raise ValueError("Unexpected frozen V4.1 ranker bundle")
    if metadata.get("model_bundle_id") != expected_bundle_id:
        raise ValueError("Ranker metadata bundle differs from release")
    component_hashes = release.get("component_hashes") or {}
    for relative, path in (
        ("ranker/ranker.json", model_path),
        ("ranker/metadata.json", metadata_path),
    ):
        if file_sha256(path) != component_hashes.get(relative):
            raise ValueError(f"Frozen ranker component hash mismatch: {relative}")
    if metadata.get("ranker_model_sha256") != file_sha256(model_path):
        raise ValueError("Ranker metadata model hash mismatch")
    trained_retrieval_signature = str(metadata.get("retrieval_signature") or "")
    if trained_retrieval_signature != str(release.get("retrieval_signature") or ""):
        raise ValueError("Ranker training retrieval signatures disagree")
    feature_order = [str(item) for item in release.get("ranker_feature_order") or []]
    if feature_order != [str(item) for item in metadata.get("feature_order") or []]:
        raise ValueError("Frozen ranker feature orders disagree")
    if not feature_order:
        raise ValueError("Frozen ranker feature order is empty")
    ranker = xgb.XGBRanker()
    ranker.load_model(model_path)
    identity = {
        "model_dir": str(model_dir),
        "release_id": str(release.get("release_id") or ""),
        "ranker_bundle_id": expected_bundle_id,
        "release_manifest_sha256": file_sha256(release_path),
        "ranker_model_sha256": file_sha256(model_path),
        "ranker_metadata_sha256": file_sha256(metadata_path),
        "feature_order": feature_order,
        "trained_retrieval_signature": trained_retrieval_signature,
    }
    return ranker, feature_order, identity


def validate_v42_retrieval_inputs(
    *,
    experiment_manifest_path: Path,
    config: V41RetrievalConfig,
    partitions_dir: Path,
    global_store_path: Path,
    state_snapshot_path: Path,
) -> dict[str, Any]:
    """Pin replay inputs to the already evaluated V4.2 variant-B pipeline."""

    manifest_path = Path(experiment_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "sireto-v4.2-representative-retrieval-evaluation-1"
    ):
        raise ValueError("Unsupported V4.2 retrieval experiment manifest")
    retrieval = manifest.get("retrieval") or {}
    if retrieval.get("v41_signature") != config.signature():
        raise ValueError("Runtime retrieval config differs from frozen V4.2")
    if retrieval.get("v41_config") != config.to_dict():
        raise ValueError("Runtime retrieval config payload differs from V4.2")
    if manifest.get("positive_injection") is not False:
        raise ValueError("V4.2 experiment did not forbid positive injection")
    inputs = manifest.get("inputs") or {}
    observed = {
        "partitions": {
            "path": str(Path(partitions_dir)),
            "runtime_signature": _path_signature(Path(partitions_dir)),
        },
        "global_store": {
            "path": str(Path(global_store_path)),
            "runtime_signature": _path_signature(Path(global_store_path)),
        },
        "current_state_snapshot": {
            "path": str(Path(state_snapshot_path)),
            "sha256": file_sha256(Path(state_snapshot_path)),
        },
    }
    for name, runtime in observed.items():
        frozen = inputs.get(name) or {}
        signature_key = (
            "sha256" if name == "current_state_snapshot" else "runtime_signature"
        )
        if runtime[signature_key] != frozen.get(signature_key):
            raise ValueError(f"Runtime {name} differs from frozen V4.2")
        if Path(runtime["path"]).resolve() != Path(str(frozen.get("path"))).resolve():
            raise ValueError(f"Runtime {name} path differs from frozen V4.2")
    return {
        "path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "retrieval_signature": config.signature(),
        "partitions_signature": observed["partitions"]["runtime_signature"],
        "global_store_signature": observed["global_store"]["runtime_signature"],
        "state_snapshot_sha256": observed["current_state_snapshot"]["sha256"],
        "verdict": str(manifest.get("verdict") or ""),
    }


def build_crm_row(row: Mapping[str, Any]) -> dict[str, str]:
    service_id = str(row["service_id"])
    query_id = str(row["audit_case_id"])
    city = _first_text(
        row.get("COMMUNE"),
        row.get("SITE_CLI_COMMUNE"),
        row.get("crm_city"),
    )
    return {
        "query_id": query_id,
        "crm_id": service_id,
        "crm_name": _first_text(row.get("SITE"), row.get("crm_name")),
        "crm_address": _first_text(
            row.get("SITE_CLI_ADRESSE"), row.get("crm_address")
        ),
        "crm_city": city,
        "crm_city_addr": _first_text(row.get("SITE_CLI_COMMUNE"), city),
        "postcode": _first_text(row.get("CODE_POSTAL"), row.get("postcode")),
        "insee": _first_text(row.get("CODE_INSEE"), row.get("insee")),
    }


def replay_ranked_scene(
    *,
    row: Mapping[str, Any],
    retriever: Any,
    ranker: Any,
    ranker_feature_order: Sequence[str],
    persistent_cache: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Replay retrieval B, ranker V4.1 and the common 80-feature scene builder."""

    service_id = str(row["service_id"])
    query_id = str(row["audit_case_id"])
    crm_row = build_crm_row(row)
    crm_pre = preprocess_crm_row(crm_row)
    result = retriever.build(
        crm_row=crm_row,
        crm_pre=crm_pre,
        input_siret=_first_text(row.get("SIRET"), row.get("input_siret")),
        gt_siret=None,
        persistent_cache=persistent_cache,
    )
    if result.sparse_result.gt_was_injected:
        raise ValueError("Positive injection is forbidden in V4.5 replay")
    candidates = [_annotate_candidate(candidate) for candidate in result.candidates]
    if len(candidates) > 100:
        raise ValueError("V4.5 replay exceeded 100 candidates")
    if len({str(item.get("siret") or "") for item in candidates}) != len(candidates):
        raise ValueError("V4.5 replay returned duplicate SIRETs")
    if any(str(item.get("etat_admin") or "").upper() != "A" for item in candidates):
        raise ValueError("V4.5 replay returned a non-active candidate")

    set_global_name_idf_map(
        result.sparse_result.idf_map,
        result.sparse_result.default_idf,
    )
    legacy_rows = make_feature_rows_from_preprocessed(
        crm_pre,
        candidates,
        include_semantic=False,
    )
    feature_rows = [
        {
            **build_legacy_55_features(legacy, candidate),
            **build_v41_candidate_features(
                candidate,
                input_siret=result.input_siret.normalized_siret,
            ),
        }
        for candidate, legacy in zip(candidates, legacy_rows, strict=True)
    ]
    if feature_rows:
        missing_features = set(ranker_feature_order) - set(feature_rows[0])
        if missing_features:
            raise ValueError(f"Ranker features missing: {sorted(missing_features)}")
    scores = (
        np.asarray(
            ranker.predict(
                pd.DataFrame(feature_rows)[list(ranker_feature_order)]
                .astype(float)
                .to_numpy()
            ),
            dtype=float,
        )
        if feature_rows
        else np.asarray([], dtype=float)
    )
    ranked_pairs = sorted(
        zip(candidates, feature_rows, scores, strict=True),
        key=lambda item: (-float(item[2]), str(item[0].get("siret") or "")),
    )
    ranked: list[dict[str, Any]] = []
    for rank, (candidate, features, score) in enumerate(ranked_pairs, start=1):
        candidate_siret = str(candidate.get("siret") or "")
        ranked.append(
            {
                **candidate,
                **features,
                "query_id": query_id,
                "candidate_siret": candidate_siret,
                "candidate_siren": str(
                    candidate.get("siren") or candidate_siret[:9]
                ),
                "candidate_state": str(candidate.get("etat_admin") or "").upper(),
                "rank": rank,
                "score": float(score),
                "rrf_score": float(candidate.get("rrf_score") or 0.0),
                "retrieval_agreement": int(
                    int(candidate.get("retrieval_channel_count") or 0) >= 2
                ),
                "prediction_origin": (
                    "frozen_v41_ranker_on_v42b_experimental"
                ),
            }
        )
    predictions = pd.DataFrame(ranked)
    scene = build_inference_scene(query_id, predictions)
    feature_values = {name: scene[name] for name in V9_SCENE_FEATURE_NAMES}
    if list(feature_values) != list(V9_SCENE_FEATURE_NAMES):
        raise ValueError("Common scene builder returned a different feature order")
    replay = {
        "service_id": service_id,
        "query_id": query_id,
        "replayed_top1_siret": _normalise_siret(scene.get("predicted_siret")),
        "replayed_top1_siren": scene.get("predicted_siren"),
        "candidate_count": len(ranked),
        "input_siret": result.input_siret.normalized_siret,
        "input_siret_state": result.input_siret.state.value,
        "candidate_pool_sirens_json": json.dumps(
            sorted({item["candidate_siren"] for item in ranked if item["candidate_siren"]})
        ),
        "channels_json": json.dumps(result.channels, sort_keys=True),
    }
    return replay, ranked, feature_values


def bind_adjudication_to_replay(
    adjudication: Mapping[str, Any],
    replayed_top1_siret: Any,
) -> dict[str, Any]:
    """Bind a label only when it still describes the exact replayed top-1."""

    frozen = _normalise_siret(adjudication.get("frozen_top1_siret"))
    replayed = _normalise_siret(replayed_top1_siret)
    matches = frozen == replayed and frozen is not None
    status = "SCENE_COMPATIBLE" if matches else "SCENE_DRIFT"
    eligible = bool(adjudication.get("acceptor_eligible")) and bool(
        adjudication.get("evidence_validated")
    )
    label = str(adjudication.get("adjudication_label") or "")
    bound = matches and eligible and label != "UNRESOLVED"
    target = adjudication.get("acceptor_target") if bound else None
    if target is not None and not pd.isna(target):
        target = int(target)
    else:
        target = None
    return {
        "scene_status": status,
        "label_bound_to_replayed_top1": bound,
        "scene_adjudication_label": label if bound else None,
        "scene_acceptor_target": target,
        "scene_training_eligible": bound,
        "frozen_adjudication_label": label,
    }


def _candidate_export_row(
    *,
    audit_case_id: str,
    service_id: str,
    candidate: Mapping[str, Any],
    ranker_feature_order: Sequence[str],
) -> dict[str, Any]:
    return {
        "audit_case_id": audit_case_id,
        "service_id": service_id,
        "candidate_siret": candidate["candidate_siret"],
        "candidate_siren": candidate["candidate_siren"],
        "candidate_state": candidate["candidate_state"],
        "rank": int(candidate["rank"]),
        "ranker_score": float(candidate["score"]),
        "rrf_score": float(candidate["rrf_score"]),
        "retrieval_channel_count": int(
            candidate.get("retrieval_channel_count") or 0
        ),
        "retrieval_agreement": int(candidate["retrieval_agreement"]),
        "retrieval_provenance_json": json.dumps(
            candidate.get("v41_channel_ranks") or {}, sort_keys=True
        ),
        **{name: float(candidate[name]) for name in ranker_feature_order},
    }


def build_artifact(
    *,
    gate_dir: Path,
    hard_label_queue_path: Path,
    crm_source_path: Path,
    model_dir: Path,
    partitions_dir: Path,
    global_store_path: Path,
    state_snapshot_path: Path,
    retrieval_experiment_manifest_path: Path,
    cache_dir: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
) -> Path:
    """Replay all canonical cases and publish one immutable external artifact."""

    adjudications, gate_record, provenance = load_canonical_gate(gate_dir)
    adjudications = link_v43_hard_label_queue(
        adjudications, hard_label_queue_path
    )
    joined = join_crm_rows(adjudications, crm_source_path)
    ranker, feature_order, model_identity = load_frozen_ranker(model_dir)
    config = V41RetrievalConfig(
        variant=V41RetrievalVariant.B_INPUT_EVIDENCE,
        max_candidates=100,
    )
    retrieval_identity = validate_v42_retrieval_inputs(
        experiment_manifest_path=retrieval_experiment_manifest_path,
        config=config,
        partitions_dir=partitions_dir,
        global_store_path=global_store_path,
        state_snapshot_path=state_snapshot_path,
    )
    if (
        model_identity["trained_retrieval_signature"]
        == retrieval_identity["retrieval_signature"]
    ):
        raise ValueError(
            "V4.5 expected an explicit cross-retrieval replay, but signatures match"
        )
    input_identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "gate_manifest_sha256": gate_record["manifest_sha256"],
        "hard_label_queue_sha256": file_sha256(Path(hard_label_queue_path)),
        "crm_source_sha256": file_sha256(Path(crm_source_path)),
        "ranker_bundle_id": model_identity["ranker_bundle_id"],
        "ranker_model_sha256": model_identity["ranker_model_sha256"],
        "retrieval_signature": config.signature(),
        "retrieval_experiment_manifest_sha256": retrieval_identity[
            "manifest_sha256"
        ],
        "partitions_signature": retrieval_identity["partitions_signature"],
        "global_store_signature": retrieval_identity["global_store_signature"],
        "state_snapshot_sha256": retrieval_identity["state_snapshot_sha256"],
        "contract_sha256": file_sha256(Path(contract_path)),
        "scene_feature_order": list(V9_SCENE_FEATURE_NAMES),
    }
    build_id = hashlib.sha256(
        json.dumps(input_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root).resolve() / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.5 artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    started = time.perf_counter()
    scene_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    try:
        persistent_cache = TfidfPersistentCache(
            config.sparse_config().tfidf_artifact_hash(),
            cache_dir=cache_dir,
        )
        partitioned_store = PartitionedCandidateStore(partitions_dir)
        with (
            V41GlobalCandidateStore(global_store_path) as global_store,
            V41CurrentStateStore(state_snapshot_path) as state_store,
        ):
            retriever = V41CandidateRetriever(
                partitioned_store=partitioned_store,
                global_store=global_store,
                current_state_store=state_store,
                config=config,
            )
            for record in joined.sort_values("audit_case_id").to_dict("records"):
                replay, ranked, features = replay_ranked_scene(
                    row=record,
                    retriever=retriever,
                    ranker=ranker,
                    ranker_feature_order=feature_order,
                    persistent_cache=persistent_cache,
                )
                binding = bind_adjudication_to_replay(
                    record, replay["replayed_top1_siret"]
                )
                audit_case_id = str(record["audit_case_id"])
                scene_rows.append(
                    {
                        "audit_case_id": audit_case_id,
                        "query_id": str(record["query_id"]),
                        "service_id": str(record["service_id"]),
                        "sampling_stratum": str(record["sampling_stratum"]),
                        "frozen_top1_siret": _normalise_siret(
                            record["frozen_top1_siret"]
                        ),
                        "frozen_top1_siren": str(
                            record.get("frozen_top1_siren") or ""
                        ),
                        "frozen_model_bundle_id": str(
                            record.get("frozen_model_bundle_id") or ""
                        ),
                        "frozen_retrieval_signature": str(
                            record.get("frozen_retrieval_signature") or ""
                        ),
                        **replay,
                        **binding,
                        **features,
                    }
                )
                candidate_rows.extend(
                    _candidate_export_row(
                        audit_case_id=audit_case_id,
                        service_id=str(record["service_id"]),
                        candidate=candidate,
                        ranker_feature_order=feature_order,
                    )
                    for candidate in ranked
                )
        scenes = pd.DataFrame(scene_rows)
        candidates = pd.DataFrame(candidate_rows)
        if list(scenes[list(V9_SCENE_FEATURE_NAMES)].columns) != list(
            V9_SCENE_FEATURE_NAMES
        ):
            raise ValueError("Published scene feature order is not the frozen order")
        drift = scenes["scene_status"].eq("SCENE_DRIFT")
        if scenes.loc[drift, "scene_acceptor_target"].notna().any():
            raise ValueError("A label crossed a SCENE_DRIFT boundary")
        if scenes.loc[drift, "scene_adjudication_label"].notna().any():
            raise ValueError("An adjudication crossed a SCENE_DRIFT boundary")
        if int(scenes["candidate_count"].max()) > 100:
            raise ValueError("Published candidate count exceeds 100")

        scenes_path = staging / "scene_compatibility.parquet"
        candidates_path = staging / "candidates.parquet"
        scenes.to_parquet(scenes_path, index=False)
        candidates.to_parquet(candidates_path, index=False)
        summary = {
            "case_count": int(len(scenes)),
            "candidate_row_count": int(len(candidates)),
            "scene_compatible_count": int((~drift).sum()),
            "scene_drift_count": int(drift.sum()),
            "label_bound_count": int(
                scenes["label_bound_to_replayed_top1"].sum()
            ),
            "training_eligible_count": int(
                scenes["scene_training_eligible"].sum()
            ),
            "max_candidate_count": int(scenes["candidate_count"].max()),
            "zero_candidate_count": int(scenes["candidate_count"].eq(0).sum()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        summary_path = staging / "summary.json"
        _json_dump(summary_path, summary)
        outputs = {
            path.name: file_sha256(path)
            for path in (scenes_path, candidates_path, summary_path)
        }
        manifest = {
            **input_identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "gate": gate_record,
            "adjudication_inputs": provenance,
            "inputs": {
                "crm_source": {
                    "path": str(Path(crm_source_path).resolve()),
                    "sha256": input_identity["crm_source_sha256"],
                },
                "hard_label_queue": {
                    "path": str(Path(hard_label_queue_path).resolve()),
                    "sha256": input_identity["hard_label_queue_sha256"],
                },
                "model": model_identity,
                "partitions": {"path": str(Path(partitions_dir).resolve())},
                "global_store": {"path": str(Path(global_store_path).resolve())},
                "state_snapshot": {
                    "path": str(Path(state_snapshot_path).resolve()),
                    "sha256": input_identity["state_snapshot_sha256"],
                },
                "contract": {
                    "path": str(Path(contract_path).resolve()),
                    "sha256": input_identity["contract_sha256"],
                },
            },
            "retrieval": {
                "config": config.to_dict(),
                "signature": config.signature(),
                "candidate_ceiling": 100,
                "frozen_v42_experiment": retrieval_identity,
            },
            "architecture_compatibility": {
                "compatibility_status": "EXPERIMENTAL_CROSS_RETRIEVAL",
                "ranker_trained_retrieval_signature": model_identity[
                    "trained_retrieval_signature"
                ],
                "replay_retrieval_signature": retrieval_identity[
                    "retrieval_signature"
                ],
                "ranker_loader": "RANKER_ONLY_HASH_VALIDATED",
                "production_release_loader_used": False,
            },
            "outputs": outputs,
            "summary": summary,
            "invariants": {
                "positive_injection": False,
                "test_final_read": False,
                "training_performed": False,
                "acceptor_loaded_or_scored": False,
                "threshold_selected_or_applied": False,
                "scene_features_built_with_common_train_serve_function": True,
                "drift_labels_copied": False,
                "candidate_ceiling": 100,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_artifact(artifact_dir: Path) -> None:
    """Validate hashes and the no-label-on-drift publication invariant."""

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported V4.5 hard-scene schema")
    for filename, expected in (manifest.get("outputs") or {}).items():
        if file_sha256(artifact_dir / filename) != expected:
            raise ValueError(f"V4.5 output hash mismatch: {filename}")
    scenes = pd.read_parquet(artifact_dir / "scene_compatibility.parquet")
    if list(manifest.get("scene_feature_order") or []) != list(
        V9_SCENE_FEATURE_NAMES
    ):
        raise ValueError("Manifest scene feature order drifted")
    if not set(V9_SCENE_FEATURE_NAMES).issubset(scenes.columns):
        raise ValueError("V4.5 scenes are missing common scene features")
    drift = scenes["scene_status"].eq("SCENE_DRIFT")
    if scenes.loc[drift, "scene_acceptor_target"].notna().any():
        raise ValueError("Drift scene carries an acceptor target")
    if scenes.loc[drift, "scene_adjudication_label"].notna().any():
        raise ValueError("Drift scene carries an adjudication label")
    if int(scenes["candidate_count"].max()) > 100:
        raise ValueError("V4.5 artifact exceeds the candidate ceiling")


def compute_scene_compatibility_gate(scenes: pd.DataFrame) -> dict[str, Any]:
    """Apply the preregistered V4.5 scene-compatibility thresholds."""

    required = {
        "audit_case_id",
        "scene_status",
        "sampling_stratum",
        "frozen_adjudication_label",
        "label_bound_to_replayed_top1",
    }
    missing = required - set(scenes.columns)
    if missing:
        raise ValueError(f"Compatibility scenes missing: {sorted(missing)}")
    if scenes["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("Compatibility scenes must be unique by audit_case_id")
    if len(scenes) != 172:
        raise ValueError("Compatibility gate requires exactly 172 V4.4 scenes")
    compatible = scenes["scene_status"].astype(str).eq("SCENE_COMPATIBLE")
    drift = scenes["scene_status"].astype(str).eq("SCENE_DRIFT")
    if not (compatible | drift).all():
        raise ValueError("Unsupported scene compatibility status")
    bound = scenes["label_bound_to_replayed_top1"].astype(bool)
    if (bound & ~compatible).any():
        raise ValueError("A bound label cannot belong to SCENE_DRIFT")
    labels = scenes["frozen_adjudication_label"].astype(str)
    random = scenes["sampling_stratum"].astype(str).eq("RANDOM_POPULATION")
    targeted = ~random
    reliable = labels.ne("UNRESOLVED")
    negative = labels.isin({"TOP1_WRONG", "AMBIGUOUS"})

    counts = {
        "total_scenes": int(len(scenes)),
        "scene_compatible": int(compatible.sum()),
        "scene_drift": int(drift.sum()),
        "random_total": int(random.sum()),
        "random_compatible": int((random & compatible).sum()),
        "random_reliable_total": int((random & reliable).sum()),
        "random_reliable_compatible": int((random & reliable & bound).sum()),
        "random_negative_total": int((random & negative & reliable).sum()),
        "random_negative_compatible": int(
            (random & negative & reliable & bound).sum()
        ),
        "targeted_top1_correct_total": int(
            (targeted & labels.eq("TOP1_CORRECT")).sum()
        ),
        "targeted_top1_correct_compatible": int(
            (targeted & labels.eq("TOP1_CORRECT") & bound).sum()
        ),
        "targeted_top1_wrong_total": int(
            (targeted & labels.eq("TOP1_WRONG")).sum()
        ),
        "targeted_top1_wrong_compatible": int(
            (targeted & labels.eq("TOP1_WRONG") & bound).sum()
        ),
        "targeted_ambiguous_total": int(
            (targeted & labels.eq("AMBIGUOUS")).sum()
        ),
        "targeted_ambiguous_compatible": int(
            (targeted & labels.eq("AMBIGUOUS") & bound).sum()
        ),
    }
    frozen_totals = {
        "random_total": 57,
        "random_reliable_total": 53,
        "random_negative_total": 6,
        "targeted_top1_correct_total": 67,
        "targeted_top1_wrong_total": 37,
        "targeted_ambiguous_total": 5,
    }
    for key, expected in frozen_totals.items():
        if counts[key] != expected:
            raise ValueError(
                f"Compatibility population mismatch for {key}: "
                f"{counts[key]} != {expected}"
            )
    checks = {
        "all_random_reliable_compatible": (
            counts["random_reliable_compatible"] == 53
        ),
        "all_random_negatives_compatible": (
            counts["random_negative_compatible"] == 6
        ),
        "targeted_top1_wrong_compatible_gte_30": (
            counts["targeted_top1_wrong_compatible"] >= 30
        ),
        "targeted_ambiguous_compatible_gte_4": (
            counts["targeted_ambiguous_compatible"] >= 4
        ),
        "targeted_top1_correct_compatible_gte_55": (
            counts["targeted_top1_correct_compatible"] >= 55
        ),
    }
    verdict = (
        "GO_SCENE_COMPATIBILITY"
        if all(checks.values())
        else "PIVOT_SCENE_DRIFT"
    )
    return {
        "verdict": verdict,
        "counts": counts,
        "checks": checks,
        "thresholds": {
            "random_reliable_compatible": 53,
            "random_negative_compatible": 6,
            "targeted_top1_wrong_compatible_min": 30,
            "targeted_ambiguous_compatible_min": 4,
            "targeted_top1_correct_compatible_min": 55,
        },
        "training_authorized": False,
    }


def build_compatibility_gate_artifact(
    *,
    scene_artifact_dir: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
) -> Path:
    """Publish the compatibility verdict separately from immutable scenes."""

    scene_artifact_dir = Path(scene_artifact_dir).resolve()
    validate_artifact(scene_artifact_dir)
    scene_manifest_path = scene_artifact_dir / "manifest.json"
    scenes_path = scene_artifact_dir / "scene_compatibility.parquet"
    identity = {
        "schema_version": COMPATIBILITY_GATE_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "scene_manifest_sha256": file_sha256(scene_manifest_path),
        "scenes_sha256": file_sha256(scenes_path),
        "contract_sha256": file_sha256(Path(contract_path)),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root).resolve() / build_id
    if target.exists():
        raise FileExistsError(f"Immutable compatibility gate exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    try:
        report = {
            "build_id": build_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            **compute_scene_compatibility_gate(pd.read_parquet(scenes_path)),
        }
        report_path = staging / "gate_report.json"
        _json_dump(report_path, report)
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "scene_artifact": {
                    "path": str(scene_artifact_dir),
                    "manifest_sha256": identity["scene_manifest_sha256"],
                    "scenes_sha256": identity["scenes_sha256"],
                },
                "contract": {
                    "path": str(Path(contract_path).resolve()),
                    "sha256": identity["contract_sha256"],
                },
            },
            "outputs": {"gate_report.json": file_sha256(report_path)},
            "verdict": report["verdict"],
            "invariants": {
                "training_performed": False,
                "acceptor_loaded_or_scored": False,
                "threshold_selected_or_applied": False,
                "test_final_read": False,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_compatibility_gate_artifact(artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    if manifest.get("schema_version") != COMPATIBILITY_GATE_SCHEMA_VERSION:
        raise ValueError("Unsupported V4.5 compatibility-gate schema")
    report_path = artifact_dir / "gate_report.json"
    if file_sha256(report_path) != (manifest.get("outputs") or {}).get(
        "gate_report.json"
    ):
        raise ValueError("Compatibility gate report hash mismatch")
    source = manifest.get("inputs", {}).get("scene_artifact") or {}
    scene_dir = Path(str(source.get("path") or ""))
    if file_sha256(scene_dir / "manifest.json") != source.get("manifest_sha256"):
        raise ValueError("Compatibility gate scene manifest changed")
    scenes_path = scene_dir / "scene_compatibility.parquet"
    if file_sha256(scenes_path) != source.get("scenes_sha256"):
        raise ValueError("Compatibility gate scenes changed")
    expected = compute_scene_compatibility_gate(pd.read_parquet(scenes_path))
    observed = json.loads(report_path.read_text())
    for key in ("verdict", "counts", "checks", "thresholds", "training_authorized"):
        if observed.get(key) != expected[key]:
            raise ValueError(f"Compatibility gate result mismatch: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path)
    parser.add_argument("--crm-source", type=Path)
    parser.add_argument("--hard-label-queue", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--partitions-dir", type=Path)
    parser.add_argument("--global-store", type=Path)
    parser.add_argument("--state-snapshot", type=Path)
    parser.add_argument("--retrieval-experiment-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--evaluate-compatibility", type=Path)
    parser.add_argument("--gate-output-root", type=Path)
    parser.add_argument("--validate-compatibility-gate", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_compatibility_gate:
        validate_compatibility_gate_artifact(args.validate_compatibility_gate)
        print(args.validate_compatibility_gate)
        return
    if args.evaluate_compatibility:
        if args.gate_output_root is None:
            raise SystemExit("--gate-output-root is required")
        output = build_compatibility_gate_artifact(
            scene_artifact_dir=args.evaluate_compatibility,
            output_root=args.gate_output_root,
            contract_path=args.contract,
        )
        print(output)
        return
    if args.validate_artifact:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    required = {
        name: getattr(args, name)
        for name in (
            "gate_dir",
            "crm_source",
            "hard_label_queue",
            "model_dir",
            "partitions_dir",
            "global_store",
            "state_snapshot",
            "retrieval_experiment_manifest",
            "cache_dir",
            "output_root",
        )
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    output = build_artifact(
        gate_dir=args.gate_dir,
        hard_label_queue_path=args.hard_label_queue,
        crm_source_path=args.crm_source,
        model_dir=args.model_dir,
        partitions_dir=args.partitions_dir,
        global_store_path=args.global_store,
        state_snapshot_path=args.state_snapshot,
        retrieval_experiment_manifest_path=args.retrieval_experiment_manifest,
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        contract_path=args.contract,
    )
    print(output)


if __name__ == "__main__":
    main()
