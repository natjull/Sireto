#!/usr/bin/env python3
"""Run the frozen V4.1 release locally in leak-safe CRM shadow mode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v41_training_dataset import (  # noqa: E402
    FEATURE_ORDER,
    LEGACY_55_FEATURE_NAMES,
    _path_signature,
    build_legacy_55_features,
    retrieval_signature,
    validate_retrieval_gate,
)
from src.xgb_matcher.features import (  # noqa: E402
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.naming import build_candidate_names  # noqa: E402
from src.xgb_matcher.features import build_address  # noqa: E402
from src.xgb_matcher.partitioned_store import (  # noqa: E402
    PartitionedCandidateStore,
)
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v41_acceptor import (  # noqa: E402
    V41RawLogisticAcceptor,
)
from src.xgb_matcher.v41_decision import decide_v41  # noqa: E402
from src.xgb_matcher.v41_features import (  # noqa: E402
    build_v41_candidate_features,
)
from src.xgb_matcher.v41_release import V41ReleaseManifest  # noqa: E402
from src.xgb_matcher.v41_retrieval import (  # noqa: E402
    InputSiretQualification,
    InputSiretState,
    V41CandidateRetriever,
    V41GlobalCandidateStore,
    V41RetrievalConfig,
)
from src.xgb_matcher.v41_shadow import (  # noqa: E402
    DenylistSpec,
    build_shadow_inventory,
    select_scoreable_rows,
    write_shadow_run,
)
from src.xgb_matcher.v9_dataset import file_sha256, read_table  # noqa: E402
from src.xgb_matcher.v9_scene import build_inference_scene  # noqa: E402


CURRENT_SOURCE_SHA256 = (
    "f770215cd0d0fcc654b750b90dbba835acbf4efb5c74ed269d339e046c2b049d"
)
CURRENT_EXPECTED_DECISION_COUNT = 19_025
CURRENT_EXPECTED_PANEL_COUNT = 500
CHECKPOINT_SCHEMA_VERSION = "sireto-v4.1-shadow-checkpoint-1"


@dataclass
class V41RuntimeBundle:
    dataset_manifest: dict[str, Any]
    release: V41ReleaseManifest
    ranker_metadata: dict[str, Any]
    ranker: Any
    acceptor: V41RawLogisticAcceptor
    retrieval_config: V41RetrievalConfig
    model_dir: Path | None = None
    dataset_dir: Path | None = None
    artifact_hashes: dict[str, str] | None = None


def reconcile_inventory_qualification(
    *,
    inventory_row: Mapping[str, Any],
    runtime: InputSiretQualification,
) -> InputSiretQualification:
    """Use the frozen full SIRENE inventory as state authority.

    The global candidate store is optimized for retrieval and can omit a SIRET
    that exists in the complete stock.  In that case, preserve the inventory's
    ACTIVE or CLOSED state without inventing candidate details.  Any
    contradictory current-state evidence remains a hard failure.
    """

    state_text = _first_text(
        inventory_row.get("input_siret_state"),
        InputSiretState.INVALID.value,
    ).upper()
    try:
        inventory_state = InputSiretState(state_text)
    except ValueError as exc:
        raise ValueError(f"Unsupported inventory SIRET state: {state_text}") from exc
    inventory_siret = _first_text(inventory_row.get("input_siret")) or None
    inventory_siren = _first_text(inventory_row.get("input_siren")) or (
        inventory_siret[:9] if inventory_siret else None
    )
    if inventory_siret != runtime.normalized_siret:
        raise ValueError("Inventory/runtime normalized SIRET mismatch")
    if inventory_state == runtime.state:
        return runtime
    if (
        inventory_state in {InputSiretState.ACTIVE, InputSiretState.CLOSED}
        and runtime.state == InputSiretState.NOT_FOUND
    ):
        return InputSiretQualification(
            raw_value=runtime.raw_value,
            normalized_siret=inventory_siret,
            siren=inventory_siren,
            state=inventory_state,
            candidate=None,
        )
    raise ValueError(
        "Input SIRET state drift: "
        f"inventory={inventory_state.value}, runtime={runtime.state.value}"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(value: Any) -> str:
    """Normalize missing CRM cells without ever serializing them as ``nan``."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _first_text(*values: Any) -> str:
    for value in values:
        normalized = _text(value)
        if normalized:
            return normalized
    return ""


def _current_code_commit() -> str:
    override = os.environ.get("SIRETO_CODE_COMMIT", "").strip()
    if override:
        return override
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def validate_inventory_chain(
    *,
    source_path: Path,
    inventory_path: Path,
    panel_path: Path,
    inventory_manifest_path: Path,
) -> dict[str, Any]:
    """Verify the frozen CRM/denylists/inventory/panel chain and its ID policy."""

    manifest = _read_json(inventory_manifest_path)
    if manifest.get("schema_version") != "sireto-shadow-v4.1-inventory-1":
        raise ValueError("Unsupported V4.1 inventory manifest")
    if (manifest.get("source") or {}).get("sha256") != file_sha256(source_path):
        raise ValueError("Inventory manifest CRM hash mismatch")
    outputs = manifest.get("outputs") or {}
    for name, path in (
        ("inventory.parquet", inventory_path),
        ("panel500.parquet", panel_path),
    ):
        if outputs.get(name) != file_sha256(path):
            raise ValueError(f"Inventory manifest output hash mismatch: {name}")

    entries = manifest.get("denylists") or []
    by_name = {str(entry.get("name")): entry for entry in entries}
    if set(by_name) != {"OLD_TEST", "FRESH_HOLDOUT"}:
        raise ValueError("Inventory manifest must bind OLD_TEST and FRESH_HOLDOUT")
    denylist_ids: dict[str, set[str]] = {}
    for name, entry in by_name.items():
        denylist_path = Path(str(entry.get("path") or ""))
        if not denylist_path.is_file() or file_sha256(denylist_path) != entry.get(
            "sha256"
        ):
            raise ValueError(f"Inventory denylist hash mismatch: {name}")
        manifest_path_value = entry.get("manifest_path")
        if manifest_path_value:
            denylist_manifest_path = Path(str(manifest_path_value))
            if (
                not denylist_manifest_path.is_file()
                or file_sha256(denylist_manifest_path)
                != entry.get("manifest_sha256")
            ):
                raise ValueError(f"Inventory denylist manifest mismatch: {name}")
        spec = DenylistSpec(
            name=name,
            path=denylist_path,
            id_column=str(entry.get("id_column") or "crm_record_id"),
            split_column=entry.get("split_column"),
            split_value=entry.get("split_value"),
        )
        denylist_ids[name] = spec.load_ids()

    source = read_table(source_path)
    inventory = read_table(inventory_path)
    rebuilt = build_shadow_inventory(source, denylists=denylist_ids)
    authorization_columns = [
        "source_row_number",
        "service_id",
        "eligible_for_shadow",
        "exclusion_reason",
        "denylist_sources",
    ]
    if len(rebuilt) != len(inventory):
        raise ValueError("Inventory row cardinality differs from CRM source")
    for column in authorization_columns:
        if column not in inventory:
            raise ValueError(f"Inventory authorization column missing: {column}")
        left = rebuilt[column].fillna("").astype(str).tolist()
        right = inventory[column].fillna("").astype(str).tolist()
        if left != right:
            raise ValueError(f"Inventory ID authorization drift: {column}")

    panel = read_table(panel_path)
    if "service_id" not in panel:
        raise ValueError("Panel is missing service_id")
    panel_ids = panel["service_id"].map(_text)
    if (panel_ids == "").any() or panel_ids.duplicated().any():
        raise ValueError("Panel SERVICE IDs must be present and unique")
    eligible = set(
        inventory.loc[inventory["eligible_for_shadow"].astype(bool), "service_id"]
        .map(_text)
        .tolist()
    )
    if not set(panel_ids) <= eligible:
        raise ValueError("Panel contains substituted or excluded SERVICE IDs")
    return manifest


def _release_identity_is_valid(release: V41ReleaseManifest) -> bool:
    rebuilt = V41ReleaseManifest.build(
        retrieval_signature=release.retrieval_signature,
        ranker_bundle_id=release.ranker_bundle_id,
        acceptor_bundle_id=release.acceptor_bundle_id,
        ranker_dataset_manifest_id=release.ranker_dataset_manifest_id,
        acceptor_dataset_manifest_id=release.acceptor_dataset_manifest_id,
        ranker_feature_order=release.ranker_feature_order,
        acceptor_feature_order=release.acceptor_feature_order,
        ranker_variant=release.ranker_variant,
        component_hashes=release.component_hashes,
    )
    return rebuilt.release_id == release.release_id


def validate_model_bundle_hashes(
    model_dir: Path,
    *,
    release: V41ReleaseManifest,
) -> dict[str, str]:
    """Verify every immutable model-bundle output before deserialization."""

    model_manifest = _read_json(model_dir / "model_manifest.json")
    if (
        model_manifest.get("schema_version")
        != "v4.1-model-bundle-manifest-1"
    ):
        raise ValueError("Unsupported V4.1 model bundle manifest")
    if model_manifest.get("release_id") != release.release_id:
        raise ValueError("Model manifest/release identity mismatch")
    declared_artifacts = model_manifest.get("outputs") or {}
    expected_model_outputs = {
        "ranker/ranker.json",
        "ranker/metadata.json",
        "acceptor/acceptor_model.joblib",
        "acceptor/metadata.json",
        "ranker_predictions.parquet",
        "acceptor_scenes.parquet",
        "split_assignments.parquet",
        "training_report.json",
        "release_manifest.json",
    }
    if set(declared_artifacts) != expected_model_outputs:
        raise ValueError("V4.1 model manifest output set is incomplete")
    artifact_hashes: dict[str, str] = {}
    for name, expected in declared_artifacts.items():
        path = model_dir / str(name)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"V4.1 model artifact hash mismatch: {name}")
        artifact_hashes[str(name)] = str(expected)
    return artifact_hashes


def validate_runtime_contract(
    *,
    dataset_manifest: Mapping[str, Any],
    release: V41ReleaseManifest,
    ranker_metadata: Mapping[str, Any],
    acceptor_metadata: Mapping[str, Any],
    retrieval_config: V41RetrievalConfig,
    partitions_signature: str,
    global_store_signature: str,
    component_hashes: Mapping[str, str] | None = None,
) -> None:
    """Fail closed before reading any eligible CRM row."""

    if not _release_identity_is_valid(release):
        raise ValueError("V4.1 release_id does not match release contents")
    if dataset_manifest.get("positive_injection") is not False:
        raise ValueError("V4.1 shadow refuses a positive-injection dataset")
    if dataset_manifest.get("build_id") != release.ranker_dataset_manifest_id:
        raise ValueError("Release/ranker dataset manifest mismatch")
    gate_chain = dataset_manifest.get("retrieval_gate") or {}
    if not gate_chain.get("manifest_sha256"):
        raise ValueError("V4.1 dataset is not chained to a retrieval gate")
    if gate_chain.get("selected_variant") != retrieval_config.variant.value:
        raise ValueError("Dataset retrieval gate variant mismatch")
    if list(dataset_manifest.get("feature_order") or []) != FEATURE_ORDER:
        raise ValueError("V4.1 dataset feature order differs from builder contract")
    expected_ranker_order = (
        FEATURE_ORDER
        if release.ranker_variant == "R1"
        else LEGACY_55_FEATURE_NAMES
    )
    if release.ranker_feature_order != expected_ranker_order:
        raise ValueError("Release ranker feature order is not the frozen R0/R1 order")
    observed_signature = retrieval_signature(
        retrieval_config,
        partitions_signature=partitions_signature,
        global_store_signature=global_store_signature,
    )
    if observed_signature != dataset_manifest.get("retrieval_signature"):
        raise ValueError("Runtime retrieval signature differs from training")
    if observed_signature != release.retrieval_signature:
        raise ValueError("Release retrieval signature differs from runtime")
    input_hashes = dataset_manifest.get("input_hashes") or {}
    if input_hashes.get("partitions") != partitions_signature:
        raise ValueError("Runtime partitions differ from training")
    if input_hashes.get("global_store") != global_store_signature:
        raise ValueError("Runtime global candidate store differs from training")
    if ranker_metadata.get("positive_injection") is not False:
        raise ValueError("V4.1 ranker metadata must declare positive_injection=false")
    release.validate_components(
        ranker_metadata=ranker_metadata,
        acceptor_metadata=acceptor_metadata,
        component_hashes=component_hashes,
    )


def load_runtime_bundle(
    *,
    dataset_dir: Path,
    model_dir: Path,
    partitions_signature: str,
    global_store_signature: str,
    retrieval_gate_manifest_path: Path,
) -> V41RuntimeBundle:
    """Load and validate the dataset manifest, model files and release."""

    dataset_manifest = _read_json(dataset_dir / "manifest.json")
    if dataset_manifest.get("schema_version") != "sireto-v4.1-training-dataset-1":
        raise ValueError("Unsupported V4.1 dataset schema")
    output_hashes = (
        dataset_manifest.get("outputs")
        or dataset_manifest.get("output_hashes")
        or {}
    )
    for filename in ("queries.parquet", "labels.parquet", "candidates.parquet"):
        expected = output_hashes.get(filename)
        path = dataset_dir / filename
        if not expected or file_sha256(path) != expected:
            raise ValueError(f"V4.1 dataset output hash mismatch: {filename}")

    release_path = model_dir / "release_manifest.json"
    release = V41ReleaseManifest.load(release_path)
    artifact_hashes = validate_model_bundle_hashes(model_dir, release=release)
    component_paths = {
        "ranker/ranker.json": model_dir / "ranker" / "ranker.json",
        "ranker/metadata.json": model_dir / "ranker" / "metadata.json",
        "acceptor/acceptor_model.joblib": (
            model_dir / "acceptor" / "acceptor_model.joblib"
        ),
        "acceptor/metadata.json": model_dir / "acceptor" / "metadata.json",
    }
    component_hashes = {
        name: file_sha256(path) for name, path in component_paths.items()
    }
    ranker_metadata = _read_json(model_dir / "ranker" / "metadata.json")
    acceptor_metadata = _read_json(model_dir / "acceptor" / "metadata.json")
    retrieval_config = V41RetrievalConfig(
        **dict(dataset_manifest["retrieval_config"])
    )
    gate_chain = validate_retrieval_gate(
        gate_manifest_path=retrieval_gate_manifest_path,
        retrieval_config=retrieval_config,
        crm_source_sha256=str(
            (dataset_manifest.get("input_hashes") or {}).get("crm_source") or ""
        ),
        partitions_signature=partitions_signature,
        global_store_signature=global_store_signature,
    )
    declared_gate = dataset_manifest.get("retrieval_gate") or {}
    if gate_chain["manifest_sha256"] != declared_gate.get("manifest_sha256"):
        raise ValueError("Runtime retrieval gate differs from training dataset")
    if gate_chain["retrieval_signature"] != dataset_manifest.get(
        "retrieval_signature"
    ):
        raise ValueError("Runtime retrieval gate signature differs from dataset")
    validate_runtime_contract(
        dataset_manifest=dataset_manifest,
        release=release,
        ranker_metadata=ranker_metadata,
        acceptor_metadata=acceptor_metadata,
        retrieval_config=retrieval_config,
        partitions_signature=partitions_signature,
        global_store_signature=global_store_signature,
        component_hashes=component_hashes,
    )
    ranker = xgb.XGBRanker()
    ranker.load_model(model_dir / "ranker" / "ranker.json")
    acceptor = V41RawLogisticAcceptor.load(model_dir / "acceptor")
    return V41RuntimeBundle(
        dataset_manifest=dataset_manifest,
        release=release,
        ranker_metadata=ranker_metadata,
        ranker=ranker,
        acceptor=acceptor,
        retrieval_config=retrieval_config,
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        artifact_hashes=artifact_hashes,
    )


class ShadowCheckpoint:
    """Transactional per-query checkpoint with immutable run identity."""

    def __init__(self, path: Path, *, identity: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                service_id TEXT PRIMARY KEY,
                decision_json TEXT NOT NULL,
                candidates_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            )
            """
        )
        canonical_identity = json.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                **dict(identity),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        if existing is not None and existing[0] != canonical_identity:
            self.connection.close()
            raise ValueError("Checkpoint identity differs from requested shadow run")
        if existing is None:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('identity', ?)",
                [canonical_identity],
            )
            self.connection.commit()

    def service_ids(self) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT service_id FROM records"
            ).fetchall()
        }

    def put(
        self,
        *,
        service_id: str,
        decision: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        evidence: Mapping[str, Any],
    ) -> None:
        payloads = [
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            for value in (dict(decision), list(candidates), dict(evidence))
        ]
        existing = self.connection.execute(
            """
            SELECT decision_json, candidates_json, evidence_json
            FROM records WHERE service_id=?
            """,
            [service_id],
        ).fetchone()
        if existing is not None:
            if list(existing) != payloads:
                raise ValueError(
                    f"Checkpoint already contains different data for {service_id}"
                )
            return
        self.connection.execute(
            """
            INSERT INTO records(
                service_id, decision_json, candidates_json, evidence_json
            ) VALUES (?, ?, ?, ?)
            """,
            [service_id, *payloads],
        )
        self.connection.commit()

    def get(self, service_id: str) -> tuple[dict, list[dict], dict]:
        row = self.connection.execute(
            """
            SELECT decision_json, candidates_json, evidence_json
            FROM records WHERE service_id=?
            """,
            [service_id],
        ).fetchone()
        if row is None:
            raise KeyError(service_id)
        return (
            json.loads(row[0]),
            json.loads(row[1]),
            json.loads(row[2]),
        )

    def mark_complete(self, output_dir: Path) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO metadata(key, value)
            VALUES ('completed_output', ?)
            """,
            [str(output_dir)],
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def _candidate_provenance(candidate: Mapping[str, Any]) -> dict[str, Any]:
    ranks = dict(candidate.get("v41_channel_ranks") or {})
    return {
        "retrieval_source": str(candidate.get("retrieval_source") or "v41"),
        "retrieval_channel_count": int(
            candidate.get("retrieval_channel_count") or len(ranks)
        ),
        "channel_ranks": ranks,
    }


def _annotate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(candidate)
    ranks = set((output.get("v41_channel_ranks") or {}).keys())
    output["candidate_from_sparse"] = "sparse_active" in ranks
    output["candidate_from_input_siret"] = "input_siret_active" in ranks
    output["candidate_from_input_siren"] = "input_siren_active_sites" in ranks
    output["candidate_from_closed_alias"] = bool(
        {"closed_alias_name", "closed_alias_address"} & ranks
    )
    output["is_direct_candidate"] = bool(
        output["candidate_from_input_siret"]
        or output["candidate_from_input_siren"]
    )
    output["has_direct_evidence"] = output["is_direct_candidate"]
    tiers = []
    for enabled, name in (
        (output["candidate_from_input_siret"], "INPUT_SIRET"),
        (output["candidate_from_input_siren"], "INPUT_SIREN"),
        (output["candidate_from_closed_alias"], "CLOSED_ALIAS"),
        (output["candidate_from_sparse"], "SPARSE"),
    ):
        if enabled:
            tiers.append(name)
    output["evidence_tier"] = "+".join(tiers) if tiers else "RETRIEVAL"
    return output


def _candidate_evidence(
    *,
    service_id: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    names = [
        {
            "text": name.text,
            "source": name.source.value,
            "is_ul_name": bool(name.is_ul_name),
            "is_sigle": bool(name.is_sigle),
        }
        for name in build_candidate_names(dict(candidate))
    ]
    return {
        "service_id": service_id,
        "candidate_siret": str(candidate.get("siret") or ""),
        "candidate_siren": str(
            candidate.get("siren")
            or str(candidate.get("siret") or "")[:9]
        ),
        "rank": int(candidate["rank"]),
        "ranker_score": float(candidate["score"]),
        "etat_admin": str(candidate.get("etat_admin") or "").upper(),
        "candidate_names_json": json.dumps(names, ensure_ascii=False),
        "candidate_address": build_address(dict(candidate)),
        "postcode": str(candidate.get("postcode") or ""),
        "city": str(candidate.get("city") or ""),
        "insee": str(candidate.get("insee") or ""),
        "evidence_tier": str(candidate.get("evidence_tier") or ""),
        "retrieval_source": str(candidate.get("retrieval_source") or ""),
        "retrieval_channel_count": int(
            candidate.get("retrieval_channel_count") or 0
        ),
        "retrieval_provenance_json": json.dumps(
            _candidate_provenance(candidate),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def score_one_query(
    *,
    row: Mapping[str, Any],
    retriever: Any,
    ranker: Any,
    ranker_feature_order: Sequence[str],
    acceptor: V41RawLogisticAcceptor,
    persistent_cache: Any,
    run_id: str,
    input_qualification: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Retrieve, rank, build the scene and decide one authorized CRM row."""

    started = time.perf_counter()
    service_id = str(row["service_id"])
    crm_row = {
        "query_id": service_id,
        "crm_id": service_id,
        "crm_name": _first_text(row.get("SITE"), row.get("crm_name")),
        "crm_address": _first_text(
            row.get("SITE_CLI_ADRESSE"), row.get("crm_address")
        ),
        "crm_city": _first_text(
            row.get("COMMUNE"),
            row.get("SITE_CLI_COMMUNE"),
            row.get("crm_city"),
        ),
        "postcode": _first_text(row.get("CODE_POSTAL"), row.get("postcode")),
        "insee": _first_text(row.get("CODE_INSEE"), row.get("insee")),
    }
    retrieval_kwargs = {
        "crm_row": crm_row,
        "crm_pre": preprocess_crm_row(crm_row),
        "input_siret": _first_text(row.get("SIRET"), row.get("input_siret")),
        "gt_siret": None,
        "persistent_cache": persistent_cache,
    }
    if input_qualification is not None:
        retrieval_kwargs["input_qualification"] = input_qualification
    result = retriever.build(**retrieval_kwargs)
    if result.sparse_result.gt_was_injected:
        raise ValueError("Positive injection is forbidden in shadow inference")
    candidates = [_annotate_candidate(candidate) for candidate in result.candidates]
    if len(candidates) > 100:
        raise ValueError("Shadow retrieval exceeded 100 candidates")
    if len({str(candidate.get("siret")) for candidate in candidates}) != len(
        candidates
    ):
        raise ValueError("Shadow retrieval returned duplicate SIRETs")
    if any(
        str(candidate.get("etat_admin") or "").upper() != "A"
        for candidate in candidates
    ):
        raise ValueError("Shadow retrieval returned a non-active candidate")

    set_global_name_idf_map(
        result.sparse_result.idf_map,
        result.sparse_result.default_idf,
    )
    crm_pre = preprocess_crm_row(crm_row)
    legacy_rows = make_feature_rows_from_preprocessed(
        crm_pre,
        candidates,
        include_semantic=False,
    )
    feature_rows: list[dict[str, float]] = []
    for candidate, legacy_row in zip(candidates, legacy_rows, strict=True):
        feature_rows.append(
            {
                **build_legacy_55_features(legacy_row, candidate),
                **build_v41_candidate_features(
                    candidate,
                    input_siret=result.input_siret.normalized_siret,
                ),
            }
        )
    if feature_rows:
        matrix = pd.DataFrame(feature_rows)[list(ranker_feature_order)].astype(float)
        scores = np.asarray(ranker.predict(matrix.to_numpy()), dtype=float)
    else:
        scores = np.asarray([], dtype=float)
    if len(scores) != len(candidates):
        raise ValueError("Ranker output cardinality differs from candidates")
    ranked_pairs = sorted(
        zip(candidates, feature_rows, scores, strict=True),
        key=lambda item: (-float(item[2]), str(item[0].get("siret") or "")),
    )
    ranked: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for rank, (candidate, features, score) in enumerate(ranked_pairs, start=1):
        candidate = {
            **candidate,
            **features,
            "candidate_siret": str(candidate.get("siret") or ""),
            "candidate_siren": str(
                candidate.get("siren")
                or str(candidate.get("siret") or "")[:9]
            ),
            "candidate_state": str(
                candidate.get("etat_admin") or ""
            ).upper(),
            "rank": rank,
            "score": float(score),
        }
        ranked.append(candidate)
        prediction_rows.append(
            {
                "query_id": service_id,
                **candidate,
                "rrf_score": float(candidate.get("rrf_score") or 0.0),
                "retrieval_agreement": int(
                    int(candidate.get("retrieval_channel_count") or 0) >= 2
                ),
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    scene = build_inference_scene(service_id, predictions)
    decision = decide_v41(
        query_id=service_id,
        input_siret=result.input_siret.normalized_siret,
        input_siret_state=result.input_siret.state.value,
        candidates=ranked,
        scene=scene,
        acceptor=acceptor,
        shadow_run_id=run_id,
    ).to_dict()
    decision["service_id"] = decision.pop("crm_id")
    decision["release_id"] = None
    decision["latency_ms"] = (time.perf_counter() - started) * 1000.0

    top10 = [
        _candidate_evidence(service_id=service_id, candidate=candidate)
        for candidate in ranked[:10]
    ]
    evidence = {
        "service_id": service_id,
        "crm": {
            "name": crm_row["crm_name"],
            "address": crm_row["crm_address"],
            "city": crm_row["crm_city"],
            "postcode": crm_row["postcode"],
            "insee": crm_row["insee"],
            "input_siret": result.input_siret.normalized_siret,
            "input_siret_state": result.input_siret.state.value,
        },
        "decision": {
            "decision": decision["decision"],
            "predicted_siret": decision["predicted_siret"],
            "confidence": decision["confidence"],
            "review_reason": decision["review_reason"],
        },
        "top_candidates": top10,
    }
    return decision, top10, evidence


def run_shadow(
    *,
    source_path: Path,
    inventory_path: Path,
    inventory_manifest_path: Path,
    panel_path: Path,
    runtime: V41RuntimeBundle,
    retriever: Any,
    persistent_cache: Any,
    output_root: Path,
    checkpoint_path: Path,
    run_id: str,
    input_artifacts: Mapping[str, Path],
    expected_source_sha256: str | None = None,
    expected_decision_count: int | None = None,
    expected_panel_count: int | None = None,
) -> Path:
    """Run or safely resume the complete shadow population."""

    source_hash = file_sha256(source_path)
    if expected_source_sha256 and source_hash != expected_source_sha256:
        raise ValueError("CRM source hash differs from the authorized source")
    inventory_manifest = validate_inventory_chain(
        source_path=source_path,
        inventory_path=inventory_path,
        panel_path=panel_path,
        inventory_manifest_path=inventory_manifest_path,
    )
    source = read_table(source_path)
    inventory = read_table(inventory_path)
    panel = read_table(panel_path)
    scoreable = select_scoreable_rows(source, inventory)
    eligible_ids = scoreable["service_id"].astype(str).tolist()
    if len(eligible_ids) != len(set(eligible_ids)):
        raise ValueError("Eligible shadow SERVICE IDs must be unique")
    if source_hash == CURRENT_SOURCE_SHA256:
        if len(eligible_ids) != CURRENT_EXPECTED_DECISION_COUNT:
            raise ValueError(
                "Current CRM source must contain exactly "
                f"{CURRENT_EXPECTED_DECISION_COUNT} eligible rows"
            )
        if expected_decision_count not in {
            None,
            CURRENT_EXPECTED_DECISION_COUNT,
        }:
            raise ValueError("Expected decision count contradicts current source")
        expected_decision_count = CURRENT_EXPECTED_DECISION_COUNT
        expected_panel_count = (
            CURRENT_EXPECTED_PANEL_COUNT
            if expected_panel_count is None
            else expected_panel_count
        )
    expected_decision_count = (
        len(eligible_ids)
        if expected_decision_count is None
        else expected_decision_count
    )
    if len(eligible_ids) != expected_decision_count:
        raise ValueError(
            f"Eligible decision population {len(eligible_ids)} "
            f"!= expected {expected_decision_count}"
        )
    if expected_panel_count is not None and len(panel) != expected_panel_count:
        raise ValueError(
            f"Panel row count {len(panel)} != expected {expected_panel_count}"
        )
    panel_ids = set(panel["service_id"].astype(str))
    if len(panel_ids) != len(panel) or not panel_ids <= set(eligible_ids):
        raise ValueError("Panel must contain unique eligible SERVICE IDs only")

    identity = {
        "run_id": run_id,
        "release_id": runtime.release.release_id,
        "retrieval_signature": runtime.release.retrieval_signature,
        "source_sha256": source_hash,
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_manifest_sha256": file_sha256(inventory_manifest_path),
        "panel_sha256": file_sha256(panel_path),
        "denylist_sha256": {
            str(entry["name"]): str(entry["sha256"])
            for entry in inventory_manifest["denylists"]
        },
        "model_artifact_sha256": dict(runtime.artifact_hashes or {}),
        "code_commit": _current_code_commit(),
        "runner_sha256": file_sha256(Path(__file__)),
        "eligible_ids_sha256": hashlib.sha256(
            "\n".join(eligible_ids).encode("utf-8")
        ).hexdigest(),
    }
    checkpoint = ShadowCheckpoint(checkpoint_path, identity=identity)
    try:
        completed = checkpoint.service_ids()
        if not completed <= set(eligible_ids):
            raise ValueError("Checkpoint contains excluded or unknown SERVICE IDs")

        inventory_by_id = (
            inventory.dropna(subset=["service_id"])
            .assign(service_id=lambda frame: frame["service_id"].astype(str))
            .set_index("service_id")
        )
        qualifications: dict[str, Any] = {}
        if hasattr(retriever, "global_store") and hasattr(
            retriever.global_store, "qualify_input_sirets"
        ):
            scoreable_records = scoreable.to_dict("records")
            runtime_qualifications = retriever.global_store.qualify_input_sirets(
                [
                    _first_text(row.get("SIRET"), row.get("input_siret"))
                    for row in scoreable_records
                ]
            )
            qualifications = {
                service_id: reconcile_inventory_qualification(
                    inventory_row=inventory_by_id.loc[service_id],
                    runtime=runtime_qualification,
                )
                for service_id, runtime_qualification in zip(
                    eligible_ids,
                    runtime_qualifications,
                    strict=True,
                )
            }
        for row in scoreable.to_dict("records"):
            service_id = str(row["service_id"])
            if service_id in completed:
                continue
            decision, top10, evidence = score_one_query(
                row=row,
                retriever=retriever,
                ranker=runtime.ranker,
                ranker_feature_order=runtime.release.ranker_feature_order,
                acceptor=runtime.acceptor,
                persistent_cache=persistent_cache,
                run_id=run_id,
                input_qualification=qualifications.get(service_id),
            )
            inventory_state = (
                _first_text(
                    inventory_by_id.loc[service_id].get("input_siret_state"),
                    "UNKNOWN",
                ).upper()
            )
            runtime_state = str(decision["input_siret_state"]).upper()
            if inventory_state not in {"", "UNKNOWN"} and (
                inventory_state != runtime_state
            ):
                raise ValueError(
                    f"Input SIRET state drift for {service_id}: "
                    f"inventory={inventory_state}, runtime={runtime_state}"
                )
            decision["release_id"] = runtime.release.release_id
            checkpoint.put(
                service_id=service_id,
                decision=decision,
                candidates=top10,
                evidence=evidence,
            )

        decisions = []
        top10_rows = []
        evidence_records = []
        for service_id in eligible_ids:
            decision, candidates, evidence = checkpoint.get(service_id)
            decisions.append(decision)
            top10_rows.extend(candidates)
            evidence_records.append(evidence)
        if len(decisions) != expected_decision_count:
            raise ValueError("Checkpoint does not cover the full eligible population")
        result = write_shadow_run(
            output_root=output_root,
            run_id=run_id,
            inventory=inventory,
            decisions=pd.DataFrame(decisions),
            candidates_top10=pd.DataFrame(top10_rows),
            evidence=evidence_records,
            panel=panel,
            input_artifacts=input_artifacts,
            run_metadata={
                "release_id": runtime.release.release_id,
                "retrieval_signature": runtime.release.retrieval_signature,
                "ranker_bundle_id": runtime.release.ranker_bundle_id,
                "acceptor_bundle_id": runtime.release.acceptor_bundle_id,
                "ranker_variant": runtime.release.ranker_variant,
                "retrieval_variant": runtime.retrieval_config.variant.value,
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "crm_writes_performed": False,
            },
        )
        checkpoint.mark_complete(result)
        return result
    finally:
        checkpoint.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--retrieval-gate-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--global-store", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--expected-source-sha256",
        default=CURRENT_SOURCE_SHA256,
    )
    parser.add_argument(
        "--expected-decision-count",
        type=int,
        default=CURRENT_EXPECTED_DECISION_COUNT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partitions_signature = _path_signature(args.partitions)
    global_store_signature = _path_signature(args.global_store)
    runtime = load_runtime_bundle(
        dataset_dir=args.dataset,
        model_dir=args.model_bundle,
        partitions_signature=partitions_signature,
        global_store_signature=global_store_signature,
        retrieval_gate_manifest_path=args.retrieval_gate_manifest,
    )
    cache = TfidfPersistentCache(
        config_hash=runtime.retrieval_config.sparse_config().tfidf_artifact_hash(),
        cache_dir=args.cache_dir,
    )
    store = PartitionedCandidateStore(args.partitions)
    with V41GlobalCandidateStore(args.global_store) as global_store:
        retriever = V41CandidateRetriever(
            partitioned_store=store,
            global_store=global_store,
            config=runtime.retrieval_config,
        )
        input_artifacts = {
            "crm_source": args.source,
            "inventory": args.inventory,
            "inventory_manifest": args.inventory_manifest,
            "panel": args.panel,
            "dataset_manifest": args.dataset / "manifest.json",
            "retrieval_gate_manifest": args.retrieval_gate_manifest,
            "release_manifest": args.model_bundle / "release_manifest.json",
            "model_manifest": args.model_bundle / "model_manifest.json",
            "ranker_metadata": args.model_bundle / "ranker" / "metadata.json",
            "ranker_model": args.model_bundle / "ranker" / "ranker.json",
            "acceptor_metadata": args.model_bundle / "acceptor" / "metadata.json",
            "acceptor_model": (
                args.model_bundle / "acceptor" / "acceptor_model.joblib"
            ),
        }
        result = run_shadow(
            source_path=args.source,
            inventory_path=args.inventory,
            inventory_manifest_path=args.inventory_manifest,
            panel_path=args.panel,
            runtime=runtime,
            retriever=retriever,
            persistent_cache=cache,
            output_root=args.output_root,
            checkpoint_path=args.checkpoint_dir / f"{args.run_id}.sqlite",
            run_id=args.run_id,
            input_artifacts=input_artifacts,
            expected_source_sha256=args.expected_source_sha256,
            expected_decision_count=args.expected_decision_count,
            expected_panel_count=CURRENT_EXPECTED_PANEL_COUNT,
        )
    print(result)


if __name__ == "__main__":
    main()
