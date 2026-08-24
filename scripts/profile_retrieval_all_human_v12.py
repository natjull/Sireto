#!/usr/bin/env python3
"""Produce a sealed warm end-to-end runtime profile for retrieval v12.

This command deliberately replays the serving path.  It does not time a
precomputed Master parquet and it never reads a human SIRET.  For every query
in the sealed 512-query sidecar it runs official/v11 candidate generation,
the nested-OOF v12 parent cascade for that query's outer fold, complete-union
feature computation, the corresponding OOF site LambdaMART and adaptive
top-100 selection.  The measured output must be byte-identical to the sealed
OOF selection before latency can be published as gate eligible.

The default path refuses any machine other than an Apple M4 Pro.  The explicit
``--allow-non-m4-test-fixture`` escape hatch exists only for stub fixtures: it
marks the artifact non-gate-eligible and therefore makes the admission
validator reject it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieve_all_human_v11_union import LABEL_COLUMNS  # noqa: E402
from scripts.run_retrieval_hierarchical_admission_v12 import (  # noqa: E402
    _validate_label_ledger_manifest_contract,
)
from scripts.run_retrieval_top100_cascade_v12 import (  # noqa: E402
    MANIFEST_SCHEMA as CASCADE_MANIFEST_SCHEMA,
    _load_parent_model_recipe,
    _validate_source_manifest_contract,
    sha256_file,
)
from src.xgb_matcher.official_evidence_retrieval import (  # noqa: E402
    OfficialEvidenceRetrievalConfig,
    OfficialEvidenceRetriever,
)
from src.xgb_matcher.retrieval_all_human_v11 import (  # noqa: E402
    AllHumanV11Retriever,
    RetrievalV11Config,
)
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (  # noqa: E402
    AdmissionConfigV12,
    HierarchicalModelsV12,
    canonicalize_union,
    score_canonical_candidates,
    select_top100,
)
from src.xgb_matcher.retrieval_sidecars_v11 import RetrievalSidecarReader  # noqa: E402
from src.xgb_matcher.retrieval_top100_cascade_v12 import (  # noqa: E402
    RetrievalTop100CascadeV12,
    RetrievalTop100CascadeV12Config,
)


SCHEMA_VERSION = "sireto-retrieval-all-human-v12-runtime-profile-1"
QUERY_SOURCE_MANIFEST_SCHEMA = (
    "sireto-retrieval-v12-label-blind-query-projection-1"
)
SAMPLE_SELECTION_RULE = "SHA256_QUERY_ID_ASC_FIRST_512"
SAMPLE_ROWS = 512
ALLOWED_FOLDS = frozenset({2, 3, 4})
PARITY_COLUMNS = ["query_id", "candidate_siret", "selected_rank"]
PARITY_SIDECAR_COLUMNS = [
    "query_id",
    "fold",
    "reference_sha256",
    "replay_sha256",
    "byte_identical",
]
REQUIRED_STAGES = [
    "candidate_generation",
    "cascade",
    "canonicalization",
    "feature_computation",
    "site_lambdamart",
    "top100_selection",
]
DEFAULT_CONFIG = ROOT / "config/retrieval_all_human_v12.json"
LABEL_LEDGER_SCHEMA = "sireto-retrieval-all-human-v12-label-ledger-1"

_EXTRA_TRUTH_COLUMNS = frozenset(
    {
        "target_siret",
        "matched_siret",
        "expected_siret",
        "truth_siret",
        "label_siret",
        "is_positive",
        "site_label",
        "parent_label",
    }
)


def _is_truth_column(name: str) -> bool:
    normalised = str(name).strip().lower()
    return (
        normalised in LABEL_COLUMNS
        or normalised in _EXTRA_TRUTH_COLUMNS
        or "ground_truth" in normalised
        or normalised.startswith("human_siret")
        or normalised.startswith("human_ground_truth")
        or normalised.startswith("acceptable_siret")
    )


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _contract_path(config_path: Path) -> Path:
    raw = _json(config_path)
    declared = Path(str(raw.get("contract_path") or ""))
    if not declared.is_absolute():
        declared = ROOT / declared
    return _require_file(declared, "v12 contract")


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("runtime replay produced no latency values")
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def selection_bytes(rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> bytes:
    """Canonical bytes for ``candidate_siret + selected_rank`` in rank order."""

    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if frame.empty:
        return b""
    missing = set(PARITY_COLUMNS[1:]).difference(frame.columns)
    if missing:
        raise ValueError(f"selection is missing columns: {sorted(missing)}")
    frame = frame[["candidate_siret", "selected_rank"]].copy()
    frame["candidate_siret"] = frame["candidate_siret"].astype(str)
    frame["selected_rank"] = pd.to_numeric(
        frame["selected_rank"], errors="raise"
    ).astype(int)
    if (
        frame["candidate_siret"].str.fullmatch(r"[0-9]{14}").ne(True).any()
        or frame["candidate_siret"].duplicated().any()
        or frame["selected_rank"].duplicated().any()
        or len(frame) > 100
    ):
        raise ValueError("invalid v12 top100 selection")
    frame = frame.sort_values(["selected_rank", "candidate_siret"], kind="stable")
    if frame["selected_rank"].tolist() != list(range(1, len(frame) + 1)):
        raise ValueError("selected_rank must be contiguous and one-based")
    return b"".join(
        f"{row.candidate_siret}\t{int(row.selected_rank)}\n".encode("ascii")
        for row in frame.itertuples(index=False)
    )


def selection_sha256(rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(selection_bytes(rows)).hexdigest()


def parity_aggregate_digest(rows: pd.DataFrame, column: str) -> str:
    """Mirror the admission validator's aggregate parity digest exactly."""

    digest = hashlib.sha256()
    for row in rows.sort_values(["query_id", "fold"], kind="stable").itertuples(
        index=False
    ):
        digest.update(str(row.query_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(row.fold)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(getattr(row, column)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_query_sample(path: Path) -> pd.DataFrame:
    sample = pd.read_parquet(_require_file(path, "runtime query sample"))
    if list(sample.columns) != ["query_id", "fold"] or len(sample) != SAMPLE_ROWS:
        raise ValueError(
            "runtime query sample must contain exactly query_id/fold and 512 rows"
        )
    sample = sample.copy()
    sample["query_id"] = sample["query_id"].astype(str)
    sample["fold"] = pd.to_numeric(sample["fold"], errors="raise").astype(int)
    if (
        sample["query_id"].eq("").any()
        or sample["query_id"].duplicated().any()
        or set(sample["fold"]) != ALLOWED_FOLDS
    ):
        raise ValueError("runtime query sample may contain only unique folds 2/3/4")
    keys = sample["query_id"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    expected_order = (
        sample.assign(__sample_key=keys)
        .sort_values(["__sample_key", "query_id"], kind="stable")
        .index.tolist()
    )
    if expected_order != list(sample.index):
        raise ValueError(
            "runtime query sample is not ordered by SHA256(query_id), query_id"
        )
    return sample.reset_index(drop=True)


def _validate_label_ledger_sample_manifest(
    path: Path, *, sample_path: Path, config_path: Path
) -> Mapping[str, Any]:
    """Reuse the strict ledger contract, without reading human-label rows."""

    manifest_path = _require_file(path, "v12 label-ledger manifest")
    manifest = _json(manifest_path)
    artifacts = manifest.get("artifacts")
    ledger = artifacts.get("label_ledger") if isinstance(artifacts, Mapping) else None
    ledger_name = ledger.get("path") if isinstance(ledger, Mapping) else None
    if not isinstance(ledger_name, str) or Path(ledger_name).name != ledger_name:
        raise ValueError("runtime sample manifest lacks its sealed label-ledger path")
    resolved = _validate_label_ledger_manifest_contract(
        manifest=manifest,
        manifest_path=manifest_path,
        ledger_path=manifest_path.parent / ledger_name,
        config_path=config_path,
    )
    declared_path = resolved["runtime_query_sample"]
    actual_sample = _require_file(sample_path, "runtime query sample")
    if declared_path != actual_sample:
        raise ValueError("runtime query sample does not match label-ledger artifact")
    return manifest


def _load_label_blind_queries(path: Path, sample: pd.DataFrame) -> dict[str, dict[str, Any]]:
    source = _require_file(path, "label-blind query source")
    parquet = pq.ParquetFile(source)
    forbidden = sorted(name for name in parquet.schema_arrow.names if _is_truth_column(name))
    if forbidden:
        raise ValueError(
            "runtime query source contains human-label columns: " + ", ".join(forbidden)
        )
    required = {"query_id"}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError("runtime query source has no query_id")
    wanted = set(sample["query_id"])
    output: dict[str, dict[str, Any]] = {}
    for batch in parquet.iter_batches(batch_size=2_048):
        for raw in batch.to_pylist():
            query_id = str(raw.get("query_id") or "")
            if query_id not in wanted:
                continue
            if query_id in output:
                raise ValueError(f"runtime query source repeats query_id {query_id}")
            # Match the v11 generator's routing precedence exactly.
            fold = int(raw.get("oof_fold", raw.get("fold", -1)))
            if fold != int(sample.loc[sample["query_id"].eq(query_id), "fold"].iloc[0]):
                raise ValueError(f"runtime query/source fold mismatch for {query_id}")
            output[query_id] = dict(raw)
    missing = sorted(wanted.difference(output))
    if missing:
        raise ValueError(f"runtime query source misses {len(missing)} sampled queries")
    return output


def _load_reference_selections(
    selected_paths: Mapping[int, Path], sample: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    wanted = set(sample["query_id"])
    records: dict[str, list[dict[str, Any]]] = {query_id: [] for query_id in wanted}
    for fold, path in sorted(selected_paths.items()):
        parquet = pq.ParquetFile(_require_file(path, f"fold{fold} selected output"))
        columns = set(parquet.schema_arrow.names)
        required = {"query_id", "fold", "candidate_siret", "selected_rank"}
        if not required.issubset(columns):
            raise ValueError(f"fold{fold} selected output lacks parity columns")
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=8_192):
            for raw in batch.to_pylist():
                query_id = str(raw.get("query_id") or "")
                if query_id not in wanted:
                    continue
                if int(raw.get("fold", -1)) != fold:
                    raise ValueError("sealed selected output mixes outer folds")
                records[query_id].append(
                    {
                        "candidate_siret": str(raw.get("candidate_siret") or ""),
                        "selected_rank": int(raw.get("selected_rank")),
                    }
                )
    return {query_id: pd.DataFrame(rows) for query_id, rows in records.items()}


_APPLE_CHIP_LINE_RE = re.compile(
    r"^\s*(?:Chip|Puce)\s*:\s*(Apple M[0-9]+(?: (?:Pro|Max|Ultra))?)\s*$",
    re.IGNORECASE,
)


def _system_profiler_chip() -> str:
    """Return only the Apple chip name, never raw hardware-identifying output."""

    executable = shutil.which("system_profiler") or "/usr/sbin/system_profiler"
    try:
        completed = subprocess.run(
            [executable, "SPHardwareDataType", "-detailLevel", "mini"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in completed.stdout.splitlines():
        match = _APPLE_CHIP_LINE_RE.fullmatch(line)
        if match:
            return match.group(1)
    return ""


def _sysctl_chip() -> str:
    """Fallback for macOS versions exposing an Apple chip as a CPU brand."""

    try:
        value = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return value if re.fullmatch(r"Apple M[0-9]+(?: (?:Pro|Max|Ultra))?", value) else ""


def _hardware_identity() -> dict[str, str]:
    chip = ""
    if platform.system() == "Darwin":
        chip = _system_profiler_chip() or _sysctl_chip() or platform.processor().strip()
    return {
        "computer": "Mac" if platform.system() == "Darwin" else platform.system(),
        "chip": chip,
        "os": platform.platform(),
        "python": platform.python_version(),
        "xgboost": str(xgb.__version__),
    }


def _is_target_hardware(hardware: Mapping[str, str]) -> bool:
    return hardware.get("computer") == "Mac" and hardware.get("chip") == "Apple M4 Pro"


def _index_manifest_hashes(
    paths: Path | Sequence[Path] | None,
) -> list[str]:
    if paths is None:
        return []
    values = [paths] if isinstance(paths, Path) else list(paths)
    output: list[str] = []
    for path in values:
        directory = _require_directory(path, "official index")
        manifest = _require_file(
            directory / "manifest.json", "official index manifest"
        )
        output.append(sha256_file(manifest))
    return output


def _validate_query_source_manifest(
    path: Path,
    *,
    query_path: Path,
    query_columns: Sequence[str],
    query_rows: int,
    v11_candidate_manifest_path: Path,
    v11_candidate_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind the label-free projection to the exact source used by v11.

    The runtime process never opens that truth-bearing source.  Projection is
    a prior, separately sealed preparation step; its manifest attests that
    only columns were removed and ties its input hash to the v11 candidate
    manifest.  Altered query text therefore cannot be silently substituted in
    the timing run.
    """

    manifest_path = _require_file(path, "label-blind query projection manifest")
    value = _json(manifest_path)
    required = {
        "schema_version": QUERY_SOURCE_MANIFEST_SCHEMA,
        "run_kind": "LABEL_BLIND_QUERY_PROJECTION",
        "projection_operation": "ARROW_ALLOWLIST_AND_FOLDS_234_FILTER_ONLY",
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "labels_present": False,
        "positive_injection": False,
        "query_id_unique": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("query projection manifest is not a sealed label-blind folds-2/3/4 source")
    if (
        value.get("query_parquet_sha256") != sha256_file(query_path)
        or value.get("rows") != int(query_rows)
        or value.get("projection_columns") != list(query_columns)
        or value.get("source_input_sha256")
        != v11_candidate_manifest.get("source_input_sha256")
        or value.get("v11_candidate_manifest_sha256")
        != sha256_file(v11_candidate_manifest_path)
    ):
        raise ValueError("query projection lineage/content does not match sealed v11 generation")
    producer_path = ROOT / "scripts/build_retrieval_all_human_v12_runtime_queries.py"
    projection_code_sha = value.get("projection_code_sha256")
    if projection_code_sha != sha256_file(_require_file(producer_path, "query projection producer")):
        raise ValueError("query projection producer hash does not match this runtime")
    return value


@dataclass(frozen=True)
class SealedOOFInputs:
    oof_manifest_path: Path
    oof_manifest: Mapping[str, Any]
    cascade_manifests: Mapping[int, Path]
    selected_paths: Mapping[int, Path]
    site_model_paths: Mapping[int, Path]
    lineage: Mapping[str, Any]


def _candidate_path_for_v11_manifest(manifest_path: Path) -> Path:
    suffix = ".manifest.json"
    raw = str(manifest_path.resolve())
    if not raw.endswith(suffix):
        raise ValueError("v11 candidate manifest must be the candidate parquet sidecar")
    return _require_file(Path(raw[: -len(suffix)]), "v11 candidate parquet")


def _validate_oof_label_ledger_lineage(
    sealed: SealedOOFInputs,
    ledger_manifest: Mapping[str, Any],
    ledger_manifest_path: Path,
) -> None:
    """Bind runtime replay to the exact label population used for OOF fit."""

    expected_ledger_sha = ledger_manifest.get("label_ledger_sha256")
    expected_manifest_sha = sha256_file(
        _require_file(ledger_manifest_path, "v12 label-ledger manifest")
    )
    if (
        not isinstance(expected_ledger_sha, str)
        or sealed.oof_manifest.get("source_labels_sha256")
        != expected_ledger_sha
        or sealed.oof_manifest.get("source_labels_manifest_sha256")
        != expected_manifest_sha
    ):
        raise ValueError(
            "sealed OOF was fitted/evaluated on another label-ledger snapshot"
        )


def _validate_oof_inputs(args: argparse.Namespace) -> SealedOOFInputs:
    config_path = _require_file(args.config, "v12 config")
    config_sha = sha256_file(config_path)
    oof_dir = _require_directory(args.oof_dir, "sealed v12 OOF directory")
    oof_manifest_path = _require_file(oof_dir / "manifest.json", "v12 OOF manifest")
    oof = _json(oof_manifest_path)
    if (
        oof.get("run_kind") != "NESTED_OUTER_OOF"
        or oof.get("config_sha256") != config_sha
        or oof.get("positive_injection") is not False
        or oof.get("fold0_opened") is not False
        or oof.get("fold1_opened") is not False
        or set(oof.get("opened_folds") or ()) != ALLOWED_FOLDS
    ):
        raise ValueError("runtime profiler requires the exact sealed folds-2/3/4 OOF run")
    outputs = oof.get("outputs")
    outer = oof.get("outer_cascades")
    if not isinstance(outputs, Mapping) or not isinstance(outer, Mapping):
        raise ValueError("sealed OOF manifest lacks output/cascade lineage")

    cascade_dirs = {
        2: args.outer2_cascade,
        3: args.outer3_cascade,
        4: args.outer4_cascade,
    }
    parent_bundle = _require_file(args.parent_model_bundle, "nested parent bundle")
    parent_bundle_sha = sha256_file(parent_bundle)
    candidate_identity: dict[str, set[Any]] = {
        key: set()
        for key in (
            "source_candidate_sha256",
            "source_manifest_sha256",
            "sidecar_manifest_sha256",
            "parent_model_bundle_sha256",
        )
    }
    cascade_manifests: dict[int, Path] = {}
    selected_paths: dict[int, Path] = {}
    model_paths: dict[int, Path] = {}
    lineage_cascades: dict[str, Any] = {}
    lineage_models: dict[str, Any] = {}
    lineage_selected: dict[str, Any] = {}
    for fold, raw_directory in sorted(cascade_dirs.items()):
        directory = _require_directory(raw_directory, f"outer{fold} cascade")
        manifest_path = _require_file(directory / "manifest.json", f"outer{fold} manifest")
        manifest = _json(manifest_path)
        master = _require_file(
            directory / "master_references.parquet", f"outer{fold} Master"
        )
        parent_candidates = _require_file(
            directory / "parent_candidates.parquet", f"outer{fold} parents"
        )
        declared = outer.get(str(fold))
        if not isinstance(declared, Mapping) or (
            declared.get("manifest_sha256") != sha256_file(manifest_path)
            or declared.get("master_sha256") != sha256_file(master)
            or declared.get("parent_candidates_sha256") != sha256_file(parent_candidates)
        ):
            raise ValueError(f"outer{fold} cascade does not match sealed OOF lineage")
        if (
            manifest.get("schema_version") != CASCADE_MANIFEST_SCHEMA
            or manifest.get("run_kind") != "LABEL_BLIND_TOP100_CASCADE"
            or manifest.get("config_sha256") != config_sha
            or manifest.get("gate_eligible") is not True
            or manifest.get("outer_heldout_fold") != fold
            or manifest.get("fold0_opened") is not False
            or manifest.get("fold1_opened") is not False
            or manifest.get("positive_injection") is not False
            or manifest.get("human_labels_opened") is not False
            or set(manifest.get("opened_folds") or ()) != ALLOWED_FOLDS
            or manifest.get("parent_model_bundle_sha256") != parent_bundle_sha
        ):
            raise ValueError(f"outer{fold} is not a gate-eligible label-blind nested cascade")
        for key in candidate_identity:
            candidate_identity[key].add(manifest.get(key))
        selected = _require_file(
            oof_dir / f"fold{fold}_selected_candidates.parquet",
            f"fold{fold} selected output",
        )
        model = _require_file(
            oof_dir / f"fold{fold}_site_ranker.json", f"fold{fold} site model"
        )
        if (
            outputs.get(selected.name) != sha256_file(selected)
            or outputs.get(model.name) != sha256_file(model)
        ):
            raise ValueError(f"fold{fold} OOF model/selection hash mismatch")
        cascade_manifests[fold] = manifest_path
        selected_paths[fold] = selected
        model_paths[fold] = model
        lineage_cascades[str(fold)] = {
            "manifest_sha256": sha256_file(manifest_path),
            "master_sha256": sha256_file(master),
        }
        lineage_models[str(fold)] = {"site_model_sha256": sha256_file(model)}
        lineage_selected[str(fold)] = {
            "selected_output_sha256": sha256_file(selected)
        }
    if any(len(values) != 1 or None in values for values in candidate_identity.values()):
        raise ValueError("outer cascades do not share one sealed retrieval lineage")

    source_manifest = _require_file(args.v11_candidate_manifest, "v11 candidate manifest")
    source_value = _json(source_manifest)
    _validate_source_manifest_contract(
        _candidate_path_for_v11_manifest(source_manifest), source_value
    )
    only_source_manifest_sha = next(iter(candidate_identity["source_manifest_sha256"]))
    if (
        sha256_file(source_manifest) != only_source_manifest_sha
        or source_value.get("candidate_parquet_sha256")
        != next(iter(candidate_identity["source_candidate_sha256"]))
        or source_value.get("positive_injection") is not False
        or source_value.get("human_labels_opened_by_retrieval") is not False
        or set(source_value.get("folds") or ()) != ALLOWED_FOLDS
        or source_value.get("config_sha256") != sha256_file(args.v11_config)
        or source_value.get("official_config_sha256") != sha256_file(args.official_config)
        or source_value.get("sidecar_manifest_sha256")
        != sha256_file(_require_file(args.sidecars / "manifest.json", "sidecar manifest"))
    ):
        raise ValueError("live v11 inputs do not match the sealed candidate-generation manifest")

    outcomes = _require_file(oof_dir / "query_outcomes.parquet", "OOF query outcomes")
    if outputs.get(outcomes.name) != sha256_file(outcomes):
        raise ValueError("OOF query outcomes hash mismatch")
    lineage = {
        "outer_cascades": lineage_cascades,
        "site_models": lineage_models,
        "selected_outputs": lineage_selected,
        "oof_manifest_sha256": sha256_file(oof_manifest_path),
        "query_outcomes_sha256": sha256_file(outcomes),
    }
    return SealedOOFInputs(
        oof_manifest_path=oof_manifest_path,
        oof_manifest=oof,
        cascade_manifests=cascade_manifests,
        selected_paths=selected_paths,
        site_model_paths=model_paths,
        lineage=lineage,
    )


class RuntimePipeline:
    """The real label-blind serving stack kept open across warm/replay passes."""

    def __init__(self, args: argparse.Namespace, sealed: SealedOOFInputs) -> None:
        self.args = args
        self.admission_config = AdmissionConfigV12.load(args.config)
        self.cascade_config = RetrievalTop100CascadeV12Config.load(args.config)
        v11_config = RetrievalV11Config.load(args.v11_config)
        official_config = OfficialEvidenceRetrievalConfig.load(args.official_config)
        official_config = replace(
            official_config,
            enable_direct_parent_expansion=False,
            enable_consensus_parent_expansion=False,
            relation_seed_limit=v11_config.relation_seed_limit,
            search_workers=int(args.search_workers),
        )
        self.official = OfficialEvidenceRetriever.from_index_paths(
            args.base_index,
            args.overlay_index,
            official_config,
            temporal_overlay_index_path=args.temporal_overlay_index,
            legal_entity_role_index_path=args.legal_entity_role_index,
            sector_official_index_path=args.sector_official_index,
        )
        self.sidecars = RetrievalSidecarReader(args.sidecars)
        self.retriever = AllHumanV11Retriever(self.official, self.sidecars, v11_config)
        self.cascades: dict[int, RetrievalTop100CascadeV12] = {}
        self.models: dict[int, HierarchicalModelsV12] = {}
        for fold in sorted(ALLOWED_FOLDS):
            recipe = _load_parent_model_recipe(
                args.parent_model_bundle,
                recipe_name=f"outer{fold}",
                required_folds={fold},
            )
            if recipe.outer_fold != fold or recipe.config_sha256 != sha256_file(args.config):
                raise ValueError("nested parent route/config does not match outer fold")
            self.cascades[fold] = RetrievalTop100CascadeV12(
                self.sidecars,
                self.cascade_config,
                rescue_provider=None,
                parent_scorer=recipe.scorer_for_fold(fold),
            )
            site = xgb.XGBRanker()
            site.load_model(sealed.site_model_paths[fold])
            self.models[fold] = HierarchicalModelsV12(
                parent=None,
                site=site,
                diagnostics={"runtime_replay": True, "outer_fold": fold},
            )

    def close(self) -> None:
        self.sidecars.close()
        pool = getattr(self.official, "_search_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)

    def replay(self, row: Mapping[str, Any], fold: int) -> pd.DataFrame:
        query_id = str(row.get("query_id") or "")
        result = self.retriever.retrieve(row)
        if not result.candidates:
            return pd.DataFrame(columns=PARITY_COLUMNS)
        metadata = {
            "fold": int(fold),
            "crm_name": result.query.name,
            "crm_address": result.query.address,
            "crm_insee": result.query.insee,
            "crm_postcode": result.query.postcode,
        }
        v11_rows = result.candidate_rows(query_id, metadata)
        cascade_result = self.cascades[fold].cascade(
            v11_rows, query=result.query, query_id=query_id
        )
        master_rows = cascade_result.master_rows(query_id, metadata)
        if not master_rows:
            return pd.DataFrame(columns=PARITY_COLUMNS)
        canonical = canonicalize_union(
            pd.DataFrame(master_rows), self.admission_config, labels_allowed=False
        )
        scored = score_canonical_candidates(
            self.models[fold], canonical, self.admission_config
        )
        selected = select_top100(scored, self.admission_config)
        return selected[PARITY_COLUMNS].copy()


def _run_replay_pass(
    sample: pd.DataFrame,
    queries: Mapping[str, Mapping[str, Any]],
    replay: Callable[[Mapping[str, Any], int], pd.DataFrame],
    *,
    measure: bool,
) -> tuple[dict[str, pd.DataFrame], list[float]]:
    outputs: dict[str, pd.DataFrame] = {}
    latencies: list[float] = []
    for row in sample.itertuples(index=False):
        started = time.perf_counter()
        selected = replay(queries[row.query_id], int(row.fold))
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        # Canonicalisation here validates caps, SIRET shape and selected ranks.
        selection_bytes(selected)
        outputs[str(row.query_id)] = selected
        if measure:
            latencies.append(elapsed_ms)
    return outputs, latencies


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def profile(
    args: argparse.Namespace,
    *,
    pipeline_factory: Callable[[argparse.Namespace, SealedOOFInputs], Any] = RuntimePipeline,
    hardware_provider: Callable[[], dict[str, str]] = _hardware_identity,
) -> Path:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= int(args.search_workers) <= 16:
        raise ValueError("search_workers must be between 1 and 16")

    hardware = hardware_provider()
    target_hardware = _is_target_hardware(hardware)
    fixture_mode = bool(args.allow_non_m4_test_fixture)
    if not target_hardware and not fixture_mode:
        raise RuntimeError(
            "runtime gate requires computer=Mac and chip=Apple M4 Pro; "
            "--allow-non-m4-test-fixture can only produce a rejected fixture"
        )

    config_path = _require_file(args.config, "v12 config")
    _contract_path(config_path)
    _require_file(args.v11_config, "v11 retrieval config")
    _require_file(args.official_config, "official retrieval config")
    _require_directory(args.sidecars, "retrieval sidecars")
    sealed = _validate_oof_inputs(args)
    ledger_manifest = _validate_label_ledger_sample_manifest(
        args.label_ledger_manifest,
        sample_path=args.query_sample,
        config_path=config_path,
    )
    _validate_oof_label_ledger_lineage(
        sealed, ledger_manifest, args.label_ledger_manifest
    )
    sample = _validate_query_sample(args.query_sample)
    query_path = _require_file(args.queries, "label-blind query source")
    query_parquet = pq.ParquetFile(query_path)
    v11_candidate_manifest_path = _require_file(
        args.v11_candidate_manifest, "v11 candidate manifest"
    )
    _validate_query_source_manifest(
        args.query_source_manifest,
        query_path=query_path,
        query_columns=query_parquet.schema_arrow.names,
        query_rows=query_parquet.metadata.num_rows,
        v11_candidate_manifest_path=v11_candidate_manifest_path,
        v11_candidate_manifest=_json(v11_candidate_manifest_path),
    )
    queries = _load_label_blind_queries(query_path, sample)
    reference = _load_reference_selections(sealed.selected_paths, sample)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    pipeline: Any | None = None
    try:
        pipeline = pipeline_factory(args, sealed)
        replay_method = getattr(pipeline, "replay", None)
        if not callable(replay_method):
            raise RuntimeError(
                "E2E_RUNTIME_INTERFACE_UNAVAILABLE: pipeline has no replay(query, fold)"
            )
        # Warm every sampled query through every serving stage.  Only the full
        # second pass contributes latency or parity.
        _run_replay_pass(sample, queries, replay_method, measure=False)
        observed, latencies = _run_replay_pass(
            sample, queries, replay_method, measure=True
        )
        if len(latencies) != SAMPLE_ROWS or set(observed) != set(sample["query_id"]):
            raise RuntimeError("runtime replay did not measure all 512 queries")

        parity_records: list[dict[str, Any]] = []
        mismatches: list[str] = []
        for row in sample.itertuples(index=False):
            expected_bytes = selection_bytes(reference[str(row.query_id)])
            replay_bytes = selection_bytes(observed[str(row.query_id)])
            expected_hash = hashlib.sha256(expected_bytes).hexdigest()
            replay_hash = hashlib.sha256(replay_bytes).hexdigest()
            identical = expected_bytes == replay_bytes
            if not identical:
                mismatches.append(str(row.query_id))
            parity_records.append(
                {
                    "query_id": str(row.query_id),
                    "fold": int(row.fold),
                    "reference_sha256": expected_hash,
                    "replay_sha256": replay_hash,
                    "byte_identical": identical,
                }
            )
        if mismatches:
            raise AssertionError(
                f"runtime replay changed candidate/order for {len(mismatches)} queries"
            )
        parity_rows = pd.DataFrame(parity_records, columns=PARITY_SIDECAR_COLUMNS)

        sample_path = stage / "query_sample.parquet"
        parity_path = stage / "parity.parquet"
        _write_parquet_atomic(sample[["query_id", "fold"]], sample_path)
        _write_parquet_atomic(parity_rows, parity_path)
        fold_counts = {
            str(int(key)): int(value)
            for key, value in sample["fold"].value_counts().sort_index().items()
        }
        reference_digest = parity_aggregate_digest(parity_rows, "reference_sha256")
        replay_digest = parity_aggregate_digest(parity_rows, "replay_sha256")
        latency = {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        }
        contract = _contract_path(config_path)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_kind": (
                "WARM_END_TO_END_OOF_REPLAY"
                if target_hardware and not fixture_mode
                else "NON_GATE_ELIGIBLE_TEST_FIXTURE"
            ),
            "folds": [2, 3, 4],
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "gate_eligible": bool(target_hardware and not fixture_mode),
            "config_sha256": sha256_file(config_path),
            "contract_sha256": sha256_file(contract),
            "code_sha256": {
                "admission_cli": sha256_file(
                    ROOT / "scripts/run_retrieval_hierarchical_admission_v12.py"
                ),
                "admission_module": sha256_file(
                    ROOT / "src/xgb_matcher/retrieval_hierarchical_admission_v12.py"
                ),
                "cascade_module": sha256_file(
                    ROOT / "src/xgb_matcher/retrieval_top100_cascade_v12.py"
                ),
                "runtime_profiler": sha256_file(Path(__file__).resolve()),
            },
            "lineage": dict(sealed.lineage),
            "query_sample": {
                "path": sample_path.name,
                "sha256": sha256_file(sample_path),
                "rows": SAMPLE_ROWS,
                "fold_counts": fold_counts,
                "selection_rule": SAMPLE_SELECTION_RULE,
                "labels_present": False,
            },
            "hardware": hardware,
            "protocol": {
                "cache_state": "WARM",
                "warm": True,
                "warmup_queries": SAMPLE_ROWS,
                "repetitions": 1,
                "concurrency": 1,
                "end_to_end": True,
                "measured_queries": SAMPLE_ROWS,
                "missing_queries": 0,
                "stages": list(REQUIRED_STAGES),
            },
            "parity": {
                "path": parity_path.name,
                "sha256": sha256_file(parity_path),
                "rows": SAMPLE_ROWS,
                "columns": list(PARITY_COLUMNS),
                "candidates_and_order_byte_identical": True,
                "reference_sha256": reference_digest,
                "replay_sha256": replay_digest,
            },
            "latency_ms": latency,
            "producer_lineage": {
                "query_source_sha256": sha256_file(args.queries),
                "query_source_manifest_sha256": sha256_file(
                    args.query_source_manifest
                ),
                "query_sample_source_sha256": sha256_file(args.query_sample),
                "label_ledger_manifest_sha256": sha256_file(
                    args.label_ledger_manifest
                ),
                "v11_config_sha256": sha256_file(args.v11_config),
                "official_config_sha256": sha256_file(args.official_config),
                "v11_candidate_manifest_sha256": sha256_file(
                    args.v11_candidate_manifest
                ),
                "sidecar_manifest_sha256": sha256_file(
                    args.sidecars / "manifest.json"
                ),
                "nested_parent_bundle_sha256": sha256_file(
                    args.parent_model_bundle
                ),
                "index_manifest_sha256": {
                    "base": _index_manifest_hashes(args.base_index),
                    "overlay": _index_manifest_hashes(args.overlay_index),
                    "temporal_overlay": _index_manifest_hashes(
                        args.temporal_overlay_index
                    ),
                    "legal_entity_role": _index_manifest_hashes(
                        args.legal_entity_role_index
                    ),
                    "sector_official": _index_manifest_hashes(
                        args.sector_official_index
                    ),
                },
                "code_sha256": {
                    "profiler": sha256_file(Path(__file__).resolve()),
                    "query_projection_builder": sha256_file(
                        ROOT
                        / "scripts/build_retrieval_all_human_v12_runtime_queries.py"
                    ),
                    "v11_retriever": sha256_file(
                        ROOT / "src/xgb_matcher/retrieval_all_human_v11.py"
                    ),
                    "official_retriever": sha256_file(
                        ROOT / "src/xgb_matcher/official_evidence_retrieval.py"
                    ),
                    "cascade_runner": sha256_file(
                        ROOT / "scripts/run_retrieval_top100_cascade_v12.py"
                    ),
                },
            },
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()
        pipeline = None
        os.replace(stage, output)
        return output / "manifest.json"
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if pipeline is not None:
            close = getattr(pipeline, "close", None)
            if callable(close):
                close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-source-manifest", type=Path, required=True)
    parser.add_argument("--query-sample", type=Path, required=True)
    parser.add_argument("--label-ledger-manifest", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--outer2-cascade", type=Path, required=True)
    parser.add_argument("--outer3-cascade", type=Path, required=True)
    parser.add_argument("--outer4-cascade", type=Path, required=True)
    parser.add_argument("--parent-model-bundle", type=Path, required=True)
    parser.add_argument("--v11-candidate-manifest", type=Path, required=True)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--overlay-index", type=Path, action="append", default=[])
    parser.add_argument("--temporal-overlay-index", type=Path)
    parser.add_argument("--legal-entity-role-index", type=Path)
    parser.add_argument(
        "--sector-official-index", type=Path, action="append", default=[]
    )
    parser.add_argument("--sidecars", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v11-config", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument(
        "--allow-non-m4-test-fixture",
        action="store_true",
        help="Produce a manifest that is explicitly ineligible for the runtime gate.",
    )
    return parser.parse_args(argv)


def main() -> None:
    print(profile(parse_args()))


if __name__ == "__main__":
    main()
