#!/usr/bin/env python3
"""Freeze the blind 800-case representative V4.1 shadow audit sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd

from src.xgb_matcher.v9_dataset import file_sha256


SCHEMA_VERSION = "sireto-v4.1-representative-audit-sample-1"
SAMPLE_NAMESPACE = "v4.1-representative-audit:42:"
QUOTAS = {
    "RANDOM_POPULATION": 250,
    "NO_ACTIVE_CANDIDATE": 50,
    "AUTO_NEAR_THRESHOLD": 100,
    "AUTO_HIGH_SCORE": 150,
    "REVIEW_NEAR_THRESHOLD": 150,
    "REVIEW_LOW_SCORE": 100,
}


def _stable_key(service_id: str) -> str:
    return hashlib.sha256(
        f"{SAMPLE_NAMESPACE}{service_id}".encode("utf-8")
    ).hexdigest()


def _take(
    frame: pd.DataFrame,
    *,
    count: int,
    used: set[str],
    order: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    available = frame[~frame["service_id"].isin(used)].copy()
    selected = available.sort_values(
        order,
        ascending=ascending,
        kind="mergesort",
    ).head(count)
    if len(selected) != count:
        raise ValueError(f"Stratum requires {count} rows, found {len(selected)}")
    used.update(selected["service_id"].astype(str))
    return selected


def select_sample(decisions: pd.DataFrame) -> pd.DataFrame:
    """Select the preregistered disjoint diagnostic strata."""

    required = {
        "service_id",
        "decision",
        "confidence",
        "candidate_count",
        "predicted_siret",
        "review_reason",
    }
    missing = required - set(decisions)
    if missing:
        raise ValueError(f"Shadow decisions missing columns: {sorted(missing)}")
    frame = decisions.copy()
    frame["service_id"] = frame["service_id"].astype(str)
    if frame["service_id"].duplicated().any():
        raise ValueError("Shadow decisions require unique service_id")
    frame["sampling_key"] = frame["service_id"].map(_stable_key)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="raise")
    used: set[str] = set()
    parts: list[pd.DataFrame] = []

    random_part = _take(
        frame,
        count=QUOTAS["RANDOM_POPULATION"],
        used=used,
        order=["sampling_key"],
        ascending=[True],
    )
    random_part["sampling_stratum"] = "RANDOM_POPULATION"
    parts.append(random_part)

    no_candidate = _take(
        frame[frame["candidate_count"].eq(0)],
        count=QUOTAS["NO_ACTIVE_CANDIDATE"],
        used=used,
        order=["sampling_key"],
        ascending=[True],
    )
    no_candidate["sampling_stratum"] = "NO_ACTIVE_CANDIDATE"
    parts.append(no_candidate)

    auto = frame[frame["decision"].eq("AUTO_MATCH")]
    auto_near = _take(
        auto,
        count=QUOTAS["AUTO_NEAR_THRESHOLD"],
        used=used,
        order=["confidence", "sampling_key"],
        ascending=[True, True],
    )
    auto_near["sampling_stratum"] = "AUTO_NEAR_THRESHOLD"
    parts.append(auto_near)

    auto_high = _take(
        auto,
        count=QUOTAS["AUTO_HIGH_SCORE"],
        used=used,
        order=["confidence", "sampling_key"],
        ascending=[False, True],
    )
    auto_high["sampling_stratum"] = "AUTO_HIGH_SCORE"
    parts.append(auto_high)

    review = frame[
        frame["decision"].eq("REVIEW") & frame["candidate_count"].gt(0)
    ]
    review_near = _take(
        review,
        count=QUOTAS["REVIEW_NEAR_THRESHOLD"],
        used=used,
        order=["confidence", "sampling_key"],
        ascending=[False, True],
    )
    review_near["sampling_stratum"] = "REVIEW_NEAR_THRESHOLD"
    parts.append(review_near)

    review_low = _take(
        review,
        count=QUOTAS["REVIEW_LOW_SCORE"],
        used=used,
        order=["confidence", "sampling_key"],
        ascending=[True, True],
    )
    review_low["sampling_stratum"] = "REVIEW_LOW_SCORE"
    parts.append(review_low)

    output = pd.concat(parts, ignore_index=True)
    if len(output) != sum(QUOTAS.values()):
        raise AssertionError("Representative audit sample cardinality drift")
    if output["service_id"].duplicated().any():
        raise AssertionError("Representative audit strata overlap")
    output["audit_case_id"] = output["sampling_key"].str[:16]
    return output


def build_blind_cases(
    sample: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Join raw CRM evidence without exposing any model output."""

    inv = inventory.dropna(subset=["service_id"]).copy()
    inv["service_id"] = inv["service_id"].astype(str)
    inv = inv[inv["service_id"].isin(set(sample["service_id"].astype(str)))]
    if inv["service_id"].duplicated().any():
        raise ValueError("Inventory service_id must be unique")
    selected = sample[["service_id", "audit_case_id"]].merge(
        inv,
        on="service_id",
        how="left",
        validate="one_to_one",
    )
    if selected["eligible_for_shadow"].ne(True).any():  # noqa: E712
        raise ValueError("Blind sample contains a shadow-ineligible row")
    forbidden = {
        "decision",
        "routing_status",
        "confidence",
        "predicted_siret",
        "predicted_siren",
        "review_reason",
        "sampling_stratum",
        "sampling_key",
    }
    leaked = forbidden & set(selected)
    if leaked:
        raise ValueError(f"Blind cases leak model outputs: {sorted(leaked)}")
    return selected.sort_values("audit_case_id").reset_index(drop=True)


def _manifest_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def freeze_sample(
    *,
    shadow_dir: Path,
    output_root: Path,
) -> Path:
    shadow_dir = Path(shadow_dir)
    shadow_manifest_path = shadow_dir / "manifest.json"
    shadow_manifest = json.loads(shadow_manifest_path.read_text(encoding="utf-8"))
    if shadow_manifest.get("schema_version") != "sireto-shadow-v4.1-1":
        raise ValueError("Unsupported shadow manifest")
    decisions_path = shadow_dir / "decisions.parquet"
    inventory_path = shadow_dir / "inventory.parquet"
    outputs = shadow_manifest.get("outputs") or {}
    for path in (decisions_path, inventory_path):
        if outputs.get(path.name) != file_sha256(path):
            raise ValueError(f"Shadow output hash mismatch: {path.name}")

    decisions = pd.read_parquet(decisions_path)
    inventory = pd.read_parquet(inventory_path)
    if len(decisions) != 19_025:
        raise ValueError("Current V4.1 shadow must contain 19,025 decisions")
    sample = select_sample(decisions)
    blind = build_blind_cases(sample, inventory)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "shadow_manifest_sha256": file_sha256(shadow_manifest_path),
        "decisions_sha256": file_sha256(decisions_path),
        "inventory_sha256": file_sha256(inventory_path),
        "quotas": QUOTAS,
        "namespace": SAMPLE_NAMESPACE,
        "service_ids_sha256": hashlib.sha256(
            "\n".join(sorted(sample["service_id"])).encode()
        ).hexdigest(),
    }
    build_id = _manifest_id(identity)
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable audit sample already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    try:
        registry_path = staging / "sample_registry.parquet"
        blind_path = staging / "blind_cases.parquet"
        sample.sort_values("audit_case_id").to_parquet(registry_path, index=False)
        blind.to_parquet(blind_path, index=False)
        summary = {
            "sample_count": int(len(sample)),
            "blind_case_count": int(len(blind)),
            "stratum_counts": {
                str(key): int(value)
                for key, value in sample["sampling_stratum"]
                .value_counts()
                .sort_index()
                .items()
            },
            "input_siret_state_counts": {
                str(key): int(value)
                for key, value in blind["input_siret_state"]
                .value_counts(dropna=False)
                .sort_index()
                .items()
            },
            "blind_model_output_columns": [],
        }
        summary_path = staging / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            **identity,
            "build_id": build_id,
            "source_shadow_dir": str(shadow_dir),
            "positive_labels_read": False,
            "adjudication_started": False,
            "outputs": {
                path.name: file_sha256(path)
                for path in (registry_path, blind_path, summary_path)
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        staging.rename(target)
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(freeze_sample(shadow_dir=args.shadow_dir, output_root=args.output_root))


if __name__ == "__main__":
    main()
