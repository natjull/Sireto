#!/usr/bin/env python3
"""Collect the official registry views for the frozen V4.7 docket."""

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
from typing import Any, Callable
from urllib.parse import urlencode

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.collect_v44_official_evidence import (  # noqa: E402
    API_URL,
    _request_json,
    build_queries,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.7-official-evidence-1"
EXPECTED_DOCKET_SHA256 = (
    "7ee6bf58ed60e3d7d9b94577ae8da62eca2d076fb61459d7c1625522d92a7104"
)
EXPECTED_CASE_COUNT = 37
Requester = Callable[..., tuple[int, dict[str, Any]]]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collect(
    *,
    docket_path: Path,
    output_root: Path,
    requests_per_second: float = 5.0,
    timeout_seconds: float = 20.0,
    enforce_canonical: bool = True,
    requester: Requester = _request_json,
) -> Path:
    if not 0.1 <= requests_per_second <= 6.0:
        raise ValueError("requests_per_second must be between 0.1 and 6.0")
    docket_path = Path(docket_path).resolve()
    docket_hash = file_sha256(docket_path)
    if enforce_canonical and docket_hash != EXPECTED_DOCKET_SHA256:
        raise ValueError("V4.7 docket hash mismatch")
    docket = pd.read_parquet(docket_path).copy()
    required = {
        "audit_case_id",
        "service_id",
        "siret_to_adjudicate",
        "input_siret",
        "SITE",
        "CODE_INSEE",
        "CODE_POSTAL",
    }
    missing = required - set(docket.columns)
    if missing:
        raise ValueError(f"V4.7 docket missing columns: {sorted(missing)}")
    if enforce_canonical and len(docket) != EXPECTED_CASE_COUNT:
        raise ValueError("V4.7 official collection requires exactly 37 dossiers")
    if docket["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("V4.7 docket has duplicate audit_case_id values")

    queue = docket.rename(columns={"siret_to_adjudicate": "top1_siret"})
    identity = {
        "schema_version": SCHEMA_VERSION,
        "docket_sha256": docket_hash,
        "api_url": API_URL,
        "query_policy": "current-top1-input-name-geo-v1",
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root).resolve() / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.7 official evidence exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent))
    interval = 1.0 / requests_per_second
    records: list[dict[str, Any]] = []
    try:
        last_started = 0.0
        for row in queue.sort_values("audit_case_id").itertuples(index=False):
            for query in build_queries(row):
                wait_seconds = interval - (time.monotonic() - last_started)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                last_started = time.monotonic()
                collected_at = datetime.now(timezone.utc).isoformat()
                status, payload = requester(
                    query["params"],
                    timeout_seconds=timeout_seconds,
                )
                records.append(
                    {
                        "audit_case_id": str(row.audit_case_id),
                        "service_id": str(row.service_id),
                        "siret_to_adjudicate": str(row.top1_siret),
                        "query_kind": query["query_kind"],
                        "query_params_json": json.dumps(
                            query["params"],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "http_status": int(status),
                        "result_count": int(payload.get("total_results") or 0),
                        "payload_json": json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "payload_sha256": hashlib.sha256(
                            json.dumps(
                                payload,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest(),
                        "collected_at": collected_at,
                        "source_url": f"{API_URL}?{urlencode(query['params'])}",
                        "source_family": (
                            "SIRENE_DERIVED_RECHERCHE_ENTREPRISES_API"
                        ),
                        "independence_group": "SIRENE_REGISTRY",
                    }
                )
        evidence = pd.DataFrame(records).sort_values(
            ["audit_case_id", "query_kind"]
        )
        if evidence["audit_case_id"].nunique() != len(docket):
            raise ValueError("Official evidence collection missed V4.7 dossiers")
        evidence_path = staging / "official_evidence.parquet"
        evidence.to_parquet(evidence_path, index=False)
        summary = {
            "case_count": int(evidence["audit_case_id"].nunique()),
            "request_count": int(len(evidence)),
            "http_status_counts": {
                str(key): int(value)
                for key, value in evidence["http_status"].value_counts().items()
            },
            "requests_with_results": int(evidence["result_count"].gt(0).sum()),
            "independent_evidence_group_count": 1,
            "adjudications_created": 0,
        }
        summary_path = staging / "summary.json"
        _json_dump(summary_path, summary)
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {"path": str(docket_path), "sha256": docket_hash},
            "outputs": {
                evidence_path.name: file_sha256(evidence_path),
                summary_path.name: file_sha256(summary_path),
            },
            "summary": summary,
        }
        _json_dump(staging / "manifest.json", manifest)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docket", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        collect(
            docket_path=args.docket,
            output_root=args.output_root,
            requests_per_second=args.requests_per_second,
            timeout_seconds=args.timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
