#!/usr/bin/env python3
"""Collect official API evidence for the frozen V4.4 AUTO population."""

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
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.4-official-evidence-1"
API_URL = "https://recherche-entreprises.api.gouv.fr/search"
USER_AGENT = "SIRETO-V4.4-evidence/1.0"
EXPECTED_AUTO_COUNT = 172


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _siret(value: Any) -> str | None:
    digits = "".join(character for character in _text(value) if character.isdigit())
    return digits if len(digits) == 14 else None


def build_queries(row: Any) -> list[dict[str, Any]]:
    """Create deterministic direct and name/geography API queries."""

    common = {
        "minimal": "true",
        "include": "matching_etablissements,score",
        "limite_matching_etablissements": "10",
    }
    queries: list[dict[str, Any]] = []
    seen_sirets: set[str] = set()
    for kind, value in (
        ("TOP1_SIRET", getattr(row, "top1_siret", None)),
        ("INPUT_SIRET", getattr(row, "input_siret", None)),
    ):
        normalized = _siret(value)
        if normalized is None or normalized in seen_sirets:
            continue
        seen_sirets.add(normalized)
        queries.append(
            {
                "query_kind": kind,
                "params": {
                    **common,
                    "q": normalized,
                    "per_page": "1",
                },
            }
        )
    name = _text(getattr(row, "SITE", None))
    if name:
        params = {
            **common,
            "q": name,
            "per_page": "10",
        }
        insee = _text(getattr(row, "CODE_INSEE", None))
        postcode = _text(getattr(row, "CODE_POSTAL", None))
        if insee:
            params["code_commune"] = insee
        elif postcode:
            params["code_postal"] = postcode
        queries.append({"query_kind": "CRM_NAME_GEO", "params": params})
    return queries


def _request_json(
    params: dict[str, str],
    *,
    timeout_seconds: float,
    retries: int = 3,
) -> tuple[int, dict[str, Any]]:
    url = f"{API_URL}?{urlencode(params)}"
    for attempt in range(retries + 1):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return int(response.status), json.load(response)
        except HTTPError as error:
            if error.code == 429 and attempt < retries:
                retry_after = float(error.headers.get("Retry-After") or 1.0)
                time.sleep(max(1.0, retry_after))
                continue
            payload = {
                "error": f"HTTP_{error.code}",
                "message": error.read().decode("utf-8", errors="replace")[:1000],
            }
            return int(error.code), payload
        except (URLError, TimeoutError) as error:
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
                continue
            return 0, {"error": type(error).__name__, "message": str(error)}
    raise AssertionError("Unreachable request retry state")


def collect(
    *,
    queue_path: Path,
    output_root: Path,
    requests_per_second: float = 5.0,
    timeout_seconds: float = 20.0,
) -> Path:
    if not 0.1 <= requests_per_second <= 6.0:
        raise ValueError("requests_per_second must be between 0.1 and 6.0")
    queue_hash = file_sha256(queue_path)
    queue = pd.read_parquet(queue_path)
    auto = queue.loc[queue["decision"].astype(str).eq("AUTO_MATCH")].copy()
    if len(auto) != EXPECTED_AUTO_COUNT:
        raise ValueError(
            f"Frozen AUTO population changed: {len(auto)} "
            f"!= {EXPECTED_AUTO_COUNT}"
        )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "queue_sha256": queue_hash,
        "api_url": API_URL,
        "query_policy": "top1-input-name-geo-v1",
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.4 evidence exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    interval = 1.0 / requests_per_second
    records: list[dict[str, Any]] = []
    try:
        last_started = 0.0
        for row in auto.sort_values("audit_case_id").itertuples(index=False):
            for query in build_queries(row):
                wait_seconds = interval - (time.monotonic() - last_started)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                last_started = time.monotonic()
                collected_at = datetime.now(timezone.utc).isoformat()
                status, payload = _request_json(
                    query["params"],
                    timeout_seconds=timeout_seconds,
                )
                records.append(
                    {
                        "audit_case_id": str(row.audit_case_id),
                        "service_id": str(row.service_id),
                        "query_kind": query["query_kind"],
                        "query_params_json": json.dumps(
                            query["params"],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "http_status": status,
                        "result_count": int(payload.get("total_results") or 0),
                        "payload_json": json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "collected_at": collected_at,
                        "source_url": (
                            f"{API_URL}?{urlencode(query['params'])}"
                        ),
                    }
                )
        evidence = pd.DataFrame(records).sort_values(
            ["audit_case_id", "query_kind"]
        )
        evidence_path = staging / "official_evidence.parquet"
        evidence.to_parquet(evidence_path, index=False)
        status_counts = {
            str(key): int(value)
            for key, value in evidence["http_status"].value_counts().items()
        }
        summary = {
            "case_count": int(evidence["audit_case_id"].nunique()),
            "request_count": int(len(evidence)),
            "query_kind_counts": {
                str(key): int(value)
                for key, value in evidence["query_kind"].value_counts().items()
            },
            "http_status_counts": status_counts,
            "requests_with_results": int(evidence["result_count"].gt(0).sum()),
            "source": "API_RECHERCHE_ENTREPRISES_OFFICIAL",
            "adjudications_created": 0,
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
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": {"path": str(queue_path), "sha256": queue_hash},
            "source": {
                "url": API_URL,
                "user_agent": USER_AGENT,
                "requests_per_second": requests_per_second,
            },
            "outputs": {
                evidence_path.name: file_sha256(evidence_path),
                summary_path.name: file_sha256(summary_path),
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        collect(
            queue_path=args.queue,
            output_root=args.output_root,
            requests_per_second=args.requests_per_second,
            timeout_seconds=args.timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
