#!/usr/bin/env python3
"""Collect producer evidence for sector identifiers seen in V4.4 payloads.

The collector records observations and producer responses only.  It deliberately
does not create labels and never decides whether a CRM-to-SIRET prediction is
correct.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.4-sector-evidence-1"
POLICY_VERSION = "producer-responses-no-adjudication-v1"
USER_AGENT = "SIRETO-V4.4-sector-evidence/1.0"

UAI_API = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-annuaire-education/records"
)
FINESS_DATASET_API = (
    "https://www.data.gouv.fr/api/1/datasets/finess-structures-1/"
)
BIO_API = "https://opendata.agencebio.org/api/gouv/operateurs/"
RGE_API = (
    "https://data.ademe.fr/data-fair/api/v1/datasets/"
    "liste-des-entreprises-rge-2-new/lines"
)

FIELD_TO_KIND = {
    "liste_uai": "UAI",
    "liste_finess": "FINESS",
    "liste_id_bio": "BIO",
    "liste_rge": "RGE",
}
PRODUCERS = {
    "UAI": "Ministère de l'Éducation nationale",
    "FINESS": "Agence du Numérique en Santé (ANS)",
    "BIO": "Agence Bio",
    "RGE": "ADEME",
}


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    final_url: str
    collected_at: str


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(_text(value) or "{}")
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _siret(value: Any) -> str:
    digits = "".join(character for character in _text(value) if character.isdigit())
    return digits if len(digits) == 14 else ""


def extract_sector_observations(evidence: pd.DataFrame) -> pd.DataFrame:
    """Extract and deduplicate sector identifiers without interpreting them."""

    required = {
        "audit_case_id",
        "service_id",
        "query_kind",
        "payload_json",
    }
    missing = required - set(evidence.columns)
    if missing:
        raise ValueError(f"Missing evidence columns: {sorted(missing)}")

    observations: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in evidence.itertuples(index=False):
        payload = _json(row.payload_json)
        for result in payload.get("results") or []:
            if not isinstance(result, Mapping):
                continue
            for establishment in result.get("matching_etablissements") or []:
                if not isinstance(establishment, Mapping):
                    continue
                observed_siret = _siret(establishment.get("siret"))
                for field, kind in FIELD_TO_KIND.items():
                    for raw_identifier in establishment.get(field) or []:
                        identifier = _text(raw_identifier)
                        if not identifier:
                            continue
                        key = (
                            _text(row.audit_case_id),
                            _text(row.service_id),
                            observed_siret,
                            kind,
                            identifier,
                        )
                        item = observations.setdefault(
                            key,
                            {
                                "audit_case_id": key[0],
                                "service_id": key[1],
                                "observed_siret": observed_siret,
                                "identifier_kind": kind,
                                "identifier": identifier,
                                "query_kinds": set(),
                                "occurrence_count": 0,
                            },
                        )
                        item["query_kinds"].add(_text(row.query_kind))
                        item["occurrence_count"] += 1

    rows = []
    for item in observations.values():
        request_key = _request_key(
            item["identifier_kind"],
            item["identifier"],
            item["observed_siret"],
        )
        rows.append(
            {
                "audit_case_id": item["audit_case_id"],
                "service_id": item["service_id"],
                "observed_siret": item["observed_siret"],
                "identifier_kind": item["identifier_kind"],
                "identifier": item["identifier"],
                "producer_request_key": request_key,
                "origin_query_kinds_json": json.dumps(
                    sorted(item["query_kinds"]), ensure_ascii=False
                ),
                "origin_occurrence_count": int(item["occurrence_count"]),
            }
        )
    columns = [
        "audit_case_id",
        "service_id",
        "observed_siret",
        "identifier_kind",
        "identifier",
        "producer_request_key",
        "origin_query_kinds_json",
        "origin_occurrence_count",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["identifier_kind", "identifier", "observed_siret", "audit_case_id"]
    ).reset_index(drop=True)


def _request_key(kind: str, identifier: str, observed_siret: str) -> str:
    # RGE qualification codes are not company identifiers.  Preserve the pair.
    value = f"{kind}|{identifier}"
    if kind == "RGE":
        value += f"|{observed_siret}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def build_request_specs(observations: pd.DataFrame) -> list[dict[str, str]]:
    """Build one producer request per unique identifier (or RGE code/SIRET)."""

    specs: dict[str, dict[str, str]] = {}
    for row in observations.itertuples(index=False):
        kind = _text(row.identifier_kind)
        identifier = _text(row.identifier)
        siret = _siret(row.observed_siret)
        request_key = _request_key(kind, identifier, siret)
        if kind == "UAI":
            params = {
                "where": f'identifiant_de_l_etablissement="{identifier}"',
                "limit": "10",
            }
            url = f"{UAI_API}?{urlencode(params)}"
        elif kind == "BIO":
            url = f"{BIO_API}?{urlencode({'numeroBio': identifier})}"
        elif kind == "RGE":
            if not siret:
                raise ValueError(
                    f"RGE observation lacks a valid SIRET: {identifier}"
                )
            query = (
                f'code_qualification:"{identifier}" AND siret:"{siret}"'
            )
            url = f"{RGE_API}?{urlencode({'size': '100', 'qs': query})}"
        elif kind == "FINESS":
            # FINESS is supplied as one daily producer snapshot, fetched below.
            url = FINESS_DATASET_API
        else:
            raise ValueError(f"Unsupported sector identifier kind: {kind}")
        specs[request_key] = {
            "request_key": request_key,
            "identifier_kind": kind,
            "identifier": identifier,
            "observed_siret": siret if kind == "RGE" else "",
            "producer": PRODUCERS[kind],
            "requested_url": url,
        }
    return sorted(
        specs.values(),
        key=lambda item: (
            item["identifier_kind"],
            item["identifier"],
            item["observed_siret"],
        ),
    )


def _fetch_url(url: str, timeout_seconds: float, retries: int = 3) -> HttpResponse:
    for attempt in range(retries + 1):
        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return HttpResponse(
                    status=int(response.status),
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=response.read(),
                    final_url=response.geturl(),
                    collected_at=datetime.now(timezone.utc).isoformat(),
                )
        except HTTPError as error:
            body = error.read()
            if error.code == 429 and attempt < retries:
                time.sleep(float(error.headers.get("Retry-After") or 1.0))
                continue
            return HttpResponse(
                status=int(error.code),
                headers={
                    key.lower(): value for key, value in error.headers.items()
                },
                body=body,
                final_url=url,
                collected_at=datetime.now(timezone.utc).isoformat(),
            )
        except (URLError, TimeoutError, ConnectionError, OSError):
            if attempt == retries:
                raise
            time.sleep(float(attempt + 1))
    raise AssertionError("Unreachable request retry state")


def _payload_result_count(kind: str, payload: Any) -> int:
    if kind == "UAI" and isinstance(payload, Mapping):
        return int(payload.get("total_count") or 0)
    if kind == "BIO":
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, Mapping):
            return len(payload.get("items") or [])
    if kind == "RGE" and isinstance(payload, Mapping):
        return int(payload.get("total") or 0)
    return 0


def _find_finess_entities(
    node: Any,
    wanted: set[str],
    found: dict[str, list[dict[str, Any]]],
) -> None:
    if isinstance(node, Mapping):
        identifier = _text(node.get("numFinessEge"))
        if identifier in wanted:
            found.setdefault(identifier, []).append(dict(node))
        for value in node.values():
            _find_finess_entities(value, wanted, found)
    elif isinstance(node, list):
        for value in node:
            _find_finess_entities(value, wanted, found)


def _raw_filename(kind: str, request_key: str, compressed: bool = False) -> str:
    suffix = ".json.gz" if compressed else ".json"
    return f"raw/{kind.lower()}-{request_key}{suffix}"


def _write_raw(staging: Path, relative_path: str, body: bytes) -> str:
    path = staging / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return file_sha256(path)


def _decode_json(body: bytes) -> Any:
    return json.loads(body.decode("utf-8"))


def _response_record(
    spec: Mapping[str, str],
    response: HttpResponse,
    *,
    raw_path: str,
    raw_sha256: str,
    result_count: int,
    response_excerpt: Any,
    producer_data_date: str = "",
) -> dict[str, Any]:
    return {
        **dict(spec),
        "response_url": response.final_url,
        "http_status": int(response.status),
        "collected_at": response.collected_at,
        "producer_data_date": producer_data_date,
        "result_count": int(result_count),
        "raw_response_path": raw_path,
        "raw_response_sha256": raw_sha256,
        "response_headers_json": json.dumps(
            response.headers, ensure_ascii=False, sort_keys=True
        ),
        "response_excerpt_json": json.dumps(
            response_excerpt, ensure_ascii=False, sort_keys=True
        ),
    }


def collect(
    *,
    evidence_path: Path,
    output_root: Path,
    requests_per_second: float = 5.0,
    timeout_seconds: float = 30.0,
    fetcher: Callable[[str, float], HttpResponse] = _fetch_url,
) -> Path:
    """Collect an immutable, label-free producer evidence build."""

    if not 0.1 <= requests_per_second <= 10.0:
        raise ValueError("requests_per_second must be between 0.1 and 10.0")
    evidence_hash = file_sha256(evidence_path)
    evidence = pd.read_parquet(evidence_path)
    observations = extract_sector_observations(evidence)
    specs = build_request_specs(observations)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "evidence_sha256": evidence_hash,
        "producer_endpoints": {
            "UAI": UAI_API,
            "FINESS": FINESS_DATASET_API,
            "BIO": BIO_API,
            "RGE": RGE_API,
        },
        "request_policy": "exact-sector-id-rge-id-plus-siret-v1",
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.4 sector evidence exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    interval = 1.0 / requests_per_second
    last_started = 0.0

    def throttled_fetch(url: str) -> HttpResponse:
        nonlocal last_started
        wait_seconds = interval - (time.monotonic() - last_started)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        last_started = time.monotonic()
        return fetcher(url, timeout_seconds)

    records: list[dict[str, Any]] = []
    try:
        observations_path = staging / "sector_identifier_observations.parquet"
        observations.to_parquet(observations_path, index=False)

        non_finess_specs = [
            spec for spec in specs if spec["identifier_kind"] != "FINESS"
        ]
        for spec in non_finess_specs:
            response = throttled_fetch(spec["requested_url"])
            raw_path = _raw_filename(
                spec["identifier_kind"], spec["request_key"]
            )
            raw_sha = _write_raw(staging, raw_path, response.body)
            try:
                payload = _decode_json(response.body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {"raw_response_not_json": True}
            records.append(
                _response_record(
                    spec,
                    response,
                    raw_path=raw_path,
                    raw_sha256=raw_sha,
                    result_count=_payload_result_count(
                        spec["identifier_kind"], payload
                    ),
                    response_excerpt=payload,
                )
            )

        finess_specs = [
            spec for spec in specs if spec["identifier_kind"] == "FINESS"
        ]
        if finess_specs:
            metadata_response = throttled_fetch(FINESS_DATASET_API)
            metadata_path = "raw/finess-dataset-metadata.json"
            metadata_sha = _write_raw(
                staging, metadata_path, metadata_response.body
            )
            metadata = _decode_json(metadata_response.body)
            resources = metadata.get("resources") or []
            if metadata_response.status != 200 or not resources:
                raise RuntimeError("FINESS dataset metadata has no resource")
            resource = max(
                resources,
                key=lambda item: _text(item.get("last_modified"))
                or _text(item.get("created_at")),
            )
            resource_url = _text(resource.get("latest") or resource.get("url"))
            if not resource_url:
                raise RuntimeError("FINESS latest resource URL is absent")
            snapshot_response = throttled_fetch(resource_url)
            snapshot_path = "raw/finess-structures-snapshot.json.gz"
            snapshot_sha = _write_raw(
                staging, snapshot_path, snapshot_response.body
            )
            try:
                snapshot = json.loads(
                    gzip.decompress(snapshot_response.body).decode("utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("Invalid FINESS JSON.GZ snapshot") from error
            wanted = {spec["identifier"] for spec in finess_specs}
            found: dict[str, list[dict[str, Any]]] = {}
            _find_finess_entities(snapshot, wanted, found)
            producer_date = _text(snapshot.get("generatedAt"))
            combined_headers = {
                **snapshot_response.headers,
                "dataset-metadata-sha256": metadata_sha,
                "dataset-metadata-path": metadata_path,
            }
            for spec in finess_specs:
                selected = found.get(spec["identifier"], [])
                response_for_record = HttpResponse(
                    status=snapshot_response.status,
                    headers=combined_headers,
                    body=b"",
                    final_url=snapshot_response.final_url,
                    collected_at=snapshot_response.collected_at,
                )
                records.append(
                    _response_record(
                        spec,
                        response_for_record,
                        raw_path=snapshot_path,
                        raw_sha256=snapshot_sha,
                        result_count=len(selected),
                        response_excerpt=selected,
                        producer_data_date=producer_date,
                    )
                )

        responses = pd.DataFrame(records).sort_values(
            ["identifier_kind", "identifier", "observed_siret"]
        )
        responses_path = staging / "producer_responses.parquet"
        responses.to_parquet(responses_path, index=False)
        capabilities = {
            "UAI": {
                "producer": PRODUCERS["UAI"],
                "retrieval_key": "identifiant_de_l_etablissement",
                "mode": "exact API query",
                "url": UAI_API,
            },
            "FINESS": {
                "producer": PRODUCERS["FINESS"],
                "retrieval_key": "numFinessEge",
                "mode": "daily JSON.GZ snapshot, then exact local extraction",
                "url": FINESS_DATASET_API,
            },
            "BIO": {
                "producer": PRODUCERS["BIO"],
                "retrieval_key": "numeroBio",
                "mode": "exact API query",
                "url": BIO_API,
            },
            "RGE": {
                "producer": PRODUCERS["RGE"],
                "retrieval_key": "code_qualification + observed SIRET",
                "mode": "exact paired API query",
                "url": RGE_API,
                "caveat": (
                    "code_qualification is shared by many companies and is "
                    "not an establishment identifier"
                ),
            },
            "adjudication_policy": (
                "No producer response is converted into a CRM-match label."
            ),
        }
        capabilities_path = staging / "producer_capabilities.json"
        capabilities_path.write_text(
            json.dumps(
                capabilities, indent=2, ensure_ascii=False, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        summary = {
            "observation_count": int(len(observations)),
            "case_count": int(observations["audit_case_id"].nunique()),
            "request_count": int(len(responses)),
            "identifier_counts": {
                kind: int(
                    observations.loc[
                        observations["identifier_kind"].eq(kind), "identifier"
                    ].nunique()
                )
                for kind in sorted(FIELD_TO_KIND.values())
            },
            "request_counts": {
                str(key): int(value)
                for key, value in responses[
                    "identifier_kind"
                ].value_counts().items()
            },
            "http_status_counts": {
                str(key): int(value)
                for key, value in responses["http_status"].value_counts().items()
            },
            "responses_with_results": int(responses["result_count"].gt(0).sum()),
            "adjudications_created": 0,
            "correctness_labels_created": 0,
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
            "input": {"path": str(evidence_path), "sha256": evidence_hash},
            "user_agent": USER_AGENT,
            "requests_per_second": requests_per_second,
            "outputs": {
                path.name: file_sha256(path)
                for path in (
                    observations_path,
                    responses_path,
                    capabilities_path,
                    summary_path,
                )
            },
            "raw_response_count": len(list((staging / "raw").glob("*"))),
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
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--requests-per-second", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = collect(
        evidence_path=args.evidence,
        output_root=args.output_root,
        requests_per_second=args.requests_per_second,
        timeout_seconds=args.timeout_seconds,
    )
    print(target)


if __name__ == "__main__":
    main()
