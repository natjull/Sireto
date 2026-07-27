from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from scripts.collect_v44_sector_evidence import (
    FINESS_DATASET_API,
    HttpResponse,
    build_request_specs,
    collect,
    extract_sector_observations,
)


def _evidence(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_case_id": "case-1",
                "service_id": "service-1",
                "query_kind": "TOP1_SIRET",
                "payload_json": json.dumps(payload),
            },
            {
                "audit_case_id": "case-1",
                "service_id": "service-1",
                "query_kind": "CRM_NAME_GEO",
                "payload_json": json.dumps(payload),
            },
        ]
    )


def test_extract_sector_observations_preserves_origin_without_labels() -> None:
    payload = {
        "results": [
            {
                "matching_etablissements": [
                    {
                        "siret": "12345678900011",
                        "liste_uai": ["0141396S"],
                        "liste_finess": ["440002947"],
                        "liste_id_bio": [110415],
                        "liste_rge": ["43SPVRGE"],
                    }
                ]
            }
        ]
    }

    observations = extract_sector_observations(_evidence(payload))

    assert len(observations) == 4
    assert set(observations["identifier_kind"]) == {
        "UAI",
        "FINESS",
        "BIO",
        "RGE",
    }
    assert set(observations["origin_occurrence_count"]) == {2}
    assert all(
        json.loads(value) == ["CRM_NAME_GEO", "TOP1_SIRET"]
        for value in observations["origin_query_kinds_json"]
    )
    assert not any(
        "correct" in column.lower() or "label" in column.lower()
        for column in observations.columns
    )


def test_request_specs_use_exact_ids_and_pair_rge_with_siret() -> None:
    payload = {
        "results": [
            {
                "matching_etablissements": [
                    {
                        "siret": "12345678900011",
                        "liste_uai": ["0141396S"],
                        "liste_finess": ["440002947"],
                        "liste_id_bio": ["110415"],
                        "liste_rge": ["43SPVRGE"],
                    }
                ]
            }
        ]
    }
    observations = extract_sector_observations(_evidence(payload))

    specs = build_request_specs(observations)
    by_kind = {spec["identifier_kind"]: spec for spec in specs}

    assert "identifiant_de_l_etablissement" in by_kind["UAI"]["requested_url"]
    assert "numeroBio=110415" in by_kind["BIO"]["requested_url"]
    assert "finess-structures-1" in by_kind["FINESS"]["requested_url"]
    assert "43SPVRGE" in by_kind["RGE"]["requested_url"]
    assert "12345678900011" in by_kind["RGE"]["requested_url"]
    assert by_kind["RGE"]["observed_siret"] == "12345678900011"


def test_request_specs_deduplicate_same_identifier_across_source_views() -> None:
    payload = {
        "results": [
            {
                "matching_etablissements": [
                    {
                        "siret": "12345678900011",
                        "liste_uai": ["0141396S"],
                    }
                ]
            }
        ]
    }
    observations = extract_sector_observations(_evidence(payload))

    specs = build_request_specs(observations)

    assert len(specs) == 1
    assert specs[0]["identifier"] == "0141396S"


def test_collect_writes_immutable_raw_responses_without_adjudication(
    tmp_path,
) -> None:
    payload = {
        "results": [
            {
                "matching_etablissements": [
                    {
                        "siret": "12345678900011",
                        "liste_uai": ["0141396S"],
                        "liste_finess": ["440002947"],
                        "liste_id_bio": ["110415"],
                        "liste_rge": ["43SPVRGE"],
                    }
                ]
            }
        ]
    }
    evidence_path = tmp_path / "official_evidence.parquet"
    _evidence(payload).to_parquet(evidence_path, index=False)
    finess_snapshot = {
        "schemaVersion": "v1",
        "generatedAt": "2026-07-27T02:06:47Z",
        "pmej": [
            {
                "ege": [
                    {
                        "informationsGenerales": {
                            "numFinessEge": "440002947",
                            "siret": "12345678900011",
                        }
                    }
                ]
            }
        ],
    }

    def fake_fetcher(url: str, timeout_seconds: float) -> HttpResponse:
        del timeout_seconds
        if url == FINESS_DATASET_API:
            body = json.dumps(
                {
                    "resources": [
                        {
                            "latest": "https://producer.test/finess.json.gz",
                            "last_modified": "2026-07-27",
                        }
                    ]
                }
            ).encode()
        elif url == "https://producer.test/finess.json.gz":
            body = gzip.compress(json.dumps(finess_snapshot).encode())
        elif "annuaire-education" in url:
            body = json.dumps(
                {
                    "total_count": 1,
                    "results": [
                        {
                            "identifiant_de_l_etablissement": "0141396S",
                            "siren_siret": "12345678900011",
                        }
                    ],
                }
            ).encode()
        elif "agencebio" in url:
            body = json.dumps(
                [{"numeroBio": 110415, "siret": "12345678900011"}]
            ).encode()
        elif "data.ademe" in url:
            body = json.dumps(
                {
                    "total": 1,
                    "results": [
                        {
                            "code_qualification": "43SPVRGE",
                            "siret": "12345678900011",
                        }
                    ],
                }
            ).encode()
        else:
            raise AssertionError(f"Unexpected URL: {url}")
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=body,
            final_url=url,
            collected_at="2026-07-27T12:00:00+00:00",
        )

    output_root = tmp_path / "builds"
    target = collect(
        evidence_path=evidence_path,
        output_root=output_root,
        requests_per_second=10.0,
        fetcher=fake_fetcher,
    )

    responses = pd.read_parquet(target / "producer_responses.parquet")
    summary = json.loads((target / "summary.json").read_text())
    assert len(responses) == 4
    assert responses["result_count"].eq(1).all()
    assert len(list((target / "raw").glob("*"))) == 5
    assert summary["adjudications_created"] == 0
    assert summary["correctness_labels_created"] == 0
    assert not any(
        "correct" in column.lower() or "label" in column.lower()
        for column in responses.columns
    )
    with pytest.raises(FileExistsError, match="Immutable"):
        collect(
            evidence_path=evidence_path,
            output_root=output_root,
            requests_per_second=10.0,
            fetcher=fake_fetcher,
        )
