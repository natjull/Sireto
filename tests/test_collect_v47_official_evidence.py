from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.collect_v47_official_evidence import collect


def test_collect_v47_official_evidence_is_bound_to_current_top1(
    tmp_path: Path,
) -> None:
    docket = pd.DataFrame(
        [
            {
                "audit_case_id": "case-1",
                "service_id": "svc-1",
                "siret_to_adjudicate": "22222222200002",
                "input_siret": "22222222200001",
                "SITE": "EXEMPLE",
                "CODE_INSEE": "75101",
                "CODE_POSTAL": "75001",
            }
        ]
    )
    docket_path = tmp_path / "docket.parquet"
    docket.to_parquet(docket_path, index=False)
    observed: list[dict[str, str]] = []

    def requester(params, *, timeout_seconds):
        observed.append(dict(params))
        return 200, {"total_results": 1, "results": []}

    target = collect(
        docket_path=docket_path,
        output_root=tmp_path / "out",
        requests_per_second=6.0,
        enforce_canonical=False,
        requester=requester,
    )
    evidence = pd.read_parquet(target / "official_evidence.parquet")
    assert set(evidence["query_kind"]) == {
        "TOP1_SIRET",
        "INPUT_SIRET",
        "CRM_NAME_GEO",
    }
    assert observed[0]["q"] == "22222222200002"
    assert evidence["siret_to_adjudicate"].eq("22222222200002").all()
    assert evidence["independence_group"].eq("SIRENE_REGISTRY").all()
