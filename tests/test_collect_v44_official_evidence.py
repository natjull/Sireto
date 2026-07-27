from __future__ import annotations

from types import SimpleNamespace

from scripts.collect_v44_official_evidence import build_queries


def test_v44_queries_cover_distinct_sirets_and_name_geo() -> None:
    row = SimpleNamespace(
        top1_siret="78983652500020",
        input_siret="21770445100210",
        SITE="Médiathèque Jacques Prévert",
        CODE_INSEE="77445",
        CODE_POSTAL="77176",
    )

    queries = build_queries(row)

    assert [query["query_kind"] for query in queries] == [
        "TOP1_SIRET",
        "INPUT_SIRET",
        "CRM_NAME_GEO",
    ]
    assert queries[0]["params"]["q"] == "78983652500020"
    assert queries[2]["params"]["code_commune"] == "77445"
    assert queries[2]["params"]["per_page"] == "10"


def test_v44_queries_do_not_repeat_same_siret() -> None:
    row = SimpleNamespace(
        top1_siret="11111111100011",
        input_siret="11111111100011",
        SITE="Entreprise exemple",
        CODE_INSEE="",
        CODE_POSTAL="75001",
    )

    queries = build_queries(row)

    assert [query["query_kind"] for query in queries] == [
        "TOP1_SIRET",
        "CRM_NAME_GEO",
    ]
    assert queries[1]["params"]["code_postal"] == "75001"
