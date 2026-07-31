from __future__ import annotations

from pathlib import Path


CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "docs/v4_12_service_parity_latency_contract.md"
)


def test_service_contract_freezes_the_real_stack_and_label_boundary() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    for required in (
        "retrieval sparse V4.12, top 100 strict",
        "45 features Ranker C",
        "80 features de scène V4.11",
        "seuil 0.8720916706888049",
        "veto V4.12-G",
        "1 456 requêtes `dev`",
        "145 236 candidats",
        "`is_ground_truth`",
        "`acceptor_target`",
        "tolérance `1e-15`",
        "`sealed_key_miss_count == 0`",
        "`lookup_missing_count == 0`",
        "`p95(V4.12-G) < 2 × p95(V4.11)`",
        "pic RSS de chaque processus inférieur à 8 Gio",
        "`GO_V412_SERVICE_FREEZE`",
    ):
        assert required in text


def test_service_contract_cannot_claim_product_precision() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "ne prouve pas une précision réelle de" in text
    assert "nouvel export CRM indépendant" in normalized
    assert "aucun nouveau test" in text
