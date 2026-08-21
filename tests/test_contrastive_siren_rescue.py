from __future__ import annotations

import numpy as np
import pandas as pd

from src.xgb_matcher.contrastive_siren_rescue import (
    AdapterConfig,
    apply_adapter,
    candidate_text,
    fit_linear_adapters,
    query_text,
    rank_sites,
    siren_documents,
    top_sirens,
)


def test_texts_never_include_identifiers() -> None:
    query = query_text(
        {"crm_name": "École Alpha", "crm_address": "1 rue B", "crm_insee": "75056"}
    )
    candidate = candidate_text(
        {
            "siret": "12345678900010",
            "siren": "123456789",
            "denomination": "Commune Alpha",
            "libelleVoie": "B",
        }
    )
    assert "75056" in query
    assert "123456789" not in candidate


def test_siren_documents_are_bounded_and_deterministic() -> None:
    frame = pd.DataFrame(
        [
            {"siren": "123456789", "denomination": "Alpha", "siret": "1"},
            {"siren": "123456789", "denomination": "Beta", "siret": "2"},
            {"siren": "987654321", "denomination": "Gamma", "siret": "3"},
        ]
    )
    documents = siren_documents(frame)
    assert documents["siren"].tolist() == ["123456789", "987654321"]
    assert "Alpha" in documents.iloc[0]["document_text"]


def test_adapter_and_top_sirens() -> None:
    queries = np.eye(3, dtype=np.float32)
    positives = queries.copy()
    negatives = np.roll(queries, 1, axis=0)
    q_weight, d_weight, history = fit_linear_adapters(
        queries,
        positives,
        negatives,
        AdapterConfig(),
    )
    q = apply_adapter(queries, q_weight)
    d = apply_adapter(positives, d_weight)
    ranking = top_sirens(q[:1], d, ["1", "2", "3"], 2)[0]
    assert ranking[0][0] == "1"
    assert history[-1]["triplet_margin_loss"] <= history[0]["triplet_margin_loss"]


def test_site_ranking_prefers_address_before_headquarters() -> None:
    sites = pd.DataFrame(
        [
            {
                "siren": "123456789",
                "siret": "12345678900001",
                "numeroVoie": "9",
                "typeVoie": "RUE",
                "libelleVoie": "LOIN",
                "etat_admin": "A",
                "is_siege": True,
            },
            {
                "siren": "123456789",
                "siret": "12345678900002",
                "numeroVoie": "1",
                "typeVoie": "RUE",
                "libelleVoie": "ALPHA",
                "etat_admin": "F",
                "is_siege": False,
            },
        ]
    )
    ranked = rank_sites(
        sites,
        {"crm_number": "1", "crm_address": "RUE ALPHA"},
        "123456789",
        2,
    )
    assert ranked.iloc[0]["siret"] == "12345678900002"
