from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.official_source_sync import canonical_json, sha256_file
from src.xgb_matcher.retrieval_sidecars_v11 import (
    MAX_QUERY_SITES,
    RETRIEVAL_SIDECAR_SCHEMA_VERSION,
    RetrievalSidecarReader,
    build_retrieval_sidecars,
)
from src.xgb_matcher.siren_dossier import SIREN_DOSSIER_SCHEMA_VERSION


def _site(
    *,
    siren: str,
    nic: str,
    insee: str,
    postcode: str,
    number: str,
    street_type: str,
    street: str,
    state: str = "A",
    headquarters: bool = False,
) -> dict[str, object]:
    siret = siren + nic
    address = " ".join(value for value in (number, street_type, street) if value)
    return {
        "siren": siren,
        "siret": siret,
        "nic": nic,
        "administrative_state": state,
        "is_headquarters": headquarters,
        "creation_date": None,
        "current_period_start": None,
        "insee": insee,
        "postcode": postcode,
        "street_number": number,
        "street_number_suffix": "",
        "street_type": street_type,
        "street_name": street,
        "address_complement": "",
        "current_address_normalized": address,
        "main_activity": "",
    }


def _write_dossier_v3(tmp_path: Path) -> Path:
    dossier = tmp_path / "dossier"
    dossier.mkdir()
    rows = [
        # Small commune: the all-sites shortcut fits a cap of three.
        _site(
            siren="100000001", nic="00011", insee="01001", postcode="01000",
            number="1", street_type="RUE", street="DES FLEURS", headquarters=True,
        ),
        _site(
            siren="100000002", nic="00012", insee="01001", postcode="01000",
            number="2", street_type="RUE", street="DES LILAS",
        ),
        # Large commune. The first SIREN deliberately has a closed site whose
        # address should outrank its active headquarters for a matching query.
        _site(
            siren="200000001", nic="00011", insee="69381", postcode="69001",
            number="10", street_type="RUE", street="ALPHA", headquarters=True,
        ),
        _site(
            siren="200000001", nic="00029", insee="69381", postcode="69001",
            number="12", street_type="RUE", street="BETA", state="F",
        ),
        _site(
            siren="200000002", nic="00013", insee="69381", postcode="69001",
            number="10", street_type="RUE", street="ALPHA",
        ),
        _site(
            siren="200000003", nic="00014", insee="69381", postcode="69001",
            number="10", street_type="RUE", street="GAMMA",
        ),
    ]
    establishments = dossier / "establishments.parquet"
    pq.write_table(pa.Table.from_pylist(rows), establishments)
    files = {
        "establishments.parquet": {
            "size_bytes": establishments.stat().st_size,
            "sha256": sha256_file(establishments),
        }
    }
    manifest = {
        "schema_version": SIREN_DOSSIER_SCHEMA_VERSION,
        "build_id": "d" * 64,
        "policy": {
            "crm_labels_present": False,
            "model_scores_present": False,
        },
        "files": files,
    }
    (dossier / "manifest.json").write_bytes(canonical_json(manifest))
    return dossier


def _build(tmp_path: Path):
    dossier = _write_dossier_v3(tmp_path)
    build = build_retrieval_sidecars(
        dossier_dir=dossier,
        output_root=tmp_path / "sidecars",
        threads=1,
        memory_limit="256MB",
    )
    return dossier, build


def test_build_is_content_addressed_deterministic_and_label_free(tmp_path: Path):
    dossier = _write_dossier_v3(tmp_path)
    first = build_retrieval_sidecars(
        dossier_dir=dossier,
        output_root=tmp_path / "first",
        threads=1,
        memory_limit="256MB",
    )
    second = build_retrieval_sidecars(
        dossier_dir=dossier,
        output_root=tmp_path / "second",
        threads=1,
        memory_limit="256MB",
    )
    assert first.build_id == second.build_id
    assert first.counts == second.counts == {
        "geo_site_cells": 6,
        "siren_local_sites": 6,
    }
    first_manifest = json.loads(first.manifest_path.read_text())
    second_manifest = json.loads(second.manifest_path.read_text())
    assert first_manifest["schema_version"] == RETRIEVAL_SIDECAR_SCHEMA_VERSION
    assert first_manifest["projection"]["crm_labels_present"] is False
    assert first_manifest["projection"]["model_scores_present"] is False
    assert first_manifest["files"] == second_manifest["files"]
    assert first_manifest["query_contract"]["address_precedes_headquarters"] is True
    for filename in ("geo_site_cells.parquet", "siren_local_sites.parquet"):
        columns = set(pq.read_schema(first.output_dir / filename).names)
        assert not any(
            marker in column.lower()
            for column in columns
            for marker in ("label", "ground_truth", "crm", "model_score")
        )
    # Restarting the same publication validates and reuses it.
    restarted = build_retrieval_sidecars(
        dossier_dir=dossier,
        output_root=tmp_path / "first",
        threads=1,
        memory_limit="256MB",
    )
    assert restarted.output_dir == first.output_dir


def test_small_commune_returns_all_sites_only_when_count_fits(tmp_path: Path):
    _dossier, build = _build(tmp_path)
    with RetrievalSidecarReader(build.output_dir) as reader:
        result = reader.all_sites_by_insee_if_fits("01001", max_sites=3)
        assert result.strategy == "INSEE_ALL"
        assert result.capped is False
        assert result.matched_count == 2
        assert [site.siret for site in result.sites] == [
            "10000000100011",
            "10000000200012",
        ]
        capped = reader.all_sites_by_insee_if_fits("01001", max_sites=1)
        assert capped.strategy == "INSEE_TOO_LARGE"
        assert capped.capped is True
        assert capped.matched_count == 2
        assert capped.sites == ()


def test_large_commune_uses_exact_street_then_number_fallback(tmp_path: Path):
    _dossier, build = _build(tmp_path)
    with RetrievalSidecarReader(build.output_dir) as reader:
        exact = reader.geo_candidates(
            insee="69381",
            postcode="69001",
            street_number="10",
            street="rue alpha",
            max_sites=2,
        )
        assert exact.strategy == "EXACT_GEO_CELL"
        assert exact.capped is False
        assert {site.siret for site in exact.sites} == {
            "20000000100011",
            "20000000200013",
        }

        number_only = reader.geo_candidates(
            insee="69381",
            postcode="69001",
            street_number="10",
            street="RUE INCONNUE",
            max_sites=3,
        )
        assert number_only.strategy == "NUMBER_CELL"
        assert [site.street_normalized for site in number_only.sites] == [
            "RUE ALPHA",
            "RUE ALPHA",
            "RUE GAMMA",
        ]

        too_many = reader.geo_candidates(
            insee="69381",
            postcode="69001",
            street_number="10",
            street="RUE INCONNUE",
            max_sites=2,
        )
        assert too_many.strategy == "NUMBER_CELL_TOO_LARGE"
        assert too_many.capped is True
        assert too_many.sites == ()

        insufficient = reader.geo_candidates(insee="69381", max_sites=2)
        assert insufficient.strategy == "INSUFFICIENT_LARGE_COMMUNE_GEO"
        assert insufficient.capped is True


def test_siren_geo_keeps_active_and_closed_and_orders_address_before_hq(tmp_path: Path):
    _dossier, build = _build(tmp_path)
    with RetrievalSidecarReader(build.output_dir) as reader:
        result = reader.sites_by_siren_geo(
            siren="200000001",
            insee="69381",
            address="12 rue beta",
            street_number="12",
            street="RUE BETA",
        )
        assert [site.siret for site in result.sites] == [
            "20000000100029",
            "20000000100011",
        ]
        assert result.sites[0].active is False
        assert result.sites[1].is_headquarters is True

        active_only = reader.sites_by_siren_geo(
            siren="200000001",
            insee="69381",
            include_closed=False,
        )
        assert [site.siret for site in active_only.sites] == ["20000000100011"]
        closed_only = reader.sites_by_siren_geo(
            siren="200000001",
            postcode="69001",
            include_active=False,
        )
        assert [site.siret for site in closed_only.sites] == ["20000000100029"]


def test_reader_cache_and_caps_are_bounded_and_sidecars_remain_unchanged(tmp_path: Path):
    _dossier, build = _build(tmp_path)
    paths = [
        build.output_dir / "geo_site_cells.parquet",
        build.output_dir / "siren_local_sites.parquet",
        build.output_dir / "manifest.json",
    ]
    before = {path: (path.stat().st_mtime_ns, sha256_file(path)) for path in paths}
    with RetrievalSidecarReader(build.output_dir, cache_size=2) as reader:
        first = reader.all_sites_by_insee_if_fits("01001", max_sites=3)
        first_info = reader.cache_info()["all_sites_by_insee"]
        second = reader.all_sites_by_insee_if_fits("01001", max_sites=3)
        second_info = reader.cache_info()["all_sites_by_insee"]
        assert second == first
        assert second_info.hits == first_info.hits + 1
        assert second_info.currsize == 1
        with pytest.raises(ValueError):
            reader.all_sites_by_insee_if_fits("01001", max_sites=0)
        with pytest.raises(ValueError):
            reader.all_sites_by_insee_if_fits(
                "01001", max_sites=MAX_QUERY_SITES + 1
            )
        with pytest.raises(ValueError):
            reader.sites_by_siren_geo(siren="200000001")
    after = {path: (path.stat().st_mtime_ns, sha256_file(path)) for path in paths}
    assert after == before


def test_builder_rejects_non_v3_or_unsealed_label_policy(tmp_path: Path):
    dossier = _write_dossier_v3(tmp_path)
    manifest_path = dossier / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "sireto-siren-dossier-v2"
    manifest_path.write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="require a SIREN dossier v3"):
        build_retrieval_sidecars(
            dossier_dir=dossier, output_root=tmp_path / "sidecars"
        )

    manifest["schema_version"] = SIREN_DOSSIER_SCHEMA_VERSION
    manifest["policy"]["crm_labels_present"] = True
    manifest_path.write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match="CRM labels are absent"):
        build_retrieval_sidecars(
            dossier_dir=dossier, output_root=tmp_path / "sidecars"
        )
