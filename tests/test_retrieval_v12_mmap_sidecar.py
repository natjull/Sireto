from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from src.xgb_matcher.retrieval_sidecars_v11 import (
    RETRIEVAL_SIDECAR_SCHEMA_VERSION,
    RetrievalSidecarReader,
)
from src.xgb_matcher.retrieval_v12_mmap_sidecar import (
    MMAP_SIDECAR_MANIFEST_VERSION,
    MMapLocalSitesReader,
    SITE_COLUMNS,
    build_mmap_local_sites,
    sha256_file,
)
from src.xgb_matcher.retrieval_v12_reader_parity import (
    load_mmap_reader_parity_evidence_v12,
    run_mmap_reader_parity_v12,
)


def _site(
    siret: str,
    *,
    insee: str,
    postcode: str,
    number: str,
    street: str,
    address: str,
    state: str,
    headquarters: bool,
) -> dict[str, object]:
    return {
        "siret": siret,
        "siren": siret[:9],
        "insee": insee,
        "postcode": postcode,
        "street_number": number,
        "street_number_suffix": "",
        "street_normalized": street,
        "current_address_normalized": address,
        "administrative_state": state,
        "is_headquarters": headquarters,
    }


def _sidecars(tmp_path: Path) -> tuple[Path, pa.Table]:
    rows = [
        _site(
            "11111111100001",
            insee="75056",
            postcode="75001",
            number="10",
            street="RUE ALPHA",
            address="10 RUE ALPHA 75001 PARIS",
            state="F",
            headquarters=True,
        ),
        _site(
            "11111111100002",
            insee="75056",
            postcode="75001",
            number="10",
            street="RUE ALPHA",
            address="10 RUE ALPHA 75001 PARIS",
            state="A",
            headquarters=False,
        ),
        _site(
            "11111111100003",
            insee="75056",
            postcode="75001",
            number="12",
            street="RUE BETA",
            address="12 RUE BETA 75001 PARIS",
            state="A",
            headquarters=True,
        ),
        _site(
            "11111111100004",
            insee="92050",
            postcode="92000",
            number="1",
            street="RUE GAMMA",
            address="1 RUE GAMMA 92000 NANTERRE",
            state="A",
            headquarters=False,
        ),
        _site(
            "22222222200001",
            insee="75056",
            postcode="75001",
            number="10",
            street="RUE ALPHA",
            address="10 RUE ALPHA 75001 PARIS",
            state="A",
            headquarters=True,
        ),
        _site(
            "33333333300001",
            insee="33063",
            # The national sidecar contains a few non-canonical postcode
            # strings. Reader parity must prefer the valid INSEE block and
            # must not forward such a value to the strict query normalizer.
            postcode="33000 CEDEX",
            number="5",
            street="COURS TEST",
            address="5 COURS TEST 33000 BORDEAUX",
            state="F",
            headquarters=False,
        ),
    ]
    rows.sort(
        key=lambda row: (
            row["siren"],
            row["insee"],
            row["postcode"],
            row["street_number"],
            row["street_normalized"],
            row["siret"],
        )
    )
    local = pa.Table.from_pylist(rows)
    sidecar_dir = tmp_path / "v11-sidecars"
    sidecar_dir.mkdir()
    local_path = sidecar_dir / "siren_local_sites.parquet"
    pq.write_table(local, local_path, row_group_size=2)

    insee_counts: dict[str, int] = {}
    for row in rows:
        insee_counts[str(row["insee"])] = insee_counts.get(str(row["insee"]), 0) + 1
    geo_rows = [dict(row, insee_site_count=insee_counts[str(row["insee"])]) for row in rows]
    geo_rows.sort(
        key=lambda row: (
            row["insee"],
            row["postcode"],
            row["street_number"],
            row["street_normalized"],
            row["siret"],
        )
    )
    geo_path = sidecar_dir / "geo_site_cells.parquet"
    pq.write_table(pa.Table.from_pylist(geo_rows), geo_path, row_group_size=2)
    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (geo_path, local_path)
    }
    (sidecar_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": RETRIEVAL_SIDECAR_SCHEMA_VERSION,
                "files": files,
                "counts": {
                    "geo_site_cells": len(rows),
                    "siren_local_sites": len(rows),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return sidecar_dir, local


def test_mmap_store_preserves_source_content_and_is_content_addressed(
    tmp_path: Path,
) -> None:
    sidecar_dir, source = _sidecars(tmp_path)
    output_root = tmp_path / "external-mmap"
    first = build_mmap_local_sites(
        sidecar_dir=sidecar_dir, output_root=output_root, batch_size=2
    )
    second = build_mmap_local_sites(
        sidecar_dir=sidecar_dir, output_root=output_root, batch_size=2
    )
    assert first.output_dir == second.output_dir
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MMAP_SIDECAR_MANIFEST_VERSION
    assert manifest["site_order"] == "SOURCE_ORDER_PRESERVED"
    assert manifest["human_labels_opened"] is False
    assert manifest["site_rows"] == source.num_rows
    assert manifest["distinct_sirens"] == 3

    with pa.memory_map(str(first.output_dir / "sites.arrow"), "r") as mapped:
        reader = ipc.open_file(mapped)
        replay = pa.Table.from_batches(
            [reader.get_batch(index) for index in range(reader.num_record_batches)]
        )
    assert replay.select(list(SITE_COLUMNS)).equals(source.select(list(SITE_COLUMNS)))


def test_mmap_lookup_has_exact_parity_with_current_reader(tmp_path: Path) -> None:
    sidecar_dir, _source = _sidecars(tmp_path)
    build = build_mmap_local_sites(
        sidecar_dir=sidecar_dir,
        output_root=tmp_path / "external-mmap",
        # SIREN 111... crosses several batches: this exercises global slices.
        batch_size=2,
    )
    cases = [
        {
            "siren": "111111111",
            "insee": "75056",
            "address": "10 rue alpha 75001 Paris",
            "street_number": "10",
            "street": "rue alpha",
            "include_active": True,
            "include_closed": True,
            "max_sites": 2,
        },
        {
            "siren": "111111111",
            "postcode": "75001",
            "address": "12 rue beta 75001 Paris",
            "street_number": "12",
            "street": "rue beta",
            "include_active": True,
            "include_closed": False,
            "max_sites": 20,
        },
        {
            "siren": "111111111",
            "insee": "75056",
            "include_active": False,
            "include_closed": True,
            "max_sites": 1,
        },
        {
            "siren": "999999999",
            "insee": "75056",
            "include_active": True,
            "include_closed": True,
            "max_sites": 5,
        },
    ]
    with RetrievalSidecarReader(sidecar_dir) as current, MMapLocalSitesReader(
        build.output_dir
    ) as mmap_reader:
        for case in cases:
            assert mmap_reader.sites_by_siren_geo(**case) == current.sites_by_siren_geo(
                **case
            )

        batch_args = {
            "sirens": ["222222222", "111111111", "999999999"],
            "insee": "75056",
            "address": "10 RUE ALPHA 75001 PARIS",
            "street_number": "10",
            "street": "RUE ALPHA",
            "include_active": True,
            "include_closed": True,
            "max_sites_per_siren": 2,
        }
        assert mmap_reader.sites_by_sirens_geo(
            **batch_args
        ) == current.sites_by_sirens_geo(**batch_args)


def test_mmap_lookup_empty_state_policy_matches_current_reader(tmp_path: Path) -> None:
    sidecar_dir, _source = _sidecars(tmp_path)
    build = build_mmap_local_sites(
        sidecar_dir=sidecar_dir,
        output_root=tmp_path / "external-mmap",
        batch_size=3,
    )
    arguments = {
        "sirens": ["111111111", "222222222"],
        "insee": "75056",
        "include_active": False,
        "include_closed": False,
        "max_sites_per_siren": 7,
    }
    with RetrievalSidecarReader(sidecar_dir) as current, MMapLocalSitesReader(
        build.output_dir
    ) as mmap_reader:
        assert mmap_reader.sites_by_sirens_geo(
            **arguments
        ) == current.sites_by_sirens_geo(**arguments)


def test_reader_parity_is_separate_hash_bound_and_recomputed(tmp_path: Path) -> None:
    import pytest

    sidecar_dir, _source = _sidecars(tmp_path)
    build = build_mmap_local_sites(
        sidecar_dir=sidecar_dir,
        output_root=tmp_path / "external-mmap",
        batch_size=2,
    )
    evidence_path = run_mmap_reader_parity_v12(
        sidecar_dir=sidecar_dir,
        mmap_dir=build.output_dir,
        output_dir=tmp_path / "reader-parity",
        case_count=4,
    )
    official_manifest = json.loads(
        (sidecar_dir / "manifest.json").read_text(encoding="utf-8")
    )
    mmap_manifest = json.loads(build.manifest_path.read_text(encoding="utf-8"))
    local_record = official_manifest["files"]["siren_local_sites.parquet"]
    loaded, digest = load_mmap_reader_parity_evidence_v12(
        evidence_path,
        expected_official_manifest_sha256=sha256_file(
            sidecar_dir / "manifest.json"
        ),
        expected_mmap_manifest_sha256=sha256_file(build.manifest_path),
        expected_source_local_sites_sha256=local_record["sha256"],
        expected_source_rows=official_manifest["counts"]["siren_local_sites"],
        expected_mmap_build_id=mmap_manifest["build_id"],
        expected_case_count=4,
    )
    assert loaded["status"] == "PASS"
    assert len(digest) == 64

    rows_path = evidence_path.parent / "reader_parity_rows.parquet"
    frame = pq.read_table(rows_path).to_pandas()
    frame.loc[0, "order_identical"] = False
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), rows_path)
    with pytest.raises(Exception, match="row hash mismatch"):
        load_mmap_reader_parity_evidence_v12(
            evidence_path,
            expected_official_manifest_sha256=sha256_file(
                sidecar_dir / "manifest.json"
            ),
            expected_mmap_manifest_sha256=sha256_file(build.manifest_path),
            expected_source_local_sites_sha256=local_record["sha256"],
            expected_source_rows=official_manifest["counts"]["siren_local_sites"],
            expected_mmap_build_id=mmap_manifest["build_id"],
            expected_case_count=4,
        )


def test_reader_parity_rejects_rehashed_false_row_and_wrong_lineage(
    tmp_path: Path,
) -> None:
    import pytest

    sidecar_dir, _source = _sidecars(tmp_path)
    build = build_mmap_local_sites(
        sidecar_dir=sidecar_dir,
        output_root=tmp_path / "external-mmap",
        batch_size=2,
    )
    evidence_path = run_mmap_reader_parity_v12(
        sidecar_dir=sidecar_dir,
        mmap_dir=build.output_dir,
        output_dir=tmp_path / "reader-parity",
        case_count=4,
    )
    official_manifest = json.loads(
        (sidecar_dir / "manifest.json").read_text(encoding="utf-8")
    )
    mmap_manifest = json.loads(build.manifest_path.read_text(encoding="utf-8"))
    local_sha = official_manifest["files"]["siren_local_sites.parquet"]["sha256"]
    common = {
        "expected_official_manifest_sha256": sha256_file(
            sidecar_dir / "manifest.json"
        ),
        "expected_mmap_manifest_sha256": sha256_file(build.manifest_path),
        "expected_source_local_sites_sha256": local_sha,
        "expected_source_rows": official_manifest["counts"]["siren_local_sites"],
        "expected_mmap_build_id": mmap_manifest["build_id"],
        "expected_case_count": 4,
    }
    with pytest.raises(Exception, match="lineage mismatch"):
        load_mmap_reader_parity_evidence_v12(
            evidence_path,
            **{**common, "expected_mmap_build_id": "0" * 64},
        )

    rows_path = evidence_path.parent / "reader_parity_rows.parquet"
    frame = pq.read_table(rows_path).to_pandas()
    frame.loc[0, "result_byte_identical"] = False
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), rows_path)
    manifest = json.loads(evidence_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["reader_parity_rows"]["sha256"] = sha256_file(
        rows_path
    )
    evidence_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(Exception, match="row claims drifted"):
        load_mmap_reader_parity_evidence_v12(evidence_path, **common)


def test_reader_parity_publication_is_byte_deterministic(tmp_path: Path) -> None:
    sidecar_dir, _source = _sidecars(tmp_path)
    build = build_mmap_local_sites(
        sidecar_dir=sidecar_dir,
        output_root=tmp_path / "external-mmap",
        batch_size=2,
    )
    first = run_mmap_reader_parity_v12(
        sidecar_dir=sidecar_dir,
        mmap_dir=build.output_dir,
        output_dir=tmp_path / "reader-parity-first",
        case_count=4,
    )
    second = run_mmap_reader_parity_v12(
        sidecar_dir=sidecar_dir,
        mmap_dir=build.output_dir,
        output_dir=tmp_path / "reader-parity-second",
        case_count=4,
    )
    assert first.read_bytes() == second.read_bytes()
    assert sha256_file(first.parent / "reader_parity_rows.parquet") == sha256_file(
        second.parent / "reader_parity_rows.parquet"
    )
