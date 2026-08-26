#!/usr/bin/env python3
"""Run the label-blind native-SIREN v16 parent workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery, TantivyBackend
from src.xgb_matcher.official_evidence_tantivy import OfficialEvidenceTantivyBackend
from src.xgb_matcher.official_sector_evidence import OfficialSectorTantivyBackend
from src.xgb_matcher.rne_legal_entity_roles import LegalEntityRoleTantivyBackend
from src.xgb_matcher.retrieval_parent_v16 import (
    ParentChannelSpec,
    generate_batch_signals,
    load_query_selection,
    sha256_file,
    write_generation,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache


BACKEND_CLASSES = {
    "base": TantivyBackend,
    "official_overlay": OfficialEvidenceTantivyBackend,
    "temporal": OfficialEvidenceTantivyBackend,
    "legal_entity_roles": LegalEntityRoleTantivyBackend,
    "sector": OfficialSectorTantivyBackend,
}

GENERIC_NAME_FIELDS = {"names", "names_exact", "name_ngrams"}
GENERIC_ADDRESS_FIELDS = {"addresses", "addresses_exact", "address_ngrams"}
ALLOWED_FIELDS_BY_KIND = {
    "base": {
        "legal_names", "trade_names", "site_names",
        "legal_names_exact", "trade_names_exact", "site_names_exact",
        "current_name_ngrams", *GENERIC_ADDRESS_FIELDS,
    },
    "official_overlay": GENERIC_NAME_FIELDS | GENERIC_ADDRESS_FIELDS,
    "temporal": GENERIC_NAME_FIELDS | GENERIC_ADDRESS_FIELDS,
    "legal_entity_roles": GENERIC_NAME_FIELDS,
    "sector": GENERIC_NAME_FIELDS | GENERIC_ADDRESS_FIELDS,
}
EXPECTED_MANIFESTS = {
    "base": ("sireto-hierarchical-index-v3", None),
    "official_overlay": ("sireto-official-evidence-index-v1", None),
    "temporal": ("sireto-sirene-temporal-index-v2", "SIRENE_TEMPORAL_RESCUE"),
    "legal_entity_roles": ("sireto-rne-legal-entity-role-index-v1", "RNE_LEGAL_ENTITY_ROLE"),
    "sector": ("sireto-official-sector-index-v1", "OFFICIAL_SECTOR_EVIDENCE"),
}


def _swapouts() -> int:
    try:
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
        for line in output.splitlines():
            if line.startswith("Swapouts:"):
                return int(line.split(":", 1)[1].strip().rstrip("."))
    except Exception:
        return -1
    return -1


def _validate_authorities(config: dict, config_path: Path, queries_path: Path) -> None:
    authorities = config["authorities"]
    checks = [
        (queries_path, authorities["cohort_queries"]["sha256"]),
        (Path(authorities["cohort_manifest"]["path"]), authorities["cohort_manifest"]["sha256"]),
        (Path(authorities["scene_ir"]["path"]) / "manifest.json", authorities["scene_ir"]["manifest_sha256"]),
    ]
    checks.extend(
        (Path(record["path"]) / "manifest.json", record["manifest_sha256"])
        for record in authorities["backends"].values()
    )
    for path, expected in checks:
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"authority drifted: {path}")
    if sha256_file(config_path) == "":
        raise AssertionError("unreachable config hash")


def _build_backends(config: dict) -> dict[str, object]:
    output: dict[str, object] = {}
    for name, record in config["authorities"]["backends"].items():
        kind = str(record["kind"])
        if kind not in BACKEND_CLASSES:
            raise ValueError(f"unknown v16 backend kind: {kind}")
        backend = BACKEND_CLASSES[kind](record["path"])
        expected_schema, expected_overlay = EXPECTED_MANIFESTS[kind]
        if backend.manifest.get("schema_version") != expected_schema:
            raise ValueError(f"unexpected manifest schema for {name}")
        if expected_overlay and backend.manifest.get("overlay_kind") != expected_overlay:
            raise ValueError(f"unexpected overlay kind for {name}")
        output[name] = backend
    return output


def _validate_channel_fields(config: dict, specs: tuple[ParentChannelSpec, ...]) -> None:
    records = config["authorities"]["backends"]
    for spec in specs:
        if spec.backend not in records:
            raise ValueError(f"channel {spec.name} references unknown backend {spec.backend}")
        kind = str(records[spec.backend]["kind"])
        unknown = set(spec.fields) - ALLOWED_FIELDS_BY_KIND[kind]
        if unknown:
            raise ValueError(f"channel {spec.name} references unsupported fields: {sorted(unknown)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native-parent SIREN v16 workbench")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-query-count", type=int, default=0)
    args = parser.parse_args()
    config_path, queries_path = args.config.resolve(), args.queries.resolve()
    config = json.loads(config_path.read_text())
    _validate_authorities(config, config_path, queries_path)
    query_rows = load_query_selection(queries_path)
    if len(query_rows) != 1034:
        raise ValueError("sealed v16 workbench requires the 1,034-query v15 cohort")
    scope = "OFFICIAL_1034_QUERY_WORKBENCH"
    if args.preflight_query_count:
        if not 1 <= args.preflight_query_count <= 16:
            raise ValueError("preflight is limited to 1..16 queries")
        query_rows = query_rows[: args.preflight_query_count]
        scope = "NON_DECISIONAL_PREFLIGHT"
    query_fold = dict(query_rows)
    specs = tuple(ParentChannelSpec.from_mapping(raw) for raw in config["channels"])
    if len(specs) != len({spec.name for spec in specs}):
        raise ValueError("parent channel names must be unique")
    _validate_channel_fields(config, specs)
    backends = _build_backends(config)
    scene_cache = SceneIRCache(Path(config["authorities"]["scene_ir"]["path"]), verify_source_hash=False)
    columns = [
        "query_id", "fold", "crm_name", "crm_address", "crm_number",
        "crm_postcode", "crm_insee", "postcode", "insee",
    ]
    started, swap_before = time.perf_counter(), _swapouts()
    retrieval_queries: list[tuple[str, int, RetrievalQuery]] = []
    seen_queries: set[str] = set()
    for scene in scene_cache.iter_selected_scenes([query_id for query_id, _fold in query_rows], columns=columns):
        if query_fold[scene.query_id] != scene.fold:
            raise ValueError("query fold differs from sealed SceneIR")
        first = scene.rows.slice(0, 1).to_pylist()[0]
        query = RetrievalQuery.from_mapping(first)
        retrieval_queries.append((scene.query_id, scene.fold, query))
        seen_queries.add(scene.query_id)
    if seen_queries != set(query_fold):
        raise ValueError("SceneIR did not emit the complete selected population")
    signal_rows = generate_batch_signals(
        queries=retrieval_queries,
        backends=backends,
        specs=specs,
        workers=int(config["resources"]["search_workers"]),
    )
    wall, swap_after = time.perf_counter() - started, _swapouts()
    resources = {
        "wall_seconds": wall,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "swapout_pages_before": swap_before,
        "swapout_pages_after": swap_after,
        "swapout_pages_delta": swap_after - swap_before if min(swap_before, swap_after) >= 0 else None,
    }
    limits = config["resources"]
    resources["pass"] = bool(
        wall <= float(limits["maximum_wall_seconds"])
        and resources["peak_rss_bytes"] <= int(limits["maximum_peak_rss_gib"]) * 1024**3
        and resources["swapout_pages_delta"] == int(limits["maximum_swapout_pages"])
    )
    manifest = write_generation(
        args.output_dir,
        signal_rows,
        {
            "config_sha256": sha256_file(config_path),
            "queries_sha256": sha256_file(queries_path),
            "query_count": len(query_rows),
            "scope": scope,
            "channel_count": len(specs),
            "resources": resources,
            "truth_columns_opened": False,
            "positive_injection": False,
            "fold0_opened": False,
            "fold1_opened": False,
        },
    )
    if scope == "OFFICIAL_1034_QUERY_WORKBENCH" and not resources["pass"]:
        raise RuntimeError(f"STOP_RESOURCE_POLICY: {resources}")
    print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest), "resources": resources}, sort_keys=True))


if __name__ == "__main__":
    main()
