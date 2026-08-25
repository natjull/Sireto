#!/usr/bin/env python3
"""Run the label-blind 1,034-query Master@2000 feasibility generation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_evidence_retrieval import OfficialEvidenceRetrievalConfig, OfficialEvidenceRetriever
from src.xgb_matcher.retrieval_master_v15 import MasterV15Recipe, compose_query, load_query_selection, sha256_file, write_generation
from src.xgb_matcher.retrieval_sidecars_v11 import RetrievalSidecarReader
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIRCache


def _swapouts() -> int:
    try:
        output = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
        for line in output.splitlines():
            if line.startswith("Swapouts:"):
                return int(line.split(":", 1)[1].strip().rstrip("."))
    except Exception:
        return -1
    return -1


def _validate_authorities(config: dict, config_path: Path) -> None:
    authorities = config["authorities"]
    checks = [
        (config_path.parent.parent / authorities["official_config"]["path"], authorities["official_config"]["sha256"]),
        (Path(authorities["raw_union_manifest"]["path"]), authorities["raw_union_manifest"]["sha256"]),
        (Path(authorities["scene_ir"]["path"]) / "manifest.json", authorities["scene_ir"]["manifest_sha256"]),
        (Path(authorities["sidecars"]["path"]) / "manifest.json", authorities["sidecars"]["manifest_sha256"]),
        (Path(authorities["base_index"]["path"]) / "manifest.json", authorities["base_index"]["manifest_sha256"]),
    ]
    checks.extend((Path(record["path"]) / "manifest.json", record["manifest_sha256"]) for record in authorities["overlay_indices"])
    for path, expected in checks:
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"authority drifted: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run label-blind Master v15 feasibility generation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    _validate_authorities(config, config_path)
    query_rows = load_query_selection(args.queries)
    if len(query_rows) != 1034:
        raise ValueError("official pilot requires exactly 1,034 queries")
    query_fold = dict(query_rows)
    recipe = MasterV15Recipe.from_config(config)
    authorities = config["authorities"]
    official_config_path = ROOT / authorities["official_config"]["path"]
    official_config = replace(
        OfficialEvidenceRetrievalConfig.load(official_config_path),
        enable_direct_parent_expansion=False,
        enable_consensus_parent_expansion=False,
        search_workers=12,
    )
    official = OfficialEvidenceRetriever.from_index_paths(
        authorities["base_index"]["path"],
        [record["path"] for record in authorities["overlay_indices"]],
        official_config,
    )
    started, swap_before = time.perf_counter(), _swapouts()
    candidates: list[dict] = []
    parents: list[dict] = []
    scene_cache = SceneIRCache(Path(authorities["scene_ir"]["path"]), verify_source_hash=False)
    columns = ["query_id", "fold", "crm_name", "crm_address", "crm_number", "crm_postcode", "crm_insee", "postcode", "insee", "union_rank", "siret", "siren"]
    try:
        with RetrievalSidecarReader(Path(authorities["sidecars"]["path"]), verify_hashes=True) as sidecars:
            for scene in scene_cache.iter_selected_scenes([query_id for query_id, _ in query_rows], columns=columns):
                if query_fold[scene.query_id] != scene.fold:
                    raise ValueError("query fold differs from sealed SceneIR")
                query_candidates, query_parents = compose_query(
                    query_id=scene.query_id, fold=scene.fold, source_rows=scene.rows.to_pylist(),
                    backends=official.backends, sidecars=sidecars, recipe=recipe,
                )
                candidates.extend(query_candidates)
                parents.extend(query_parents)
    finally:
        official._search_pool.shutdown(wait=True)
    wall = time.perf_counter() - started
    swap_after = _swapouts()
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB. This project executes on macOS.
    resources = {
        "wall_seconds": wall, "peak_rss_bytes": peak_rss_bytes,
        "swapout_pages_before": swap_before, "swapout_pages_after": swap_after,
        "swapout_pages_delta": (swap_after - swap_before if min(swap_before, swap_after) >= 0 else None),
    }
    limits = config["resources"]
    resources["pass"] = bool(
        wall <= float(limits["maximum_wall_seconds"])
        and peak_rss_bytes <= int(limits["maximum_peak_rss_gib"]) * 1024**3
        and resources["swapout_pages_delta"] == int(limits["maximum_swapout_pages"])
    )
    manifest = write_generation(
        args.output_dir, candidates, parents,
        {"config_sha256": sha256_file(config_path), "queries_sha256": sha256_file(args.queries), "query_count": len(query_rows), "recipe": config["recipe"], "resources": resources, "truth_columns_opened": False, "positive_injection": False},
    )
    if not resources["pass"]:
        raise RuntimeError(f"STOP_RESOURCE_POLICY: {resources}")
    print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest), "resources": resources}, sort_keys=True))


if __name__ == "__main__":
    main()
