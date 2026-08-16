#!/usr/bin/env python3
"""Run counted balanced synthetic-GT batches until the sealed target is met.

This coordinator never creates or edits CRM text.  It only composes the
existing selector, Luna driver, deterministic validators, full-SIRENE audit,
exact promoter, and cumulative registry.  Every stage is restartable from its
sealed artifact; a failed command stops the loop before the next batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "synthetic_gt_corpus/balanced_v1"
)
DEFAULT_TEMP_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/"
    "duckdb_synthetic_gt_full_exact_balanced_v1"
)


def run(command: list[str]) -> None:
    print(json.dumps({"exec": command}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def checkpoint(db: Path) -> None:
    connection = sqlite3.connect(db)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    finally:
        connection.close()


def next_attempted_variants(
    promoted: int, target: int, batch_target_count: int,
) -> int:
    if not 0 <= promoted < target or batch_target_count <= 0:
        raise ValueError("invalid residual production counts")
    return min(batch_target_count * 3, target - promoted)


def batch_paths(output_root: Path, batch_id: str) -> dict[str, Path]:
    return {
        "seed": output_root / f"{batch_id}_seed_input.jsonl",
        "db": output_root / f"{batch_id}.sqlite",
        "artifacts": output_root / f"{batch_id}_artifacts",
        "export": output_root / f"{batch_id}_export",
        "audit": output_root / f"{batch_id}_full_sirene_audit.json",
        "promoted": output_root / f"{batch_id}_promoted",
    }


def ensure_batch(
    args: argparse.Namespace, batch_number: int, attempted_variants: int,
) -> int:
    batch_id = f"P{batch_number:03d}"
    run_id = f"synthetic-gt-balanced-v1-{batch_id}"
    paths = batch_paths(args.output_root, batch_id)

    if not paths["seed"].exists():
        run([
            sys.executable, "scripts/select_synthetic_gt_balanced_production.py",
            "--plan", str(args.selection_plan),
            "--output", str(paths["seed"]),
            "--batch-id", batch_id,
            "--target-count", str(args.batch_target_count),
            "--variant-count", str(attempted_variants),
            "--selection-seed", f"SIRETO-BALANCED-{batch_id}",
            "--production-registry", str(args.registry),
            "--selection-pool-limit", str(args.selection_pool_limit),
        ])
    seed_manifest = paths["seed"].with_suffix(
        paths["seed"].suffix + ".manifest.json"
    )
    if not seed_manifest.exists():
        raise FileNotFoundError(seed_manifest)
    planned_variants = int(load_json(seed_manifest).get("planned_variants", -1))
    if planned_variants != attempted_variants:
        raise RuntimeError(
            f"{batch_id} frozen seed plans {planned_variants} variants but the "
            f"current residual requires {attempted_variants}"
        )

    if not paths["db"].exists():
        run([
            sys.executable, "scripts/run_synthetic_gt_agentic_loop.py",
            "--db", str(paths["db"]), "init",
            "--run-id", run_id,
            "--seeds", str(paths["seed"]),
            "--schema", str(args.message_schema),
            "--critic-mode", "all",
            "--max-generator-attempts", "3",
            "--generator-task-mode", "per-variant",
            "--allow-partial-seed-promotion",
        ])

    if not (paths["export"] / "manifest.json").exists():
        run([
            "caffeinate", "-dimsu", sys.executable,
            "scripts/run_synthetic_gt_luna_driver.py",
            "--db", str(paths["db"]),
            "--run-id", run_id,
            "--artifacts", str(paths["artifacts"]),
            "--export", str(paths["export"]),
            "--schema", str(args.message_schema),
            "--model", "gpt-5.6-luna",
            "--generator-reasoning-effort", "low",
            "--critic-reasoning-effort", "high",
            "--adjudicator-reasoning-effort", "max",
            "--concurrency", str(args.concurrency),
            "--lease-ttl-seconds", "1800",
            "--agent-timeout-seconds", "180",
            "--transport-retries", "2",
        ])

    checkpoint(paths["db"])
    if not paths["audit"].exists():
        run([
            sys.executable, "scripts/audit_synthetic_gt_full_sirene_exact.py",
            "--db", str(paths["db"]),
            "--run-id", run_id,
            "--plan", str(args.corpus_plan),
            "--output", str(paths["audit"]),
            "--temp-directory", str(args.temp_root / batch_id),
        ])

    promotion_manifest = paths["promoted"] / "manifest.json"
    if not promotion_manifest.exists():
        run([
            sys.executable, "scripts/promote_synthetic_gt_full_exact.py",
            "--db", str(paths["db"]),
            "--run-id", run_id,
            "--diagnostic-export", str(paths["export"]),
            "--full-audit", str(paths["audit"]),
            "--seed-input", str(paths["seed"]),
            "--output", str(paths["promoted"]),
            "--promotion-mode", "per-variant",
        ])

    manifest = load_json(promotion_manifest)
    promoted = int(manifest.get("promoted_variants", 0))
    if promoted < int(attempted_variants * args.minimum_promotion_rate):
        raise RuntimeError(
            f"{batch_id} promotion yield {promoted}/{attempted_variants} is below "
            f"{args.minimum_promotion_rate:.1%}"
        )
    run([
        sys.executable, "scripts/manage_synthetic_gt_balanced_registry.py",
        "register", "--registry", str(args.registry),
        "--target", str(args.target),
        "--batch-id", batch_id,
        "--seed-input", str(paths["seed"]),
        "--promoted", str(paths["promoted"] / "promoted.jsonl"),
        "--promotion-manifest", str(promotion_manifest),
    ])
    return promoted


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    result.add_argument(
        "--registry", type=Path,
        default=DEFAULT_OUTPUT_ROOT / "production_registry.json",
    )
    result.add_argument(
        "--selection-plan", type=Path,
        default=ROOT / "config/synthetic_gt_balanced_v1_plan.json",
    )
    result.add_argument(
        "--corpus-plan", type=Path,
        default=ROOT / "config/synthetic_gt_corpus_plan.json",
    )
    result.add_argument(
        "--message-schema", type=Path,
        default=ROOT / "config/synthetic_gt_agentic_message_schema_v2.json",
    )
    result.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP_ROOT)
    result.add_argument("--target", type=int, default=20_000)
    result.add_argument("--batch-target-count", type=int, default=200)
    result.add_argument("--selection-pool-limit", type=int, default=6000)
    result.add_argument("--start-batch", type=int, default=1)
    result.add_argument("--concurrency", type=int, default=64)
    result.add_argument("--minimum-promotion-rate", type=float, default=0.70)
    result.add_argument(
        "--maximum-batches", type=int, default=0,
        help="Optional safety bound; zero means continue until the target.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if not 1 <= args.concurrency <= 128:
        raise ValueError("concurrency must be between 1 and 128")
    if not 0 < args.minimum_promotion_rate <= 1:
        raise ValueError("minimum promotion rate must be in (0, 1]")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.temp_root.mkdir(parents=True, exist_ok=True)
    batch_number = args.start_batch
    completed_batches = 0
    while True:
        registry = load_json(args.registry)
        promoted = int(registry.get("summary", {}).get("promoted_variants", 0))
        if promoted >= args.target:
            print(json.dumps({"status": "COMPLETE", "promoted": promoted}, indent=2))
            return
        if args.maximum_batches and completed_batches >= args.maximum_batches:
            print(json.dumps({"status": "BOUND_REACHED", "promoted": promoted}, indent=2))
            return
        attempted_variants = next_attempted_variants(
            promoted, args.target, args.batch_target_count,
        )
        ensure_batch(args, batch_number, attempted_variants)
        batch_number += 1
        completed_batches += 1


if __name__ == "__main__":
    main()
