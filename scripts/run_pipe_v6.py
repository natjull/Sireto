"""CLI entry point to run the Pipe V6 pipeline (task 10.1)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging

# Ensure local src/ is importable when the script is run from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipe_v6.config import load_config
from pipe_v6.logging_utils import setup_logging
from pipe_v6.pipeline import run_pipeline
from pipe_v6.exporter import export_results, compute_stats, save_stats_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pipe V6 pipeline")
    parser.add_argument("--config", "-c", type=Path, help="Path to config YAML")
    parser.add_argument("--crm-path", type=Path, help="Override CRM input path")
    parser.add_argument("--output-path", "-o", type=Path, help="Override output path (.csv or .json)")
    parser.add_argument("--max-rows", type=int, help="Limit number of CRM rows processed")
    parser.add_argument("--stats-json", type=Path, help="Optional path to write stats JSON")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level (also applied to file logger)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Load configuration (YAML/env/defaults)
    config = load_config(args.config)

    # Apply CLI overrides
    if args.crm_path:
        config.crm_path = args.crm_path.expanduser().resolve()
    if args.output_path:
        config.output_path = args.output_path.expanduser().resolve()
    if args.log_level:
        config.log_level = args.log_level

    logger = setup_logging(config)

    try:
        df_results = run_pipeline(
            config,
            max_rows=args.max_rows,
            logger=logger,
        )

        # Export results (format inferred from suffix)
        export_results(df_results, config, args.output_path)

        stats = compute_stats(df_results)
        logger.info("Total rows: %d", stats["total_rows"])
        logger.info(
            "Match rate: %.1f%% — Review rate: %.1f%% — No-match rate: %.1f%%",
            stats["status"]["percentages"].get("MATCH", 0) * 100,
            stats["status"]["percentages"].get("REVIEW", 0) * 100,
            stats["status"]["percentages"].get("NO_MATCH", 0) * 100,
        )

        if args.stats_json:
            save_stats_json(stats, args.stats_json)
            logger.info("Stats JSON written to %s", args.stats_json)

    except Exception as exc:  # noqa: BLE001 - surface full context
        logger.exception("Pipeline failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
