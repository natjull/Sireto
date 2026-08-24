#!/usr/bin/env python3
"""Seal historical v11 reader versus v12 mmap parity on external scratch."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (  # noqa: E402
    FoldFusedInvariantError,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import (  # noqa: E402
    load_executor_config_v12,
)
from src.xgb_matcher.retrieval_v12_reader_parity import (  # noqa: E402
    READER_PARITY_CASE_COUNT,
    run_mmap_reader_parity_v12,
)


DEFAULT_CONFIG = ROOT / "config/retrieval_all_human_v12.json"
DEFAULT_EXECUTOR_CONFIG = (
    ROOT / "config/retrieval_all_human_v12_fold_fused_executor.json"
)


def run(args: argparse.Namespace) -> Path:
    executor = load_executor_config_v12(
        args.executor_config,
        scientific_config_path=args.config,
        workspace_root=ROOT,
    )
    output = Path(args.output_dir).resolve()
    scratch = executor.scratch_root.resolve()
    if not scratch.is_dir() or output == scratch or scratch not in output.parents:
        raise FoldFusedInvariantError(
            "reader parity output must be a child of configured external scratch"
        )
    manifest = run_mmap_reader_parity_v12(
        sidecar_dir=args.sidecars,
        mmap_dir=args.mmap_sidecar,
        output_dir=output,
        case_count=READER_PARITY_CASE_COUNT,
    )
    print(str(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--sidecars", type=Path, required=True)
    value.add_argument("--mmap-sidecar", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--executor-config", type=Path, default=DEFAULT_EXECUTOR_CONFIG)
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
