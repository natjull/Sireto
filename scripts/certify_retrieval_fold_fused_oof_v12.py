#!/usr/bin/env python3
"""Join one immutable fold-fused OOF run with its exact POSTFIT parity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.retrieval_fold_fused_certification_v12 import (  # noqa: E402
    certify_fold_fused_oof_postfit_v12,
)
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (  # noqa: E402
    FoldFusedInvariantError,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import (  # noqa: E402
    load_executor_config_v12,
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
    output = Path(args.output).resolve()
    scratch = executor.scratch_root.resolve()
    if not scratch.is_dir() or output == scratch or scratch not in output.parents:
        raise FoldFusedInvariantError(
            "postfit certificate must be written under configured external scratch"
        )
    result = certify_fold_fused_oof_postfit_v12(
        oof_manifest_path=args.oof_manifest,
        postfit_parity_manifest_path=args.postfit_parity_manifest,
        output_path=output,
    )
    print(str(result))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--oof-manifest", type=Path, required=True)
    value.add_argument("--postfit-parity-manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    value.add_argument("--executor-config", type=Path, default=DEFAULT_EXECUTOR_CONFIG)
    return value


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
