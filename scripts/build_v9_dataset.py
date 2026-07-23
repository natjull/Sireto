#!/usr/bin/env python3
"""Build a canonical, immutable SIRETO V9 dataset bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.retrieval_config import RetrievalConfigV1
from src.xgb_matcher.v9_dataset import build_canonical_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/v9"))
    parser.add_argument("--sirene-snapshot-id", required=True)
    parser.add_argument("--sirene-snapshot", type=Path)
    parser.add_argument("--tokenizer-model", type=Path)
    parser.add_argument("--retrieval-config", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-source", default="provided_ground_truth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_config = RetrievalConfigV1()
    if args.retrieval_config:
        retrieval_config = RetrievalConfigV1.from_dict(
            json.loads(args.retrieval_config.read_text(encoding="utf-8"))
        )
    output = build_canonical_dataset(
        query_source_path=args.queries,
        label_source_path=args.labels,
        candidate_source_path=args.candidates,
        output_root=args.output_root,
        sirene_snapshot_id=args.sirene_snapshot_id,
        sirene_snapshot_path=args.sirene_snapshot,
        tokenizer_model_path=args.tokenizer_model,
        retrieval_config=retrieval_config,
        seed=args.seed,
        default_label_source=args.label_source,
    )
    print(output)


if __name__ == "__main__":
    main()
