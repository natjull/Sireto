#!/usr/bin/env python3
"""Export the detailed <=2,000-candidate official-evidence union to Parquet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_evidence_retrieval import (  # noqa: E402
    OfficialEvidenceRetrievalConfig,
    OfficialEvidenceRetriever,
    retrieve_official_evidence_union_to_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--overlay-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-id-field", default="query_id")
    parser.add_argument("--max-union-candidates", type=int, default=2000)
    parser.add_argument("--exact-limit", type=int, default=500)
    parser.add_argument("--word-limit", type=int, default=1000)
    parser.add_argument("--character-limit", type=int, default=1000)
    parser.add_argument("--siren-limit", type=int, default=10)
    parser.add_argument("--sites-per-siren", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def _rows(path: Path, batch_size: int) -> Iterator[Mapping[str, Any]]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()
        return
    if path.suffix.lower() in {".csv", ".tsv"}:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(
                stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ","
            )
        return
    raise ValueError(f"unsupported retrieval input: {path}")


def main() -> None:
    args = parse_args()
    config = OfficialEvidenceRetrievalConfig(
        max_union_candidates=args.max_union_candidates,
        exact_limit=args.exact_limit,
        word_limit=args.word_limit,
        character_limit=args.character_limit,
        siren_limit=args.siren_limit,
        sites_per_siren=args.sites_per_siren,
    )
    retriever = OfficialEvidenceRetriever.from_index_paths(
        args.base_index, args.overlay_index, config
    )
    output = retrieve_official_evidence_union_to_parquet(
        retriever,
        _rows(args.input, args.batch_size),
        args.output,
        query_id_field=args.query_id_field,
        batch_size=args.batch_size,
    )
    print(output)


if __name__ == "__main__":
    main()
