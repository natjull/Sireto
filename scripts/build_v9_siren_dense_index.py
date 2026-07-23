#!/usr/bin/env python3
"""Build the streaming global dense SIREN ANN index used by V9."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.dataset as ds


NAME_COLUMNS = [
    "siren",
    "sigleUniteLegale",
    "prenomUsuelUniteLegale",
    "nomUniteLegale",
    "nomUsageUniteLegale",
    "denominationUniteLegale",
    "denominationUsuelle1UniteLegale",
    "denominationUsuelle2UniteLegale",
    "denominationUsuelle3UniteLegale",
]


def siren_text(row: dict) -> str:
    names = []
    for column in NAME_COLUMNS[1:]:
        value = str(row.get(column) or "").strip()
        if value and value.lower() != "nan" and value not in names:
            names.append(value)
    return " | ".join(names)


def iter_entity_batches(
    source: Path,
    *,
    batch_size: int,
    max_rows: int = 0,
) -> Iterable[tuple[list[str], list[str]]]:
    dataset = ds.dataset(source, format="parquet")
    available = set(dataset.schema.names)
    columns = [column for column in NAME_COLUMNS if column in available]
    if "siren" not in columns:
        raise ValueError("Source parquet must contain a siren column")

    emitted = 0
    for record_batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        frame = record_batch.to_pandas()
        sirens: list[str] = []
        texts: list[str] = []
        for row in frame.to_dict("records"):
            siren = "".join(char for char in str(row.get("siren") or "") if char.isdigit())
            text = siren_text(row)
            if not siren or not text:
                continue
            sirens.append(siren.zfill(9))
            texts.append(text)
            emitted += 1
            if max_rows and emitted >= max_rows:
                break
        if sirens:
            yield sirens, texts
        if max_rows and emitted >= max_rows:
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/semantic/siret-bert-deploy"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--training-rows", type=int, default=100_000)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--pq-subquantizers", type=int, default=48)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        batches = iter_entity_batches(
            args.source,
            batch_size=args.read_batch_size,
            max_rows=min(args.max_rows or 100, 100),
        )
        count = sum(len(sirens) for sirens, _ in batches)
        print(json.dumps({"valid_preview_rows": count, "source": str(args.source)}))
        return
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output directory already exists: {args.output_dir}"
        )

    os.environ["XGB_SEMANTIC_ENABLED"] = "1"
    os.environ["XGB_SEMANTIC_MODEL"] = str(args.model)
    os.environ["XGB_SEMANTIC_DEVICE"] = args.device

    import faiss
    from src.xgb_matcher.semantic import _get_encoder
    from src.xgb_matcher.v9_dataset import file_sha256, tokenizer_fingerprint

    encoder = _get_encoder(str(args.model))
    if encoder is None:
        raise RuntimeError("Unable to load the semantic encoder")

    training_texts: list[str] = []
    for _sirens, texts in iter_entity_batches(
        args.source,
        batch_size=args.read_batch_size,
        max_rows=args.training_rows,
    ):
        training_texts.extend(texts)
        if len(training_texts) >= args.training_rows:
            training_texts = training_texts[: args.training_rows]
            break
    if not training_texts:
        raise ValueError("No valid SIREN entity text found")

    training_vectors = encoder.encode(
        training_texts,
        batch_size=args.encode_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    dimension = int(training_vectors.shape[1])
    subquantizers = args.pq_subquantizers
    if dimension % subquantizers != 0:
        raise ValueError(
            f"Embedding dimension {dimension} is not divisible by "
            f"pq-subquantizers={subquantizers}"
        )
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        args.nlist,
        subquantizers,
        8,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.train(training_vectors)
    index.nprobe = min(32, args.nlist)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile(
        dir=args.output_dir,
        prefix="siren_ids_",
        suffix=".bin",
        delete=False,
    ) as ids_handle:
        ids_path = Path(ids_handle.name)

    total = 0
    try:
        with ids_path.open("ab") as ids_handle:
            for sirens, texts in iter_entity_batches(
                args.source,
                batch_size=args.read_batch_size,
                max_rows=args.max_rows,
            ):
                vectors = encoder.encode(
                    texts,
                    batch_size=args.encode_batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)
                index.add(vectors)
                np.asarray(sirens, dtype="S9").tofile(ids_handle)
                total += len(sirens)

        raw_ids = np.memmap(ids_path, dtype="S9", mode="r", shape=(total,))
        np.save(args.output_dir / "siren_ids.npy", raw_ids)
        faiss.write_index(index, str(args.output_dir / "siren_faiss.index"))
    finally:
        ids_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": "v9-siren-dense-1",
        "source": str(args.source),
        "source_sha256": file_sha256(args.source),
        "model": str(args.model),
        "tokenizer_fingerprint": tokenizer_fingerprint(args.model),
        "entity_count": total,
        "dimension": dimension,
        "index_type": "IndexIVFPQ",
        "nlist": args.nlist,
        "nprobe": index.nprobe,
        "pq_subquantizers": subquantizers,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
