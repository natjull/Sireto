#!/usr/bin/env python3
"""Materialize groupwise training scenes for the V4.12-N rerankers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_CORPUS = BASE / "datasets/v4_12_neural_text_corpus/02b8668f8050c5e9"
DEFAULT_OUTPUT_ROOT = BASE / "datasets/v4_12_neural_training_groups"
SCHEMA_VERSION = "sireto-v4.12-neural-training-groups-1"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _duckdb_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _validate_group_stats(stats: tuple[Any, ...], negative_count: int) -> None:
    rows, scenes, min_group, max_group, min_pos, max_pos, folds = map(int, stats)
    if not rows or not scenes:
        raise ValueError("Training groups are empty")
    if min_pos != 1 or max_pos != 1:
        raise ValueError("A group does not contain exactly one retrieved truth")
    if max_group > negative_count + 1:
        raise ValueError("A group exceeds the registered size")
    if folds != 3:
        raise ValueError("Training groups do not contain exactly folds 2/3/4")


def build(args: argparse.Namespace) -> Path:
    manifest = json.loads((args.corpus / "manifest.json").read_text())
    for name in ("queries_text.parquet", "candidates_text.parquet", "labels.parquet"):
        if file_sha256(args.corpus / name) != manifest.get("outputs", {}).get(name):
            raise ValueError(f"Corpus hash mismatch: {name}")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Training groups require a non-injected corpus")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.json"),
        "train_folds": [2, 3, 4],
        "positive_count_per_scene": 1,
        "max_same_siren_negatives": 5,
        "max_negative_count": args.negative_count,
        "xgboost_mining": False,
        "positive_injection": False,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text())
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    output = temporary / "training_groups.parquet"
    try:
        with duckdb.connect() as connection:
            connection.execute("SET threads TO 8")
            connection.execute(
                f"""
                COPY (
                    WITH exact AS (
                        SELECT CAST(query_id AS VARCHAR) AS query_id,
                               ground_truth_siret, ground_truth_siren, oof_fold,
                               label_is_human_validated, ranker_weight
                        FROM read_parquet('{_duckdb_path(args.corpus / "labels.parquet")}')
                        WHERE label_kind = 'MATCH_EXACT' AND oof_fold IN (2, 3, 4)
                    ),
                    eligible AS (
                        SELECT e.*
                        FROM exact e
                        INNER JOIN (
                            SELECT query_id,
                                   sum(CASE WHEN is_ground_truth = 1 THEN 1 ELSE 0 END) AS truths
                            FROM read_parquet('{_duckdb_path(args.corpus / "candidates_text.parquet")}')
                            GROUP BY query_id
                        ) c USING (query_id)
                        WHERE c.truths = 1
                    ),
                    joined AS (
                        SELECT c.*, q.query_text, e.ground_truth_siren, e.oof_fold,
                               e.label_is_human_validated, e.ranker_weight
                        FROM read_parquet('{_duckdb_path(args.corpus / "candidates_text.parquet")}') c
                        INNER JOIN eligible e USING (query_id)
                        INNER JOIN read_parquet('{_duckdb_path(args.corpus / "queries_text.parquet")}') q
                            USING (query_id)
                    ),
                    positive AS (
                        SELECT *, 0 AS negative_priority
                        FROM joined WHERE is_ground_truth = 1
                    ),
                    same_siren AS (
                        SELECT *, 0 AS negative_priority,
                               row_number() OVER (
                                   PARTITION BY query_id ORDER BY retrieval_rank, candidate_siret
                               ) AS local_rank
                        FROM joined
                        WHERE is_ground_truth = 0 AND candidate_siren = ground_truth_siren
                    ),
                    retrieval AS (
                        SELECT *, 1 AS negative_priority,
                               row_number() OVER (
                                   PARTITION BY query_id ORDER BY retrieval_rank, candidate_siret
                               ) AS local_rank
                        FROM joined WHERE is_ground_truth = 0
                    ),
                    negative_union AS (
                        SELECT * EXCLUDE(local_rank) FROM same_siren WHERE local_rank <= 5
                        UNION ALL
                        SELECT * EXCLUDE(local_rank) FROM retrieval
                    ),
                    negative_unique AS (
                        SELECT * EXCLUDE(dedup_rank)
                        FROM (
                            SELECT *, row_number() OVER (
                                PARTITION BY query_id, candidate_siret
                                ORDER BY negative_priority, retrieval_rank
                            ) AS dedup_rank
                            FROM negative_union
                        ) WHERE dedup_rank = 1
                    ),
                    negative AS (
                        SELECT * EXCLUDE(group_rank), group_rank AS group_position
                        FROM (
                            SELECT *, row_number() OVER (
                                PARTITION BY query_id
                                ORDER BY negative_priority, retrieval_rank, candidate_siret
                            ) AS group_rank
                            FROM negative_unique
                        ) WHERE group_rank <= {int(args.negative_count)}
                    )
                    SELECT query_id, oof_fold, label_is_human_validated, ranker_weight,
                           candidate_siret, candidate_siren, retrieval_rank,
                           CAST(0 AS SMALLINT) AS group_position,
                           CAST(1 AS TINYINT) AS is_positive,
                           query_text, candidate_text
                    FROM positive
                    UNION ALL
                    SELECT query_id, oof_fold, label_is_human_validated, ranker_weight,
                           candidate_siret, candidate_siren, retrieval_rank,
                           CAST(group_position AS SMALLINT),
                           CAST(0 AS TINYINT) AS is_positive,
                           query_text, candidate_text
                    FROM negative
                    ORDER BY query_id, group_position
                ) TO '{_duckdb_path(output)}'
                    (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
                """
            )
            stats = connection.execute(
                """
                SELECT count(*) AS rows,
                       count(DISTINCT query_id) AS scenes,
                       min(group_size) AS min_group_size,
                       max(group_size) AS max_group_size,
                       min(positive_count) AS min_positive_count,
                       max(positive_count) AS max_positive_count,
                       count(DISTINCT oof_fold) AS folds
                FROM (
                    SELECT *, count(*) OVER (PARTITION BY query_id) AS group_size,
                           sum(is_positive) OVER (PARTITION BY query_id) AS positive_count
                    FROM read_parquet(?)
                )
                """,
                [str(output)],
            ).fetchone()
            leakage = connection.execute(
                """
                SELECT count(*) FROM (
                    SELECT candidate_siren
                    FROM read_parquet(?)
                    GROUP BY candidate_siren
                    HAVING count(DISTINCT oof_fold) > 1
                )
                """,
                [str(output)],
            ).fetchone()[0]

        _validate_group_stats(stats, args.negative_count)
        rows, scenes, min_group, max_group, min_pos, max_pos, folds = map(int, stats)
        # Candidate SIRENs may legitimately be retrieved as negatives in many
        # folds. Leakage is controlled by truth SIREN components, not by the
        # arbitrary negative pool; publish the diagnostic without rejecting.
        report = (
            "# Groupes d'apprentissage V4.12-N\n\n"
            f"- scènes entraînables : {scenes:,} ;\n"
            f"- lignes : {rows:,} ;\n"
            f"- taille des groupes : {min_group} à {max_group} ;\n"
            "- vérité par groupe : exactement 1 ;\n"
            "- folds : 2, 3 et 4 uniquement ;\n"
            "- minage XGBoost : non ;\n"
            "- positif injecté : non.\n\n"
            "Les négatifs sont d'abord les autres sites du même SIREN, puis "
            "les candidats les mieux placés par le retrieval gelé.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        output_names = ["training_groups.parquet", "report.md"]
        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "row_counts": {"scenes": scenes, "rows": rows},
            "group_size": {"min": min_group, "max": max_group},
            "truth_per_group": 1,
            "candidate_sirens_crossing_folds_as_negatives": int(leakage),
            "truth_siren_component_policy": "inherited_from_corpus",
            "positive_injection": False,
            "final_test_opened": False,
            "outputs": {name: file_sha256(temporary / name) for name in output_names},
        }
        _json_dump(temporary / "manifest.json", output_manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--negative-count", type=int, default=15)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
