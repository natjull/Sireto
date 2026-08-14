#!/usr/bin/env python3
"""Build the preregistered hard-negative groups for V4.12-BGE."""

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
DEFAULT_BUSINESS = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_RANKER = BASE / "experiments/v4_12_learned_oof_rankers/839ef55308d5077e"
DEFAULT_OUTPUT_ROOT = BASE / "datasets/v4_12_bge_training_groups"
SCHEMA_VERSION = "sireto-v4.12-bge-training-groups-1"
TRAIN_FOLDS = [2, 3, 4]
HOMONYM_THRESHOLD = 0.90


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _validate_manifest(root: Path, required: tuple[str, ...]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in required:
        expected = manifest.get("outputs", {}).get(name)
        if not expected or file_sha256(root / name) != expected:
            raise ValueError(f"Manifest mismatch: {root / name}")
    return manifest


def _validate_stats(stats: tuple[Any, ...], negative_count: int) -> dict[str, int]:
    keys = (
        "rows",
        "scenes",
        "min_group_size",
        "max_group_size",
        "min_positive_count",
        "max_positive_count",
        "fold_count",
    )
    values = {key: int(value) for key, value in zip(keys, stats, strict=True)}
    if values["rows"] <= 0 or values["scenes"] <= 0:
        raise ValueError("BGE training groups are empty")
    if values["min_positive_count"] != 1 or values["max_positive_count"] != 1:
        raise ValueError("Every BGE scene must contain exactly one positive")
    if values["max_group_size"] > negative_count + 1:
        raise ValueError("A BGE scene exceeds the preregistered group size")
    if values["fold_count"] != len(TRAIN_FOLDS):
        raise ValueError("BGE groups must contain folds 2/3/4 only")
    return values


def build(args: argparse.Namespace) -> Path:
    corpus_manifest = _validate_manifest(
        args.corpus,
        ("queries_text.parquet", "candidates_text.parquet", "labels.parquet"),
    )
    business_manifest = _validate_manifest(
        args.business,
        ("candidates_business.parquet", "labels.parquet"),
    )
    ranker_manifest = _validate_manifest(
        args.ranker,
        ("business_learned_oof_candidates.parquet",),
    )
    if corpus_manifest.get("positive_injection") is not False:
        raise ValueError("The text corpus is not certified non-injected")
    if business_manifest.get("positive_injection") is not False:
        raise ValueError("The business dataset is not certified non-injected")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "corpus_manifest_sha256": file_sha256(args.corpus / "manifest.json"),
        "business_manifest_sha256": file_sha256(args.business / "manifest.json"),
        "ranker_manifest_sha256": file_sha256(args.ranker / "manifest.json"),
        "train_folds": TRAIN_FOLDS,
        "positive_count_per_scene": 1,
        "max_negative_count": args.negative_count,
        "xgb_top_negative_count": 5,
        "same_siren_negative_count": 3,
        "homonym_negative_count": 3,
        "state_competitor_negative_count": 2,
        "homonym_threshold": HOMONYM_THRESHOLD,
        "positive_injection": False,
        "truth_siren_component_policy": "inherited_from_corpus",
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
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
                               ground_truth_siret, ground_truth_siren,
                               ground_truth_state, oof_fold,
                               label_is_human_validated, ranker_weight
                        FROM read_parquet('{_sql_path(args.corpus / 'labels.parquet')}')
                        WHERE label_kind = 'MATCH_EXACT' AND oof_fold IN (2, 3, 4)
                    ),
                    eligible AS (
                        SELECT e.*
                        FROM exact e
                        INNER JOIN (
                            SELECT query_id,
                                   sum(CASE WHEN is_ground_truth = 1 THEN 1 ELSE 0 END) AS truths
                            FROM read_parquet('{_sql_path(args.corpus / 'candidates_text.parquet')}')
                            GROUP BY query_id
                        ) c USING (query_id)
                        WHERE c.truths = 1
                    ),
                    joined AS (
                        SELECT c.query_id, e.oof_fold, e.ground_truth_siret,
                               e.ground_truth_siren, e.ground_truth_state,
                               e.label_is_human_validated, e.ranker_weight,
                               c.candidate_siret, c.candidate_siren,
                               c.retrieval_rank, c.is_ground_truth,
                               q.query_text, c.candidate_text,
                               r.ranker_score AS business_ranker_score,
                               r.ranker_rank AS business_ranker_rank,
                               b.source_name_score, b.name_jaro_max,
                               b.addr_jaro, b.postcode_match,
                               CASE
                                   WHEN c.candidate_text LIKE '%État établissement : A.%' THEN 'A'
                                   WHEN c.candidate_text LIKE '%État établissement : F.%' THEN 'F'
                                   ELSE '?'
                               END AS candidate_state
                        FROM read_parquet('{_sql_path(args.corpus / 'candidates_text.parquet')}') c
                        INNER JOIN eligible e USING (query_id)
                        INNER JOIN read_parquet('{_sql_path(args.corpus / 'queries_text.parquet')}') q
                            USING (query_id)
                        INNER JOIN read_parquet('{_sql_path(args.business / 'candidates_business.parquet')}') b
                            USING (query_id, candidate_siret, candidate_siren)
                        INNER JOIN read_parquet('{_sql_path(args.ranker / 'business_learned_oof_candidates.parquet')}') r
                            USING (query_id, candidate_siret, candidate_siren)
                    ),
                    negative_base AS (
                        SELECT *, row_number() OVER (
                            PARTITION BY query_id
                            ORDER BY business_ranker_rank, retrieval_rank, candidate_siret
                        ) AS negative_xgb_rank
                        FROM joined
                        WHERE is_ground_truth = 0
                    ),
                    xgb_top AS (
                        SELECT *, 0 AS negative_priority,
                               'xgb_top' AS negative_category
                        FROM negative_base WHERE negative_xgb_rank <= 5
                    ),
                    same_siren AS (
                        SELECT * EXCLUDE(category_rank), 1 AS negative_priority,
                               'same_siren' AS negative_category
                        FROM (
                            SELECT *, row_number() OVER (
                                PARTITION BY query_id
                                ORDER BY business_ranker_rank, retrieval_rank, candidate_siret
                            ) AS category_rank
                            FROM negative_base n
                            WHERE n.negative_xgb_rank > 5
                              AND n.candidate_siren = n.ground_truth_siren
                        ) WHERE category_rank <= 3
                    ),
                    selected_1 AS (
                        SELECT * FROM xgb_top
                        UNION ALL BY NAME SELECT * FROM same_siren
                    ),
                    homonym AS (
                        SELECT * EXCLUDE(category_rank), 2 AS negative_priority,
                               'homonym_or_same_address' AS negative_category
                        FROM (
                            SELECT n.*, row_number() OVER (
                                PARTITION BY n.query_id
                                ORDER BY n.business_ranker_rank, n.retrieval_rank,
                                         n.candidate_siret
                            ) AS category_rank
                            FROM negative_base n
                            WHERE NOT EXISTS (
                                SELECT 1 FROM selected_1 s
                                WHERE s.query_id = n.query_id
                                  AND s.candidate_siret = n.candidate_siret
                            )
                              AND (
                                  greatest(n.source_name_score, n.name_jaro_max) >= {HOMONYM_THRESHOLD}
                                  OR (n.addr_jaro >= 0.98 AND n.postcode_match = 1)
                              )
                        ) WHERE category_rank <= 3
                    ),
                    selected_2 AS (
                        SELECT * FROM selected_1
                        UNION ALL BY NAME SELECT * FROM homonym
                    ),
                    state_competitor AS (
                        SELECT * EXCLUDE(category_rank), 3 AS negative_priority,
                               'state_competitor' AS negative_category
                        FROM (
                            SELECT n.*, row_number() OVER (
                                PARTITION BY n.query_id
                                ORDER BY n.business_ranker_rank, n.retrieval_rank,
                                         n.candidate_siret
                            ) AS category_rank
                            FROM negative_base n
                            WHERE NOT EXISTS (
                                SELECT 1 FROM selected_2 s
                                WHERE s.query_id = n.query_id
                                  AND s.candidate_siret = n.candidate_siret
                            )
                              AND n.candidate_state IN ('A', 'F')
                              AND n.candidate_state <> n.ground_truth_state
                        ) WHERE category_rank <= 2
                    ),
                    selected_3 AS (
                        SELECT * FROM selected_2
                        UNION ALL BY NAME SELECT * FROM state_competitor
                    ),
                    retrieval_fill AS (
                        SELECT n.*, 4 AS negative_priority,
                               'retrieval_fill' AS negative_category
                        FROM negative_base n
                        WHERE NOT EXISTS (
                            SELECT 1 FROM selected_3 s
                            WHERE s.query_id = n.query_id
                              AND s.candidate_siret = n.candidate_siret
                        )
                    ),
                    categorized AS (
                        SELECT * FROM selected_3
                        UNION ALL BY NAME SELECT * FROM retrieval_fill
                    ),
                    negative AS (
                        SELECT * EXCLUDE(group_position), group_position
                        FROM (
                            SELECT *, row_number() OVER (
                                PARTITION BY query_id
                                ORDER BY negative_priority, business_ranker_rank,
                                         retrieval_rank, candidate_siret
                            ) AS group_position
                            FROM categorized
                        ) WHERE group_position <= {int(args.negative_count)}
                    )
                    SELECT query_id, oof_fold, label_is_human_validated, ranker_weight,
                           candidate_siret, candidate_siren, retrieval_rank,
                           business_ranker_score, business_ranker_rank,
                           CAST(0 AS SMALLINT) AS group_position,
                           CAST(1 AS TINYINT) AS is_positive,
                           CAST('positive' AS VARCHAR) AS negative_category,
                           query_text, candidate_text
                    FROM joined WHERE is_ground_truth = 1
                    UNION ALL
                    SELECT query_id, oof_fold, label_is_human_validated, ranker_weight,
                           candidate_siret, candidate_siren, retrieval_rank,
                           business_ranker_score, business_ranker_rank,
                           CAST(group_position AS SMALLINT),
                           CAST(0 AS TINYINT), negative_category,
                           query_text, candidate_text
                    FROM negative
                    ORDER BY query_id, group_position
                ) TO '{_sql_path(output)}'
                  (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
                """
            )
            stats = connection.execute(
                """
                SELECT count(*), count(DISTINCT query_id), min(group_size), max(group_size),
                       min(positive_count), max(positive_count), count(DISTINCT oof_fold)
                FROM (
                    SELECT *, count(*) OVER (PARTITION BY query_id) AS group_size,
                           sum(is_positive) OVER (PARTITION BY query_id) AS positive_count
                    FROM read_parquet(?)
                )
                """,
                [str(output)],
            ).fetchone()
            category_rows = connection.execute(
                """
                SELECT negative_category, count(*) AS rows,
                       count(DISTINCT query_id) AS scenes
                FROM read_parquet(?)
                GROUP BY negative_category ORDER BY negative_category
                """,
                [str(output)],
            ).fetchall()
            leakage = connection.execute(
                """
                SELECT count(*) FROM (
                    SELECT ground_truth_siren
                    FROM (
                        SELECT query_id, any_value(candidate_siren) FILTER (WHERE is_positive = 1)
                            AS ground_truth_siren, any_value(oof_fold) AS oof_fold
                        FROM read_parquet(?) GROUP BY query_id
                    )
                    GROUP BY ground_truth_siren
                    HAVING count(DISTINCT oof_fold) > 1
                )
                """,
                [str(output)],
            ).fetchone()[0]

        values = _validate_stats(stats, args.negative_count)
        if int(leakage) != 0:
            raise ValueError("A truth SIREN crosses BGE training folds")
        categories = {
            str(category): {"rows": int(rows), "scenes": int(scenes)}
            for category, rows, scenes in category_rows
        }
        report = (
            "# Groupes difficiles V4.12-BGE\n\n"
            f"- scènes : {values['scenes']:,} ;\n"
            f"- lignes : {values['rows']:,} ;\n"
            f"- taille : {values['min_group_size']} à {values['max_group_size']} ;\n"
            "- folds : 2/3/4 uniquement ;\n"
            "- vérité : exactement une par scène, déjà présente dans le pool ;\n"
            "- fuite SIREN vérité : 0 ;\n"
            "- injection positive : non.\n\n"
            f"Catégories : `{json.dumps(categories, ensure_ascii=False, sort_keys=True)}`.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        outputs = ("training_groups.parquet", "report.md")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "row_counts": {"scenes": values["scenes"], "rows": values["rows"]},
            "group_size": {
                "min": values["min_group_size"],
                "max": values["max_group_size"],
            },
            "negative_categories": categories,
            "truth_per_group": 1,
            "truth_sirens_crossing_folds": 0,
            "positive_injection": False,
            "confirmation_fold_opened": False,
            "final_test_opened": False,
            "outputs": {name: file_sha256(temporary / name) for name in outputs},
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--business", type=Path, default=DEFAULT_BUSINESS)
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--negative-count", type=int, default=15)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
