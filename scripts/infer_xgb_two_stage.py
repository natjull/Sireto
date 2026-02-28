"""Two-stage XGBoost inference (SSOT, no legacy modes)."""

from __future__ import annotations

# CRITICAL: enable semantic before imports
import os

os.environ.setdefault("XGB_SEMANTIC_ENABLED", "1")

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import xgboost as xgb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.xgb_matcher.features import (
    FEATURE_NAMES,
    FAST_RANKER_FEATURE_NAMES,
    make_features_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.infer import XgbInferenceEngine, CrmInput
from src.xgb_matcher.profile import InferenceProfile
from src.xgb_matcher.retrieval import build_candidate_pool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer XGB two-stage matcher (SSOT).")
    p.add_argument("--crm-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, default=Path("reports/xgb_two_stage_topk.csv"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--partitions-dir", type=Path, default=Path("data/candidates_v7_all"))
    p.add_argument("--pool-mode", choices=["insee_then_postcode"], default="insee_then_postcode")
    p.add_argument("--prefilter-k", type=int, default=500)
    p.add_argument("--min-candidates", type=int, default=100)
    p.add_argument(
        "--stage1-top-n",
        type=int,
        default=50,
        help="Top-N candidates after Stage1 ranker (SSOT=50).",
    )
    p.add_argument("--char-top-k", type=int, default=200)
    p.add_argument("--tfidf-name-mode", choices=["bag"], default="bag")
    p.add_argument("--siren-siblings", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--drop-unnamed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--exclude-closed", action="store_true", default=False)
    p.add_argument("--export-routing-features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--allow-no-semantic",
        action="store_true",
        default=False,
        help="Allow running without semantic embeddings (semantic features will be zero).",
    )
    p.add_argument("--meta-path", type=Path, default=None)
    p.add_argument("--ranker-model", type=Path, default=None)
    p.add_argument("--ranker-fast-model", type=Path, default=None)
    p.add_argument("--decider-model", type=Path, default=None)
    p.add_argument("--calibrator-path", type=Path, default=None)
    p.add_argument(
        "--override-retrieval",
        action="store_true",
        default=False,
        help="Allow overriding retrieval knobs even when --meta-path is provided.",
    )
    p.add_argument("--debug-gt", action="store_true", default=False)
    p.add_argument("--gt-column", type=str, default="ground_truth_siret")
    p.add_argument(
        "--debug-gt-output",
        type=Path,
        default=None,
        help="Path for GT debug CSV output (default: <output>_gt_debug.csv).",
    )
    return p.parse_args()


def load_crm(path: Path) -> pd.DataFrame:
    import csv

    sample = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ";"

    df = pd.read_csv(path, sep=delimiter, dtype=str)
    df = df.rename(
        columns={
            "Client final": "crm_name",
            "Adresse": "crm_address",
            "Commune": "crm_city",
            "Code Postal": "postcode",
            "Code INSEE": "insee",
            "crm_insee": "insee",
            "crm_cp": "postcode",
            "crm_adresse": "crm_address",
            "crm_commune": "crm_city",
        }
    )
    for col in ["postcode", "insee"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else x)
    if "crm_id" not in df.columns:
        df["crm_id"] = df.index
    return df


def _find_latest_meta(model_dir: Path) -> Path | None:
    candidates = sorted(model_dir.glob("xgb_two_stage_meta_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _build_profile(args: argparse.Namespace, logger: logging.Logger) -> InferenceProfile:
    if args.allow_no_semantic:
        os.environ["XGB_ALLOW_NO_SEMANTIC"] = "1"

    model_dir = Path("models")
    meta_path = args.meta_path or _find_latest_meta(model_dir)

    if meta_path and meta_path.exists():
        profile = InferenceProfile.from_meta(meta_path, strict=not args.allow_no_semantic)
    else:
        if not args.ranker_model or not args.decider_model:
            raise FileNotFoundError(
                "Meta path not provided and ranker/decider paths not set. "
                "Provide --meta-path or both --ranker-model and --decider-model."
            )
        ranker_is_fast = "fast" in args.ranker_model.name.lower()
        profile = InferenceProfile(
            ranker_path=args.ranker_model,
            ranker_fast_path=args.ranker_model if ranker_is_fast else None,
            decider_path=args.decider_model,
            calibrator_path=args.calibrator_path,
            feature_order=FEATURE_NAMES,
            ranker_feature_order=FEATURE_NAMES,
            ranker_fast_feature_order=FAST_RANKER_FEATURE_NAMES,
            tfidf_name_mode=args.tfidf_name_mode,
            siren_siblings=args.siren_siblings,
            prefilter_k=args.prefilter_k,
            char_top_k=args.char_top_k,
            min_candidates=args.min_candidates,
            stage1_top_n=args.stage1_top_n,
            drop_unnamed=args.drop_unnamed,
            exclude_closed=args.exclude_closed,
            partitions_dir=args.partitions_dir,
            semantic_required=not args.allow_no_semantic,
            use_ranker_fast=ranker_is_fast,
        )

    if not (meta_path and meta_path.exists()) or args.override_retrieval:
        profile.prefilter_k = args.prefilter_k
        profile.tfidf_name_mode = args.tfidf_name_mode
        profile.siren_siblings = args.siren_siblings
        profile.drop_unnamed = args.drop_unnamed
        profile.exclude_closed = args.exclude_closed
        profile.char_top_k = args.char_top_k
        profile.min_candidates = args.min_candidates
        profile.stage1_top_n = args.stage1_top_n
        profile.partitions_dir = args.partitions_dir
        profile.semantic_required = not args.allow_no_semantic
    if args.ranker_model:
        profile.ranker_path = args.ranker_model
    if args.ranker_fast_model:
        profile.ranker_fast_path = args.ranker_fast_model
    if args.decider_model:
        profile.decider_path = args.decider_model
    if args.calibrator_path:
        profile.calibrator_path = args.calibrator_path

    logger.info("Using profile with stage1_top_n=%d", profile.stage1_top_n)
    return profile


def _infer_debug(
    engine: XgbInferenceEngine,
    profile: InferenceProfile,
    records: List[dict],
    gt_column: str,
    top_k: int,
    export_routing_features: bool,
) -> tuple[List[dict], List[dict]]:
    rows_out: List[dict] = []
    debug_rows: List[dict] = []

    progress = tqdm(total=len(records), desc="Infer")
    for r in records:
        orig_index = r.get("_orig_index")
        gt_siret = str(r.get(gt_column) or "").strip()
        gt_siret = gt_siret if gt_siret and gt_siret not in ("", "nan", "None") else None

        crm_in = CrmInput(
            crm_id=str(r.get("crm_id") or orig_index),
            crm_name=str(r.get("crm_name") or ""),
            crm_address=str(r.get("crm_address") or ""),
            postcode=str(r.get("postcode") or ""),
            insee=str(r.get("insee") or ""),
            crm_city=str(r.get("crm_city") or ""),
        )
        crm_pre = preprocess_crm_row(r)

        result = build_candidate_pool(
            store=engine.store,
            crm_row=r,
            crm_pre=crm_pre,
            config=profile.build_retrieval_config(),
            tfidf_cache={},
            gt_siret=gt_siret,
        )
        set_global_name_idf_map(result.idf_map, result.default_idf)

        in_stage1_topn = False
        stage1_rank = -1
        stage1_topn_size = 0
        if result.candidates:
            cand_sirets = [str(c.get("siret") or "").zfill(14) for c in result.candidates]
            feats_stage1 = [
                make_features_from_preprocessed(crm_pre, c, skip_semantic=True)
                for c in result.candidates
            ]
            ranker_feature_order = engine.ranker_feature_order or engine.feature_order
            X1 = pd.DataFrame(feats_stage1)[ranker_feature_order]
            scores_stage1 = engine.ranker.predict(
                xgb.DMatrix(X1.values, feature_names=ranker_feature_order)
            )
            order = list(reversed(scores_stage1.argsort()))
            stage1_top_n = min(engine.stage1_top_n, len(order))
            top_n_idx = order[:stage1_top_n]
            stage1_topn_size = len(top_n_idx)
            if gt_siret:
                gt_norm = str(gt_siret).zfill(14)
                in_stage1_topn = gt_norm in {cand_sirets[i] for i in top_n_idx}
                if result.gt_in_tfidf_pool and not in_stage1_topn:
                    for rank_s1, idx_s1 in enumerate(order, start=1):
                        if cand_sirets[idx_s1] == gt_norm:
                            stage1_rank = rank_s1
                            break

        topk_rows = engine.infer_topk(
            crm_in,
            top_k=top_k,
            pool_mode="insee_then_postcode",
            export_routing_features=export_routing_features,
        )
        for tk in topk_rows:
            row_dict = tk.to_dict()
            row_dict["_orig_index"] = orig_index
            rows_out.append(row_dict)

        in_stage2_topk = False
        stage2_rank = -1
        if gt_siret:
            gt_norm = str(gt_siret).zfill(14)
            for tk in topk_rows:
                if str(tk.siret_candidate).zfill(14) == gt_norm:
                    in_stage2_topk = True
                    stage2_rank = tk.rank
                    break

        loss_reason = result.loss_reason or ""
        if gt_siret and result.gt_in_tfidf_pool and not in_stage1_topn:
            loss_reason = f"PRUNED_BY_STAGE1_RANK_{stage1_rank}"
        elif gt_siret and in_stage1_topn and not in_stage2_topk:
            loss_reason = "PRUNED_BY_STAGE2_TOPK"

        debug_rows.append(
            {
                "crm_id": str(r.get("crm_id") or ""),
                "crm_name": str(r.get("crm_name") or ""),
                "gt_siret": gt_siret or "",
                "loc_key": str(r.get("loc_key") or f"{r.get('insee','')}|{r.get('postcode','')}")
                if gt_siret
                else "",
                "in_base_pool": bool(result.gt_in_base_pool),
                "in_filtered_pool": bool(result.gt_in_filtered_pool),
                "in_tfidf_pool": bool(result.gt_in_tfidf_pool),
                "in_stage1_topn": bool(in_stage1_topn),
                "in_stage2_topk": bool(in_stage2_topk),
                "stage1_rank": int(stage1_rank),
                "stage2_rank": int(stage2_rank),
                "base_pool_size": int(result.pool_sizes.get("base", 0)),
                "filtered_pool_size": int(result.pool_sizes.get("filtered", 0)),
                "tfidf_pool_size": int(result.pool_sizes.get("tfidf", 0)),
                "stage1_topn_size": int(stage1_topn_size),
                "loss_reason": loss_reason,
            }
        )
        progress.update(1)
    progress.close()
    return rows_out, debug_rows


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger(__name__)

    if args.pool_mode != "insee_then_postcode":
        raise ValueError("pool_mode must be 'insee_then_postcode' (SSOT).")

    crm = load_crm(args.crm_path)
    crm_records = crm.to_dict("records")
    for idx, r in enumerate(crm_records):
        r["_orig_index"] = idx

    # Sort queries geometrically to maximize Parquet LRU cache hits. This reduces I/O by 99%
    crm_records.sort(key=lambda x: (str(x.get("postcode") or ""), str(x.get("insee") or "")))

    profile = _build_profile(args, logger)
    engine = XgbInferenceEngine.from_profile(profile)

    rows_out: List[dict] = []
    debug_rows: List[dict] = []

    if args.debug_gt:
        rows_out, debug_rows = _infer_debug(
            engine,
            profile,
            crm_records,
            args.gt_column,
            args.top_k,
            args.export_routing_features,
        )
    else:
        for r in tqdm(crm_records, total=len(crm_records), desc="Infer"):
            orig_index = r.get("_orig_index")
            crm_in = CrmInput(
                crm_id=str(r.get("crm_id") or orig_index),
                crm_name=str(r.get("crm_name") or ""),
                crm_address=str(r.get("crm_address") or ""),
                postcode=str(r.get("postcode") or ""),
                insee=str(r.get("insee") or ""),
                crm_city=str(r.get("crm_city") or ""),
            )
            topk_rows = engine.infer_topk(
                crm_in,
                top_k=args.top_k,
                pool_mode=args.pool_mode,
                export_routing_features=args.export_routing_features,
            )
            for tk in topk_rows:
                row_dict = tk.to_dict()
                row_dict["_orig_index"] = orig_index
                rows_out.append(row_dict)

    if not rows_out:
        logger.warning("No inference results produced. Output will be empty.")
        out_df = pd.DataFrame(columns=["crm_id", "crm_name", "siret_candidate", "score", "rank"])
    else:
        out_df = pd.DataFrame(rows_out)
        if "_orig_index" in out_df.columns:
            out_df = out_df.sort_values(["_orig_index", "rank"], ascending=[True, True])
            out_df = out_df.drop(columns=["_orig_index"])
        else:
            out_df = out_df.sort_values(["crm_name", "rank", "score"], ascending=[True, True, False])

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    logger.info("Saved top-%d results to %s (%d rows)", args.top_k, args.output_path, len(out_df))

    if args.debug_gt and debug_rows:
        debug_output = args.debug_gt_output or args.output_path.with_name(args.output_path.stem + "_gt_debug.csv")
        debug_df = pd.DataFrame(debug_rows)
        debug_df.to_csv(debug_output, index=False)

        n_total = len(debug_df)
        n_in_base = int(debug_df["in_base_pool"].sum()) if n_total else 0
        n_in_filtered = int(debug_df["in_filtered_pool"].sum()) if n_total else 0
        n_in_tfidf = int(debug_df["in_tfidf_pool"].sum()) if n_total else 0
        n_in_stage1 = int(debug_df["in_stage1_topn"].sum()) if n_total else 0
        n_in_topk = int(debug_df["in_stage2_topk"].sum()) if n_total else 0
        logger.info("GT Debug summary (%d queries with GT):", n_total)
        logger.info("  in_base_pool: %d (%.1f%%)", n_in_base, 100 * n_in_base / n_total if n_total else 0)
        logger.info("  in_filtered_pool: %d (%.1f%%)", n_in_filtered, 100 * n_in_filtered / n_total if n_total else 0)
        logger.info("  in_tfidf_pool: %d (%.1f%%)", n_in_tfidf, 100 * n_in_tfidf / n_total if n_total else 0)
        logger.info("  in_stage1_topn: %d (%.1f%%)", n_in_stage1, 100 * n_in_stage1 / n_total if n_total else 0)
        logger.info("  in_stage2_topk: %d (%.1f%%)", n_in_topk, 100 * n_in_topk / n_total if n_total else 0)

        loss_counts = debug_df["loss_reason"].value_counts()
        logger.info("  Loss reasons:")
        for reason, count in loss_counts.items():
            if reason:
                logger.info("    %s: %d", reason, count)
        logger.info("Saved GT debug to %s", debug_output)


if __name__ == "__main__":
    main()
