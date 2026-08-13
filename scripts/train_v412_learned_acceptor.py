#!/usr/bin/env python3
"""Nested-OOF selective acceptor for the V4.12-L query scenes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import _binary_metric  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.12-learned-acceptor-1"
BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_SCENES = BASE / "datasets/v4_12_learned_scenes/2f2bb2b0208241e0"
DEFAULT_LOCAL_LABELS = Path("reports/v412_review_local_identifiable_labels_279.csv")
TARGET_PRECISION = 0.998
SEED = 42


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _select_threshold(
    scores: np.ndarray,
    correct: np.ndarray,
    audited_open: np.ndarray,
    *,
    target_precision: float = TARGET_PRECISION,
) -> tuple[float, dict[str, Any]]:
    """Maximise calibration coverage under precision and audited-open gates."""
    thresholds = np.unique(scores)[::-1]
    best: tuple[int, float, int, int] | None = None
    for threshold in thresholds:
        accepted = scores >= threshold
        count = int(accepted.sum())
        successes = int(correct[accepted].sum())
        audited_count = int(audited_open[accepted].sum())
        precision = successes / count if count else 1.0
        if precision >= target_precision and audited_count == 0:
            candidate = (count, float(threshold), successes, audited_count)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return float("inf"), {
            "accepted": 0,
            "correct": 0,
            "precision": None,
            "audited_open_auto": 0,
        }
    count, threshold, successes, audited_count = best
    return threshold, {
        "accepted": count,
        "correct": successes,
        "precision": successes / count,
        "audited_open_auto": audited_count,
    }


def _metrics(frame: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, Any]:
    view = frame if mask is None else frame[mask]
    accepted = view["auto_match"].astype(bool)
    accepted_count = int(accepted.sum())
    correct_count = int(view.loc[accepted, "top1_correct"].sum())
    return {
        "query_count": len(view),
        "auto_count": accepted_count,
        "auto_coverage": accepted_count / len(view) if len(view) else None,
        "auto_correct_count": correct_count,
        "auto_error_count": accepted_count - correct_count,
        "auto_precision": correct_count / accepted_count if accepted_count else None,
        "top1_correct_count": int(view["top1_correct"].sum()),
        "oracle_max_auto_coverage": float(view["top1_correct"].mean()) if len(view) else None,
        "top1_correct_wilson_95": _binary_metric(view["top1_correct"].astype(bool)).get("wilson_95") if len(view) else None,
    }


def _review_reason(frame: pd.DataFrame) -> pd.Series:
    missing = frame[[
        "missing_crm_name",
        "missing_crm_address",
        "missing_crm_postcode",
        "missing_crm_city",
        "missing_crm_insee",
    ]].sum(axis=1)
    reason = pd.Series("LOW_CONFIDENCE", index=frame.index, dtype="object")
    reason.loc[frame["top1_alt_siren_score_gap"].lt(0.05)] = "AMBIGUOUS_SIREN"
    reason.loc[
        frame["top1_same_siren_count"].gt(1)
        & frame["top1_top2_score_gap"].lt(0.05)
    ] = "AMBIGUOUS_SITE"
    reason.loc[frame["top1_retrieval_rank"].gt(20)] = "RETRIEVAL_DISAGREEMENT"
    reason.loc[missing.ge(3)] = "OUT_OF_DISTRIBUTION"
    return reason


def _risk_coverage_curve(decisions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scores = decisions["accept_margin"].to_numpy(dtype=float)
    for requested_coverage in np.linspace(0.01, 1.0, 100):
        threshold = float(np.quantile(scores, 1.0 - requested_coverage, method="higher"))
        accepted = decisions["accept_margin"].ge(threshold)
        count = int(accepted.sum())
        correct = int(decisions.loc[accepted, "top1_correct"].sum())
        audited_open = int(
            (
                accepted
                & decisions["label_is_human_validated"].astype(bool)
                & decisions["label_kind"].isin(["AMBIGUOUS", "UNRESOLVED"])
            ).sum()
        )
        rows.append(
            {
                "model": model_name,
                "requested_coverage": requested_coverage,
                "margin_threshold": threshold,
                "auto_count": count,
                "auto_coverage": count / len(decisions),
                "auto_correct_count": correct,
                "auto_error_count": count - correct,
                "auto_precision": correct / count if count else np.nan,
                "audited_open_auto": audited_open,
            }
        )
    return pd.DataFrame(rows)


def _factories() -> dict[str, Callable[[], Any]]:
    return {
        "LOGISTIC": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, max_iter=1500, random_state=SEED),
        ),
        "XGBOOST": lambda: xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=450,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=10.0,
            reg_alpha=0.25,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def _fit_predict_nested(
    scenes: pd.DataFrame,
    features: list[str],
    model_name: str,
    factory: Callable[[], Any],
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts: list[pd.DataFrame] = []
    fold_details: dict[str, Any] = {}
    for outer_fold in range(5):
        calibration_fold = (outer_fold + 1) % 5
        train = scenes[~scenes["oof_fold"].astype(int).isin([outer_fold, calibration_fold])]
        calibration = scenes[scenes["oof_fold"].astype(int).eq(calibration_fold)]
        validation = scenes[scenes["oof_fold"].astype(int).eq(outer_fold)]
        model = factory()
        started = time.perf_counter()
        sample_weight = train["acceptor_weight"].astype(np.float32).to_numpy()
        fit_kwargs = (
            {"logisticregression__sample_weight": sample_weight}
            if model_name == "LOGISTIC"
            else {"sample_weight": sample_weight}
        )
        model.fit(
            train[features].to_numpy(dtype=np.float32),
            train["top1_correct"].astype(np.int8).to_numpy(),
            **fit_kwargs,
        )
        calibration_scores = model.predict_proba(
            calibration[features].to_numpy(dtype=np.float32)
        )[:, 1]
        audited_calibration = (
            calibration["label_is_human_validated"].astype(bool)
            & calibration["label_kind"].isin(["AMBIGUOUS", "UNRESOLVED"])
        ).to_numpy()
        threshold, calibration_metric = _select_threshold(
            calibration_scores,
            calibration["top1_correct"].astype(bool).to_numpy(),
            audited_calibration,
        )
        scores = model.predict_proba(validation[features].to_numpy(dtype=np.float32))[:, 1]
        output = validation[[
            "query_id",
            "oof_fold",
            "label_kind",
            "label_is_human_validated",
            "ground_truth_siret",
            "predicted_siret",
            "predicted_siren",
            "top1_correct",
        ]].copy()
        output["acceptor_score"] = scores.astype(np.float32)
        output["fold_threshold"] = np.float32(threshold)
        output["accept_margin"] = output["acceptor_score"] - output["fold_threshold"]
        output["auto_match"] = output["acceptor_score"].ge(output["fold_threshold"])
        output["model"] = model_name
        parts.append(output)
        model_path = model_dir / f"outer_fold_{outer_fold}"
        if model_name == "XGBOOST":
            model.get_booster().save_model(model_path.with_suffix(".json"))
        else:
            joblib.dump(model, model_path.with_suffix(".joblib"))
        fold_details[str(outer_fold)] = {
            "training_folds": sorted(set(range(5)) - {outer_fold, calibration_fold}),
            "calibration_fold": calibration_fold,
            "validation_fold": outer_fold,
            "training_rows": len(train),
            "calibration_rows": len(calibration),
            "validation_rows": len(validation),
            "threshold": threshold,
            "calibration": calibration_metric,
            "elapsed_seconds": time.perf_counter() - started,
        }
    output = pd.concat(parts, ignore_index=True).sort_values("query_id").reset_index(drop=True)
    if len(output) != len(scenes) or output["query_id"].duplicated().any():
        raise ValueError("Nested acceptor predictions do not cover scenes exactly once")
    return output, fold_details


def run(args: argparse.Namespace) -> Path:
    manifest_path = args.scenes / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes_path = args.scenes / "scenes.parquet"
    if manifest["outputs"].get("scenes.parquet") != file_sha256(scenes_path):
        raise ValueError("Scene dataset hash mismatch")
    if manifest.get("all_ranker_predictions_oof") is not True:
        raise ValueError("Acceptor requires OOF ranker scenes")
    if manifest.get("deterministic_promotions") is not False:
        raise ValueError("Acceptor refuses deterministic promotions")
    features = list(manifest["feature_order"])
    identity = {
        "schema_version": SCHEMA_VERSION,
        "runner_sha256": file_sha256(Path(__file__)),
        "scenes_manifest_sha256": file_sha256(manifest_path),
        "local_labels_sha256": file_sha256(args.local_labels),
        "features": features,
        "target_precision": TARGET_PRECISION,
        "nested_policy": "outer=f; calibration=(f+1)%5; train=remaining3",
        "models": sorted(_factories()),
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    scenes = pd.read_parquet(scenes_path)
    local = pd.read_csv(args.local_labels, dtype=str, keep_default_na=False)
    local_ids = set(local["query_id"].astype(str))
    local_open_ids = set(local.loc[local["label_kind"].isin(["AMBIGUOUS", "UNRESOLVED"]), "query_id"].astype(str))
    if scenes[features].isna().any().any() or not np.isfinite(scenes[features].to_numpy(dtype=np.float32)).all():
        raise ValueError("Acceptor scene matrix is not finite")
    if set(scenes["oof_fold"].astype(int)) != set(range(5)):
        raise ValueError("Acceptor requires five component folds")

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    results: dict[str, Any] = {}
    decisions_by_model: dict[str, pd.DataFrame] = {}
    curve_parts: list[pd.DataFrame] = []
    try:
        for name, factory in _factories().items():
            model_dir = temporary / f"models_{name.lower()}"
            model_dir.mkdir()
            decisions, folds = _fit_predict_nested(scenes, features, name, factory, model_dir)
            decisions["review_reason"] = _review_reason(
                scenes.set_index("query_id").reindex(decisions["query_id"]).reset_index()
            ).to_numpy()
            decisions.loc[decisions["auto_match"], "review_reason"] = ""
            decisions["decision"] = np.where(decisions["auto_match"], "AUTO_MATCH", "REVIEW")
            decisions["routing_status"] = np.where(decisions["auto_match"], "AUTO", "REVIEW")
            decisions_by_model[name] = decisions
            curve_parts.append(_risk_coverage_curve(decisions, name))
            metrics = {
                "all_queries": _metrics(decisions),
                "identifiable_exact": _metrics(decisions, decisions["label_kind"].eq("MATCH_EXACT")),
                "audited_279": _metrics(decisions, decisions["query_id"].isin(local_ids)),
                "audited_open": _metrics(decisions, decisions["query_id"].isin(local_open_ids)),
                "folds": {
                    str(fold): _metrics(decisions, decisions["oof_fold"].astype(int).eq(fold))
                    for fold in range(5)
                },
            }
            results[name] = {"metrics": metrics, "nested_folds": folds}
            decisions.to_parquet(temporary / f"decisions_{name.lower()}.parquet", index=False)

        eligible = [
            name for name, result in results.items()
            if (result["metrics"]["all_queries"]["auto_precision"] or 0.0) >= TARGET_PRECISION
            and result["metrics"]["audited_open"]["auto_count"] == 0
        ]
        if eligible:
            winner = max(
                eligible,
                key=lambda name: results[name]["metrics"]["all_queries"]["auto_coverage"],
            )
        else:
            winner = max(
                results,
                key=lambda name: (
                    results[name]["metrics"]["all_queries"]["auto_precision"] or 0.0,
                    results[name]["metrics"]["all_queries"]["auto_coverage"],
                ),
            )
        winner_metrics = results[winner]["metrics"]["all_queries"]
        gate = {
            "observed_precision_at_least_99_8": {
                "passed": (winner_metrics["auto_precision"] or 0.0) >= TARGET_PRECISION,
                "observed": winner_metrics["auto_precision"],
            },
            "coverage_between_88_and_92_percent": {
                "passed": 0.88 <= winner_metrics["auto_coverage"] <= 0.92,
                "observed": winner_metrics["auto_coverage"],
            },
            "zero_audited_open_auto": {
                "passed": results[winner]["metrics"]["audited_open"]["auto_count"] == 0,
                "observed": results[winner]["metrics"]["audited_open"]["auto_count"],
            },
            "ranker_gate_225_of_241": {"passed": False, "observed": 220},
        }
        verdict = "GO_FINAL_TEST" if all(value["passed"] for value in gate.values()) else "PIVOT_RANKER_UPPER_BOUND"
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "CONSUMED_DEVELOPMENT_NESTED_OOF",
            "independent_certification": False,
            "final_test_opened": False,
            "winner": winner,
            "target_precision": TARGET_PRECISION,
            "verdict": verdict,
            "gate": gate,
            "variants": results,
        }
        _json_dump(temporary / "evaluation.json", evaluation)
        winner_decisions = decisions_by_model[winner].copy()
        winner_decisions.to_parquet(temporary / "decisions.parquet", index=False)
        pd.concat(curve_parts, ignore_index=True).to_csv(
            temporary / "risk_coverage.csv", index=False
        )
        report_lines = [
            "# Accepteur appris V4.12-L",
            "",
            f"Verdict : **{verdict}**. Winner : **{winner}**.",
            "",
            "| Modèle | AUTO | Couverture | Précision observée | Erreurs AUTO | Audités ouverts AUTO | Borne oracle |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, result in results.items():
            m = result["metrics"]["all_queries"]
            opened = result["metrics"]["audited_open"]["auto_count"]
            report_lines.append(
                f"| {name} | {m['auto_count']} | {m['auto_coverage']:.2%} | "
                f"{m['auto_precision']:.3%} | {m['auto_error_count']} | {opened} | "
                f"{m['oracle_max_auto_coverage']:.2%} |"
            )
        report_lines.extend([
            "",
            "Chaque prédiction externe est exclue du train et de la calibration. Le seuil de chaque fold est choisi sur un autre fold, lui-même exclu du train. L'accepteur ne remplace jamais le top1.",
            "",
            "Le test final reste fermé : le gate ranker et la couverture AUTO ne sont pas franchis.",
        ])
        (temporary / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        output_hashes = {
            str(path.relative_to(temporary)): file_sha256(path)
            for path in temporary.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        _json_dump(
            temporary / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "build_id": build_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "build_identity": identity,
                "winner": winner,
                "verdict": verdict,
                "final_test_opened": False,
                "deterministic_promotions": False,
                "outputs": output_hashes,
            },
        )
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--local-labels", type=Path, default=DEFAULT_LOCAL_LABELS)
    parser.add_argument("--output-root", type=Path, default=BASE / "experiments/v4_12_learned_acceptor")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
