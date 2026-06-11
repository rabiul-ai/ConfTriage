from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    ci: Dict[str, Tuple[float, float]]
    confusion: Dict[str, int]


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "average_precision": _safe_ap(y_true, y_prob),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    rounds: int = 300,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    names = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    storage: Dict[str, List[float]] = {k: [] for k in names}

    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for k in names:
            v = m[k]
            if np.isfinite(v):
                storage[k].append(v)

    lower = alpha / 2.0
    upper = 1.0 - alpha / 2.0
    out: Dict[str, Tuple[float, float]] = {}
    for k in names:
        vals = storage[k]
        if not vals:
            out[k] = (float("nan"), float("nan"))
        else:
            out[k] = (float(np.quantile(vals, lower)), float(np.quantile(vals, upper)))
    return out


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    bootstrap_rounds: int,
    bootstrap_alpha: float,
    seed: int,
) -> EvalResult:
    metrics = compute_metrics(y_true, y_pred, y_prob)
    ci = bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        rounds=bootstrap_rounds,
        alpha=bootstrap_alpha,
        seed=seed,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return EvalResult(
        metrics=metrics,
        ci=ci,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )


def make_result_row(model_id: str, family: str, ablation: str, split: str, eval_result: EvalResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model_id": model_id,
        "family": family,
        "ablation": ablation,
        "split": split,
    }
    row.update(eval_result.metrics)
    for metric_name, (lo, hi) in eval_result.ci.items():
        row[f"{metric_name}_ci_low"] = lo
        row[f"{metric_name}_ci_high"] = hi
    row.update(eval_result.confusion)
    return row


def collect_feature_importance(train_df: pd.DataFrame, numeric_cols: List[str], target_col: str) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(columns=["feature", "abs_correlation"])
    y = train_df[target_col].astype(float)
    y_filled = y.fillna(y.median())
    y_std = float(np.std(y_filled))
    rows = []
    for c in numeric_cols:
        x = pd.to_numeric(train_df[c], errors="coerce")
        corr = np.nan
        if x.notna().sum() > 2:
            x_filled = x.fillna(x.median())
            x_std = float(np.std(x_filled))
            if x_std > 0 and y_std > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = np.corrcoef(x_filled, y_filled)[0, 1]
        rows.append({"feature": c, "abs_correlation": abs(float(corr)) if np.isfinite(corr) else np.nan})
    out = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return out


def _paired_bootstrap_metric_differences(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_prob_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_prob_b: np.ndarray,
    metrics: List[str],
    rounds: int,
    alpha: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    obs_a = compute_metrics(y_true, y_pred_a, y_prob_a)
    obs_b = compute_metrics(y_true, y_pred_b, y_prob_b)
    observed = {m: float(obs_a[m] - obs_b[m]) for m in metrics}

    rng = np.random.default_rng(seed)
    n = len(y_true)
    storage: Dict[str, List[float]] = {m: [] for m in metrics}

    for _ in range(rounds):
        idx = rng.integers(0, n, size=n)
        m_a = compute_metrics(y_true[idx], y_pred_a[idx], y_prob_a[idx])
        m_b = compute_metrics(y_true[idx], y_pred_b[idx], y_prob_b[idx])
        for metric in metrics:
            a_val = m_a.get(metric, float("nan"))
            b_val = m_b.get(metric, float("nan"))
            if np.isfinite(a_val) and np.isfinite(b_val):
                storage[metric].append(float(a_val - b_val))

    lower = alpha / 2.0
    upper = 1.0 - alpha / 2.0
    out: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        vals = np.asarray(storage[metric], dtype=float)
        if vals.size == 0:
            out[metric] = {
                "observed_diff": observed[metric],
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p_value": float("nan"),
                "valid_bootstrap_samples": 0.0,
            }
            continue
        ci_low = float(np.quantile(vals, lower))
        ci_high = float(np.quantile(vals, upper))
        p_left = float(np.mean(vals <= 0.0))
        p_right = float(np.mean(vals >= 0.0))
        p_val = min(1.0, 2.0 * min(p_left, p_right))
        out[metric] = {
            "observed_diff": observed[metric],
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_val,
            "valid_bootstrap_samples": float(vals.size),
        }
    return out


def build_pairwise_bootstrap_stat_tests(
    predictions_df: pd.DataFrame,
    metrics: List[str],
    rounds: int,
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    required = {
        "ablation",
        "model_id",
        "model_short_name",
        "nodule ID",
        "true_label",
        "pred_label",
        "pred_prob",
    }
    if predictions_df.empty or not required.issubset(set(predictions_df.columns)):
        return pd.DataFrame(
            columns=[
                "ablation",
                "metric",
                "model_a",
                "model_a_short",
                "model_b",
                "model_b_short",
                "n_samples",
                "observed_diff_a_minus_b",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value_fdr_bh",
                "significant_p_0_05",
                "significant_fdr_0_05",
                "better_model_by_observed_diff",
                "valid_bootstrap_samples",
            ]
        )

    rows: List[Dict[str, Any]] = []
    pred = predictions_df.copy()
    pred["ablation"] = pred["ablation"].astype(str)
    pred["model_id"] = pred["model_id"].astype(str)
    pred["model_short_name"] = pred["model_short_name"].astype(str)

    for ablation, ab_df in pred.groupby("ablation", sort=False):
        model_ids = list(dict.fromkeys(ab_df["model_id"].tolist()))
        for model_a, model_b in combinations(model_ids, 2):
            a_df = ab_df[ab_df["model_id"] == model_a][
                ["nodule ID", "true_label", "pred_label", "pred_prob", "model_short_name"]
            ].copy()
            b_df = ab_df[ab_df["model_id"] == model_b][
                ["nodule ID", "true_label", "pred_label", "pred_prob", "model_short_name"]
            ].copy()

            merged = a_df.merge(
                b_df,
                on="nodule ID",
                suffixes=("_a", "_b"),
                how="inner",
            )
            if merged.empty:
                continue

            y_true = merged["true_label_a"].astype(int).to_numpy()
            y_pred_a = merged["pred_label_a"].astype(int).to_numpy()
            y_prob_a = merged["pred_prob_a"].astype(float).to_numpy()
            y_pred_b = merged["pred_label_b"].astype(int).to_numpy()
            y_prob_b = merged["pred_prob_b"].astype(float).to_numpy()

            diff_stats = _paired_bootstrap_metric_differences(
                y_true=y_true,
                y_pred_a=y_pred_a,
                y_prob_a=y_prob_a,
                y_pred_b=y_pred_b,
                y_prob_b=y_prob_b,
                metrics=metrics,
                rounds=rounds,
                alpha=alpha,
                seed=seed,
            )

            model_a_short = str(merged["model_short_name_a"].iloc[0])
            model_b_short = str(merged["model_short_name_b"].iloc[0])
            for metric in metrics:
                stat = diff_stats[metric]
                obs = stat["observed_diff"]
                better = ""
                if np.isfinite(obs):
                    if obs > 0:
                        better = model_a_short
                    elif obs < 0:
                        better = model_b_short
                    else:
                        better = "tie"
                rows.append(
                    {
                        "ablation": ablation,
                        "metric": metric,
                        "model_a": model_a,
                        "model_a_short": model_a_short,
                        "model_b": model_b,
                        "model_b_short": model_b_short,
                        "n_samples": int(len(merged)),
                        "observed_diff_a_minus_b": float(obs),
                        "ci_low": float(stat["ci_low"]),
                        "ci_high": float(stat["ci_high"]),
                        "p_value": float(stat["p_value"]),
                        "q_value_fdr_bh": float("nan"),
                        "significant_p_0_05": bool(np.isfinite(stat["p_value"]) and stat["p_value"] < 0.05),
                        "significant_fdr_0_05": False,
                        "better_model_by_observed_diff": better,
                        "valid_bootstrap_samples": int(stat["valid_bootstrap_samples"]),
                    }
                )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    valid_idx = out["p_value"].notna()
    if valid_idx.any():
        _, q_vals, _, _ = multipletests(
            out.loc[valid_idx, "p_value"].astype(float).to_numpy(),
            alpha=0.05,
            method="fdr_bh",
        )
        out.loc[valid_idx, "q_value_fdr_bh"] = q_vals
        out.loc[valid_idx, "significant_fdr_0_05"] = out.loc[valid_idx, "q_value_fdr_bh"] < 0.05
    return out
