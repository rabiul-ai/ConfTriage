"""
June 4, 2026. Md Rabiul Islam, ECEN, TAMU.
LIDC reader-comparison vs leave-one-out consensus + ConfTriage / DL backstop.

Inputs (code inputs/triage8 conftriage comp with radiologist/):
  - LIDC-IDRI FINAL metadata 955.xlsx  (conftriage_prob, certain_net_pred_prob)
  - LIDC-IDRI radiologists annotation 955.xlsx  (long per-reader malignancy)

Outputs (code outputs/triage8 conftriage comp with radiologist/<run_id>/):
  tables/, plots/, report/
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# ----------------------------- Paths ---------------------------------------

input_folder = (
    r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs"
    r"\Nodule Classification\Codes\code inputs\triage8 conftriage comp with radiologist"
)
output_root = (
    r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs"
    r"\Nodule Classification\Codes\code outputs\triage8 conftriage comp with radiologist"
)

INPUT_METADATA = os.path.join(input_folder, "LIDC-IDRI FINAL metadata 955.xlsx")
INPUT_RAD_LONG = os.path.join(input_folder, "LIDC-IDRI radiologists annotation 955.xlsx")


@dataclass(frozen=True)
class Config:
    nodule_id_col_meta: str = "nodule ID"
    nodule_selected_col: str = "nodule selected"
    nodule_selected_yes: str = "Yes"
    conftriage_prob_col: str = "conftriage_prob"
    dl_backstop_prob_col: str = "certain_net_pred_prob"
    method_name: str = "ConfTriage"
    backstop_name: str = "DL_Backstop"
    method_thresholds: Tuple[float, float, float] = (0.30, 0.50, 0.70)
    exclude_ambiguous_malignancy_3: bool = True
    max_readers_per_nodule: int = 4
    bootstrap_n: int = 1000
    bootstrap_seed: int = 7
    fig_dpi: int = 300


CFG = Config()


def _now_run_id() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


RUN_ID = _now_run_id()
OUT_DIR = os.path.join(output_root, RUN_ID)
OUT_TABLES = os.path.join(OUT_DIR, "tables")
OUT_PLOTS = os.path.join(OUT_DIR, "plots")
OUT_REPORT = os.path.join(OUT_DIR, "report")

for d in [OUT_DIR, OUT_TABLES, OUT_PLOTS, OUT_REPORT]:
    os.makedirs(d, exist_ok=True)

with open(os.path.join(output_root, "latest_run_path.txt"), "w", encoding="utf-8") as f:
    f.write(OUT_DIR)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


save_json(os.path.join(OUT_DIR, "run_config.json"), asdict(CFG))


# ----------------------------- Utilities -----------------------------------


def set_pub_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": CFG.fig_dpi,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


set_pub_style()


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def malignancy_to_binary(score: float) -> Optional[int]:
    """Protocol: >3 malignant, <3 benign, =3 ambiguous (excluded upstream)."""
    if pd.isna(score):
        return None
    s = float(score)
    if s > 3.0:
        return 1
    if s < 3.0:
        return 0
    return None


def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def safe_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = safe_confusion(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = sensitivity
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else np.nan
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cohen_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    if len(y1) == 0:
        return np.nan
    p0 = np.mean(y1 == y2)
    p_yes_1 = np.mean(y1 == 1)
    p_yes_2 = np.mean(y2 == 1)
    p_no_1 = 1 - p_yes_1
    p_no_2 = 1 - p_yes_2
    pe = p_yes_1 * p_yes_2 + p_no_1 * p_no_2
    if pe == 1:
        return np.nan
    return (p0 - pe) / (1 - pe)


def bootstrap_ci(
    df: pd.DataFrame,
    group_col: str,
    fn,
    n_boot: int,
    seed: int,
    min_required: int = 20,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    groups = df[group_col].dropna().unique()
    if len(groups) < 2 or len(df) < min_required:
        return {}

    metrics_samples: Dict[str, List[float]] = {}
    for _ in range(n_boot):
        boot_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = df[df[group_col].isin(boot_groups)]
        out = fn(sample)
        for k, v in out.items():
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                continue
            metrics_samples.setdefault(k, []).append(float(v))

    cis: Dict[str, Tuple[float, float]] = {}
    for k, vals in metrics_samples.items():
        if len(vals) < max(50, n_boot * 0.6):
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        cis[k] = (float(lo), float(hi))
    return cis


def save_df(df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(OUT_TABLES, f"{name}.csv"), index=False)


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_PLOTS, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_PLOTS, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


# ----------------------------- Load ----------------------------------------


meta = pd.read_excel(INPUT_METADATA)
meta.columns = [c.strip() for c in meta.columns]
save_df(meta, "00_metadata_raw")

df_raw = pd.read_excel(INPUT_RAD_LONG)
df_raw.columns = [c.strip() for c in df_raw.columns]
save_df(df_raw, "00_radiologist_long_raw")

required_cols = [
    "nodule_global_id",
    "patient_id",
    "nodule_index_in_patient",
    "radiologist_index_in_nodule",
    "malignancy",
]
missing_required = [c for c in required_cols if c not in df_raw.columns]
if missing_required:
    raise ValueError(f"Missing required columns in radiologist file: {missing_required}")

prob_cols = [CFG.conftriage_prob_col, CFG.dl_backstop_prob_col]
missing_prob = [c for c in prob_cols if c not in meta.columns]
if missing_prob:
    raise ValueError(f"Missing probability columns in metadata file: {missing_prob}")

# ----------------------------- Validation --------------------------------


df = df_raw.copy()
df["nodule_global_id"] = coerce_numeric(df["nodule_global_id"]).astype("Int64")
df["nodule_index_in_patient"] = coerce_numeric(df["nodule_index_in_patient"]).astype("Int64")
df["radiologist_index_in_nodule"] = coerce_numeric(df["radiologist_index_in_nodule"]).astype("Int64")
df["malignancy"] = coerce_numeric(df["malignancy"])

meta_sel = meta[meta[CFG.nodule_selected_col].astype(str).str.strip() == CFG.nodule_selected_yes].copy()
meta_sel[CFG.nodule_id_col_meta] = coerce_numeric(meta_sel[CFG.nodule_id_col_meta]).astype("Int64")
selected_ids = set(meta_sel[CFG.nodule_id_col_meta].dropna().astype(int).tolist())

df = df[df["nodule_global_id"].notna()].copy()
df = df[df["nodule_global_id"].astype(int).isin(selected_ids)].copy()
save_df(meta_sel, "00_metadata_selected_only")

df = (
    df.sort_values(["nodule_global_id", "radiologist_index_in_nodule"])
    .groupby("nodule_global_id", sort=False, as_index=False)
    .head(CFG.max_readers_per_nodule)
    .copy()
)
save_df(df, "01_filtered_selected_nodules_and_capped_readers")

stats: Dict = {}
stats["total_patients"] = int(df["patient_id"].nunique(dropna=True))
stats["total_nodules"] = int(df["nodule_global_id"].nunique(dropna=True))
stats["total_annotations"] = int(len(df))
stats["max_readers_per_nodule_applied"] = int(CFG.max_readers_per_nodule)

readers_per_nodule = df.groupby("nodule_global_id")["radiologist_index_in_nodule"].nunique(dropna=True)
stats["nodules_with_n_readers"] = {int(k): int(v) for k, v in readers_per_nodule.value_counts().sort_index().items()}
stats["malignancy_score_distribution"] = {
    str(k): int(v) for k, v in df["malignancy"].value_counts(dropna=False).sort_index().items()
}

dup_mask = df.duplicated(subset=["nodule_global_id", "radiologist_index_in_nodule"], keep=False)
stats["duplicate_rows_on_nodule_reader"] = int(dup_mask.sum())
save_df(df.loc[dup_mask].sort_values(["nodule_global_id", "radiologist_index_in_nodule"]), "01_duplicates_nodule_reader")
save_json(os.path.join(OUT_REPORT, "01_dataset_stats.json"), stats)

# Model probabilities (one row per nodule)
model_probs = meta_sel[
    [CFG.nodule_id_col_meta, CFG.conftriage_prob_col, CFG.dl_backstop_prob_col]
].rename(columns={CFG.nodule_id_col_meta: "nodule_global_id"}).copy()
model_probs["nodule_global_id"] = model_probs["nodule_global_id"].astype(int)
model_probs = model_probs.drop_duplicates(subset=["nodule_global_id"])
save_df(model_probs, "01_model_probs_per_nodule")

# ----------------------------- Preprocessing -------------------------------


df_pp = df.copy()
if CFG.exclude_ambiguous_malignancy_3:
    df_pp = df_pp[df_pp["malignancy"] != 3].copy()

df_pp["reader_binary"] = df_pp["malignancy"].apply(malignancy_to_binary).astype("Int64")
save_df(df_pp, "02_preprocessed_excluding_m3" if CFG.exclude_ambiguous_malignancy_3 else "02_preprocessed")


# -------------------- Leave-one-out consensus construction -----------------


def build_loo_consensus(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    consensus_scores: List[Optional[float]] = []
    consensus_bin: List[Optional[int]] = []
    valid_consensus: List[bool] = []

    by_nodule = {nid: g for nid, g in d.groupby("nodule_global_id", sort=False)}

    for _, r in d.iterrows():
        nid = r["nodule_global_id"]
        ridx = r["radiologist_index_in_nodule"]
        g = by_nodule.get(nid)
        if g is None:
            consensus_scores.append(np.nan)
            consensus_bin.append(np.nan)
            valid_consensus.append(False)
            continue

        others = g[g["radiologist_index_in_nodule"] != ridx]
        others_scores = others["malignancy"].dropna().to_numpy(dtype=float)
        if len(others_scores) == 0:
            consensus_scores.append(np.nan)
            consensus_bin.append(np.nan)
            valid_consensus.append(False)
            continue

        med = float(np.median(others_scores))
        consensus_scores.append(med)
        consensus_bin.append(malignancy_to_binary(med))
        valid_consensus.append(malignancy_to_binary(med) is not None)

    d["consensus_malignancy_median_loo"] = consensus_scores
    d["consensus_binary_loo"] = pd.Series(consensus_bin, index=d.index).astype("Int64")
    d["has_valid_loo_consensus"] = valid_consensus
    return d


df_loo = build_loo_consensus(df_pp)
df_loo_valid = df_loo[df_loo["has_valid_loo_consensus"]].copy()
save_df(df_loo, "03_loo_with_consensus_all")
save_df(df_loo_valid, "03_loo_with_consensus_valid_only")

# Nodule-level full median consensus (for model ROC / threshold metrics)
def build_nodule_median_consensus(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nid, sub in d.groupby("nodule_global_id", sort=False):
        scores = sub["malignancy"].dropna().to_numpy(dtype=float)
        if len(scores) == 0:
            continue
        med = float(np.median(scores))
        y = malignancy_to_binary(med)
        if y is None:
            continue
        rows.append(
            {
                "nodule_global_id": int(nid),
                "consensus_malignancy_median": med,
                "consensus_binary": int(y),
                "n_readers": int(len(scores)),
            }
        )
    return pd.DataFrame(rows)


df_nodule_consensus = build_nodule_median_consensus(df_pp)
df_nodule_model = df_nodule_consensus.merge(model_probs, on="nodule_global_id", how="inner")
save_df(df_nodule_model, "03_nodule_level_consensus_with_model_probs")

# LOO rows with model scores (for ConfTriage metrics vs same consensus as readers)
df_loo_model = df_loo_valid.merge(model_probs, on="nodule_global_id", how="inner")
save_df(df_loo_model, "03_loo_valid_with_model_probs")

# --------------------- Reader-vs-consensus analysis ------------------------


def reader_metrics_for_df(d: pd.DataFrame) -> Dict[str, float]:
    y_true = d["consensus_binary_loo"].astype(int).to_numpy()
    y_pred = d["reader_binary"].astype(int).to_numpy()
    out = compute_binary_metrics(y_true, y_pred)

    y_score = d["malignancy"].to_numpy(dtype=float)
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else np.nan
    except Exception:
        out["auc"] = np.nan

    out["kappa"] = float(cohen_kappa(y_pred, y_true))
    out["n"] = float(len(d))
    return out


reader_ids = sorted(df_loo_valid["radiologist_index_in_nodule"].dropna().unique())
reader_rows = []
confmats = {}
roc_curves = {}
boot_cis_rows = []

for rid in reader_ids:
    sub = df_loo_valid[df_loo_valid["radiologist_index_in_nodule"] == rid].copy()
    if len(sub) == 0:
        continue

    metrics = reader_metrics_for_df(sub)
    metrics["reader"] = f"R{int(rid)}"
    reader_rows.append(metrics)

    y_true = sub["consensus_binary_loo"].astype(int).to_numpy()
    y_pred = sub["reader_binary"].astype(int).to_numpy()
    y_score = sub["malignancy"].to_numpy(dtype=float)
    confmats[f"R{int(rid)}"] = safe_confusion(y_true, y_pred)

    try:
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_curves[f"R{int(rid)}"] = (fpr, tpr)
    except Exception:
        pass

    cis = bootstrap_ci(
        sub,
        group_col="nodule_global_id",
        fn=reader_metrics_for_df,
        n_boot=CFG.bootstrap_n,
        seed=CFG.bootstrap_seed + int(rid) * 31,
        min_required=50,
    )
    for k, (lo, hi) in cis.items():
        boot_cis_rows.append({"reader": f"R{int(rid)}", "metric": k, "ci_low": lo, "ci_high": hi})


df_reader_metrics = pd.DataFrame(reader_rows).sort_values("reader")
save_df(df_reader_metrics, "04_reader_vs_consensus_metrics")
save_df(pd.DataFrame(boot_cis_rows).sort_values(["reader", "metric"]), "04_reader_vs_consensus_bootstrap_cis")

for rname, (tn, fp, fn, tp) in confmats.items():
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix: {rname} vs LOO consensus")
    ax.set_xlabel("Predicted (Reader)")
    ax.set_ylabel("True (Consensus)")
    ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_yticklabels(["Benign", "Malignant"], rotation=0)
    save_fig(fig, f"cm_{rname}")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
for rname, (fpr, tpr) in roc_curves.items():
    auc_val = df_reader_metrics.loc[df_reader_metrics["reader"] == rname, "auc"]
    auc_str = f"{float(auc_val.iloc[0]):.3f}" if len(auc_val) else "NA"
    ax.plot(fpr, tpr, linewidth=2, label=f"{rname} (AUC={auc_str})")
ax.set_title("ROC Curves: Reader vs Leave-One-Out Consensus")
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.legend(loc="lower right")
save_fig(fig, "roc_per_reader")

fig, ax = plt.subplots(figsize=(6, 5))
tmp = df_reader_metrics.copy()
tmp["fpr"] = 1 - tmp["specificity"]
sns.scatterplot(data=tmp, x="fpr", y="sensitivity", hue="reader", s=80, ax=ax)
ax.set_title("Reader Operating Points")
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(title="Reader", loc="lower right")
save_fig(fig, "operating_points")

# -------------------- Model metrics (nodule-level consensus) ---------------


def model_metrics_at_threshold(
    d: pd.DataFrame, score_col: str, thr: float, label_col: str = "consensus_binary_loo"
) -> Dict[str, float]:
    y_true = d[label_col].astype(int).to_numpy()
    scores = d[score_col].to_numpy(dtype=float)
    y_pred = (scores >= thr).astype(int)
    out = compute_binary_metrics(y_true, y_pred)
    out["kappa"] = float(cohen_kappa(y_pred, y_true))
    out["auc"] = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) >= 2 else np.nan
    out["n"] = float(len(d))
    out["threshold"] = float(thr)
    return out


def model_roc(
    d: pd.DataFrame, score_col: str, label_col: str = "consensus_binary_loo"
) -> Tuple[np.ndarray, np.ndarray, float]:
    y_true = d[label_col].astype(int).to_numpy()
    scores = d[score_col].to_numpy(dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.array([0, 1]), np.array([0, 1]), np.nan
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr, float(roc_auc_score(y_true, scores))


fpr_m, tpr_m, auc_m = model_roc(df_loo_model, CFG.conftriage_prob_col)
fpr_b, tpr_b, auc_b = model_roc(df_loo_model, CFG.dl_backstop_prob_col)

# ---------------- Reader-comparison figure (ROC + operating envelope) ---------


reader_points = []
for _, r in df_reader_metrics.iterrows():
    if pd.isna(r.get("specificity")) or pd.isna(r.get("sensitivity")):
        continue
    reader_points.append((float(1 - r["specificity"]), float(r["sensitivity"])))

hull = convex_hull(reader_points) if len(set(reader_points)) >= 3 else []

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Chance")
ax.plot(fpr_m, tpr_m, linewidth=2.5, label=f"{CFG.method_name} ROC (AUC={auc_m:.3f})")
ax.plot(fpr_b, tpr_b, linewidth=2.5, label=f"{CFG.backstop_name} ROC (AUC={auc_b:.3f})")

for _, rr in df_reader_metrics.sort_values("reader").iterrows():
    if pd.isna(rr.get("specificity")) or pd.isna(rr.get("sensitivity")):
        continue
    x = float(1 - rr["specificity"])
    y = float(rr["sensitivity"])
    ax.scatter([x], [y], s=90, zorder=5)
    ax.text(x + 0.015, y + 0.015, rr["reader"], fontsize=10)

if hull:
    hx, hy = zip(*(hull + [hull[0]]))
    ax.plot(hx, hy, color="black", linewidth=1.8, label="Reader envelope (convex hull)")

# ConfTriage operating points at tau1, tau2, tau3
for thr in CFG.method_thresholds:
    m = model_metrics_at_threshold(df_loo_model, CFG.conftriage_prob_col, thr)
    x = 1 - m["specificity"]
    y = m["sensitivity"]
    ax.scatter([x], [y], s=70, marker="D", zorder=5)
    ax.text(x + 0.015, y - 0.04, f"{CFG.method_name} τ={thr:.2f}", fontsize=9)

ax.set_title("ROC & Reader Operating Envelope (vs LOO consensus)")
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="lower right")
save_fig(fig, "reader_comparison_roc_overlay")

# ---------------- Reader-comparison table ----------------------------------


rows_comp = []
for _, r in df_reader_metrics.iterrows():
    rows_comp.append(
        {
            "system": r["reader"],
            "threshold": np.nan,
            "n": r.get("n"),
            "sensitivity": r.get("sensitivity"),
            "specificity": r.get("specificity"),
            "accuracy": r.get("accuracy"),
            "auc": r.get("auc"),
            "kappa": r.get("kappa"),
        }
    )

for thr in CFG.method_thresholds:
    m = model_metrics_at_threshold(df_loo_model, CFG.conftriage_prob_col, thr)
    rows_comp.append(
        {
            "system": CFG.method_name,
            "threshold": thr,
            "n": m.get("n"),
            "sensitivity": m.get("sensitivity"),
            "specificity": m.get("specificity"),
            "accuracy": m.get("accuracy"),
            "auc": m.get("auc"),
            "kappa": m.get("kappa"),
        }
    )

rows_comp.append(
    {
        "system": CFG.backstop_name,
        "threshold": np.nan,
        "n": float(len(df_loo_model)),
        "sensitivity": np.nan,
        "specificity": np.nan,
        "accuracy": np.nan,
        "auc": auc_b,
        "kappa": np.nan,
    }
)

df_reader_comparison_table = pd.DataFrame(rows_comp)
save_df(df_reader_comparison_table, "04_reader_comparison_table_with_method_thresholds")

# ---------------------- Inter-reader agreement -----------------------------


pairwise_rows = []
df_pairs = df_pp[["nodule_global_id", "radiologist_index_in_nodule", "reader_binary"]].copy()
wide_bin = df_pairs.pivot_table(
    index="nodule_global_id",
    columns="radiologist_index_in_nodule",
    values="reader_binary",
    aggfunc="first",
)
reader_labels = [f"R{int(r)}" for r in wide_bin.columns]
wide_bin.columns = reader_labels

for i in range(len(reader_labels)):
    for j in range(i + 1, len(reader_labels)):
        r1, r2 = reader_labels[i], reader_labels[j]
        sub = wide_bin[[r1, r2]].dropna()
        if len(sub) == 0:
            k, n = np.nan, 0
        else:
            k = cohen_kappa(sub[r1].astype(int).to_numpy(), sub[r2].astype(int).to_numpy())
            n = int(len(sub))
        pairwise_rows.append({"reader_a": r1, "reader_b": r2, "kappa": k, "n_nodules_overlap": n})

df_pairwise_kappa = pd.DataFrame(pairwise_rows)
save_df(df_pairwise_kappa, "05_pairwise_cohen_kappa")

heat = pd.DataFrame(np.nan, index=reader_labels, columns=reader_labels)
np.fill_diagonal(heat.values, 1.0)
for _, r in df_pairwise_kappa.iterrows():
    heat.loc[r["reader_a"], r["reader_b"]] = r["kappa"]
    heat.loc[r["reader_b"], r["reader_a"]] = r["kappa"]

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, ax=ax)
ax.set_title("Pairwise Cohen's Kappa (binary malignancy)")
save_fig(fig, "kappa_heatmap_pairwise")

excel_out = os.path.join(OUT_TABLES, "reader_comparison_table.xlsx")
with pd.ExcelWriter(excel_out) as writer:
    df_reader_comparison_table.to_excel(writer, sheet_name="reader_comparison", index=False)
    df_reader_metrics.to_excel(writer, sheet_name="per_reader_loo", index=False)
    df_pairwise_kappa.to_excel(writer, sheet_name="pairwise_kappa", index=False)

# ----------------------------- Report --------------------------------------


def df_to_md_table(d: pd.DataFrame, max_rows: int = 20) -> str:
    d2 = d.head(max_rows).copy()
    cols = list(d2.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for _, row in d2.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4g}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


report_lines = [
    "# LIDC Reader Comparison: ConfTriage vs Radiologists",
    "",
    "## Dataset",
    f"- Nodules (selected): {stats['total_nodules']}",
    f"- Annotations (max {CFG.max_readers_per_nodule} readers/nodule): {stats['total_annotations']}",
    f"- Excluded malignancy == 3: {CFG.exclude_ambiguous_malignancy_3}",
    "- Binary rule: >3 malignant, <3 benign.",
    "",
    "## Reader vs LOO median consensus",
    df_to_md_table(
        df_reader_metrics[
            ["reader", "n", "sensitivity", "specificity", "accuracy", "auc", "kappa"]
        ]
    ),
    "",
    "## Reader-comparison table (readers + ConfTriage at τ1–τ3)",
    df_to_md_table(df_reader_comparison_table),
    "",
    "## Outputs",
    f"- Tables: `{OUT_TABLES}`",
    f"- Plots: `{OUT_PLOTS}`",
    f"- ROC overlay: `plots/reader_comparison_roc_overlay.(png|pdf)`",
]

report_path = os.path.join(OUT_REPORT, "final_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Done.")
print(f"Saved outputs to: {OUT_DIR}")
print(f"Report: {report_path}")
