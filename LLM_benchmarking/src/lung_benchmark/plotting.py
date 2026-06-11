from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def build_model_palette(model_labels: List[str]) -> Dict[str, str]:
    unique = []
    for label in model_labels:
        if label not in unique:
            unique.append(label)
    if not unique:
        return {}
    base = sns.color_palette("tab20", n_colors=max(3, len(unique)))
    return {label: to_hex(base[i]) for i, label in enumerate(unique)}


def plot_metric_bars(
    results_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    model_col: str = "model_short_name",
    hue_order: List[str] | None = None,
    palette: Dict[str, str] | None = None,
    style: str = "whitegrid",
    dpi: int = 160,
) -> None:
    if results_df.empty:
        return
    if model_col not in results_df.columns:
        return
    sns.set_style(style)
    plt.figure(figsize=(12, 6))
    plot_df = results_df.copy()
    plot_df = plot_df.sort_values(metric, ascending=False)
    ax = sns.barplot(
        data=plot_df,
        x="ablation",
        y=metric,
        hue=model_col,
        hue_order=hue_order,
        palette=palette,
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel(metric.replace("_", " ").title())
    # Most benchmark metrics are in [0, 1], so format as percentages.
    if plot_df[metric].min() >= 0 and plot_df[metric].max() <= 1:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    # Put legend above plot area in a horizontal row.
    legend_order = hue_order or list(dict.fromkeys(plot_df[model_col].astype(str).tolist()))
    ncol = max(1, len(legend_order))
    ax.legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.1,
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_family_heatmap(results_df: pd.DataFrame, metric: str, out_path: Path, style: str = "whitegrid", dpi: int = 160) -> None:
    if results_df.empty:
        return
    sns.set_style(style)
    pivot = (
        results_df.groupby(["family", "ablation"], as_index=False)[metric]
        .mean()
        .pivot(index="family", columns="ablation", values=metric)
    )
    if pivot.empty:
        return

    label_fontsize = 17
    tick_fontsize = 16
    cell_fontsize = 20
    n_cols = max(1, int(pivot.shape[1]))
    n_rows = max(1, int(pivot.shape[0]))
    fig_w = max(12.0, 1.7 * n_cols + 3.0)
    fig_h = max(5.6, 1.35 * n_rows + 3.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    heat = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        annot_kws={"size": cell_fontsize, "weight": "semibold"},
        cbar_kws={"label": metric.replace("_", " ").title()},
        ax=ax,
    )
    cbar = heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label(metric.replace("_", " ").title(), fontsize=label_fontsize)

    ax.set_title("")
    ax.set_xlabel("ablation", fontsize=label_fontsize)
    ax.set_ylabel("family", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=90)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path, style: str = "whitegrid", dpi: int = 160, top_k: int = 20) -> None:
    if importance_df.empty:
        return
    sns.set_style(style)
    plot_df = importance_df.head(top_k).copy()
    plt.figure(figsize=(9, 6))
    sns.barplot(data=plot_df, x="abs_correlation", y="feature", orient="h")
    plt.title("Approximate feature importance (abs correlation)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_roc_curves(
    predictions_df: pd.DataFrame,
    out_path: Path,
    model_col: str = "model_short_name",
    hue_order: List[str] | None = None,
    palette: Dict[str, str] | None = None,
    style: str = "whitegrid",
    dpi: int = 160,
) -> None:
    if predictions_df.empty:
        return
    required = {"ablation", model_col, "true_label", "pred_prob"}
    if not required.issubset(set(predictions_df.columns)):
        return

    sns.set_style(style)
    ablations = list(dict.fromkeys(predictions_df["ablation"].astype(str).tolist()))
    if not ablations:
        return

    n = len(ablations)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
    flat_axes = list(np.atleast_1d(axes).ravel())

    if hue_order is None:
        hue_order = list(dict.fromkeys(predictions_df[model_col].astype(str).tolist()))
    palette = palette or {}

    for idx, ablation in enumerate(ablations):
        ax = flat_axes[idx]
        subset = predictions_df[predictions_df["ablation"].astype(str) == ablation].copy()
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        lines_plotted = 0
        for model_name in hue_order:
            model_df = subset[subset[model_col].astype(str) == str(model_name)]
            if model_df.empty:
                continue
            y_true = model_df["true_label"].astype(int).to_numpy()
            y_prob = model_df["pred_prob"].astype(float).to_numpy()
            if len(set(y_true.tolist())) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = float(auc(fpr, tpr))
            color = palette.get(str(model_name))
            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                color=color,
                label=f"{model_name} (AUC={roc_auc:.3f})",
            )
            lines_plotted += 1
        ax.set_title(f"ROC Curve - {ablation}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        if lines_plotted > 0:
            ax.legend(title="Model", loc="lower right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No valid ROC curves", ha="center", va="center")

    for j in range(len(ablations), len(flat_axes)):
        fig.delaxes(flat_axes[j])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_prompt_temperature_auc_heatmap(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    template_col: str = "template_id",
    temperature_col: str = "temperature",
    mean_auc_col: str = "auc_mean",
    std_auc_col: str = "auc_std",
    style: str = "whitegrid",
    dpi: int = 180,
) -> None:
    """
    Plot a 5x5 template x temperature AUC heatmap.
    - Base colormap encodes mean AUC.
    - Per-cell black alpha overlay encodes replicate std (darker = higher std).
    """
    required = {template_col, temperature_col, mean_auc_col, std_auc_col}
    if summary_df.empty or not required.issubset(set(summary_df.columns)):
        return

    df = summary_df.copy()
    df[template_col] = df[template_col].astype(str)
    df[temperature_col] = pd.to_numeric(df[temperature_col], errors="coerce")
    df = df.dropna(subset=[temperature_col])
    if df.empty:
        return

    temp_order = sorted(df[temperature_col].unique().tolist())
    # Keep canonical T1..T5 order when available.
    template_order = sorted(
        df[template_col].unique().tolist(),
        key=lambda x: (0, int(x[1:])) if isinstance(x, str) and x.startswith("T") and x[1:].isdigit() else (1, x),
    )

    mean_pivot = (
        df.pivot_table(index=template_col, columns=temperature_col, values=mean_auc_col, aggfunc="mean")
        .reindex(index=template_order, columns=temp_order)
    )
    std_pivot = (
        df.pivot_table(index=template_col, columns=temperature_col, values=std_auc_col, aggfunc="mean")
        .reindex(index=template_order, columns=temp_order)
    )

    label_fontsize = 21
    tick_fontsize = 19
    cell_fontsize = 22
    footnote_fontsize = 15

    sns.set_style(style)
    fig, ax = plt.subplots(figsize=(16, 10))
    heat = sns.heatmap(
        mean_pivot,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        annot=False,
        cbar_kws={"label": "Mean AUC across replicates"},
        ax=ax,
    )
    cbar = heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_fontsize)
    cbar.set_label("Mean AUC across replicates", fontsize=label_fontsize)

    std_vals = std_pivot.to_numpy(dtype=float)
    finite_std = std_vals[np.isfinite(std_vals)]
    max_std = float(np.max(finite_std)) if finite_std.size else 0.0
    min_std = float(np.min(finite_std)) if finite_std.size else 0.0

    for i in range(mean_pivot.shape[0]):
        for j in range(mean_pivot.shape[1]):
            mean_val = mean_pivot.iat[i, j]
            std_val = std_pivot.iat[i, j]
            if np.isfinite(std_val) and max_std > 0:
                alpha = min(0.55, max(0.0, float(std_val) / max_std * 0.55))
                ax.add_patch(
                    mpatches.Rectangle(
                        (j, i),
                        1,
                        1,
                        facecolor="black",
                        edgecolor="none",
                        alpha=alpha,
                    )
                )
            if np.isfinite(mean_val):
                txt = f"{mean_val:.3f}"
                if np.isfinite(std_val):
                    txt = f"{mean_val:.3f}\n±{std_val:.3f}"
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    txt,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=cell_fontsize,
                    fontweight="semibold",
                )

    ax.set_title("")
    ax.set_xlabel("Temperature", fontsize=label_fontsize)
    ax.set_ylabel("Prompt template", fontsize=label_fontsize)
    ax.tick_params(axis="x", labelrotation=0, labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    if finite_std.size:
        ax.text(
            0.01,
            -0.13,
            f"Darker overlay = higher replicate std (min={min_std:.4f}, max={max_std:.4f})",
            transform=ax.transAxes,
            fontsize=footnote_fontsize,
            ha="left",
            va="top",
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
