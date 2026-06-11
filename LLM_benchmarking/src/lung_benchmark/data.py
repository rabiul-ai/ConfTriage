from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


IMAGE_REGEX = re.compile(
    r"^nid_(?P<nid>\d+)_sel_(?P<selected>Yes|No)_s_(?P<slice>\d+)_dia_(?P<dia>\d+)_p_(?P<patient>\d+)_n_(?P<nodule>\d+)_m_(?P<malignancy>\d+)_c_(?P<cancer>[YNU])\.png$"
)


@dataclass
class DataBundle:
    metadata_df: pd.DataFrame
    image_summary_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label_col: str


def _resolve_image_dir(image_dir: str | Path) -> Path:
    path = Path(image_dir)
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Image directory not found: {path}. Add local images under '2D Nodule Views'."
    )


def _parse_filename(filename: str) -> Dict[str, Any] | None:
    match = IMAGE_REGEX.match(filename)
    if not match:
        return None
    d = match.groupdict()
    return {
        "nodule ID": int(d["nid"]),
        "slice_idx": int(d["slice"]),
        "filename": filename,
    }


def _safe_image_stats(path: Path) -> Dict[str, float]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
    return {
        "img_mean_intensity": float(np.mean(arr)),
        "img_std_intensity": float(np.std(arr)),
        "img_min_intensity": float(np.min(arr)),
        "img_max_intensity": float(np.max(arr)),
        "img_p10_intensity": float(np.percentile(arr, 10)),
        "img_p90_intensity": float(np.percentile(arr, 90)),
    }


def summarize_image_features(image_dir: str | Path) -> pd.DataFrame:
    root = _resolve_image_dir(image_dir)
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.png")):
        parsed = _parse_filename(path.name)
        if parsed is None:
            continue
        stats = _safe_image_stats(path)
        rows.append({**parsed, **stats})
    if not rows:
        return pd.DataFrame(columns=["nodule ID"])
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("nodule ID")
        .agg(
            img_view_count=("filename", "count"),
            img_mean_intensity=("img_mean_intensity", "mean"),
            img_std_intensity=("img_std_intensity", "mean"),
            img_min_intensity=("img_min_intensity", "mean"),
            img_max_intensity=("img_max_intensity", "mean"),
            img_p10_intensity=("img_p10_intensity", "mean"),
            img_p90_intensity=("img_p90_intensity", "mean"),
        )
        .reset_index()
    )
    return agg


def _prepare_label(
    df: pd.DataFrame, label_col: str, include_uncertain: bool
) -> Tuple[pd.DataFrame, str]:
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not in metadata.")
    out = df.copy()
    out[label_col] = out[label_col].astype(str).str.strip().str.upper()
    if include_uncertain:
        out = out[out[label_col].isin(["Y", "N", "U"])].copy()
        out["target_binary"] = out[label_col].map({"Y": 1, "N": 0, "U": np.nan})
        out = out.dropna(subset=["target_binary"]).copy()
    else:
        out = out[out[label_col].isin(["Y", "N"])].copy()
        out["target_binary"] = out[label_col].map({"Y": 1, "N": 0})
    out["target_binary"] = out["target_binary"].astype(int)
    return out, "target_binary"


def _apply_protocol_cohort_filters(df: pd.DataFrame, data_cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply benchmark cohort inclusion/exclusion filters before label mapping."""
    protocol_cfg = data_cfg.get("cohort_protocol", {})
    if not bool(protocol_cfg.get("enabled", False)):
        return df

    out = df.copy()

    selected_col = str(protocol_cfg.get("selected_column", "nodule selected"))
    selected_value = str(protocol_cfg.get("selected_value", "Yes"))
    if selected_col not in out.columns:
        raise KeyError(
            f"Cohort protocol enabled but selected column '{selected_col}' is missing."
        )
    selected_mask = (
        out[selected_col]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq(selected_value.strip().casefold())
    )
    out = out[selected_mask].copy()

    diameter_col = str(protocol_cfg.get("diameter_column", "nodule diameter (mm)"))
    min_diameter_mm = float(protocol_cfg.get("min_diameter_mm", 3.0))
    if diameter_col not in out.columns:
        raise KeyError(
            f"Cohort protocol enabled but diameter column '{diameter_col}' is missing."
        )
    diameter_vals = pd.to_numeric(out[diameter_col], errors="coerce")
    out = out[diameter_vals > min_diameter_mm].copy()

    malignancy_col = str(protocol_cfg.get("malignancy_column", "malignancy"))
    exclude_malignancy_values = [
        float(v) for v in protocol_cfg.get("exclude_malignancy_values", [3])
    ]
    if malignancy_col not in out.columns:
        raise KeyError(
            f"Cohort protocol enabled but malignancy column '{malignancy_col}' is missing."
        )
    malignancy_vals = pd.to_numeric(out[malignancy_col], errors="coerce")
    out = out[~malignancy_vals.isin(exclude_malignancy_values)].copy()

    return out


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _scale_columns(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cols: List[str]):
    scaler = StandardScaler()
    available = []
    for c in cols:
        if c not in train.columns:
            continue
        # Skip columns that are entirely missing/non-finite in train split.
        series = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() == 0:
            continue
        available.append(c)
    if not available:
        return train, val, test
    train[available] = scaler.fit_transform(train[available])
    val[available] = scaler.transform(val[available])
    test[available] = scaler.transform(test[available])
    return train, val, test


def _empty_image_summary_with_schema() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "nodule ID",
            "img_view_count",
            "img_mean_intensity",
            "img_std_intensity",
            "img_min_intensity",
            "img_max_intensity",
            "img_p10_intensity",
            "img_p90_intensity",
        ]
    )


def load_dataset(config: Dict[str, Any]) -> DataBundle:
    data_cfg = config["data"]
    split_cfg = config["splits"]
    prep_cfg = config["preprocessing"]

    metadata_path = Path(data_cfg["metadata_path"])
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    metadata_df = pd.read_excel(metadata_path)
    metadata_df = _apply_protocol_cohort_filters(metadata_df, data_cfg)

    metadata_df, label_out_col = _prepare_label(
        metadata_df,
        label_col=data_cfg["label_column"],
        include_uncertain=bool(data_cfg.get("include_uncertain", False)),
    )

    image_summary_df = _empty_image_summary_with_schema()
    image_dir = Path(data_cfg.get("image_dir", "2D Nodule Views"))
    if image_dir.exists():
        image_summary_df = summarize_image_features(image_dir)
    metadata_df = metadata_df.merge(image_summary_df, on="nodule ID", how="left")

    train_df, test_df = train_test_split(
        metadata_df,
        test_size=float(split_cfg["test_size"]),
        random_state=int(config["project"]["seed"]),
        stratify=metadata_df[label_out_col] if bool(split_cfg.get("stratify", True)) else None,
    )
    val_size = float(split_cfg["val_size"]) / (1.0 - float(split_cfg["test_size"]))
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=int(config["project"]["seed"]),
        stratify=train_df[label_out_col] if bool(split_cfg.get("stratify", True)) else None,
    )

    # Coerce common numeric columns to numeric type.
    numeric_candidates = [
        "subtlety",
        "internal structure",
        "calcification",
        "sphericity",
        "margin",
        "lobulation",
        "spiculation",
        "textures",
        "nodule diameter (mm)",
        "img_view_count",
        "img_mean_intensity",
        "img_std_intensity",
        "img_min_intensity",
        "img_max_intensity",
        "img_p10_intensity",
        "img_p90_intensity",
    ]
    train_df = _coerce_numeric(train_df, numeric_candidates)
    val_df = _coerce_numeric(val_df, numeric_candidates)
    test_df = _coerce_numeric(test_df, numeric_candidates)

    for col in numeric_candidates:
        if col in train_df.columns:
            fill = train_df[col].median() if prep_cfg.get("impute_strategy", "median") == "median" else train_df[col].mean()
            if pd.isna(fill):
                fill = 0.0
            train_df[col] = train_df[col].fillna(fill)
            val_df[col] = val_df[col].fillna(fill)
            test_df[col] = test_df[col].fillna(fill)

    if bool(prep_cfg.get("scale_numeric", True)):
        train_df, val_df, test_df = _scale_columns(train_df, val_df, test_df, numeric_candidates)

    return DataBundle(
        metadata_df=metadata_df,
        image_summary_df=image_summary_df,
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        label_col=label_out_col,
    )

