from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


TEXT_COLUMN = "nodule description"
TEXT_CORRUPTION_NONE = "none"
TEXT_CORRUPTION_SHUFFLE_MALIGNANT = "shuffle_within_malignant"
TEXT_CORRUPTION_NEGATE = "negate_attributes"
TEXT_CORRUPTION_PARAPHRASE_NEUTRAL = "paraphrase_neutral"
SUPPORTED_TEXT_CORRUPTIONS = {
    TEXT_CORRUPTION_NONE,
    TEXT_CORRUPTION_SHUFFLE_MALIGNANT,
    TEXT_CORRUPTION_NEGATE,
    TEXT_CORRUPTION_PARAPHRASE_NEUTRAL,
}
IMAGE_STAT_COLUMNS = [
    "img_view_count",
    "img_mean_intensity",
    "img_std_intensity",
    "img_min_intensity",
    "img_max_intensity",
    "img_p10_intensity",
    "img_p90_intensity",
]


@dataclass
class FeaturePlan:
    name: str
    include_text: bool
    include_image_stats: bool
    feature_cols: List[str]
    text_corruption: str = TEXT_CORRUPTION_NONE

    def selected_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        cols = [c for c in self.feature_cols if c in df.columns]
        if self.include_image_stats:
            cols.extend([c for c in IMAGE_STAT_COLUMNS if c in df.columns])
        # Stable unique order
        deduped: List[str] = []
        for c in cols:
            if c not in deduped:
                deduped.append(c)
        return deduped


def build_feature_plans(config: Dict[str, Any]) -> List[FeaturePlan]:
    plans: List[FeaturePlan] = []
    groups = config.get("ablations", {}).get("groups", [])
    for g in groups:
        text_corruption = str(g.get("text_corruption", TEXT_CORRUPTION_NONE))
        if text_corruption not in SUPPORTED_TEXT_CORRUPTIONS:
            raise ValueError(
                f"Unsupported text_corruption='{text_corruption}' for ablation={g.get('name')}. "
                f"Supported values: {sorted(SUPPORTED_TEXT_CORRUPTIONS)}"
            )
        plans.append(
            FeaturePlan(
                name=str(g["name"]),
                include_text=bool(g.get("include_text", False)),
                include_image_stats=bool(g.get("include_image_stats", False)),
                feature_cols=[str(c) for c in g.get("feature_cols", [])],
                text_corruption=text_corruption,
            )
        )
    if not plans:
        raise ValueError("No ablation groups configured under `ablations.groups`.")
    return plans


def _apply_case(template: str, replacement: str) -> str:
    if template.isupper():
        return replacement.upper()
    if template[:1].isupper():
        return replacement.capitalize()
    return replacement


def _replace_terms(text: str, mapping: Dict[str, str]) -> str:
    if not text:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(sorted((re.escape(k) for k in mapping.keys()), key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )

    def _repl(match: re.Match[str]) -> str:
        term = match.group(0)
        replacement = mapping.get(term.lower(), term)
        return _apply_case(term, replacement)

    return pattern.sub(_repl, text)


def _negate_description_text(text: str) -> str:
    negation_map = {
        "smooth": "spiculated",
        "spiculated": "smooth",
        "well-defined": "ill-defined",
        "well defined": "ill defined",
        "ill-defined": "well-defined",
        "ill defined": "well defined",
        "regular": "irregular",
        "irregular": "regular",
        "spherical": "irregular",
        "round": "irregular",
        "rounded": "irregular",
        "lobulated": "smooth",
        "solid": "ground-glass",
        "ground-glass": "solid",
        "homogeneous": "heterogeneous",
        "heterogeneous": "homogeneous",
        "calcified": "non-calcified",
        "non-calcified": "calcified",
        "small": "large",
        "large": "small",
        "minimal": "marked",
        "marked": "minimal",
        "low": "high",
        "high": "low",
    }
    return _replace_terms(text, negation_map)


def _neutral_paraphrase_text(text: str) -> str:
    neutral_map = {
        "spherical": "rounded",
        "round": "rounded",
        "spiculated": "with radiating margins",
        "lobulated": "with lobulated contour",
        "smooth": "with smooth margins",
        "irregular": "non-uniform",
        "homogeneous": "uniform",
        "heterogeneous": "mixed-pattern",
        "solid": "high-attenuation",
        "ground-glass": "low-attenuation",
        "well-defined": "well circumscribed",
        "ill-defined": "poorly circumscribed",
        "calcified": "showing calcification",
        "non-calcified": "without calcification",
    }
    return _replace_terms(text, neutral_map)


def apply_text_corruption(
    df: pd.DataFrame,
    *,
    label_col: str,
    corruption: str,
    text_col: str = TEXT_COLUMN,
) -> pd.DataFrame:
    if corruption == TEXT_CORRUPTION_NONE or text_col not in df.columns:
        return df
    if corruption not in SUPPORTED_TEXT_CORRUPTIONS:
        raise ValueError(f"Unsupported text corruption mode: {corruption}")

    out = df.copy()
    out[text_col] = out[text_col].fillna("").astype(str)

    if corruption == TEXT_CORRUPTION_SHUFFLE_MALIGNANT:
        if label_col not in out.columns:
            return out
        malignant_idx = out.index[out[label_col] == 1].tolist()
        if len(malignant_idx) > 1:
            if "nodule ID" in out.columns:
                malignant_idx = sorted(malignant_idx, key=lambda idx: (out.at[idx, "nodule ID"], idx))
            shuffled_values = out.loc[malignant_idx, text_col].tolist()
            shuffled_values = shuffled_values[-1:] + shuffled_values[:-1]
            out.loc[malignant_idx, text_col] = shuffled_values
        return out

    if corruption == TEXT_CORRUPTION_NEGATE:
        out[text_col] = out[text_col].map(_negate_description_text)
        return out

    if corruption == TEXT_CORRUPTION_PARAPHRASE_NEUTRAL:
        out[text_col] = out[text_col].map(_neutral_paraphrase_text)
        return out

    return out


def build_row_prompt(
    row: pd.Series,
    plan: FeaturePlan,
    numeric_cols: List[str],
    text_max_chars: int,
) -> str:
    lines = [
        "You are a lung nodule malignancy classifier.",
        "Predict whether the nodule is cancerous (binary).",
        "Return strict JSON with keys: label, probability.",
        "label must be 'Y' or 'N'. probability must be a float in [0, 1].",
        "",
        f"Ablation condition: {plan.name}",
    ]
    if plan.include_text and plan.text_corruption != TEXT_CORRUPTION_NONE:
        lines.append(f"Synthetic text control: {plan.text_corruption}")

    if numeric_cols:
        lines.append("Structured radiological/image features:")
        for col in numeric_cols:
            value = row.get(col, None)
            lines.append(f"- {col}: {value}")
    else:
        lines.append("Structured features: None included in this ablation.")

    if plan.include_text:
        text_value = str(row.get(TEXT_COLUMN, ""))
        if text_value.lower() == "nan":
            text_value = ""
        text_value = text_value[:text_max_chars]
        lines.append("")
        lines.append("Nodule description text:")
        lines.append(text_value if text_value else "<empty>")

    lines.append("")
    lines.append("Output JSON only. Do not include any extra keys.")
    return "\n".join(lines)
