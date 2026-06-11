from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from .config import ensure_output_dirs
from .data import load_dataset
from .evaluation import evaluate_predictions
from .features import FeaturePlan, TEXT_COLUMN, build_row_prompt
from .logging_utils import setup_logger
from .models import get_runner, model_specs_from_config


ATTR_COLUMNS = [
    "subtlety",
    "internal structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "textures",
    "nodule diameter (mm)",
]


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _ensure_desc_source_dirs(output_root: Path) -> Dict[str, Path]:
    root = output_root / "description_source_ablation"
    dirs = {
        "root": root,
        "tables": root / "tables",
        "figures": root / "figures",
        "logs": root / "logs",
        "predictions": root / "predictions",
        "manifests": root / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _format_attr_value(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return str(value)


def _deterministic_attribute_template(row: pd.Series, *, drop_attrs: set[str] | None = None) -> str:
    drop_attrs = drop_attrs or set()
    parts = []
    for col in ATTR_COLUMNS:
        if col in drop_attrs:
            continue
        if col in row.index:
            parts.append(f"{col}: {_format_attr_value(row[col])}")
    joined = "; ".join(parts)
    return (
        "Deterministic attribute template: "
        f"{joined}. This description is generated directly from structured radiology attributes."
    )


def _adversarial_paraphrase(text: str) -> str:
    # Deterministic lexical substitutions designed to reduce simple trigger-term matching.
    substitutions = {
        "nodule": "lesion unit",
        "deterministic": "rule-bound",
        "attribute": "descriptor field",
        "template": "rendering",
        "subtlety": "subtle-pattern index",
        "internal structure": "intra-lesion architecture",
        "calcification": "mineralization pattern",
        "sphericity": "roundness profile",
        "margin": "boundary contour",
        "lobulation": "lobed contour",
        "spiculation": "radial-edge irregularity",
        "textures": "texture signature",
        "diameter": "span",
        "missing": "not-reported",
    }
    out = text
    for src, tgt in substitutions.items():
        out = out.replace(src, tgt)
        out = out.replace(src.title(), tgt.title())
    return out


def _call_openrouter_chat(
    *,
    endpoint: str,
    api_key: str,
    model_id: str,
    prompt: str,
    temperature: float,
    seed: int,
    max_tokens: int = 300,
    timeout_sec: int = 90,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    site_url = os.getenv("OPENROUTER_SITE_URL", "")
    site_name = os.getenv("OPENROUTER_SITE_NAME", "")
    if "openrouter.ai" in endpoint:
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "seed": int(seed),
        "max_tokens": int(max_tokens),
    }
    res = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_sec)
    res.raise_for_status()
    data = res.json()
    return str(data["choices"][0]["message"]["content"])


def _build_llama_paraphrase_prompt(text: str) -> str:
    return (
        "Paraphrase the following radiology description while preserving all attribute information exactly. "
        "Do not add or remove facts. Keep it concise and natural.\n\n"
        f"Original:\n{text}\n\nParaphrase:"
    )


def _upsert_desc_source_text_cache(
    cache_path: Path,
    cache_rows: List[Dict[str, Any]],
) -> None:
    pd.DataFrame(cache_rows).to_csv(cache_path, index=False)


def _load_desc_source_text_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(cache_path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def run_description_source_ablation(config: Dict[str, Any]) -> Dict[str, Any]:
    core_dirs = ensure_output_dirs(config)
    dirs = _ensure_desc_source_dirs(core_dirs["root"])
    logger = setup_logger(dirs["logs"] / "description_source_ablation.log")

    desc_cfg = config.get("description_source_ablation", {})
    target_model_id = str(desc_cfg.get("target_model_id", "google/gemini-3.1-flash-lite-preview"))
    paraphraser_model_id = str(desc_cfg.get("paraphraser_model_id", "meta-llama/llama-3.1-8b-instruct"))
    paraphraser_endpoint = str(
        desc_cfg.get("paraphraser_endpoint", "https://openrouter.ai/api/v1/chat/completions")
    )
    paraphraser_api_key_env = str(desc_cfg.get("paraphraser_api_key_env", "OPENROUTER_API_KEY"))
    paraphraser_temperature = float(desc_cfg.get("paraphraser_temperature", 0.0))
    paraphraser_seed = int(desc_cfg.get("paraphraser_seed", 42))
    bootstrap_rounds = int(desc_cfg.get("bootstrap_rounds", config.get("evaluation", {}).get("bootstrap_rounds", 300)))
    bootstrap_alpha = float(desc_cfg.get("bootstrap_alpha", config.get("evaluation", {}).get("bootstrap_alpha", 0.05)))
    text_max_chars = int(config.get("preprocessing", {}).get("text_max_chars", 1600))
    classification_temperature = float(desc_cfg.get("classification_temperature", 0.0))
    retry_attempts = int(desc_cfg.get("paraphraser_retry_attempts", 3))
    prediction_flush_every = max(1, int(desc_cfg.get("prediction_flush_every", 5)))
    near_cfg = desc_cfg.get("near_leak_ablation", {})
    near_enabled = bool(near_cfg.get("enabled", True))

    logger.info("Loading filtered cohort for description-source ablation")
    bundle = load_dataset(config)
    full_df = pd.concat([bundle.train_df, bundle.val_df, bundle.test_df], ignore_index=True).copy()
    full_df = full_df.reset_index(drop=True)
    if len(full_df) != 955:
        logger.warning(
            "Description-source ablation expected 955 filtered nodules for consistency, found %d",
            len(full_df),
        )
    else:
        logger.info("Description-source ablation running on full filtered cohort of 955 nodules")
    y_full = full_df[bundle.label_col].astype(int).to_numpy()

    model_spec = None
    for spec in model_specs_from_config(config):
        if spec.model_id == target_model_id:
            model_spec = spec
            break
    if model_spec is None:
        raise ValueError(f"Target model '{target_model_id}' not found in config.models")
    model_spec = replace(model_spec, temperature=classification_temperature)
    if model_spec.api_key_env and not os.getenv(model_spec.api_key_env):
        raise RuntimeError(
            f"Missing API key env var '{model_spec.api_key_env}' for target model {target_model_id}"
        )

    paraphraser_api_key = os.getenv(paraphraser_api_key_env, "")
    if not paraphraser_api_key:
        raise RuntimeError(
            f"Missing API key env var '{paraphraser_api_key_env}' for paraphraser model {paraphraser_model_id}"
        )

    logger.info("Building deterministic template descriptions")
    deterministic_texts = [_deterministic_attribute_template(row) for _, row in full_df.iterrows()]
    adversarial_texts = [_adversarial_paraphrase(text) for text in deterministic_texts]
    near_remove_spiculation_texts = [
        _deterministic_attribute_template(row, drop_attrs={"spiculation"})
        for _, row in full_df.iterrows()
    ]
    near_remove_margin_texts = [
        _deterministic_attribute_template(row, drop_attrs={"margin"})
        for _, row in full_df.iterrows()
    ]
    near_remove_both_texts = [
        _deterministic_attribute_template(row, drop_attrs={"spiculation", "margin"})
        for _, row in full_df.iterrows()
    ]

    text_cache_path = dirs["tables"] / "description_source_texts.csv"
    cache_df = _load_desc_source_text_cache(text_cache_path)
    if cache_df.empty or "nodule ID" not in cache_df.columns:
        cache_df = pd.DataFrame(columns=["nodule ID", "deterministic_text", "llama_paraphrase_text", "adversarial_text"])
    cache_df["nodule ID"] = cache_df["nodule ID"].astype(str)
    cache_map = {str(row["nodule ID"]): row for _, row in cache_df.iterrows()}

    logger.info("Generating Llama paraphrases (with resume/cache support)")
    cache_rows: List[Dict[str, Any]] = []
    llama_paraphrase_texts: List[str] = []
    for idx, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Llama paraphrases", unit="nodule"):
        nodule_id = str(row["nodule ID"])
        deterministic_text = deterministic_texts[idx]
        adversarial_text = adversarial_texts[idx]

        cached = cache_map.get(nodule_id, {})
        cached_para = (
            str(cached.get("llama_paraphrase_text", "")).strip()
            if isinstance(cached, (dict, pd.Series))
            else ""
        )
        if cached_para:
            llama_text = cached_para
        else:
            prompt = _build_llama_paraphrase_prompt(deterministic_text)
            llama_text = ""
            last_err: Exception | None = None
            for attempt in range(1, retry_attempts + 1):
                try:
                    llama_text = _call_openrouter_chat(
                        endpoint=paraphraser_endpoint,
                        api_key=paraphraser_api_key,
                        model_id=paraphraser_model_id,
                        prompt=prompt,
                        temperature=paraphraser_temperature,
                        seed=paraphraser_seed,
                    ).strip()
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    if attempt < retry_attempts:
                        sleep_sec = 2 ** attempt
                        logger.warning(
                            "Paraphraser failed (nodule=%s attempt=%d/%d), retrying in %ds: %s",
                            nodule_id,
                            attempt,
                            retry_attempts,
                            sleep_sec,
                            exc,
                        )
                        time.sleep(sleep_sec)
                    else:
                        logger.error("Paraphraser failed permanently for nodule=%s: %s", nodule_id, exc)
            if not llama_text:
                raise RuntimeError(
                    f"Failed to generate Llama paraphrase for nodule {nodule_id}: {last_err}"
                )
        llama_paraphrase_texts.append(llama_text)

        cache_rows.append(
            {
                "nodule ID": nodule_id,
                "deterministic_text": deterministic_text,
                "llama_paraphrase_text": llama_text,
                "adversarial_text": adversarial_text,
            }
        )
        if (idx + 1) % 25 == 0 or (idx + 1) == len(full_df):
            _upsert_desc_source_text_cache(text_cache_path, cache_rows)

    _upsert_desc_source_text_cache(text_cache_path, cache_rows)

    source_payloads = [
        ("deterministic_template", deterministic_texts),
        ("llama_paraphrase", llama_paraphrase_texts),
        ("adversarial_paraphrase", adversarial_texts),
    ]
    near_leak_payloads = [
        ("baseline_template", deterministic_texts),
        ("remove_spiculation", near_remove_spiculation_texts),
        ("remove_margin", near_remove_margin_texts),
        ("remove_spiculation_margin", near_remove_both_texts),
    ]

    runner = get_runner(model_spec)
    x_train = np.zeros((len(bundle.train_df), 0), dtype=float)
    y_train = bundle.train_df[bundle.label_col].astype(int).to_numpy()
    x_full = np.zeros((len(full_df), 0), dtype=float)
    runner.fit(x_train, y_train)

    plan = FeaturePlan(
        name="description_source_text_only",
        include_text=True,
        include_image_stats=False,
        feature_cols=[],
    )

    metrics_rows: List[Dict[str, Any]] = []
    for source_id, source_texts in source_payloads:
        source_df = full_df.copy()
        source_df["nodule ID"] = source_df["nodule ID"].astype(str)
        source_df[TEXT_COLUMN] = [str(text)[:text_max_chars] for text in source_texts]

        prediction_path = dirs["predictions"] / f"predictions_desc_source_{_safe_name(source_id)}_full.csv"
        pred_df = pd.DataFrame()
        if prediction_path.exists():
            try:
                pred_df = pd.read_csv(prediction_path)
            except Exception:  # noqa: BLE001
                pred_df = pd.DataFrame()

        pred_cols = ["nodule ID", "true_label", "pred_label", "pred_prob", "source_id", "model_id", "split"]
        if pred_df.empty:
            pred_df = pd.DataFrame(columns=pred_cols)
        else:
            keep_cols = [c for c in pred_cols if c in pred_df.columns]
            pred_df = pred_df[keep_cols].copy()
            pred_df["nodule ID"] = pred_df["nodule ID"].astype(str)
            pred_df = pred_df.drop_duplicates(subset=["nodule ID"], keep="last").reset_index(drop=True)

        existing_ids = set(pred_df["nodule ID"].astype(str).tolist()) if "nodule ID" in pred_df.columns else set()
        missing_df = source_df[~source_df["nodule ID"].isin(existing_ids)].copy()
        present_count = int(source_df["nodule ID"].isin(existing_ids).sum())
        logger.info(
            "Running target model for description source=%s (existing=%d missing=%d total=%d)",
            source_id,
            present_count,
            len(missing_df),
            len(source_df),
        )

        if not missing_df.empty:
            buffer_rows: List[Dict[str, Any]] = []
            x_single = np.zeros((1, 0), dtype=float)
            with tqdm(
                total=len(source_df),
                initial=present_count,
                desc=f"{source_id} predictions",
                unit="nodule",
            ) as pbar:
                for _, row in missing_df.iterrows():
                    prompt = build_row_prompt(row, plan, [], text_max_chars=text_max_chars)
                    pred = runner.predict_batch([prompt], x_single)
                    buffer_rows.append(
                        {
                            "nodule ID": str(row["nodule ID"]),
                            "true_label": int(row[bundle.label_col]),
                            "pred_label": int(pred["labels"][0]),
                            "pred_prob": float(pred["probs"][0]),
                            "source_id": source_id,
                            "model_id": target_model_id,
                            "split": "full",
                        }
                    )
                    pbar.update(1)

                    if len(buffer_rows) >= prediction_flush_every:
                        pred_df = pd.concat([pred_df, pd.DataFrame(buffer_rows)], ignore_index=True)
                        pred_df = pred_df.drop_duplicates(subset=["nodule ID"], keep="last")
                        pred_df.to_csv(prediction_path, index=False)
                        buffer_rows = []

                if buffer_rows:
                    pred_df = pd.concat([pred_df, pd.DataFrame(buffer_rows)], ignore_index=True)
                    pred_df = pred_df.drop_duplicates(subset=["nodule ID"], keep="last")
                    pred_df.to_csv(prediction_path, index=False)

        pred_df["nodule ID"] = pred_df["nodule ID"].astype(str)
        ordered_pred_df = source_df[["nodule ID"]].merge(
            pred_df[["nodule ID", "pred_label", "pred_prob"]],
            on="nodule ID",
            how="left",
        )
        if ordered_pred_df["pred_label"].isna().any() or ordered_pred_df["pred_prob"].isna().any():
            missing_count = int(ordered_pred_df["pred_prob"].isna().sum())
            raise RuntimeError(
                f"Incomplete predictions for source={source_id}; missing nodule count={missing_count}"
            )
        pred_df = pd.DataFrame(
            {
                "nodule ID": source_df["nodule ID"].values,
                "true_label": y_full,
                "pred_label": ordered_pred_df["pred_label"].astype(int).to_numpy(),
                "pred_prob": ordered_pred_df["pred_prob"].astype(float).to_numpy(),
                "source_id": source_id,
                "model_id": target_model_id,
                "split": "full",
            }
        )
        pred_df.to_csv(prediction_path, index=False)

        eval_result = evaluate_predictions(
            y_true=pred_df["true_label"].astype(int).to_numpy(),
            y_pred=pred_df["pred_label"].astype(int).to_numpy(),
            y_prob=pred_df["pred_prob"].astype(float).to_numpy(),
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_alpha=bootstrap_alpha,
            seed=int(config["project"]["seed"]),
        )
        row: Dict[str, Any] = {
            "model_id": target_model_id,
            "source_id": source_id,
            "split": "full",
            "n_samples": int(len(source_df)),
        }
        row.update(eval_result.metrics)
        for metric_name, (low, high) in eval_result.ci.items():
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    metrics_df.to_csv(dirs["tables"] / "description_source_metrics.csv", index=False)

    auc_table = metrics_df[
        [
            "source_id",
            "roc_auc",
            "roc_auc_ci_low",
            "roc_auc_ci_high",
            "average_precision",
            "average_precision_ci_low",
            "average_precision_ci_high",
        ]
    ].copy()
    auc_table = auc_table.rename(
        columns={
            "source_id": "description_source",
            "roc_auc": "auc",
            "roc_auc_ci_low": "auc_ci_low",
            "roc_auc_ci_high": "auc_ci_high",
        }
    )
    auc_table.to_csv(dirs["tables"] / "description_source_auc_table.csv", index=False)

    if near_enabled:
        near_rows: List[Dict[str, Any]] = []
        for source_id, source_texts in near_leak_payloads:
            source_df = full_df.copy()
            source_df["nodule ID"] = source_df["nodule ID"].astype(str)
            source_df[TEXT_COLUMN] = [str(text)[:text_max_chars] for text in source_texts]

            prediction_path = dirs["predictions"] / f"predictions_near_leak_{_safe_name(source_id)}_full.csv"
            pred_df = pd.DataFrame()
            if prediction_path.exists():
                try:
                    pred_df = pd.read_csv(prediction_path)
                except Exception:  # noqa: BLE001
                    pred_df = pd.DataFrame()

            pred_cols = ["nodule ID", "true_label", "pred_label", "pred_prob", "source_id", "model_id", "split"]
            if pred_df.empty:
                pred_df = pd.DataFrame(columns=pred_cols)
            else:
                keep_cols = [c for c in pred_cols if c in pred_df.columns]
                pred_df = pred_df[keep_cols].copy()
                pred_df["nodule ID"] = pred_df["nodule ID"].astype(str)
                pred_df = pred_df.drop_duplicates(subset=["nodule ID"], keep="last").reset_index(drop=True)

            existing_ids = set(pred_df["nodule ID"].astype(str).tolist()) if "nodule ID" in pred_df.columns else set()
            missing_df = source_df[~source_df["nodule ID"].isin(existing_ids)].copy()
            present_count = int(source_df["nodule ID"].isin(existing_ids).sum())
            logger.info(
                "Running near-leak source=%s (existing=%d missing=%d total=%d)",
                source_id,
                present_count,
                len(missing_df),
                len(source_df),
            )

            if not missing_df.empty:
                buffer_rows: List[Dict[str, Any]] = []
                x_single = np.zeros((1, 0), dtype=float)
                with tqdm(
                    total=len(source_df),
                    initial=present_count,
                    desc=f"near_leak:{source_id}",
                    unit="nodule",
                ) as pbar:
                    for _, row in missing_df.iterrows():
                        prompt = build_row_prompt(row, plan, [], text_max_chars=text_max_chars)
                        pred = runner.predict_batch([prompt], x_single)
                        buffer_rows.append(
                            {
                                "nodule ID": str(row["nodule ID"]),
                                "true_label": int(row[bundle.label_col]),
                                "pred_label": int(pred["labels"][0]),
                                "pred_prob": float(pred["probs"][0]),
                                "source_id": source_id,
                                "model_id": target_model_id,
                                "split": "full",
                            }
                        )
                        pbar.update(1)
                        if len(buffer_rows) >= prediction_flush_every:
                            pred_df = pd.concat([pred_df, pd.DataFrame(buffer_rows)], ignore_index=True)
                            pred_df = pred_df.drop_duplicates(subset=["nodule ID"], keep="last")
                            pred_df.to_csv(prediction_path, index=False)
                            buffer_rows = []
                    if buffer_rows:
                        pred_df = pd.concat([pred_df, pd.DataFrame(buffer_rows)], ignore_index=True)
                        pred_df = pred_df.drop_duplicates(subset=["nodule ID"], keep="last")
                        pred_df.to_csv(prediction_path, index=False)

            pred_df["nodule ID"] = pred_df["nodule ID"].astype(str)
            ordered_pred_df = source_df[["nodule ID"]].merge(
                pred_df[["nodule ID", "pred_label", "pred_prob"]],
                on="nodule ID",
                how="left",
            )
            if ordered_pred_df["pred_label"].isna().any() or ordered_pred_df["pred_prob"].isna().any():
                missing_count = int(ordered_pred_df["pred_prob"].isna().sum())
                raise RuntimeError(
                    f"Incomplete near-leak predictions for source={source_id}; missing nodule count={missing_count}"
                )
            pred_df = pd.DataFrame(
                {
                    "nodule ID": source_df["nodule ID"].values,
                    "true_label": y_full,
                    "pred_label": ordered_pred_df["pred_label"].astype(int).to_numpy(),
                    "pred_prob": ordered_pred_df["pred_prob"].astype(float).to_numpy(),
                    "source_id": source_id,
                    "model_id": target_model_id,
                    "split": "full",
                }
            )
            pred_df.to_csv(prediction_path, index=False)

            eval_result = evaluate_predictions(
                y_true=pred_df["true_label"].astype(int).to_numpy(),
                y_pred=pred_df["pred_label"].astype(int).to_numpy(),
                y_prob=pred_df["pred_prob"].astype(float).to_numpy(),
                bootstrap_rounds=bootstrap_rounds,
                bootstrap_alpha=bootstrap_alpha,
                seed=int(config["project"]["seed"]),
            )
            row: Dict[str, Any] = {
                "model_id": target_model_id,
                "source_id": source_id,
                "split": "full",
                "n_samples": int(len(source_df)),
            }
            row.update(eval_result.metrics)
            for metric_name, (low, high) in eval_result.ci.items():
                row[f"{metric_name}_ci_low"] = low
                row[f"{metric_name}_ci_high"] = high
            near_rows.append(row)

        near_order = {name: idx for idx, (name, _) in enumerate(near_leak_payloads)}
        near_leak_metrics_df = pd.DataFrame(near_rows)
        near_leak_metrics_df["source_order"] = near_leak_metrics_df["source_id"].map(near_order).fillna(999).astype(int)
        near_leak_metrics_df = near_leak_metrics_df.sort_values("source_order").drop(columns=["source_order"]).reset_index(drop=True)
        near_leak_metrics_df.to_csv(dirs["tables"] / "near_leak_attribute_metrics.csv", index=False)
        baseline_auc = float(
            near_leak_metrics_df.loc[near_leak_metrics_df["source_id"] == "baseline_template", "roc_auc"].iloc[0]
        )
        near_leak_auc_table = near_leak_metrics_df[
            ["source_id", "roc_auc", "roc_auc_ci_low", "roc_auc_ci_high", "average_precision", "average_precision_ci_low", "average_precision_ci_high"]
        ].copy()
        near_leak_auc_table["delta_auc_vs_baseline"] = near_leak_auc_table["roc_auc"].astype(float) - baseline_auc
        near_leak_auc_table = near_leak_auc_table.rename(
            columns={
                "source_id": "removed_attribute_setting",
                "roc_auc": "auc",
                "roc_auc_ci_low": "auc_ci_low",
                "roc_auc_ci_high": "auc_ci_high",
            }
        )
        near_leak_auc_table.to_csv(dirs["tables"] / "near_leak_attribute_auc_drops.csv", index=False)

    manifest = {
        "target_model_id": target_model_id,
        "paraphraser_model_id": paraphraser_model_id,
        "paraphraser_seed": paraphraser_seed,
        "paraphraser_temperature": paraphraser_temperature,
        "bootstrap_rounds": bootstrap_rounds,
        "bootstrap_alpha": bootstrap_alpha,
        "n_rows_full": int(len(full_df)),
        "sources": [x[0] for x in source_payloads],
        "near_leak_ablation_enabled": near_enabled,
        "near_leak_sources": [x[0] for x in near_leak_payloads] if near_enabled else [],
        "outputs": {k: str(v) for k, v in dirs.items()},
    }
    with (dirs["manifests"] / "description_source_ablation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Description-source ablation completed successfully.")
    return manifest
