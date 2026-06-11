from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import ensure_output_dirs
from .data import DataBundle, load_dataset
from .evaluation import evaluate_predictions
from .features import TEXT_COLUMN
from .logging_utils import setup_logger
from .models import get_runner, model_specs_from_config
from .plotting import plot_prompt_temperature_auc_heatmap


TEMPLATE_NAMES: Dict[str, str] = {
    "T1": "zero_shot_direct",
    "T2": "thoracic_radiologist_persona",
    "T3": "chain_of_thought",
    "T4": "three_shot_dcalib",
    "T5": "fleischner_2017_preamble",
}


def _prompt_output_contract() -> str:
    return (
        "Return strict JSON only with keys: label, probability. "
        "label must be 'Y' for malignant or 'N' for benign. "
        "probability must be a float in [0, 1] for malignant probability."
    )


def _build_template_prompt(
    template_id: str,
    description_text: str,
    few_shot_examples: List[Dict[str, Any]],
) -> str:
    text_value = (description_text or "").strip() or "<empty>"
    base_suffix = (
        f"Nodule description:\n{text_value}\n\n"
        "Question: Is this nodule benign or malignant?\n"
        f"{_prompt_output_contract()}"
    )

    if template_id == "T1":
        return (
            "You are evaluating a pulmonary nodule.\n"
            "Give a direct binary malignancy decision.\n\n"
            f"{base_suffix}"
        )

    if template_id == "T2":
        return (
            "You are a board-certified thoracic radiologist with expertise in lung nodule risk assessment.\n"
            "Use a clinically grounded interpretation of the nodule description.\n\n"
            f"{base_suffix}"
        )

    if template_id == "T3":
        return (
            "Reason through the malignancy evidence step by step internally.\n"
            "Do not reveal chain-of-thought. Output only the final JSON answer.\n\n"
            f"{base_suffix}"
        )

    if template_id == "T4":
        example_lines = ["Calibration examples (Dcalib):"]
        for idx, ex in enumerate(few_shot_examples, start=1):
            label = "Y" if int(ex["label"]) == 1 else "N"
            prob = 0.90 if label == "Y" else 0.10
            example_lines.append(f"Example {idx}:")
            example_lines.append(f"Description: {ex['text']}")
            example_lines.append(f"Answer: {{\"label\": \"{label}\", \"probability\": {prob:.2f}}}")
        return (
            "Use these class-balanced in-context examples from Dcalib before answering.\n"
            + "\n".join(example_lines)
            + "\n\n"
            + base_suffix
        )

    if template_id == "T5":
        return (
            "Fleischner Society 2017 style preamble:\n"
            "- Consider suspicious morphology and contextual risk from the reported descriptors.\n"
            "- Prioritize features suggestive of malignancy such as irregular or spiculated patterns.\n"
            "- Use conservative probabilistic judgment when descriptors are mixed.\n\n"
            f"{base_suffix}"
        )

    raise ValueError(f"Unsupported template_id: {template_id}")


def _temperature_token(value: float) -> str:
    return f"{float(value):.1f}".replace(".", "p")


def _ensure_prompt_ablation_dirs(output_root: Path) -> Dict[str, Path]:
    root = output_root / "prompt_ablation"
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


def _split_df_from_bundle(bundle: DataBundle, split_name: str) -> pd.DataFrame:
    name = split_name.strip().lower()
    if name == "train":
        return bundle.train_df
    if name == "val":
        return bundle.val_df
    if name == "test":
        return bundle.test_df
    if name == "full":
        return pd.concat([bundle.train_df, bundle.val_df, bundle.test_df], ignore_index=True)
    raise ValueError(f"Unsupported split name for Dcalib selection: {split_name}")


def _select_few_shot_examples(
    dcalib_df: pd.DataFrame,
    *,
    label_col: str,
    text_col: str,
    n_shots: int,
) -> List[Dict[str, Any]]:
    if n_shots <= 0:
        return []
    if text_col not in dcalib_df.columns:
        raise KeyError(f"Text column '{text_col}' not found for few-shot example selection.")
    if label_col not in dcalib_df.columns:
        raise KeyError(f"Label column '{label_col}' not found for few-shot example selection.")

    work_df = dcalib_df.copy()
    work_df[text_col] = work_df[text_col].fillna("").astype(str).str.strip()
    work_df = work_df[work_df[text_col] != ""].copy()
    if "nodule ID" in work_df.columns:
        work_df = work_df.sort_values("nodule ID")

    by_class: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
    for _, row in work_df.iterrows():
        lbl = int(row[label_col])
        if lbl in by_class:
            by_class[lbl].append({"text": row[text_col], "label": lbl, "nodule ID": row.get("nodule ID", np.nan)})

    examples: List[Dict[str, Any]] = []
    class_cycle = [0, 1]
    class_indices = {0: 0, 1: 0}
    ptr = 0
    while len(examples) < n_shots:
        target_class = class_cycle[ptr % len(class_cycle)]
        ptr += 1
        idx = class_indices[target_class]
        if idx < len(by_class[target_class]):
            examples.append(by_class[target_class][idx])
            class_indices[target_class] += 1
            continue
        other = 1 - target_class
        idx_other = class_indices[other]
        if idx_other < len(by_class[other]):
            examples.append(by_class[other][idx_other])
            class_indices[other] += 1
            continue
        break
    return examples


def _load_results_map(path: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return {}
    required = {"template_id", "temperature", "replicate"}
    if not required.issubset(set(df.columns)):
        return {}
    result_map: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = (str(row["template_id"]), f"{float(row['temperature']):.3f}", int(row["replicate"]))
        result_map[key] = row.to_dict()
    return result_map


def _results_map_to_df(results_map: Dict[Tuple[str, str, int], Dict[str, Any]]) -> pd.DataFrame:
    if not results_map:
        return pd.DataFrame()
    rows = list(results_map.values())
    out = pd.DataFrame(rows)
    if {"template_id", "temperature", "replicate"}.issubset(set(out.columns)):
        out = out.sort_values(["template_id", "temperature", "replicate"]).reset_index(drop=True)
    return out


def _is_reusable_prompt_result(
    row: Dict[str, Any] | None,
    *,
    model_id: str,
    expected_n_samples: int,
    expected_bootstrap_rounds: int,
) -> bool:
    if not isinstance(row, dict) or not row:
        return False
    if str(row.get("model_id", "")) != str(model_id):
        return False
    if str(row.get("split", "")) != "full":
        return False
    try:
        n_samples = int(row.get("n_samples", -1))
    except Exception:  # noqa: BLE001
        return False
    if n_samples != int(expected_n_samples):
        return False
    if "bootstrap_rounds" in row and not pd.isna(row.get("bootstrap_rounds")):
        try:
            rounds = int(row.get("bootstrap_rounds"))
            if rounds != int(expected_bootstrap_rounds):
                return False
        except Exception:  # noqa: BLE001
            return False
    metric_keys = ("accuracy", "f1", "roc_auc", "average_precision")
    for key in metric_keys:
        value = row.get(key)
        if value is None or pd.isna(value):
            return False
    return True


def run_prompt_ablation(config: Dict[str, Any]) -> Dict[str, Any]:
    core_dirs = ensure_output_dirs(config)
    dirs = _ensure_prompt_ablation_dirs(core_dirs["root"])
    logger = setup_logger(dirs["logs"] / "prompt_ablation.log")

    prompt_cfg = config.get("prompt_ablation", {})
    model_id = str(prompt_cfg.get("model_id", "google/gemini-3.1-flash-lite-preview"))
    template_ids = [str(x) for x in prompt_cfg.get("templates", ["T1", "T2", "T3", "T4", "T5"])]
    temperatures = [float(x) for x in prompt_cfg.get("temperatures", [0.0, 0.3, 0.5, 0.7, 1.0])]
    replicates = int(prompt_cfg.get("replicates", 5))
    dcalib_split = str(prompt_cfg.get("dcalib_split", "val"))
    n_shots = int(prompt_cfg.get("n_shots", 3))
    text_max_chars = int(config.get("preprocessing", {}).get("text_max_chars", 1600))
    eval_cfg = config.get("evaluation", {})
    bootstrap_rounds = int(prompt_cfg.get("bootstrap_rounds", eval_cfg.get("bootstrap_rounds", 300)))
    bootstrap_alpha = float(prompt_cfg.get("bootstrap_alpha", eval_cfg.get("bootstrap_alpha", 0.05)))
    seed = int(config["project"]["seed"])

    for template_id in template_ids:
        if template_id not in TEMPLATE_NAMES:
            raise ValueError(f"Unsupported template ID '{template_id}'. Supported: {sorted(TEMPLATE_NAMES)}")
    if replicates < 1:
        raise ValueError("prompt_ablation.replicates must be >= 1")

    logger.info("Loading dataset for prompt ablation")
    bundle = load_dataset(config)
    full_df = pd.concat([bundle.train_df, bundle.val_df, bundle.test_df], ignore_index=True)
    y_full = full_df[bundle.label_col].astype(int).to_numpy()
    if len(full_df) != 955:
        logger.warning(
            "Prompt ablation expected 955 filtered nodules for consistency, found %d",
            len(full_df),
        )
    else:
        logger.info("Prompt ablation running on full filtered cohort of 955 nodules")

    dcalib_df = _split_df_from_bundle(bundle, dcalib_split)
    few_shot_examples = _select_few_shot_examples(
        dcalib_df,
        label_col=bundle.label_col,
        text_col=TEXT_COLUMN,
        n_shots=n_shots,
    )
    if len(few_shot_examples) < n_shots:
        logger.warning(
            "Requested %d few-shot examples, only %d available from Dcalib split '%s'",
            n_shots,
            len(few_shot_examples),
            dcalib_split,
        )
    pd.DataFrame(few_shot_examples).to_csv(dirs["tables"] / "dcalib_few_shot_examples.csv", index=False)

    model_spec = None
    for spec in model_specs_from_config(config):
        if spec.model_id == model_id:
            model_spec = spec
            break
    if model_spec is None:
        raise ValueError(f"Model '{model_id}' not found in config.models")
    if model_spec.api_key_env and not os.getenv(model_spec.api_key_env):
        raise RuntimeError(
            f"Missing API key env var '{model_spec.api_key_env}' for prompt ablation model {model_id}"
        )

    prompts_by_template: Dict[str, List[str]] = {}
    for template_id in template_ids:
        prompts: List[str] = []
        for _, row in full_df.iterrows():
            text = str(row.get(TEXT_COLUMN, ""))[:text_max_chars]
            prompts.append(_build_template_prompt(template_id, text, few_shot_examples))
        prompts_by_template[template_id] = prompts

    x_train = np.zeros((len(bundle.train_df), 0), dtype=float)
    x_full = np.zeros((len(full_df), 0), dtype=float)
    y_train = bundle.train_df[bundle.label_col].astype(int).to_numpy()

    results_path = dirs["tables"] / "prompt_template_temperature_results.csv"
    result_map = _load_results_map(results_path)
    work_items: List[Tuple[str, float, int]] = [
        (template_id, float(temp), rep)
        for template_id in template_ids
        for temp in temperatures
        for rep in range(1, replicates + 1)
    ]
    total_runs = len(work_items)
    completed_runs = 0
    for template_id, temp, rep in work_items:
        key = (template_id, f"{float(temp):.3f}", rep)
        if _is_reusable_prompt_result(
            result_map.get(key),
            model_id=model_id,
            expected_n_samples=len(full_df),
            expected_bootstrap_rounds=bootstrap_rounds,
        ):
            completed_runs += 1
    logger.info("Prompt ablation total runs=%d (templates x temperatures x replicates)", total_runs)
    logger.info("Prompt ablation resume state: reusable_completed=%d pending=%d", completed_runs, total_runs - completed_runs)

    with tqdm(total=total_runs, desc="Prompt ablation runs", unit="run") as pbar:
        for template_id in template_ids:
            prompts = prompts_by_template[template_id]
            for temp in temperatures:
                temp_token = _temperature_token(temp)
                runner = None
                for rep in range(1, replicates + 1):
                    key = (template_id, f"{float(temp):.3f}", rep)
                    if _is_reusable_prompt_result(
                        result_map.get(key),
                        model_id=model_id,
                        expected_n_samples=len(full_df),
                        expected_bootstrap_rounds=bootstrap_rounds,
                    ):
                        logger.info(
                            "Skipping completed prompt ablation run model=%s template=%s temp=%.1f replicate=%d",
                            model_id,
                            template_id,
                            temp,
                            rep,
                        )
                        pbar.set_postfix(
                            {
                                "template": template_id,
                                "temp": f"{temp:.1f}",
                                "rep": rep,
                                "status": "resume-skip",
                            }
                        )
                        pbar.update(1)
                        continue

                    if runner is None:
                        runner = get_runner(replace(model_spec, temperature=float(temp)))
                        runner.fit(x_train, y_train)
                    pred_path = (
                        dirs["predictions"]
                        / f"prompt_{template_id}_temp_{temp_token}_rep_{rep}_full.csv"
                    )
                    logger.info(
                        "Prompt ablation model=%s template=%s temp=%.1f replicate=%d",
                        model_id,
                        template_id,
                        temp,
                        rep,
                    )

                    pred_df = pd.DataFrame()
                    if pred_path.exists():
                        try:
                            pred_df = pd.read_csv(pred_path)
                        except Exception:  # noqa: BLE001
                            pred_df = pd.DataFrame()

                    if pred_df.empty or len(pred_df) != len(full_df):
                        pred = runner.predict_batch(prompts, x_full)
                        pred_df = pd.DataFrame(
                            {
                                "nodule ID": full_df["nodule ID"].values,
                                "true_label": y_full,
                                "pred_label": pred["labels"],
                                "pred_prob": pred["probs"],
                                "template_id": template_id,
                                "temperature": float(temp),
                                "replicate": rep,
                                "model_id": model_id,
                            }
                        )
                        pred_df.to_csv(pred_path, index=False)

                    eval_result = evaluate_predictions(
                        y_true=pred_df["true_label"].astype(int).to_numpy(),
                        y_pred=pred_df["pred_label"].astype(int).to_numpy(),
                        y_prob=pred_df["pred_prob"].astype(float).to_numpy(),
                        bootstrap_rounds=bootstrap_rounds,
                        bootstrap_alpha=bootstrap_alpha,
                        seed=seed + rep,
                    )
                    row: Dict[str, Any] = {
                        "model_id": model_id,
                        "template_id": template_id,
                        "template_name": TEMPLATE_NAMES[template_id],
                        "temperature": float(temp),
                        "replicate": rep,
                        "split": "full",
                        "n_samples": int(len(full_df)),
                        "bootstrap_rounds": bootstrap_rounds,
                    }
                    row.update(eval_result.metrics)
                    for metric_name, (low, high) in eval_result.ci.items():
                        row[f"{metric_name}_ci_low"] = low
                        row[f"{metric_name}_ci_high"] = high
                    row.update(eval_result.confusion)
                    result_map[key] = row
                    _results_map_to_df(result_map).to_csv(results_path, index=False)

                    pbar.set_postfix(
                        {
                            "template": template_id,
                            "temp": f"{temp:.1f}",
                            "rep": rep,
                        }
                    )
                    pbar.update(1)

    results_df = _results_map_to_df(result_map)
    summary_df = (
        results_df.groupby(["template_id", "template_name", "temperature"], as_index=False)
        .agg(
            auc_mean=("roc_auc", "mean"),
            auc_std=("roc_auc", "std"),
            auc_min=("roc_auc", "min"),
            auc_max=("roc_auc", "max"),
            replicate_count=("roc_auc", "count"),
        )
        .sort_values(["template_id", "temperature"])
        .reset_index(drop=True)
    )
    summary_df["auc_std"] = summary_df["auc_std"].fillna(0.0)
    summary_path = dirs["tables"] / "prompt_template_temperature_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    plot_prompt_temperature_auc_heatmap(
        summary_df,
        dirs["figures"] / "prompt_template_temperature_auc_heatmap.png",
        template_col="template_id",
        temperature_col="temperature",
        mean_auc_col="auc_mean",
        std_auc_col="auc_std",
        style=str(config.get("plotting", {}).get("style", "whitegrid")),
        dpi=int(config.get("plotting", {}).get("dpi", 180)),
    )

    # Self-consistency upper-bound style baseline.
    self_cfg = prompt_cfg.get("self_consistency", {})
    if bool(self_cfg.get("enabled", True)):
        sc_template = str(self_cfg.get("template_id", "T3"))
        if sc_template not in TEMPLATE_NAMES:
            raise ValueError(f"Unsupported self_consistency.template_id '{sc_template}'")
        sc_temp = float(self_cfg.get("temperature", 0.7))
        sc_n = int(self_cfg.get("n_samples", 10))
        sc_pred_path = (
            dirs["predictions"]
            / f"self_consistency_{sc_template}_temp_{_temperature_token(sc_temp)}_n_{sc_n}_full.csv"
        )
        if sc_template not in prompts_by_template:
            prompts_sc: List[str] = []
            for _, row in full_df.iterrows():
                text = str(row.get(TEXT_COLUMN, ""))[:text_max_chars]
                prompts_sc.append(_build_template_prompt(sc_template, text, few_shot_examples))
            prompts_by_template[sc_template] = prompts_sc
        prompts_sc = prompts_by_template[sc_template]

        sc_pred_df = pd.DataFrame()
        if sc_pred_path.exists():
            try:
                sc_pred_df = pd.read_csv(sc_pred_path)
            except Exception:  # noqa: BLE001
                sc_pred_df = pd.DataFrame()

        sample_metrics_rows: List[Dict[str, Any]] = []
        if sc_pred_df.empty or len(sc_pred_df) != len(full_df):
            logger.info(
                "Running self-consistency baseline template=%s temperature=%.1f n=%d",
                sc_template,
                sc_temp,
                sc_n,
            )
            runner_sc = get_runner(replace(model_spec, temperature=sc_temp))
            runner_sc.fit(x_train, y_train)

            sample_probs: List[np.ndarray] = []
            with tqdm(total=sc_n, desc="Self-consistency samples", unit="sample") as sc_bar:
                for sample_idx in range(1, sc_n + 1):
                    sample_pred = runner_sc.predict_batch(prompts_sc, x_full)
                    sample_probs.append(sample_pred["probs"])
                    sample_eval = evaluate_predictions(
                        y_true=y_full,
                        y_pred=sample_pred["labels"],
                        y_prob=sample_pred["probs"],
                        bootstrap_rounds=0,
                        bootstrap_alpha=bootstrap_alpha,
                        seed=seed + sample_idx,
                    )
                    sample_metrics_rows.append(
                        {
                            "sample_idx": sample_idx,
                            "roc_auc": sample_eval.metrics["roc_auc"],
                            "accuracy": sample_eval.metrics["accuracy"],
                            "f1": sample_eval.metrics["f1"],
                        }
                    )
                    sc_bar.update(1)
                    sc_bar.set_postfix({"sample": sample_idx})

            stacked_probs = np.vstack(sample_probs)
            ensemble_prob = np.mean(stacked_probs, axis=0)
            ensemble_label = (ensemble_prob >= 0.5).astype(int)
            sc_pred_df = pd.DataFrame(
                {
                    "nodule ID": full_df["nodule ID"].values,
                    "true_label": y_full,
                    "pred_label": ensemble_label,
                    "pred_prob": ensemble_prob,
                    "template_id": sc_template,
                    "temperature": sc_temp,
                    "n_samples_ensemble": sc_n,
                    "model_id": model_id,
                }
            )
            sc_pred_df.to_csv(sc_pred_path, index=False)
            if sample_metrics_rows:
                pd.DataFrame(sample_metrics_rows).to_csv(
                    dirs["tables"] / "self_consistency_sample_metrics.csv",
                    index=False,
                )

        sc_eval = evaluate_predictions(
            y_true=sc_pred_df["true_label"].astype(int).to_numpy(),
            y_pred=sc_pred_df["pred_label"].astype(int).to_numpy(),
            y_prob=sc_pred_df["pred_prob"].astype(float).to_numpy(),
            bootstrap_rounds=bootstrap_rounds,
            bootstrap_alpha=bootstrap_alpha,
            seed=seed,
        )
        sc_row: Dict[str, Any] = {
            "model_id": model_id,
            "template_id": sc_template,
            "template_name": TEMPLATE_NAMES[sc_template],
            "temperature": sc_temp,
            "self_consistency_n": sc_n,
            "split": "full",
            "n_samples": int(len(full_df)),
        }
        sc_row.update(sc_eval.metrics)
        for metric_name, (low, high) in sc_eval.ci.items():
            sc_row[f"{metric_name}_ci_low"] = low
            sc_row[f"{metric_name}_ci_high"] = high
        sc_row.update(sc_eval.confusion)
        pd.DataFrame([sc_row]).to_csv(dirs["tables"] / "self_consistency_results.csv", index=False)

    manifest = {
        "model_id": model_id,
        "template_ids": template_ids,
        "temperatures": temperatures,
        "replicates": replicates,
        "dataset_rows_full": int(len(full_df)),
        "bootstrap_rounds": bootstrap_rounds,
        "bootstrap_alpha": bootstrap_alpha,
        "dcalib_split": dcalib_split,
        "few_shot_n": n_shots,
        "self_consistency": self_cfg,
        "outputs": {k: str(v) for k, v in dirs.items()},
    }
    with (dirs["manifests"] / "prompt_ablation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Prompt ablation completed successfully.")
    return manifest


def run_prompt_ablation_plots_only(config: Dict[str, Any]) -> Dict[str, Any]:
    core_dirs = ensure_output_dirs(config)
    dirs = _ensure_prompt_ablation_dirs(core_dirs["root"])
    logger = setup_logger(dirs["logs"] / "prompt_ablation.log")

    summary_path = dirs["tables"] / "prompt_template_temperature_summary.csv"
    if not summary_path.exists():
        logger.warning(
            "Plots-only mode: prompt ablation summary not found at %s; skipping prompt heatmap",
            summary_path,
        )
        return {
            "task": "plots-only",
            "prompt_ablation_heatmap_rendered": False,
            "summary_path": str(summary_path),
            "figure_path": str(dirs["figures"] / "prompt_template_temperature_auc_heatmap.png"),
        }

    try:
        summary_df = pd.read_csv(summary_path)
    except Exception:  # noqa: BLE001
        logger.warning("Plots-only mode: failed reading prompt summary CSV at %s", summary_path)
        return {
            "task": "plots-only",
            "prompt_ablation_heatmap_rendered": False,
            "summary_path": str(summary_path),
            "figure_path": str(dirs["figures"] / "prompt_template_temperature_auc_heatmap.png"),
        }

    plot_prompt_temperature_auc_heatmap(
        summary_df,
        dirs["figures"] / "prompt_template_temperature_auc_heatmap.png",
        template_col="template_id",
        temperature_col="temperature",
        mean_auc_col="auc_mean",
        std_auc_col="auc_std",
        style=str(config.get("plotting", {}).get("style", "whitegrid")),
        dpi=int(config.get("plotting", {}).get("dpi", 180)),
    )
    manifest = {
        "task": "plots-only",
        "prompt_ablation_heatmap_rendered": True,
        "summary_path": str(summary_path),
        "figure_path": str(dirs["figures"] / "prompt_template_temperature_auc_heatmap.png"),
    }
    with (dirs["manifests"] / "plots_only_prompt_ablation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Plots-only prompt heatmap rendering completed successfully.")
    return manifest
