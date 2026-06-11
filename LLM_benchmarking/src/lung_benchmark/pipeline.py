from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .config import ensure_output_dirs
from .data import load_dataset
from .evaluation import (
    build_pairwise_bootstrap_stat_tests,
    collect_feature_importance,
    evaluate_predictions,
    make_result_row,
)
from .features import (
    TEXT_CORRUPTION_NONE,
    FeaturePlan,
    apply_text_corruption,
    build_feature_plans,
    build_row_prompt,
)
from .logging_utils import setup_logger
from .models import get_runner, model_specs_from_config
from .plotting import (
    build_model_palette,
    plot_family_heatmap,
    plot_feature_importance,
    plot_metric_bars,
    plot_roc_curves,
)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _short_model_name(model_id: str) -> str:
    # Keep short names readable and stable for figure legends.
    tail = str(model_id).split("/")[-1].strip()
    tail = tail.replace("-instruct", "")
    tail = tail.replace("-chat", "")
    return tail or str(model_id)


def _excluded_reporting_policy(config: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str], bool]:
    eval_cfg = config.get("evaluation", {})
    excluded_ids = {str(x) for x in eval_cfg.get("exclude_model_ids_from_reporting", [])}
    excluded_families = {str(x) for x in eval_cfg.get("exclude_model_families_from_reporting", [])}
    excluded_name_substrings = {
        str(x).strip().lower()
        for x in eval_cfg.get("exclude_model_name_substrings_from_reporting", [])
        if str(x).strip()
    }
    skip_excluded = bool(eval_cfg.get("skip_excluded_from_execution", True))
    return excluded_ids, excluded_families, excluded_name_substrings, skip_excluded


def _main_reporting_ablation_policy(config: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    eval_cfg = config.get("evaluation", {})
    excluded_names = {
        str(x).strip()
        for x in eval_cfg.get("exclude_ablation_names_from_main_reporting", [])
        if str(x).strip()
    }
    excluded_prefixes = {
        str(x).strip()
        for x in eval_cfg.get("exclude_ablation_name_prefixes_from_main_reporting", [])
        if str(x).strip()
    }
    return excluded_names, excluded_prefixes


def _is_auxiliary_ablation_for_main_reporting(
    ablation_name: str,
    excluded_names: Set[str],
    excluded_prefixes: Set[str],
) -> bool:
    name = str(ablation_name)
    if name in excluded_names:
        return True
    return any(name.startswith(prefix) for prefix in excluded_prefixes)


def _is_excluded_for_reporting(
    model_id: str,
    family: str,
    excluded_ids: Set[str],
    excluded_families: Set[str],
    excluded_name_substrings: Set[str],
) -> bool:
    model_id_l = str(model_id).lower()
    by_substring = any(substr in model_id_l for substr in excluded_name_substrings)
    return model_id in excluded_ids or family in excluded_families or by_substring


def _matrix_from_df(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=float)
    x = df[cols].astype(float).to_numpy()
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _prompts_from_df(df: pd.DataFrame, plan: FeaturePlan, numeric_cols: List[str], text_max_chars: int) -> List[str]:
    prompts: List[str] = []
    for _, row in df.iterrows():
        prompts.append(build_row_prompt(row, plan, numeric_cols, text_max_chars=text_max_chars))
    return prompts


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_existing_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        # Corrupt/partial CSV should not hard-stop the pipeline.
        return pd.DataFrame()


def _splits_by_combo(results_df: pd.DataFrame) -> Dict[Tuple[str, str], Set[str]]:
    if results_df.empty:
        return {}
    required = {"model_id", "ablation", "split"}
    if not required.issubset(set(results_df.columns)):
        return {}
    grouped = results_df.groupby(["model_id", "ablation"])["split"].apply(set)
    out: Dict[Tuple[str, str], Set[str]] = {}
    for (model_id, ablation), splits in grouped.items():
        out[(str(model_id), str(ablation))] = {str(s) for s in splits}
    return out


def _completed_combo_keys(results_df: pd.DataFrame, required_splits: Set[str]) -> Set[Tuple[str, str]]:
    if results_df.empty:
        return set()
    out: Set[Tuple[str, str]] = set()
    for (model_id, ablation), splits in _splits_by_combo(results_df).items():
        if required_splits.issubset(splits):
            out.add((str(model_id), str(ablation)))
    return out


def _upsert_result_rows(existing_rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    key_fields = ("model_id", "ablation", "split")
    new_keys = {
        tuple(str(r.get(k, "")) for k in key_fields)
        for r in new_rows
    }
    kept = [
        row
        for row in existing_rows
        if tuple(str(row.get(k, "")) for k in key_fields) not in new_keys
    ]
    kept.extend(new_rows)
    return kept


def _filter_rows_for_reporting(
    rows: List[Dict[str, Any]],
    excluded_ids: Set[str],
    excluded_families: Set[str],
    excluded_name_substrings: Set[str],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        model_id = str(row.get("model_id", ""))
        family = str(row.get("family", ""))
        if _is_excluded_for_reporting(
            model_id,
            family,
            excluded_ids,
            excluded_families,
            excluded_name_substrings,
        ):
            continue
        filtered.append(row)
    return filtered


def _prediction_pairs_from_df(pred_df: pd.DataFrame) -> Set[Tuple[str, str]]:
    if pred_df.empty:
        return set()
    required = {"model_id", "ablation"}
    if not required.issubset(set(pred_df.columns)):
        return set()
    pairs = set(
        zip(
            pred_df["model_id"].astype(str).tolist(),
            pred_df["ablation"].astype(str).tolist(),
        )
    )
    return pairs


def _filter_rows_by_prediction_presence(
    rows: List[Dict[str, Any]],
    prediction_pairs: Set[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    if not prediction_pairs:
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("model_id", "")), str(row.get("ablation", "")))
        if key in prediction_pairs:
            out.append(row)
    return out


def _split_results_for_main_reporting(
    rows: List[Dict[str, Any]],
    excluded_ablation_names: Set[str],
    excluded_ablation_prefixes: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    main_rows: List[Dict[str, Any]] = []
    auxiliary_rows: List[Dict[str, Any]] = []
    for row in rows:
        ablation_name = str(row.get("ablation", ""))
        if _is_auxiliary_ablation_for_main_reporting(
            ablation_name,
            excluded_ablation_names,
            excluded_ablation_prefixes,
        ):
            auxiliary_rows.append(row)
        else:
            main_rows.append(row)
    return main_rows, auxiliary_rows


def _split_predictions_for_main_reporting(
    pred_df: pd.DataFrame,
    excluded_ablation_names: Set[str],
    excluded_ablation_prefixes: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pred_df.empty or "ablation" not in pred_df.columns:
        return pred_df, pd.DataFrame(columns=pred_df.columns)
    aux_mask = pred_df["ablation"].astype(str).map(
        lambda name: _is_auxiliary_ablation_for_main_reporting(
            name,
            excluded_ablation_names,
            excluded_ablation_prefixes,
        )
    )
    main_df = pred_df[~aux_mask].copy()
    aux_df = pred_df[aux_mask].copy()
    return main_df, aux_df


def _append_failure(
    failures_path: Path,
    model_id: str,
    ablation: str,
    error: str,
) -> None:
    row = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "ablation": ablation,
                "error": error,
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            }
        ]
    )
    if failures_path.exists():
        row.to_csv(failures_path, mode="a", index=False, header=False)
    else:
        row.to_csv(failures_path, index=False)


def _collect_predictions_for_roc(
    predictions_dir: Path,
    model_specs: List[Any],
    plans: List[FeaturePlan],
    split_name: str = "test",
) -> pd.DataFrame:
    files = sorted(predictions_dir.glob(f"predictions_*_{split_name}.csv"))
    if not files:
        return pd.DataFrame()

    safe_model_to_id = {_safe_name(s.model_id): s.model_id for s in model_specs}
    safe_ablation_to_name = {_safe_name(p.name): p.name for p in plans}
    short_name_map = {s.model_id: _short_model_name(s.model_id) for s in model_specs}

    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or not {"true_label", "pred_prob"}.issubset(df.columns):
            continue

        if "model_id" not in df.columns or "ablation" not in df.columns:
            stem = path.stem
            split_suffix = f"_{split_name}"
            if not stem.startswith("predictions_") or not stem.endswith(split_suffix):
                continue
            middle = stem[len("predictions_") : -len(split_suffix)]
            resolved_ablation = None
            resolved_model = None
            for safe_ablation, ablation_name in sorted(
                safe_ablation_to_name.items(), key=lambda kv: len(kv[0]), reverse=True
            ):
                suffix = f"_{safe_ablation}"
                if middle.endswith(suffix):
                    safe_model = middle[: -len(suffix)]
                    resolved_ablation = ablation_name
                    resolved_model = safe_model_to_id.get(safe_model)
                    break
            if not resolved_model or not resolved_ablation:
                continue
            df["model_id"] = resolved_model
            df["ablation"] = resolved_ablation

        if "model_short_name" not in df.columns:
            df["model_short_name"] = df["model_id"].astype(str).map(short_name_map).fillna(df["model_id"])
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    keep_cols = [
        "model_id",
        "model_short_name",
        "ablation",
        "nodule ID",
        "true_label",
        "pred_label",
        "pred_prob",
    ]
    return combined[[c for c in keep_cols if c in combined.columns]]


def _prediction_path(predictions_dir: Path, model_id: str, ablation: str, split_name: str) -> Path:
    return predictions_dir / f"predictions_{_safe_name(model_id)}_{_safe_name(ablation)}_{split_name}.csv"


def _load_prediction_split(
    predictions_dir: Path,
    model_id: str,
    ablation: str,
    split_name: str,
) -> pd.DataFrame:
    path = _prediction_path(predictions_dir, model_id, ablation, split_name)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    dirs = ensure_output_dirs(config)
    logger = setup_logger(dirs["logs"] / "run.log")

    logger.info("Loading dataset and preparing splits")
    bundle = load_dataset(config)
    plans = build_feature_plans(config)
    model_specs = model_specs_from_config(config)
    short_name_map = {s.model_id: _short_model_name(s.model_id) for s in model_specs}
    seed = int(config["project"]["seed"])
    text_max_chars = int(config.get("preprocessing", {}).get("text_max_chars", 1600))
    (
        excluded_ids,
        excluded_families,
        excluded_name_substrings,
        skip_excluded,
    ) = _excluded_reporting_policy(config)
    (
        excluded_ablation_names_main,
        excluded_ablation_prefixes_main,
    ) = _main_reporting_ablation_policy(config)
    eval_cfg = config.get("evaluation", {})
    bootstrap_rounds = int(eval_cfg.get("bootstrap_rounds", 300))
    bootstrap_alpha = float(eval_cfg.get("bootstrap_alpha", 0.05))
    reporting_split = str(eval_cfg.get("reporting_split", "full")).strip() or "full"
    incremental_full_from_test = bool(eval_cfg.get("incremental_full_from_test", True))
    required_completion_splits_cfg = eval_cfg.get("required_completion_splits")
    if required_completion_splits_cfg:
        required_completion_splits = {str(x).strip() for x in required_completion_splits_cfg if str(x).strip()}
    else:
        required_completion_splits = {"val", "test", "full"}

    results_path = dirs["tables"] / "benchmark_results.csv"
    results_all_path = dirs["tables"] / "benchmark_results_all.csv"
    results_aux_path = dirs["tables"] / "benchmark_results_auxiliary.csv"
    summary_path = dirs["tables"] / "benchmark_summary_ranked.csv"
    summary_aux_path = dirs["tables"] / "benchmark_summary_auxiliary_ranked.csv"
    pairwise_path = dirs["tables"] / "pairwise_bootstrap_pvalues.csv"
    pairwise_aux_path = dirs["tables"] / "pairwise_bootstrap_pvalues_auxiliary.csv"
    failures_path = dirs["tables"] / "benchmark_failures.csv"
    existing_results_df = _load_existing_results(results_all_path)
    if existing_results_df.empty:
        existing_results_df = _load_existing_results(results_path)
    all_rows_raw: List[Dict[str, Any]] = (
        existing_results_df.to_dict(orient="records") if not existing_results_df.empty else []
    )
    all_rows_policy_filtered: List[Dict[str, Any]] = _filter_rows_for_reporting(
        all_rows_raw,
        excluded_ids,
        excluded_families,
        excluded_name_substrings,
    )
    all_rows: List[Dict[str, Any]] = all_rows_policy_filtered
    if len(all_rows_raw) != len(all_rows):
        logger.info("Filtered out %d rows by reporting policy", len(all_rows_raw) - len(all_rows))
    completed_keys = _completed_combo_keys(
        pd.DataFrame(all_rows),
        required_splits=required_completion_splits,
    )
    splits_by_combo = _splits_by_combo(pd.DataFrame(all_rows))

    logger.info("Dataset stats: train=%d val=%d test=%d", len(bundle.train_df), len(bundle.val_df), len(bundle.test_df))
    logger.info("Image summaries available for %d nodules", len(bundle.image_summary_df))
    if completed_keys:
        logger.info(
            "Resume mode: found %d completed model-ablation combinations in %s",
            len(completed_keys),
            results_path,
        )

    for plan in plans:
        logger.info("Running ablation: %s", plan.name)
        numeric_cols = plan.selected_numeric_cols(bundle.train_df)
        x_train = _matrix_from_df(bundle.train_df, numeric_cols)
        x_val = _matrix_from_df(bundle.val_df, numeric_cols)
        x_test = _matrix_from_df(bundle.test_df, numeric_cols)
        full_df = pd.concat([bundle.train_df, bundle.val_df, bundle.test_df], ignore_index=True)
        x_full = _matrix_from_df(full_df, numeric_cols)
        y_train = bundle.train_df[bundle.label_col].astype(int).to_numpy()
        y_val = bundle.val_df[bundle.label_col].astype(int).to_numpy()
        y_test = bundle.test_df[bundle.label_col].astype(int).to_numpy()
        y_full = full_df[bundle.label_col].astype(int).to_numpy()

        prompt_val_df = bundle.val_df
        prompt_test_df = bundle.test_df
        prompt_full_df = full_df
        if plan.include_text and plan.text_corruption != TEXT_CORRUPTION_NONE:
            prompt_val_df = apply_text_corruption(
                bundle.val_df,
                label_col=bundle.label_col,
                corruption=plan.text_corruption,
            )
            prompt_test_df = apply_text_corruption(
                bundle.test_df,
                label_col=bundle.label_col,
                corruption=plan.text_corruption,
            )
            prompt_full_df = apply_text_corruption(
                full_df,
                label_col=bundle.label_col,
                corruption=plan.text_corruption,
            )
            logger.info(
                "Applied synthetic text corruption='%s' for ablation=%s",
                plan.text_corruption,
                plan.name,
            )

        prompts_val = _prompts_from_df(prompt_val_df, plan, numeric_cols, text_max_chars=text_max_chars)
        prompts_test = _prompts_from_df(prompt_test_df, plan, numeric_cols, text_max_chars=text_max_chars)
        prompts_full = _prompts_from_df(prompt_full_df, plan, numeric_cols, text_max_chars=text_max_chars)

        importance_df = collect_feature_importance(bundle.train_df, numeric_cols, bundle.label_col)
        importance_df.to_csv(dirs["tables"] / f"feature_importance_{plan.name}.csv", index=False)

        for spec in model_specs:
            excluded = _is_excluded_for_reporting(
                spec.model_id,
                spec.family,
                excluded_ids,
                excluded_families,
                excluded_name_substrings,
            )
            if excluded and skip_excluded:
                logger.info(
                    "Skipping model=%s family=%s by reporting policy",
                    spec.model_id,
                    spec.family,
                )
                continue
            combo_key = (spec.model_id, plan.name)
            existing_splits = splits_by_combo.get(combo_key, set())
            missing_splits = required_completion_splits - existing_splits
            if not missing_splits:
                logger.info(
                    "Skipping already completed model=%s ablation=%s (resume checkpoint)",
                    spec.model_id,
                    plan.name,
                )
                continue
            logger.info(
                "Evaluating model=%s family=%s ablation=%s missing_splits=%s",
                spec.model_id,
                spec.family,
                plan.name,
                sorted(missing_splits),
            )
            if spec.api_key_env and not os.getenv(spec.api_key_env):
                logger.warning(
                    "Skipping model=%s because env var %s is not set",
                    spec.model_id,
                    spec.api_key_env,
                )
                continue
            runner = get_runner(spec)
            runner.fit(x_train, y_train)

            predictions_by_split: Dict[str, Dict[str, np.ndarray]] = {}
            cached_test_pred_df = pd.DataFrame()
            if incremental_full_from_test and "full" in missing_splits and "test" not in missing_splits:
                cached_test_pred_df = _load_prediction_split(
                    dirs["predictions"],
                    model_id=spec.model_id,
                    ablation=plan.name,
                    split_name="test",
                )
                if not cached_test_pred_df.empty:
                    logger.info(
                        "Using incremental full mode from existing test predictions for model=%s ablation=%s",
                        spec.model_id,
                        plan.name,
                    )
            try:
                if "val" in missing_splits:
                    val_pred = runner.predict_batch(prompts_val, x_val)
                    predictions_by_split["val"] = val_pred
                if "test" in missing_splits:
                    test_pred = runner.predict_batch(prompts_test, x_test)
                    predictions_by_split["test"] = test_pred
                if "full" in missing_splits:
                    use_incremental_full = incremental_full_from_test and "test" not in missing_splits and not cached_test_pred_df.empty
                    if use_incremental_full:
                        id_col = "nodule ID"
                        full_ids = full_df[id_col].astype(str)
                        test_ids = bundle.test_df[id_col].astype(str)
                        rest_mask = ~full_ids.isin(set(test_ids.tolist()))
                        rest_df = full_df.loc[rest_mask].reset_index(drop=True)
                        prompt_full_rest_df = prompt_full_df.loc[rest_mask].reset_index(drop=True)
                        if rest_df.empty:
                            # Degenerate case: everything already covered by test IDs.
                            use_incremental_full = False
                        else:
                            x_full_rest = _matrix_from_df(rest_df, numeric_cols)
                            prompts_full_rest = _prompts_from_df(
                                prompt_full_rest_df,
                                plan,
                                numeric_cols,
                                text_max_chars=text_max_chars,
                            )
                            rest_pred = runner.predict_batch(prompts_full_rest, x_full_rest)
                            rest_pred_df = pd.DataFrame(
                                {
                                    id_col: rest_df[id_col].astype(str).values,
                                    "pred_label": rest_pred["labels"],
                                    "pred_prob": rest_pred["probs"],
                                }
                            )
                            test_pred_df = cached_test_pred_df.copy()
                            required_test_cols = {id_col, "pred_label", "pred_prob"}
                            if not required_test_cols.issubset(set(test_pred_df.columns)):
                                use_incremental_full = False
                            else:
                                test_pred_df[id_col] = test_pred_df[id_col].astype(str)
                                combined_pred_df = pd.concat(
                                    [
                                        test_pred_df[[id_col, "pred_label", "pred_prob"]],
                                        rest_pred_df,
                                    ],
                                    ignore_index=True,
                                ).drop_duplicates(subset=[id_col], keep="last")
                                ordered = full_df[[id_col]].copy()
                                ordered[id_col] = ordered[id_col].astype(str)
                                ordered = ordered.merge(
                                    combined_pred_df,
                                    on=id_col,
                                    how="left",
                                )
                                if ordered["pred_label"].isna().any() or ordered["pred_prob"].isna().any():
                                    use_incremental_full = False
                                else:
                                    predictions_by_split["full"] = {
                                        "labels": ordered["pred_label"].astype(int).to_numpy(),
                                        "probs": ordered["pred_prob"].astype(float).to_numpy(),
                                    }
                    if not use_incremental_full:
                        full_pred = runner.predict_batch(prompts_full, x_full)
                        predictions_by_split["full"] = full_pred
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Model evaluation failed for model=%s ablation=%s; continuing. Error=%s",
                    spec.model_id,
                    plan.name,
                    exc,
                )
                _append_failure(
                    failures_path=failures_path,
                    model_id=spec.model_id,
                    ablation=plan.name,
                    error=str(exc),
                )
                continue

            new_rows: List[Dict[str, Any]] = []
            if "val" in predictions_by_split:
                val_eval = evaluate_predictions(
                    y_true=y_val,
                    y_pred=predictions_by_split["val"]["labels"],
                    y_prob=predictions_by_split["val"]["probs"],
                    bootstrap_rounds=bootstrap_rounds,
                    bootstrap_alpha=bootstrap_alpha,
                    seed=seed,
                )
                new_rows.append(make_result_row(spec.model_id, spec.family, plan.name, "val", val_eval))
            if "test" in predictions_by_split:
                test_eval = evaluate_predictions(
                    y_true=y_test,
                    y_pred=predictions_by_split["test"]["labels"],
                    y_prob=predictions_by_split["test"]["probs"],
                    bootstrap_rounds=bootstrap_rounds,
                    bootstrap_alpha=bootstrap_alpha,
                    seed=seed,
                )
                new_rows.append(make_result_row(spec.model_id, spec.family, plan.name, "test", test_eval))
            if "full" in predictions_by_split:
                full_eval = evaluate_predictions(
                    y_true=y_full,
                    y_pred=predictions_by_split["full"]["labels"],
                    y_prob=predictions_by_split["full"]["probs"],
                    bootstrap_rounds=bootstrap_rounds,
                    bootstrap_alpha=bootstrap_alpha,
                    seed=seed,
                )
                new_rows.append(make_result_row(spec.model_id, spec.family, plan.name, "full", full_eval))
            for row in new_rows:
                row["model_short_name"] = short_name_map.get(spec.model_id, spec.model_id)
            if not excluded:
                all_rows = _upsert_result_rows(all_rows, new_rows)

            if not excluded:
                if "test" in predictions_by_split:
                    pred_df = pd.DataFrame(
                        {
                            "model_id": spec.model_id,
                            "model_short_name": short_name_map.get(spec.model_id, spec.model_id),
                            "ablation": plan.name,
                            "nodule ID": bundle.test_df["nodule ID"].values,
                            "true_label": y_test,
                            "pred_label": predictions_by_split["test"]["labels"],
                            "pred_prob": predictions_by_split["test"]["probs"],
                        }
                    )
                    pred_df.to_csv(
                        dirs["predictions"]
                        / f"predictions_{_safe_name(spec.model_id)}_{_safe_name(plan.name)}_test.csv",
                        index=False,
                    )
                if "full" in predictions_by_split:
                    pred_full_df = pd.DataFrame(
                        {
                            "model_id": spec.model_id,
                            "model_short_name": short_name_map.get(spec.model_id, spec.model_id),
                            "ablation": plan.name,
                            "nodule ID": full_df["nodule ID"].values,
                            "true_label": y_full,
                            "pred_label": predictions_by_split["full"]["labels"],
                            "pred_prob": predictions_by_split["full"]["probs"],
                        }
                    )
                    pred_full_df.to_csv(
                        dirs["predictions"]
                        / f"predictions_{_safe_name(spec.model_id)}_{_safe_name(plan.name)}_full.csv",
                        index=False,
                    )
                pd.DataFrame(all_rows).to_csv(results_all_path, index=False)
            if new_rows:
                updated_splits = splits_by_combo.get(combo_key, set()).union({str(r["split"]) for r in new_rows})
                splits_by_combo[combo_key] = updated_splits
                if required_completion_splits.issubset(updated_splits):
                    completed_keys.add(combo_key)

    current_pred_df = _collect_predictions_for_roc(
        predictions_dir=dirs["predictions"],
        model_specs=model_specs,
        plans=plans,
        split_name=reporting_split,
    )
    current_prediction_pairs = _prediction_pairs_from_df(current_pred_df)
    results_all_df = pd.DataFrame(all_rows)
    if (
        not results_all_df.empty
        and "model_short_name" not in results_all_df.columns
        and "model_id" in results_all_df.columns
    ):
        results_all_df["model_short_name"] = (
            results_all_df["model_id"].astype(str).map(short_name_map).fillna(results_all_df["model_id"])
        )
    reporting_ready_rows = _filter_rows_by_prediction_presence(all_rows, current_prediction_pairs)
    main_rows, aux_rows = _split_results_for_main_reporting(
        reporting_ready_rows,
        excluded_ablation_names_main,
        excluded_ablation_prefixes_main,
    )
    results_df = pd.DataFrame(main_rows)
    results_aux_df = pd.DataFrame(aux_rows)
    for df in (results_df, results_aux_df):
        if not df.empty and "model_short_name" not in df.columns and "model_id" in df.columns:
            df["model_short_name"] = df["model_id"].astype(str).map(short_name_map).fillna(df["model_id"])

    results_all_df.to_csv(results_all_path, index=False)
    results_df.to_csv(results_path, index=False)
    if not results_aux_df.empty:
        results_aux_df.to_csv(results_aux_path, index=False)
    elif results_aux_path.exists():
        results_aux_path.unlink()

    if not results_df.empty:
        summary_df = (
            results_df[results_df["split"] == reporting_split]
            .groupby(["model_id", "family", "ablation"], as_index=False)[
                ["accuracy", "f1", "roc_auc", "average_precision"]
            ]
            .mean()
            .sort_values(["roc_auc", "f1", "accuracy"], ascending=False)
            .reset_index(drop=True)
        )
        summary_df["rank_roc_auc"] = summary_df["roc_auc"].rank(ascending=False, method="dense")
        summary_df.to_csv(summary_path, index=False)
    elif summary_path.exists():
        summary_path.unlink()
    if not results_aux_df.empty:
        summary_aux_df = (
            results_aux_df[results_aux_df["split"] == reporting_split]
            .groupby(["model_id", "family", "ablation"], as_index=False)[
                ["accuracy", "f1", "roc_auc", "average_precision"]
            ]
            .mean()
            .sort_values(["roc_auc", "f1", "accuracy"], ascending=False)
            .reset_index(drop=True)
        )
        summary_aux_df["rank_roc_auc"] = summary_aux_df["roc_auc"].rank(ascending=False, method="dense")
        summary_aux_df.to_csv(summary_aux_path, index=False)
    elif summary_aux_path.exists():
        summary_aux_path.unlink()

    # Pairwise statistical testing across models per ablation using paired bootstrap.
    # P-values are derived from bootstrap distributions of metric differences.
    supported_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    }
    metric_list = [
        str(m) for m in config.get("evaluation", {}).get("metrics", []) if str(m) in supported_metrics
    ]
    if not metric_list:
        metric_list = ["accuracy", "f1", "roc_auc", "average_precision"]
    main_pred_df, aux_pred_df = _split_predictions_for_main_reporting(
        current_pred_df,
        excluded_ablation_names_main,
        excluded_ablation_prefixes_main,
    )
    if not main_pred_df.empty:
        pval_df = build_pairwise_bootstrap_stat_tests(
            predictions_df=main_pred_df,
            metrics=metric_list,
            rounds=bootstrap_rounds,
            alpha=bootstrap_alpha,
            seed=seed,
        )
        pval_df.to_csv(pairwise_path, index=False)
    else:
        pd.DataFrame(
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
        ).to_csv(pairwise_path, index=False)
    if not aux_pred_df.empty:
        pval_aux_df = build_pairwise_bootstrap_stat_tests(
            predictions_df=aux_pred_df,
            metrics=metric_list,
            rounds=bootstrap_rounds,
            alpha=bootstrap_alpha,
            seed=seed,
        )
        pval_aux_df.to_csv(pairwise_aux_path, index=False)
    elif pairwise_aux_path.exists():
        pairwise_aux_path.unlink()

    plot_cfg = config.get("plotting", {})
    style = str(plot_cfg.get("style", "whitegrid"))
    dpi = int(plot_cfg.get("dpi", 160))
    if not results_df.empty and "split" in results_df.columns:
        test_df = results_df[results_df["split"] == reporting_split].copy()
        if "model_short_name" not in test_df.columns and "model_id" in test_df.columns:
            test_df["model_short_name"] = (
                test_df["model_id"].astype(str).map(short_name_map).fillna(test_df["model_id"])
            )
        hue_order = list(dict.fromkeys(test_df["model_short_name"].astype(str).tolist()))
        palette = build_model_palette(hue_order)
        for metric in ["accuracy", "f1", "average_precision"]:
            plot_metric_bars(
                test_df,
                metric=metric,
                out_path=dirs["figures"] / f"{metric}_bars.png",
                model_col="model_short_name",
                hue_order=hue_order,
                palette=palette,
                style=style,
                dpi=dpi,
            )
        for metric in ["accuracy", "f1", "roc_auc", "average_precision"]:
            plot_family_heatmap(
                test_df,
                metric=metric,
                out_path=dirs["figures"] / f"{metric}_family_heatmap.png",
                style=style,
                dpi=dpi,
            )
        roc_pred_df = _collect_predictions_for_roc(
            predictions_dir=dirs["predictions"],
            model_specs=model_specs,
            plans=plans,
            split_name=reporting_split,
        )
        roc_pred_df, _ = _split_predictions_for_main_reporting(
            roc_pred_df,
            excluded_ablation_names_main,
            excluded_ablation_prefixes_main,
        )
        if not roc_pred_df.empty:
            plot_roc_curves(
                roc_pred_df,
                out_path=dirs["figures"] / "roc_auc_curves.png",
                model_col="model_short_name",
                hue_order=hue_order,
                palette=palette,
                style=style,
                dpi=dpi,
            )

    if plans:
        first_cols = plans[0].selected_numeric_cols(bundle.train_df)
        imp_df = collect_feature_importance(bundle.train_df, first_cols, bundle.label_col)
        plot_feature_importance(
            imp_df,
            out_path=dirs["figures"] / "feature_importance.png",
            style=style,
            dpi=dpi,
        )

    manifest = {
        "project": config.get("project", {}),
        "data": {
            "metadata_path": config.get("data", {}).get("metadata_path"),
            "image_dir": config.get("data", {}).get("image_dir"),
            "rows_total_labeled": int(len(bundle.metadata_df)),
            "rows_train": int(len(bundle.train_df)),
            "rows_val": int(len(bundle.val_df)),
            "rows_test": int(len(bundle.test_df)),
        },
        "models": [s.__dict__ for s in model_specs],
        "ablations": [p.__dict__ for p in plans],
        "outputs": {k: str(v) for k, v in dirs.items()},
        "resume": {
            "results_path": str(results_path),
            "results_all_path": str(results_all_path),
            "failures_path": str(failures_path),
            "completed_model_ablation_pairs": len(completed_keys),
            "required_completion_splits": sorted(required_completion_splits),
            "reporting_split": reporting_split,
            "incremental_full_from_test": incremental_full_from_test,
        },
    }
    _write_json(dirs["manifests"] / "run_manifest.json", manifest)
    logger.info("Pipeline completed successfully (result rows=%d)", len(results_df))
    return manifest


def run_pipeline_plots_only(config: Dict[str, Any]) -> Dict[str, Any]:
    dirs = ensure_output_dirs(config)
    logger = setup_logger(dirs["logs"] / "plots_only.log")
    model_specs = model_specs_from_config(config)
    plans = build_feature_plans(config)
    short_name_map = {s.model_id: _short_model_name(s.model_id) for s in model_specs}
    (
        excluded_ids,
        excluded_families,
        excluded_name_substrings,
        _skip_excluded,
    ) = _excluded_reporting_policy(config)
    (
        excluded_ablation_names_main,
        excluded_ablation_prefixes_main,
    ) = _main_reporting_ablation_policy(config)
    plot_cfg = config.get("plotting", {})
    style = str(plot_cfg.get("style", "whitegrid"))
    dpi = int(plot_cfg.get("dpi", 160))
    eval_cfg = config.get("evaluation", {})
    reporting_split = str(eval_cfg.get("reporting_split", "full")).strip() or "full"

    results_path = dirs["tables"] / "benchmark_results.csv"
    results_all_path = dirs["tables"] / "benchmark_results_all.csv"
    results_raw_df = _load_existing_results(results_all_path)
    if results_raw_df.empty:
        results_raw_df = _load_existing_results(results_path)

    rendered_benchmark_plots = 0
    if results_raw_df.empty:
        logger.warning(
            "Plots-only mode: no benchmark results found at %s or %s; skipping benchmark plots",
            results_all_path,
            results_path,
        )
    else:
        all_rows_raw = results_raw_df.to_dict(orient="records")
        all_rows_policy_filtered = _filter_rows_for_reporting(
            all_rows_raw,
            excluded_ids,
            excluded_families,
            excluded_name_substrings,
        )
        current_pred_df = _collect_predictions_for_roc(
            predictions_dir=dirs["predictions"],
            model_specs=model_specs,
            plans=plans,
            split_name=reporting_split,
        )
        current_prediction_pairs = _prediction_pairs_from_df(current_pred_df)
        reporting_ready_rows = _filter_rows_by_prediction_presence(all_rows_policy_filtered, current_prediction_pairs)
        main_rows, _aux_rows = _split_results_for_main_reporting(
            reporting_ready_rows,
            excluded_ablation_names_main,
            excluded_ablation_prefixes_main,
        )
        results_df = pd.DataFrame(main_rows)
        if not results_df.empty and "split" in results_df.columns:
            test_df = results_df[results_df["split"] == reporting_split].copy()
            if "model_short_name" not in test_df.columns and "model_id" in test_df.columns:
                test_df["model_short_name"] = (
                    test_df["model_id"].astype(str).map(short_name_map).fillna(test_df["model_id"])
                )
            hue_order = list(dict.fromkeys(test_df["model_short_name"].astype(str).tolist()))
            palette = build_model_palette(hue_order)
            for metric in ["accuracy", "f1", "average_precision"]:
                plot_metric_bars(
                    test_df,
                    metric=metric,
                    out_path=dirs["figures"] / f"{metric}_bars.png",
                    model_col="model_short_name",
                    hue_order=hue_order,
                    palette=palette,
                    style=style,
                    dpi=dpi,
                )
                rendered_benchmark_plots += 1
            for metric in ["accuracy", "f1", "roc_auc", "average_precision"]:
                plot_family_heatmap(
                    test_df,
                    metric=metric,
                    out_path=dirs["figures"] / f"{metric}_family_heatmap.png",
                    style=style,
                    dpi=dpi,
                )
                rendered_benchmark_plots += 1
        else:
            logger.warning(
                "Plots-only mode: benchmark results missing reporting split '%s'; skipping bar/family plots",
                reporting_split,
            )

        roc_pred_df, _ = _split_predictions_for_main_reporting(
            current_pred_df,
            excluded_ablation_names_main,
            excluded_ablation_prefixes_main,
        )
        if not roc_pred_df.empty:
            if "model_short_name" not in roc_pred_df.columns and "model_id" in roc_pred_df.columns:
                roc_pred_df["model_short_name"] = (
                    roc_pred_df["model_id"].astype(str).map(short_name_map).fillna(roc_pred_df["model_id"])
                )
            hue_order = list(dict.fromkeys(roc_pred_df["model_short_name"].astype(str).tolist()))
            palette = build_model_palette(hue_order)
            plot_roc_curves(
                roc_pred_df,
                out_path=dirs["figures"] / "roc_auc_curves.png",
                model_col="model_short_name",
                hue_order=hue_order,
                palette=palette,
                style=style,
                dpi=dpi,
            )
            rendered_benchmark_plots += 1
        else:
            logger.warning(
                "Plots-only mode: no prediction CSVs found for split '%s'; skipping ROC curves",
                reporting_split,
            )

    feature_importance_path = (
        dirs["tables"] / f"feature_importance_{plans[0].name}.csv"
        if plans
        else None
    )
    if feature_importance_path and feature_importance_path.exists():
        try:
            imp_df = pd.read_csv(feature_importance_path)
        except Exception:  # noqa: BLE001
            imp_df = pd.DataFrame()
        if not imp_df.empty:
            plot_feature_importance(
                imp_df,
                out_path=dirs["figures"] / "feature_importance.png",
                style=style,
                dpi=dpi,
            )
            rendered_benchmark_plots += 1

    manifest = {
        "task": "plots-only",
        "reporting_split": reporting_split,
        "benchmark_plots_rendered": rendered_benchmark_plots,
        "outputs": {k: str(v) for k, v in dirs.items()},
    }
    _write_json(dirs["manifests"] / "plots_only_benchmark_manifest.json", manifest)
    logger.info("Plots-only benchmark rendering completed (figure_count=%d)", rendered_benchmark_plots)
    return manifest
