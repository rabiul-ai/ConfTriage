# lung_stresstest

Comprehensive, runnable benchmarking pipeline for lung cancer binary classification
using LIDC-IDRI nodule metadata and optional local 2D nodule images, designed for
ablation-heavy evaluation of many LLM families (closed and open).

## What this repository now provides

- End-to-end experiment runner with:
  - metadata ingestion from `LIDC-IDRI nodule metadata.xlsx`
  - optional image-stat feature extraction from local `2D Nodule Views/`
  - ablation groups (numeric radiology, text-only, image-stats-only, multimodal)
  - model-family benchmarking across API-backed closed/open models
  - metrics, confidence intervals, ranked summary tables, and publication-ready plots
- Config-driven workflow (`configs/default.yaml`) for reproducibility.
- Local-image friendly setup: large image files are ignored by git and expected locally.

## Dataset assumptions

### Metadata (included in repo)
- File: `LIDC-IDRI nodule metadata.xlsx`
- Label source: `cancer label` column (`Y`/`N` used by default, `U` excluded)
- Default cohort protocol is enforced globally before any split/model/ablation:
  - keep `nodule selected == Yes`
  - keep `nodule diameter (mm) > 3.0`
  - exclude `malignancy == 3`
  - this protocol is configured under `data.cohort_protocol`

### Images (local only, not committed)
- Place all PNG nodule views under:
  - `2D Nodule Views/`
- Filename pattern expected by parser:
  - `nid_1_sel_Yes_s_1_dia_33_p_0001_n_1_m_5_c_Y.png`

If `2D Nodule Views/` is missing, pipeline still runs (metadata/text only). If present,
image statistics are extracted and included in ablations that request them.

## Quickstart

1) Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2) (Optional) Add OpenRouter API key for hosted model benchmarking:

```bash
cp .env.example .env
# then set OPENROUTER_API_KEY
```

3) Ensure data layout:

```text
.
├── LIDC-IDRI nodule metadata.xlsx
├── 2D Nodule Views/          # local large PNGs (not tracked by git)
├── configs/default.yaml
└── run_benchmark.py
```

4) Run benchmark:

```bash
python3 run_benchmark.py --config configs/default.yaml
```

5) Run Gemini text-only prompt-template ablation (T1-T5 x temperatures):

```bash
python3 run_benchmark.py --config configs/default.yaml --task prompt-ablation
```

6) Run description-source ablation (deterministic vs Llama paraphrase vs adversarial paraphrase):

```bash
python3 run_benchmark.py --config configs/default.yaml --task description-source-ablation
```

7) Regenerate figures only from cached CSV/prediction outputs (no model inference):

```bash
python3 run_benchmark.py --config configs/default.yaml --task plots-only
```

## Output artifacts

Default output root: `outputs/default_run/`

- `tables/benchmark_results.csv`: main per model x ablation x split metrics (auxiliary ablations excluded; includes `val`, `test`, and `full`)
- `tables/benchmark_results_all.csv`: full per model x ablation x split metrics (including auxiliary ablations)
- `tables/benchmark_results_auxiliary.csv`: auxiliary-ablation metrics (if any)
- `tables/benchmark_summary_ranked.csv`: ranked summary on `evaluation.reporting_split`
- `tables/benchmark_summary_auxiliary_ranked.csv`: ranked auxiliary summary on `evaluation.reporting_split` (if any)
- `tables/feature_importance_*.csv`: ablation-specific feature analysis
- `tables/pairwise_bootstrap_pvalues.csv`: pairwise model-comparison p-values (with FDR-adjusted q-values)
- `tables/pairwise_bootstrap_pvalues_auxiliary.csv`: pairwise p-values for auxiliary ablations (if any)
- `predictions/*.csv`: per-sample predictions (including `_test.csv` and `_full.csv`)
- `figures/*`: barplots, heatmaps, feature-importance chart, ROC curve figure
- `manifests/run_manifest.json`: run configuration and data stats
- `logs/run.log`: detailed execution log
- `prompt_ablation/tables/prompt_template_temperature_results.csv`: per-template, per-temperature, per-replicate metrics
- `prompt_ablation/tables/prompt_template_temperature_summary.csv`: template x temperature AUC mean/std summary
- `prompt_ablation/tables/self_consistency_results.csv`: self-consistency baseline results (`n=10`, `T=0.7` by default)
- `prompt_ablation/figures/prompt_template_temperature_auc_heatmap.png`: 5x5 AUC heatmap (color=mean AUC, dark overlay=replicate std)
- `description_source_ablation/tables/description_source_auc_table.csv`: three-row AUC table for description-source ablation
- `description_source_ablation/tables/description_source_metrics.csv`: full metrics with CIs per description source
- `description_source_ablation/tables/near_leak_attribute_auc_drops.csv`: four-row near-leak table (baseline, -spiculation, -margin, -both) with AUC deltas vs baseline
- `description_source_ablation/tables/near_leak_attribute_metrics.csv`: full near-leak metrics with bootstrap CIs

### Ablations at a glance

Default ablations in `configs/default.yaml`:

- `radiology_numeric`
  - Uses structured radiological metadata only:
    `subtlety`, `internal structure`, `calcification`, `sphericity`, `margin`,
    `lobulation`, `spiculation`, `textures`, `nodule diameter (mm)`.
- `text_only`
  - Uses only `nodule description` text.
- `text_corruption_shuffle_malignant`
  - Synthetic corruption control on `nodule description`:
    descriptions are deterministically shuffled across malignant nodules
    within each split to preserve text marginals while breaking
    input-to-label alignment in the malignant class.
- `text_corruption_negated_attributes`
  - Synthetic corruption control on `nodule description`:
    key descriptor terms are deterministically negated to semantic opposites
    (for example, smooth $\leftrightarrow$ spiculated).
- `text_corruption_paraphrase_neutral`
  - Synthetic corruption control on `nodule description`:
    descriptor terms are deterministically paraphrased to neutral wording
    while preserving semantic content.
- `image_stats_only`
  - Uses only image-derived numeric summaries from local PNGs:
    `img_view_count`, `img_mean_intensity`, `img_std_intensity`,
    `img_min_intensity`, `img_max_intensity`, `img_p10_intensity`,
    `img_p90_intensity`.
- `img_plus_text`
  - Uses image-derived numeric summaries + `nodule description` text
    (without radiology numeric metadata columns).
- `radiology_plus_text`
  - Structured radiology numeric features + `nodule description` text.
- `radiology_plus_image_stats`
  - Structured radiology numeric features + image-derived numeric summaries.
- `all_modalities`
  - Structured radiology numeric features + `nodule description` text +
    image-derived numeric summaries.

### Figure conventions

- ROC-AUC is plotted as ROC curves (`figures/roc_auc_curves.png`) rather than bars.
- Model legends use short names (derived from model IDs) for readability.
- A fixed model-color mapping is used consistently across model-colored figures
  (bar plots and ROC curves).
- Ablations whose names match
  `evaluation.exclude_ablation_names_from_main_reporting` or
  `evaluation.exclude_ablation_name_prefixes_from_main_reporting`
  are excluded from main figures/tables and emitted as auxiliary tables.

### Gemini text-only prompt ablation

- Config section: `prompt_ablation`
- Model: defaults to `google/gemini-3.1-flash-lite-preview`
- Templates: `T1`..`T5`
  - `T1`: zero-shot direct
  - `T2`: thoracic-radiologist persona
  - `T3`: chain-of-thought style instruction (internal reasoning; JSON output only)
  - `T4`: three-shot in-context examples from `Dcalib` (default split: `val`)
  - `T5`: Fleischner 2017 preamble
- Temperatures: default `[0.0, 0.3, 0.5, 0.7, 1.0]`
- Replicates: default `5` per template-temperature setting
- Self-consistency baseline: default `template=T3`, `T=0.7`, `n=10`
- Cohort consistency: runs on the same filtered full cohort used by the pipeline.
- Runtime visibility: CLI shows tqdm progress bars for
  template-temperature-replicate runs and self-consistency sampling.
  You can also monitor logs via
  `tail -f outputs/default_run/prompt_ablation/logs/prompt_ablation.log`.
- Resume behavior: completed `template + temperature + replicate` rows in
  `prompt_template_temperature_results.csv` are reused and skipped entirely
  (no re-inference and no bootstrap recomputation), while only missing/incomplete
  runs are executed.
- Figure refresh without reruns: use `--task plots-only` to redraw
  `prompt_template_temperature_auc_heatmap.png` from
  `prompt_template_temperature_summary.csv` only.

### Description-source ablation

- Config section: `description_source_ablation`
- Evaluates three description sources on the same filtered full cohort:
  - deterministic template from structured attributes
  - Llama 3.1 8B Instruct paraphrase (fixed seed)
  - deterministic adversarial paraphrase
- Target classifier defaults to Gemini text-only inference.
- Prediction stage is crash-resumable at nodule level; partial source runs are checkpointed
  every few nodules (`description_source_ablation.prediction_flush_every`).
- Includes a near-leak attribute-removal analysis from deterministic descriptions:
  - baseline template (all radiology attributes)
  - remove `spiculation`
  - remove `margin`
  - remove both `spiculation` and `margin`
- Near-leak analysis can be toggled via
  `description_source_ablation.near_leak_ablation.enabled`.
- Main output table:
  - `outputs/default_run/description_source_ablation/tables/description_source_auc_table.csv`
  - `outputs/default_run/description_source_ablation/tables/near_leak_attribute_auc_drops.csv`

### Statistical testing output

- Pairwise tests are computed across model pairs within each ablation using the
  same bootstrap rounds/alpha configured under `evaluation`.
- Main reporting/statistical split is configurable via `evaluation.reporting_split`
  (default: `full`, which evaluates all filtered nodules).
- Main output table: `tables/pairwise_bootstrap_pvalues.csv`
- Auxiliary output table (if any): `tables/pairwise_bootstrap_pvalues_auxiliary.csv`
- Columns include:
  - `observed_diff_a_minus_b`
  - bootstrap CI (`ci_low`, `ci_high`)
  - raw `p_value`
  - FDR-corrected `q_value_fdr_bh`
  - significance flags (`significant_p_0_05`, `significant_fdr_0_05`)

## Resume / checkpoint behavior

- Results are persisted incrementally to:
  - `outputs/.../tables/benchmark_results.csv`
- After each successful `model_id + ablation` evaluation, val/test rows are written
  immediately. If a later model fails (for example, OpenRouter 429 rate limit), prior
  successful results are retained.
- Re-running with the same `output_dir` resumes from checkpoints:
  - already completed `model_id + ablation` pairs are skipped
  - default config is full-driven resume (`evaluation.required_completion_splits: ["full"]`)
  - when `evaluation.incremental_full_from_test: true`, a missing `full` split can be filled incrementally from existing `test` predictions by evaluating only the non-test remainder
  - if you configure extra required splits, partially completed pairs run only missing splits
  - unfinished/failed pairs are attempted again
- To force a fully fresh run, remove the output directory (or point config to a new
  `project.output_dir`).

## Config customization

Edit `configs/default.yaml` to:

- add/remove ablation groups
- set per-ablation `text_corruption` (one of: `none`, `shuffle_within_malignant`,
  `negate_attributes`, `paraphrase_neutral`) for synthetic corruption controls
- configure protocol cohort filtering under `data.cohort_protocol` (applies to all experiments/ablations)
- choose main reporting split via `evaluation.reporting_split` (`full`/`test`/`val`)
- configure resume completion split requirements via `evaluation.required_completion_splits`
- enable incremental full-from-test completion via `evaluation.incremental_full_from_test`
- control which ablations are treated as auxiliary (excluded from main tables/figures)
  via `evaluation.exclude_ablation_names_from_main_reporting` and
  `evaluation.exclude_ablation_name_prefixes_from_main_reporting`
- add model endpoints and IDs
- tune splits, preprocessing, bootstrap rounds
- change output directory per experiment

## Model support

- `openai-compatible` provider:
  - used for OpenRouter chat completions endpoint
  - default config benchmarks GPT, Claude, Mistral, Gemini, and Qwen model families

Models requiring API keys are automatically skipped when keys are absent (for default config, this is `OPENROUTER_API_KEY`).

## Reproducibility notes

- Seed is controlled via config (`project.seed`).
- Output directory captures all artifacts for each run.
- Keep data files stable and log config snapshots in your paper workflow.
