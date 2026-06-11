from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .description_source_ablation import run_description_source_ablation
from .pipeline import run_pipeline, run_pipeline_plots_only
from .prompt_ablation import run_prompt_ablation, run_prompt_ablation_plots_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lung cancer LLM benchmark runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to benchmark YAML config",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="benchmark",
        choices=["benchmark", "prompt-ablation", "description-source-ablation", "both", "all", "plots-only"],
        help="Which task to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    if args.task == "plots-only":
        run_pipeline_plots_only(config)
        run_prompt_ablation_plots_only(config)
        return
    if args.task in {"benchmark", "both", "all"}:
        run_pipeline(config)
    if args.task in {"prompt-ablation", "both", "all"}:
        run_prompt_ablation(config)
    if args.task in {"description-source-ablation", "all"}:
        run_description_source_ablation(config)


if __name__ == "__main__":
    main()
