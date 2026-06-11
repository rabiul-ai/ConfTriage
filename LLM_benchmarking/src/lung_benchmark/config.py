from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config as a plain dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    output_root = Path(config["project"]["output_dir"])
    dirs = {
        "root": output_root,
        "tables": output_root / "tables",
        "figures": output_root / "figures",
        "logs": output_root / "logs",
        "predictions": output_root / "predictions",
        "manifests": output_root / "manifests",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
