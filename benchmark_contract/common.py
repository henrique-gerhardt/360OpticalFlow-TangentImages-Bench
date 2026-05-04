from __future__ import annotations

import copy
import json
import os
import platform
import random
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
OUTPUTS_DIR = ROOT / "outputs"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_contract_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "raw_logs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "optional_predictions").mkdir(parents=True, exist_ok=True)


def load_all_configs() -> Dict[str, Any]:
    return {
        "manifest": load_yaml(ROOT / "manifest.yaml"),
        "datasets": load_yaml(CONFIG_DIR / "datasets.yaml"),
        "runtime": load_yaml(CONFIG_DIR / "runtime.yaml"),
        "experiment": load_yaml(CONFIG_DIR / "experiment.yaml"),
    }


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like in (None, ""):
        return None
    expanded = Path(os.path.expandvars(str(path_like)))
    if expanded.is_absolute():
        return expanded
    return PROJECT_ROOT / expanded


def resolve_runtime_config(configs: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    runtime = copy.deepcopy(configs["runtime"]["runtime"])
    scenario_overrides = runtime.pop("scenarios", {})
    deep_update(runtime, scenario_overrides.get(scenario, {}))
    return runtime


def resolve_experiment_config(configs: Dict[str, Any], scenario: str) -> Dict[str, Any]:
    experiment = copy.deepcopy(configs["experiment"]["experiment"])
    scenario_overrides = experiment.pop("scenarios", {})
    deep_update(experiment, scenario_overrides.get(scenario, {}))
    experiment["scenario"] = scenario
    return experiment


def resolve_dataset_config(configs: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset = copy.deepcopy(configs["datasets"]["datasets"][dataset_name])
    dataset["name"] = dataset_name
    root_value = dataset["root"]
    root_env = dataset.get("root_env")
    if root_env and os.environ.get(root_env):
        root_value = os.environ[root_env]
    dataset["root"] = resolve_path(root_value)
    return dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def discover_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "framework": "python",
        "framework_version": None,
        "torch_available": torch is not None,
        "torch_version": getattr(torch, "__version__", None) if torch is not None else None,
        "cuda_available": bool(torch and torch.cuda.is_available()),
        "numpy_version": package_version("numpy"),
        "opencv_version": package_version("opencv-python") or package_version("opencv-python-headless"),
        "pillow_version": package_version("Pillow"),
        "scipy_version": package_version("scipy"),
        "scikit_image_version": package_version("scikit-image"),
        "colorama_version": package_version("colorama"),
    }
    if torch is not None and torch.cuda.is_available():
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
    else:
        env["cuda_version"] = None
        env["gpu_name"] = None
        env["gpu_count"] = 0

    try:
        env["driver_version"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        env["driver_version"] = None

    return env


def get_result_paths() -> Dict[str, Path]:
    return {
        "metadata": RESULTS_DIR / "metadata.json",
        "quality": RESULTS_DIR / "quality_metrics.json",
        "efficiency": RESULTS_DIR / "efficiency_metrics.json",
        "run_config": RESULTS_DIR / "run_config.json",
        "environment": RESULTS_DIR / "environment.json",
        "predictions_npz": OUTPUTS_DIR / "predictions.npz",
        "pred_flow_npy": OUTPUTS_DIR / "pred_flow.npy",
        "pred_flow_native_npy": OUTPUTS_DIR / "pred_flow_native.npy",
        "pred_flow_flo": OUTPUTS_DIR / "pred_flow.flo",
        "pred_flow_vis": OUTPUTS_DIR / "pred_flow_vis.png",
        "gt_flow_npy": OUTPUTS_DIR / "gt_flow.npy",
        "valid_mask_npy": OUTPUTS_DIR / "valid_mask.npy",
        "inference_metadata": OUTPUTS_DIR / "inference_metadata.json",
    }


def latitude_band_masks(height: int, width: int, bands: List[Tuple[float, float]]) -> Dict[str, np.ndarray]:
    lat = ((np.arange(height, dtype=np.float32) + 0.5) / float(height)) * 180.0 - 90.0
    lat = lat[:, None]
    lat = np.repeat(lat, width, axis=1)
    masks: Dict[str, np.ndarray] = {}
    for lo, hi in bands:
        key = f"{int(lo)}_{int(hi)}"
        masks[key] = (lat >= lo) & (lat < hi)
    return masks


def epe(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
    axis = -1 if pred.ndim == 3 and pred.shape[-1] == 2 else 0
    err = np.linalg.norm(pred - gt, axis=axis)
    if valid_mask is not None:
        valid = valid_mask.astype(bool)
        if valid.sum() == 0:
            return float("nan")
        return float(err[valid].mean())
    return float(err.mean())
