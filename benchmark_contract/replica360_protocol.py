from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

import numpy as np

from common import PROJECT_ROOT

import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import flow_evaluate  # type: ignore


SUBSET_TO_LABEL = {
    "circ": "circle",
    "line": "line",
    "rand": "random",
}

SUMMARY_METRIC_KEYS = ("epe", "aae", "rmse", "sepe", "saae", "srmse")

PAPER_TABLE1_REFERENCE: Dict[str, Dict[str, float]] = {
    "circle": {
        "epe": 3.507,
        "aae": 0.1694,
        "rmse": 34.21,
        "sepe": 0.005370,
        "saae": 0.03480,
        "srmse": 0.01021,
    },
    "line": {
        "epe": 5.839,
        "aae": 0.1971,
        "rmse": 40.74,
        "sepe": 0.01063,
        "saae": 0.05951,
        "srmse": 0.02098,
    },
    "random": {
        "epe": 14.10,
        "aae": 0.2192,
        "rmse": 59.78,
        "sepe": 0.02717,
        "saae": 0.08849,
        "srmse": 0.05753,
    },
    "all": {
        "epe": 7.701,
        "aae": 0.1946,
        "rmse": 44.62,
        "sepe": 0.01411,
        "saae": 0.06027,
        "srmse": 0.02905,
    },
}


@dataclass(frozen=True)
class ProtocolSampleSpec:
    subset: str
    scene: str
    frame_idx: int
    target_idx: int
    direction: str


def normalize_flow(flow: np.ndarray) -> np.ndarray:
    array = np.asarray(flow, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Unexpected optical flow rank: {array.shape}")
    if array.shape[-1] == 2:
        return array
    if array.shape[0] == 2:
        return np.moveaxis(array, 0, -1).astype(np.float32)
    raise ValueError(f"Unexpected optical flow shape: {array.shape}")


def metric_mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    if not data:
        return None
    return float(mean(data))


def aggregate_metric(metric_name: str, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None, spherical: bool = False) -> float | None:
    if metric_name == "epe":
        values, valid = flow_evaluate.EPE_mat(gt.copy(), pred.copy(), spherical=spherical, of_mask=mask)
        if int(valid.sum()) == 0:
            return None
        return float(np.sum(values) / np.sum(valid))

    if metric_name == "rmse":
        values, valid = flow_evaluate.RMSE_mat(gt.copy(), pred.copy(), spherical=spherical, of_mask=mask)
        if int(valid.sum()) == 0:
            return None
        return float(np.sqrt(np.sum(np.square(values)) / np.sum(valid)))

    if metric_name == "aae":
        values, valid = flow_evaluate.AAE_mat(gt.copy(), pred.copy(), spherical=spherical, of_mask=mask)
        if int(valid.sum()) == 0:
            return None
        return float(np.sum(values) / np.sum(valid))

    raise ValueError(f"Unsupported metric: {metric_name}")


def try_metric(
    metric_name: str,
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray | None,
    notes: List[str],
    spherical: bool = False,
) -> float | None:
    try:
        return aggregate_metric(metric_name, gt, pred, mask, spherical=spherical)
    except Exception as exc:
        suffix = "_spherical" if spherical else ""
        notes.append(f"{metric_name}{suffix} unavailable: {exc}")
        return None


def compute_sample_metrics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None) -> Dict[str, Any]:
    valid = flow_evaluate.available_pixel(gt.copy(), mask)
    notes: List[str] = []
    metrics = {
        "aae": try_metric("aae", gt, pred, mask, notes, spherical=False),
        "epe": try_metric("epe", gt, pred, mask, notes, spherical=False),
        "rmse": try_metric("rmse", gt, pred, mask, notes, spherical=False),
        "saae": try_metric("aae", gt, pred, mask, notes, spherical=True),
        "sepe": try_metric("epe", gt, pred, mask, notes, spherical=True),
        "srmse": try_metric("rmse", gt, pred, mask, notes, spherical=True),
        "valid_pixel_count": int(valid.sum()),
        "valid_pixels_ratio": float(valid.mean()),
    }
    if notes:
        metrics["notes"] = notes
    return metrics


def infer_subset_from_scene(scene_name: str) -> str:
    for subset in SUBSET_TO_LABEL:
        if scene_name.endswith(f"_{subset}"):
            return subset
    raise ValueError(f"Unable to infer Replica360 subset from scene name: {scene_name}")


def discover_replica360_scene_groups(dataset_root: Path) -> Dict[str, List[str]]:
    groups = {subset: [] for subset in SUBSET_TO_LABEL}
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        try:
            subset = infer_subset_from_scene(child.name)
        except ValueError:
            continue
        groups[subset].append(child.name)
    for subset, scenes in groups.items():
        if not scenes:
            raise FileNotFoundError(f"No Replica360 scenes found for subset '{subset}' under {dataset_root}")
    return groups


def scene_frame_indices(scene_dir: Path) -> List[int]:
    frames = sorted(int(path.name.split("_", 1)[0]) for path in scene_dir.glob("*_rgb_pano.jpg"))
    if not frames:
        raise FileNotFoundError(f"No RGB panoramas found under {scene_dir}")
    return frames


def build_replica360_protocol_samples(dataset_root: Path) -> Dict[str, List[ProtocolSampleSpec]]:
    groups = discover_replica360_scene_groups(dataset_root)
    samples: Dict[str, List[ProtocolSampleSpec]] = {subset: [] for subset in SUBSET_TO_LABEL}
    for subset, scenes in groups.items():
        for scene in scenes:
            frames = scene_frame_indices(dataset_root / scene)
            if subset in {"circ", "rand"}:
                for index, frame_idx in enumerate(frames):
                    samples[subset].append(
                        ProtocolSampleSpec(
                            subset=subset,
                            scene=scene,
                            frame_idx=frame_idx,
                            target_idx=frames[(index + 1) % len(frames)],
                            direction="forward",
                        )
                    )
                    samples[subset].append(
                        ProtocolSampleSpec(
                            subset=subset,
                            scene=scene,
                            frame_idx=frame_idx,
                            target_idx=frames[(index - 1) % len(frames)],
                            direction="backward",
                        )
                    )
            elif subset == "line":
                for source_idx, target_idx in zip(frames[:-1], frames[1:]):
                    samples[subset].append(
                        ProtocolSampleSpec(
                            subset=subset,
                            scene=scene,
                            frame_idx=source_idx,
                            target_idx=target_idx,
                            direction="forward",
                        )
                    )
                for source_idx, target_idx in zip(frames[1:], frames[:-1]):
                    samples[subset].append(
                        ProtocolSampleSpec(
                            subset=subset,
                            scene=scene,
                            frame_idx=source_idx,
                            target_idx=target_idx,
                            direction="backward",
                        )
                    )
            else:
                raise ValueError(f"Unsupported Replica360 subset: {subset}")
    return samples


def subset_label(subset: str) -> str:
    return SUBSET_TO_LABEL[subset]


def summarize_protocol_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for subset, label in SUBSET_TO_LABEL.items():
        subset_rows = [row for row in rows if row["subset"] == subset]
        summary[label] = {
            metric: metric_mean(row[metric] for row in subset_rows if row.get(metric) is not None)
            for metric in SUMMARY_METRIC_KEYS
        }
        summary[label]["sample_count"] = len(subset_rows)
        summary[label]["scene_count"] = len({row["scene"] for row in subset_rows})

    summary["all"] = {
        metric: metric_mean(row[metric] for row in rows if row.get(metric) is not None)
        for metric in SUMMARY_METRIC_KEYS
    }
    summary["all"]["sample_count"] = len(rows)
    summary["all"]["scene_count"] = len({row["scene"] for row in rows})
    return summary


def build_paper_comparison(summary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    comparison: Dict[str, Dict[str, Any]] = {}
    for group_name, reference_metrics in PAPER_TABLE1_REFERENCE.items():
        observed = summary.get(group_name, {})
        comparison[group_name] = {
            metric: None if observed.get(metric) is None else float(observed[metric] - reference_metrics[metric])
            for metric in SUMMARY_METRIC_KEYS
        }
    return comparison


def write_protocol_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subset",
        "scene",
        "frame_idx",
        "target_idx",
        "direction",
        "aae",
        "epe",
        "rmse",
        "saae",
        "sepe",
        "srmse",
        "valid_pixel_count",
        "valid_pixels_ratio",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            if isinstance(csv_row.get("notes"), list):
                csv_row["notes"] = " | ".join(csv_row["notes"])
            writer.writerow(csv_row)
