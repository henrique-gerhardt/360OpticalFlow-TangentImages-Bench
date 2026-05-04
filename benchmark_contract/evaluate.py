from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import numpy as np

from common import PROJECT_ROOT, get_result_paths, latitude_band_masks, write_json

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import flow_evaluate  # type: ignore


def normalize_flow(flow: np.ndarray) -> np.ndarray:
    array = np.asarray(flow, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Unexpected optical flow rank: {array.shape}")
    if array.shape[-1] == 2:
        return array
    if array.shape[0] == 2:
        return np.moveaxis(array, 0, -1).astype(np.float32)
    raise ValueError(f"Unexpected optical flow shape: {array.shape}")


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
    notes: list[str],
    spherical: bool = False,
) -> float | None:
    try:
        return aggregate_metric(metric_name, gt, pred, mask, spherical=spherical)
    except Exception as exc:
        notes.append(f"{metric_name}{'_spherical' if spherical else ''} unavailable: {exc}")
        return None


def load_prediction_bundle() -> Dict[str, Any]:
    data = np.load(get_result_paths()["predictions_npz"])
    pred = normalize_flow(data["pred_flow"])
    gt = normalize_flow(data["gt_flow"]) if data["gt_flow"].size else None
    valid_mask = data["valid_mask"].astype(bool) if data["valid_mask"].size else None
    return {"pred": pred, "gt": gt, "valid_mask": valid_mask}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    bundle = load_prediction_bundle()
    pred = bundle["pred"]
    gt = bundle["gt"]
    valid_mask = bundle["valid_mask"]

    if gt is None:
        write_json(
            get_result_paths()["quality"],
            {
                "scenario": args.scenario,
                "quality_metrics_available": False,
                "notes": ["Ground-truth optical flow was not available for this sample."],
            },
        )
        return

    height, width = pred.shape[:2]
    bands = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    masks = latitude_band_masks(height, width, bands)

    global_valid = flow_evaluate.available_pixel(gt.copy(), valid_mask)
    notes: list[str] = []
    metrics: Dict[str, Any] = {
        "scenario": args.scenario,
        "quality_metrics_available": True,
        "aae_global": try_metric("aae", gt, pred, valid_mask, notes, spherical=False),
        "epe_global": try_metric("epe", gt, pred, valid_mask, notes, spherical=False),
        "rmse_global": try_metric("rmse", gt, pred, valid_mask, notes, spherical=False),
        "saae_global": try_metric("aae", gt, pred, valid_mask, notes, spherical=True),
        "sepe_global": try_metric("epe", gt, pred, valid_mask, notes, spherical=True),
        "srmse_global": try_metric("rmse", gt, pred, valid_mask, notes, spherical=True),
        "valid_pixels_ratio": float(global_valid.mean()),
        "valid_pixel_count": int(global_valid.sum()),
        "epe_by_latitude": {},
        "sepe_by_latitude": {},
    }

    for key, band_mask in masks.items():
        effective_mask = band_mask if valid_mask is None else (band_mask & valid_mask)
        metrics["epe_by_latitude"][key] = try_metric("epe", gt, pred, effective_mask, notes, spherical=False)
        metrics["sepe_by_latitude"][key] = try_metric("epe", gt, pred, effective_mask, notes, spherical=True)

    polar_mask = masks["-90_-60"] | masks["60_90"]
    equatorial_mask = masks["-30_0"] | masks["0_30"]
    if valid_mask is not None:
        polar_mask = polar_mask & valid_mask
        equatorial_mask = equatorial_mask & valid_mask

    polar_epe = try_metric("epe", gt, pred, polar_mask, notes, spherical=False)
    equatorial_epe = try_metric("epe", gt, pred, equatorial_mask, notes, spherical=False)
    polar_sepe = try_metric("epe", gt, pred, polar_mask, notes, spherical=True)
    equatorial_sepe = try_metric("epe", gt, pred, equatorial_mask, notes, spherical=True)

    metrics["epe_polar"] = polar_epe
    metrics["epe_equatorial"] = equatorial_epe
    metrics["sepe_polar"] = polar_sepe
    metrics["sepe_equatorial"] = equatorial_sepe
    metrics["regional_robustness"] = {
        "polar_minus_equatorial_epe": None if polar_epe is None or equatorial_epe is None else float(polar_epe - equatorial_epe),
        "polar_minus_equatorial_sepe": None if polar_sepe is None or equatorial_sepe is None else float(polar_sepe - equatorial_sepe),
    }
    if notes:
        metrics["notes"] = notes

    write_json(get_result_paths()["quality"], metrics)


if __name__ == "__main__":
    main()
