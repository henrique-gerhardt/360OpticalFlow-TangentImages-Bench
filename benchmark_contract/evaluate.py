from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any, Dict

import numpy as np

from common import (
    PROJECT_ROOT,
    get_result_paths,
    latitude_band_masks,
    load_all_configs,
    read_json,
    resolve_experiment_config,
    write_json,
)
from replica360_protocol import (
    PAPER_TABLE1_REFERENCE,
    build_paper_comparison,
    compute_sample_metrics,
    normalize_flow,
    summarize_protocol_rows,
    try_metric,
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import flow_evaluate  # type: ignore


def load_prediction_bundle() -> Dict[str, Any]:
    data = np.load(get_result_paths()["predictions_npz"])
    pred = normalize_flow(data["pred_flow"])
    gt = normalize_flow(data["gt_flow"]) if data["gt_flow"].size else None
    valid_mask = data["valid_mask"].astype(bool) if data["valid_mask"].size else None
    return {"pred": pred, "gt": gt, "valid_mask": valid_mask}


def build_single_sample_quality(scenario: str, pred: np.ndarray, gt: np.ndarray | None, valid_mask: np.ndarray | None) -> Dict[str, Any]:
    if gt is None:
        return {
            "scenario": scenario,
            "quality_metrics_available": False,
            "notes": ["Ground-truth optical flow was not available for this sample."],
        }

    height, width = pred.shape[:2]
    bands = [(-90, -60), (-60, -30), (-30, 0), (0, 30), (30, 60), (60, 90)]
    masks = latitude_band_masks(height, width, bands)
    global_valid = flow_evaluate.available_pixel(gt.copy(), valid_mask)

    notes: list[str] = []
    sample_metrics = compute_sample_metrics(gt, pred, valid_mask)
    notes.extend(sample_metrics.get("notes", []))
    metrics: Dict[str, Any] = {
        "scenario": scenario,
        "quality_metrics_available": True,
        "aae_global": sample_metrics["aae"],
        "epe_global": sample_metrics["epe"],
        "rmse_global": sample_metrics["rmse"],
        "saae_global": sample_metrics["saae"],
        "sepe_global": sample_metrics["sepe"],
        "srmse_global": sample_metrics["srmse"],
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
    return metrics


def build_official_protocol_quality(scenario: str) -> Dict[str, Any]:
    result_paths = get_result_paths()
    protocol_payload = read_json(result_paths["protocol_rows_json"])
    rows = protocol_payload.get("rows", [])
    if not rows:
        return {
            "scenario": scenario,
            "quality_metrics_available": False,
            "notes": ["Replica360 official protocol did not generate any per-sample rows."],
        }

    summary = summarize_protocol_rows(rows)
    paper_comparison = build_paper_comparison(summary)

    quality_metrics = {
        "scenario": scenario,
        "quality_metrics_available": True,
        "official_protocol": "replica360_table1",
        "aggregation_rule": "Unweighted mean over per-sample metrics, matching test_replica360.py and Table 1.",
        "aae_global": summary["all"]["aae"],
        "epe_global": summary["all"]["epe"],
        "rmse_global": summary["all"]["rmse"],
        "saae_global": summary["all"]["saae"],
        "sepe_global": summary["all"]["sepe"],
        "srmse_global": summary["all"]["srmse"],
        "valid_pixels_ratio": float(statistics.mean(row["valid_pixels_ratio"] for row in rows)),
        "valid_pixel_count": int(sum(int(row["valid_pixel_count"]) for row in rows)),
        "replica360_protocol": {
            "subsets": summary,
            "paper_table1_reference": PAPER_TABLE1_REFERENCE,
            "delta_vs_paper_table1": paper_comparison,
        },
        "notes": [
            "The official_reproduction scenario was aggregated over the full Replica360 protocol subsets circle, line, random, and all.",
            "Top-level global metrics correspond to the 'All' row of Table 1 and are directly comparable to the paper.",
        ],
    }
    write_json(result_paths["paper_comparison"], quality_metrics["replica360_protocol"])
    return quality_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    configs = load_all_configs()
    experiment_cfg = resolve_experiment_config(configs, args.scenario)
    official_protocol = experiment_cfg.get("official_protocol")

    if args.scenario == "official_reproduction" and official_protocol == "replica360_table1":
        write_json(get_result_paths()["quality"], build_official_protocol_quality(args.scenario))
        return

    bundle = load_prediction_bundle()
    write_json(
        get_result_paths()["quality"],
        build_single_sample_quality(
            args.scenario,
            pred=bundle["pred"],
            gt=bundle["gt"],
            valid_mask=bundle["valid_mask"],
        ),
    )


if __name__ == "__main__":
    main()
