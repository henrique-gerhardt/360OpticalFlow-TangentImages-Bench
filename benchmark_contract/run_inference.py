from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from common import (
    PROJECT_ROOT,
    ensure_contract_dirs,
    get_result_paths,
    load_all_configs,
    read_json,
    resolve_dataset_config,
    resolve_experiment_config,
    resolve_runtime_config,
    set_seed,
    write_json,
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

import flow_estimate  # type: ignore
import flow_io  # type: ignore
import flow_postproc  # type: ignore
import flow_vis  # type: ignore
import image_io  # type: ignore


@dataclass
class SamplePaths:
    sample_dir: Path
    src_image: Path
    tgt_image: Path
    gt_flow: Path | None
    mask: Path | None


@dataclass
class PreparedSample:
    paths: SamplePaths
    src_original: np.ndarray
    tgt_original: np.ndarray
    src_input: np.ndarray
    tgt_input: np.ndarray
    gt_flow: np.ndarray | None
    valid_mask: np.ndarray | None


def load_scenario_bundle(scenario: str) -> Dict[str, Any]:
    configs = load_all_configs()
    experiment = resolve_experiment_config(configs, scenario)
    runtime = resolve_runtime_config(configs, scenario)
    dataset = resolve_dataset_config(configs, experiment["dataset"])
    return {
        "manifest": configs["manifest"],
        "dataset": dataset,
        "runtime": runtime,
        "experiment": experiment,
    }


def read_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8)).save(path)


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_flow(flow: np.ndarray, width: int, height: int) -> np.ndarray:
    original_height, original_width = flow.shape[:2]
    resized = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    resized[..., 0] *= float(width) / float(original_width)
    resized[..., 1] *= float(height) / float(original_height)
    return resized


def normalize_flow(flow: np.ndarray) -> np.ndarray:
    array = np.asarray(flow, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Unexpected optical flow rank: {array.shape}")
    if array.shape[-1] == 2:
        return array
    if array.shape[0] == 2:
        return np.moveaxis(array, 0, -1).astype(np.float32)
    raise ValueError(f"Unexpected optical flow shape: {array.shape}")


def discover_sample(dataset_cfg: Dict[str, Any], experiment_cfg: Dict[str, Any]) -> SamplePaths:
    scene_name = experiment_cfg.get("scene") or dataset_cfg.get("default_scene")
    if not scene_name:
        raise ValueError("No scene configured for the selected dataset.")

    sample_dir = Path(dataset_cfg["root"]) / scene_name
    pano_subdir = dataset_cfg.get("pano_subdir")
    if pano_subdir:
        sample_dir = sample_dir / pano_subdir

    frame_idx = int(experiment_cfg["frame_idx"])
    direction = experiment_cfg.get("direction", dataset_cfg.get("default_direction", "forward"))
    offset = 1 if direction == "forward" else -1
    target_idx = frame_idx + offset

    image_pattern = dataset_cfg["image_pattern"]
    src_image = sample_dir / image_pattern.format(frame_idx=frame_idx)
    tgt_image = sample_dir / image_pattern.format(frame_idx=target_idx)

    flow_pattern = dataset_cfg.get("flow_forward_pattern") if direction == "forward" else dataset_cfg.get("flow_backward_pattern")
    gt_flow = sample_dir / flow_pattern.format(frame_idx=frame_idx) if flow_pattern else None

    mask_pattern = dataset_cfg.get("mask_pattern")
    mask = sample_dir / mask_pattern.format(frame_idx=frame_idx) if mask_pattern else None

    if not src_image.exists():
        raise FileNotFoundError(f"Source image not found: {src_image}")
    if not tgt_image.exists():
        raise FileNotFoundError(f"Target image not found: {tgt_image}")

    return SamplePaths(
        sample_dir=sample_dir,
        src_image=src_image,
        tgt_image=tgt_image,
        gt_flow=gt_flow if gt_flow is not None and gt_flow.exists() else None,
        mask=mask if mask is not None and mask.exists() else None,
    )


def build_estimator(experiment_cfg: Dict[str, Any]):
    estimator = flow_estimate.PanoOpticalFlow()
    estimator.debug_enable = False
    estimator.debug_output_dir = None
    estimator.padding_size_cubemap = float(experiment_cfg.get("padding_size", 0.3))
    estimator.padding_size_ico = float(experiment_cfg.get("padding_size", 0.3))
    estimator.flow2rotmat_method = experiment_cfg.get("flow2rotmat_method", "3D")
    estimator.tangent_image_width_ico = int(experiment_cfg.get("tangent_image_width_ico", 480))
    return estimator


def prepare_sample(bundle: Dict[str, Any]) -> PreparedSample:
    dataset_cfg = bundle["dataset"]
    experiment_cfg = bundle["experiment"]
    runtime_cfg = bundle["runtime"]

    paths = discover_sample(dataset_cfg, experiment_cfg)
    src_original = np.asarray(image_io.image_read(str(paths.src_image)))
    tgt_original = np.asarray(image_io.image_read(str(paths.tgt_image)))

    input_height = int(runtime_cfg["input_height"])
    input_width = int(runtime_cfg["input_width"])
    resize_for_efficiency = bool(experiment_cfg.get("resize_for_efficiency", False))

    if resize_for_efficiency and (src_original.shape[0] != input_height or src_original.shape[1] != input_width):
        src_input = resize_image(src_original, input_width, input_height)
        tgt_input = resize_image(tgt_original, input_width, input_height)
    else:
        src_input = src_original
        tgt_input = tgt_original

    gt_flow = None
    if paths.gt_flow is not None:
        gt_flow = normalize_flow(flow_io.read_flow_flo(str(paths.gt_flow)))

    valid_mask = read_mask(paths.mask) if paths.mask is not None else None
    return PreparedSample(
        paths=paths,
        src_original=src_original,
        tgt_original=tgt_original,
        src_input=src_input,
        tgt_input=tgt_input,
        gt_flow=gt_flow,
        valid_mask=valid_mask,
    )


def predict_flow(estimator: Any, src_image: np.ndarray, tgt_image: np.ndarray) -> np.ndarray:
    flow = estimator.estimate(src_image, tgt_image)
    return normalize_flow(flow)


def run_inference_once(bundle: Dict[str, Any], estimator: Any | None = None) -> Dict[str, Any]:
    sample = prepare_sample(bundle)

    load_started = time.perf_counter()
    model = estimator if estimator is not None else build_estimator(bundle["experiment"])
    load_ended = time.perf_counter()

    inference_started = time.perf_counter()
    pred_native = predict_flow(model, sample.src_input, sample.tgt_input)
    pred_native = normalize_flow(flow_postproc.erp_of_wraparound(pred_native))
    inference_ended = time.perf_counter()

    if sample.gt_flow is not None and pred_native.shape[:2] != sample.gt_flow.shape[:2]:
        pred_eval = resize_flow(pred_native, sample.gt_flow.shape[1], sample.gt_flow.shape[0])
    elif pred_native.shape[:2] != sample.src_original.shape[:2]:
        pred_eval = resize_flow(pred_native, sample.src_original.shape[1], sample.src_original.shape[0])
    else:
        pred_eval = pred_native.copy()

    return {
        "model": model,
        "sample": sample,
        "pred_flow_native": pred_native,
        "pred_flow": pred_eval,
        "load_time_ms": (load_ended - load_started) * 1000.0,
        "inference_time_ms": (inference_ended - inference_started) * 1000.0,
    }


def maybe_save_visualization(path: Path, flow: np.ndarray) -> None:
    try:
        vis = flow_vis.flow_to_color(flow, min_ratio=0.2, max_ratio=0.8)
        save_image(path, vis)
    except Exception:
        pass


def write_prediction_artifacts(bundle: Dict[str, Any], run_output: Dict[str, Any]) -> None:
    result_paths = get_result_paths()
    sample: PreparedSample = run_output["sample"]
    pred_flow = run_output["pred_flow"]
    pred_flow_native = run_output["pred_flow_native"]
    runtime_cfg = bundle["runtime"]
    experiment_cfg = bundle["experiment"]

    np.save(result_paths["pred_flow_npy"], pred_flow)
    np.save(result_paths["pred_flow_native_npy"], pred_flow_native)
    flow_io.flow_write(pred_flow.astype(np.float32), str(result_paths["pred_flow_flo"]))

    gt_flow = sample.gt_flow if sample.gt_flow is not None else np.empty((0, 0, 2), dtype=np.float32)
    valid_mask = sample.valid_mask if sample.valid_mask is not None else np.ones(pred_flow.shape[:2], dtype=bool)

    np.save(result_paths["gt_flow_npy"], gt_flow)
    np.save(result_paths["valid_mask_npy"], valid_mask)
    np.savez_compressed(
        result_paths["predictions_npz"],
        pred_flow=pred_flow.astype(np.float32),
        pred_flow_native=pred_flow_native.astype(np.float32),
        gt_flow=gt_flow.astype(np.float32),
        valid_mask=valid_mask.astype(bool),
    )

    if bool(runtime_cfg.get("save_visualizations", False)) and bool(experiment_cfg.get("save_optional_predictions", False)):
        maybe_save_visualization(result_paths["pred_flow_vis"], pred_flow)

    inference_metadata = {
        "sample_dir": str(sample.paths.sample_dir),
        "src_image": str(sample.paths.src_image),
        "tgt_image": str(sample.paths.tgt_image),
        "gt_flow": str(sample.paths.gt_flow) if sample.paths.gt_flow is not None else None,
        "mask": str(sample.paths.mask) if sample.paths.mask is not None else None,
        "pred_flow_flo": str(result_paths["pred_flow_flo"]),
        "pred_flow_npy": str(result_paths["pred_flow_npy"]),
        "pred_flow_native_npy": str(result_paths["pred_flow_native_npy"]),
    }
    write_json(result_paths["inference_metadata"], inference_metadata)

    metadata_path = result_paths["metadata"]
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(
        {
            "dataset_root": str(bundle["dataset"]["root"]),
            "scene": bundle["experiment"].get("scene"),
            "frame_idx": int(bundle["experiment"]["frame_idx"]),
            "direction": bundle["experiment"].get("direction"),
            "source_image": str(sample.paths.src_image),
            "target_image": str(sample.paths.tgt_image),
            "ground_truth_flow": str(sample.paths.gt_flow) if sample.paths.gt_flow is not None else None,
            "mask_path": str(sample.paths.mask) if sample.paths.mask is not None else None,
        }
    )
    write_json(metadata_path, metadata)

    write_json(
        result_paths["run_config"],
        {
            "scenario": bundle["experiment"]["scenario"],
            "dataset": bundle["dataset"]["name"],
            "dataset_root": str(bundle["dataset"]["root"]),
            "scene": bundle["experiment"].get("scene"),
            "frame_idx": int(bundle["experiment"]["frame_idx"]),
            "direction": bundle["experiment"].get("direction"),
            "batch_size": int(runtime_cfg["batch_size"]),
            "precision": runtime_cfg["precision"],
            "warmup_runs": int(runtime_cfg["warmup_runs"]),
            "measured_runs": int(runtime_cfg["measured_runs"]),
            "original_input_height": int(sample.src_original.shape[0]),
            "original_input_width": int(sample.src_original.shape[1]),
            "inference_input_height": int(sample.src_input.shape[0]),
            "inference_input_width": int(sample.src_input.shape[1]),
            "prediction_height": int(pred_flow.shape[0]),
            "prediction_width": int(pred_flow.shape[1]),
            "resize_for_efficiency": bool(bundle["experiment"].get("resize_for_efficiency", False)),
            "save_optional_predictions": bool(bundle["experiment"].get("save_optional_predictions", False)),
            "padding_size": float(bundle["experiment"].get("padding_size", 0.3)),
            "tangent_image_width_ico": int(bundle["experiment"].get("tangent_image_width_ico", 480)),
            "flow2rotmat_method": bundle["experiment"].get("flow2rotmat_method", "3D"),
            "estimator_load_wall_ms": run_output["load_time_ms"],
            "single_inference_wall_ms": run_output["inference_time_ms"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    ensure_contract_dirs()
    bundle = load_scenario_bundle(args.scenario)
    set_seed(int(bundle["runtime"]["seed"]))

    run_output = run_inference_once(bundle)
    write_prediction_artifacts(bundle, run_output)


if __name__ == "__main__":
    main()
