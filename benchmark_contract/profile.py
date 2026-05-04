from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from common import PROJECT_ROOT, get_result_paths, write_json
from run_inference import build_estimator, flow_postproc, load_scenario_bundle, prepare_sample, predict_flow


def get_peak_rss_mb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(peak / (1024.0 * 1024.0))
        return float(peak / 1024.0)
    except Exception:
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def estimate_checkpoint_size_mb(bundle: Dict[str, Any]) -> float | None:
    checkpoint = bundle["experiment"].get("checkpoint") or bundle["manifest"].get("checkpoints", {}).get("default")
    if not checkpoint:
        return None
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if checkpoint_path.exists() and checkpoint_path.is_file():
        return round(checkpoint_path.stat().st_size / (1024.0 * 1024.0), 4)
    return None


def estimate_source_artifact_size_mb() -> float:
    relevant_paths = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "test_replica360.py",
        PROJECT_ROOT / "requirements.txt",
    ]
    total_bytes = sum(path_size_bytes(path) for path in relevant_paths)
    return round(total_bytes / (1024.0 * 1024.0), 4)


def measure_latency(bundle: Dict[str, Any]) -> Dict[str, Any]:
    runtime_cfg = bundle["runtime"]
    sample_prepare_started = time.perf_counter()
    sample = prepare_sample(bundle)
    sample_prepare_ended = time.perf_counter()

    load_started = time.perf_counter()
    estimator = build_estimator(bundle["experiment"])
    load_ended = time.perf_counter()

    warmup_runs = int(runtime_cfg["warmup_runs"])
    measured_runs = int(runtime_cfg["measured_runs"])

    for _ in range(warmup_runs):
        _ = flow_postproc.erp_of_wraparound(predict_flow(estimator, sample.src_input, sample.tgt_input))

    latencies: list[float] = []
    for _ in range(measured_runs):
        started = time.perf_counter()
        _ = flow_postproc.erp_of_wraparound(predict_flow(estimator, sample.src_input, sample.tgt_input))
        ended = time.perf_counter()
        latencies.append((ended - started) * 1000.0)

    mean_ms = float(statistics.mean(latencies)) if latencies else None
    median_ms = float(statistics.median(latencies)) if latencies else None
    return {
        "estimator_load_wall_ms": float((load_ended - load_started) * 1000.0),
        "sample_prepare_wall_ms": float((sample_prepare_ended - sample_prepare_started) * 1000.0),
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_p95_ms": percentile(latencies, 95.0),
        "fps": None if mean_ms in (None, 0.0) else float(1000.0 / mean_ms),
        "max_memory_mb": get_peak_rss_mb(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    bundle = load_scenario_bundle(args.scenario)
    checkpoint_size_mb = estimate_checkpoint_size_mb(bundle)
    metrics = measure_latency(bundle)
    metrics.update(
        {
            "scenario": args.scenario,
            "parameters": None,
            "parameters_applicable": False,
            "flops_g": None,
            "flops_g_applicable": False,
            "checkpoint_size_mb": checkpoint_size_mb,
            "checkpoint_size_mb_applicable": checkpoint_size_mb is not None,
            "source_artifact_size_mb": estimate_source_artifact_size_mb(),
            "not_applicable": {
                "parameters": "The method is exposed as a classical geometric/OpenCV pipeline, not as a checkpointed torch.nn.Module.",
                "flops_g": "There is no stable tensor-graph/module path that supports meaningful FLOPs counting for the native pipeline.",
                "checkpoint_size_mb": "Not applicable when no standalone checkpoint artifact is required by the selected method configuration." if checkpoint_size_mb is None else None,
            },
            "notes": [
                "Profiling uses the native PanoOpticalFlow estimator path from src/flow_estimate.py.",
                "FLOPs and parameter count remain null because this project does not expose a torch.nn.Module checkpointed model path.",
            ],
        }
    )

    write_json(get_result_paths()["efficiency"], metrics)


if __name__ == "__main__":
    main()
