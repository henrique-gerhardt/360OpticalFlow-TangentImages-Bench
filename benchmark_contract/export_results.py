from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import (
    discover_environment,
    ensure_contract_dirs,
    get_result_paths,
    load_all_configs,
    resolve_experiment_config,
    write_json,
)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_metadata(scenario: str) -> None:
    ensure_contract_dirs()
    configs = load_all_configs()
    manifest = configs["manifest"]
    experiment = resolve_experiment_config(configs, scenario)

    payload = {
        "method_name": manifest.get("method_name"),
        "method_family": manifest.get("method_family"),
        "paper_year": manifest.get("paper_year"),
        "framework": manifest.get("framework"),
        "scenario": scenario,
        "dataset": experiment.get("dataset"),
        "checkpoint": experiment.get("checkpoint"),
        "commit": git_commit(),
    }
    write_json(get_result_paths()["metadata"], payload)
    write_json(get_result_paths()["environment"], discover_environment())


def finalize() -> None:
    # Hook futuro para consolidar hashes, checksums ou agregações.
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["metadata", "finalize"], required=True)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    if args.phase == "metadata":
        write_metadata(args.scenario)
    else:
        finalize()


if __name__ == "__main__":
    main()
