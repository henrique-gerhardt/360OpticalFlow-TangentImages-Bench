#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-official_reproduction}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="$PROJECT_ROOT/benchmark_contract/.mplconfig"

mkdir -p "$PROJECT_ROOT/benchmark_contract/results/raw_logs"
mkdir -p "$PROJECT_ROOT/benchmark_contract/results/optional_predictions"
mkdir -p "$MPLCONFIGDIR"

python "$PROJECT_ROOT/benchmark_contract/export_results.py" --phase metadata --scenario "$SCENARIO"
python "$PROJECT_ROOT/benchmark_contract/run_inference.py" --scenario "$SCENARIO"
python "$PROJECT_ROOT/benchmark_contract/evaluate.py" --scenario "$SCENARIO"
python "$PROJECT_ROOT/benchmark_contract/profile.py" --scenario "$SCENARIO"
python "$PROJECT_ROOT/benchmark_contract/export_results.py" --phase finalize --scenario "$SCENARIO"

echo "Benchmark contract finished for scenario: $SCENARIO"
