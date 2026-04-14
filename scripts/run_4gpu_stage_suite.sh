#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_ACTIVATE="${VENV_ACTIVATE:-../links/scratch/pmpp/bin/activate}"
SUITE="${SUITE:-full}"
NUM_DEVICES="${NUM_DEVICES:-4}"
SIZE_PROFILE="${SIZE_PROFILE:-h100}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/diagnostics/4gpu_stage_suite_${TIMESTAMP}}"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "[ERROR] Virtualenv activate script not found: $VENV_ACTIVATE" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

source "$VENV_ACTIVATE"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

echo "[INFO] repo_root=$REPO_ROOT"
echo "[INFO] run_dir=$RUN_DIR"
echo "[INFO] suite=$SUITE"
echo "[INFO] num_devices=$NUM_DEVICES"
echo "[INFO] size_profile=$SIZE_PROFILE"
echo "[INFO] python=$(command -v python)"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"

python scripts/run_4gpu_stage_suite.py \
  --suite "$SUITE" \
  --num-devices "$NUM_DEVICES" \
  --size-profile "$SIZE_PROFILE" \
  --python "$(command -v python)" \
  --output-dir "$RUN_DIR" |& tee "$RUN_DIR/driver.log"

tar -czf "${RUN_DIR}.tar.gz" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"

echo "[INFO] summary_json=$RUN_DIR/summary.json"
echo "[INFO] archive=${RUN_DIR}.tar.gz"
