#!/usr/bin/env bash
# Convert the latest OLMo-core checkpoint of each run to a HuggingFace model
# (written next to it as step<N>-hf). By default converts the baseline run.
# Usage: bash olmo3_sft/convert_listed_checkpoints_to_hf.sh [run_dir ...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u; source "$HERE/../env.sh"; set -u  # conda activate is not nounset-safe

CONVERTER="$OLMO_CORE/src/examples/huggingface/convert_checkpoint_to_hf.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONVERT_DEVICE="${CONVERT_DEVICE:-cuda}"
CONVERT_CUDA_VISIBLE_DEVICES="${CONVERT_CUDA_VISIBLE_DEVICES:-0}"
VALIDATE="${VALIDATE:-0}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-65536}"

if [[ $# -gt 0 ]]; then
  RUN_DIRS=("$@")
else
  RUN_DIRS=("$ROOT/repro/olmo3-7b-instruct-sft-repro-ai2data")
fi

convert_one() {
  local run_dir="$1"
  local run_name
  local checkpoint_root
  local latest_step
  local output_dir
  local cmd

  run_name="$(basename "$run_dir")"
  checkpoint_root="$run_dir/checkpoints/$run_name"

  if [[ ! -d "$checkpoint_root" ]]; then
    echo "Skipping $run_name: missing checkpoint root $checkpoint_root"
    return
  fi

  latest_step="$(
    find "$checkpoint_root" -maxdepth 1 -mindepth 1 -type d \
      -name 'step[0-9]*' ! -name '*-hf' | sort -V | tail -n 1
  )"

  if [[ -z "$latest_step" ]]; then
    echo "Skipping $run_name: no numeric step directories found under $checkpoint_root"
    return
  fi

  output_dir="${latest_step}-hf"

  if [[ -d "$output_dir" ]]; then
    echo "Skipping $run_name: HF export already exists at $output_dir"
    return
  fi

  echo
  echo "Converting $run_name"
  echo "  input : $latest_step"
  echo "  output: $output_dir"
  echo "  device: $CONVERT_DEVICE"

  cmd=(
    "$PYTHON_BIN" "$CONVERTER"
    -i "$latest_step"
    -o "$output_dir"
    -s "$MAX_SEQUENCE_LENGTH"
    --device "$CONVERT_DEVICE"
  )

  if [[ "$VALIDATE" != "1" ]]; then
    cmd+=(--skip-validation)
  fi

  if [[ "$CONVERT_DEVICE" == "cuda" ]]; then
    CUDA_VISIBLE_DEVICES="$CONVERT_CUDA_VISIBLE_DEVICES" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

for run_dir in "${RUN_DIRS[@]}"; do
  convert_one "$run_dir"
done
