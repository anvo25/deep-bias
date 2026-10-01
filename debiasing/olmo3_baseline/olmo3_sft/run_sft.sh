#!/usr/bin/env bash
# SFT Olmo-3-1025-7B on Dolci-Instruct-SFT (Ai2 tokenization) -> the Olmo3 baseline.
# Run `bash setup.sh` first, then: bash olmo3_sft/run_sft.sh
#
# Optional environment variables:
#   NUM_GPUS      GPUs on this node (default 2, the original run)
#   GPU_CLUSTER   only used to pick the GPU type for the micro-batch size:
#                 ai2/titan = B200 (default, the original run), ai2/jupiter = H100 80GB
#   RUN_NAME      run / checkpoint folder name
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u; source "$HERE/../env.sh"; set -u  # conda activate is not nounset-safe

RUN_NAME="${RUN_NAME:-olmo3-7b-instruct-sft-repro-ai2data}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_CLUSTER="${GPU_CLUSTER:-ai2/titan}"
WORK="$ROOT/repro/$RUN_NAME"

WANDB_ARGS=(--trainer.callbacks.wandb.enabled=False)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  WANDB_ARGS=(
    --trainer.callbacks.wandb.enabled=True
    --trainer.callbacks.wandb.entity="${WANDB_ENTITY}"
    --trainer.callbacks.wandb.project=my-olmo3-7b-sft
  )
fi

mkdir -p "$WORK/checkpoints"
export OLMO_SFT_ROOT_DIR="$WORK"

cd "$OLMO_CORE"
export PYTHONPATH="$OLMO_CORE/src:${PYTHONPATH:-}"

torchrun --nproc-per-node="$NUM_GPUS" \
  "$OLMO_CORE/src/scripts/train/sft/Olmo-3-7B-SFT.py" train \
  "$RUN_NAME" \
  "$BASE_CORE/model_and_optim" \
  "$GPU_CLUSTER" \
  --seq_len=32768 \
  --num_nodes=1 \
  --gpus_per_node="$NUM_GPUS" \
  --global_batch_size=1048576 \
  --dataset_path="$SFT_DATA" \
  --init_seed=33333 \
  --train_module.optim.lr=8e-5 \
  --trainer.max_duration.value=2 \
  --trainer.save_folder="$WORK/checkpoints/$RUN_NAME" \
  "${WANDB_ARGS[@]}" \
  2>&1 | tee "$WORK/train.log"
