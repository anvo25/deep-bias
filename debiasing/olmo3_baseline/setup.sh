#!/usr/bin/env bash
# Rebuild everything that is NOT stored in git: miniforge + the olmo3 conda env,
# the base model, and the tokenized SFT data.
# Usage: bash setup.sh [envs|model|data|all]   (default: all)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$REPO_DIR/olmo3_sft
CONDA_DIR=$ROOT/miniforge
STEP="${1:-all}"
export HF_HOME=${HF_HOME:-$ROOT/cache/huggingface}

activate_olmo3() {
  set +u  # conda activate is not nounset-safe
  source "$CONDA_DIR/etc/profile.d/conda.sh"
  conda activate olmo3
  set -u
}

setup_envs() {
  if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O "$ROOT/miniforge.sh"
    bash "$ROOT/miniforge.sh" -b -p "$CONDA_DIR"
  fi
  set +u
  source "$CONDA_DIR/etc/profile.d/conda.sh"

  # olmo3: training (OLMo-core + open-instruct), CUDA 12.8
  conda env list | grep -q "^olmo3 " || conda create -n olmo3 python=3.12 -y
  conda activate olmo3
  pip install --upgrade pip setuptools wheel packaging ninja
  pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
  pip install -e "$ROOT/src/OLMo-core[all]"
  pip install -e "$ROOT/src/open-instruct"
  pip install psutil
  pip install flash_attn==2.8.3 --no-build-isolation
  conda deactivate
  set -u
}

setup_model() {
  activate_olmo3
  mkdir -p "$ROOT/artifacts"

  # HF base model (needs HF_TOKEN exported)
  hf download allenai/Olmo-3-1025-7B --local-dir "$ROOT/artifacts/Olmo-3-1025-7B-hf"

  # Convert to OLMo-core format used by the SFT script ($BASE_CORE/model_and_optim)
  python "$ROOT/src/OLMo-core/src/examples/huggingface/convert_checkpoint_from_hf.py" \
    -i "$ROOT/artifacts/Olmo-3-1025-7B-hf" \
    -m olmo3_7b \
    -t dolma2 \
    -o "$ROOT/artifacts/olmo3-base-core"
}

setup_data() {
  activate_olmo3
  mkdir -p "$ROOT/artifacts"

  # Tokenize allenai/Dolci-Instruct-SFT with Ai2's recipe
  # (open-instruct/scripts/slurm/sft/prepare_dolci_instruct_data.sh). Resumable.
  cd "$ROOT/src/open-instruct"
  python scripts/data/convert_sft_data_for_olmocore.py \
    --tokenizer_name_or_path allenai/Olmo-3-7B-Instruct-SFT \
    --dataset_mixer_list allenai/Dolci-Instruct-SFT 1.0 \
    --output_dir "$ROOT/artifacts/dolci-instruct-sft-olmocore-ai2" \
    --max_seq_length 32768 \
    --visualize True \
    --resume \
    --checkpoint_interval 500000
}

case "$STEP" in
  envs) setup_envs ;;
  model) setup_model ;;
  data) setup_data ;;
  all) setup_envs; setup_model; setup_data ;;
  *) echo "Usage: bash setup.sh [envs|model|data|all]" >&2; exit 1 ;;
esac
