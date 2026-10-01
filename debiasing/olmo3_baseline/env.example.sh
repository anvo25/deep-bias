# Copy to env.sh (ignored by git), fill in the tokens, then: source env.sh
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ROOT=$REPO_DIR/olmo3_sft
export OLMO_CORE=$ROOT/src/OLMo-core
export OPEN_INSTRUCT=$ROOT/src/open-instruct
export HF_BASE=$ROOT/artifacts/Olmo-3-1025-7B-hf
export BASE_CORE=$ROOT/artifacts/olmo3-base-core
export SFT_DATA=$ROOT/artifacts/dolci-instruct-sft-olmocore-ai2
export OLMO_SFT_ROOT_DIR=$ROOT/outputs
export HF_HOME=${HF_HOME:-$ROOT/cache/huggingface}

# Secrets: never commit real values
export HF_TOKEN="hf_..."
export WANDB_API_KEY=""        # leave empty to train without Weights & Biases
export WANDB_ENTITY=""         # your W&B user or team

if [[ -f "$ROOT/miniforge/etc/profile.d/conda.sh" ]]; then
  source "$ROOT/miniforge/etc/profile.d/conda.sh"
  conda activate olmo3
fi
