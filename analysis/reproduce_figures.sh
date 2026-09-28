#!/usr/bin/env bash
# Redraw every figure in the paper from the released outputs (CPU only).
#
#   bash analysis/reproduce_figures.sh                 # data from Hugging Face
#   bash analysis/reproduce_figures.sh --data ./hf     # a local copy
#
# Arguments are passed to every script, so --out DIR works too
# (default: figures/).
set -euo pipefail
cd "$(dirname "$0")/.."

for script in \
    fig1_distributions \
    fig2_bias_types \
    fig3_training_stages \
    fig4_pretrained_vs_sft \
    fig5_case_sft_vs_lora \
    appendix_cases \
    threshold_sensitivity; do
  echo "== analysis/${script}.py"
  MPLBACKEND=Agg python "analysis/${script}.py" "$@"
done
