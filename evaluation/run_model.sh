#!/usr/bin/env bash
# Evaluate one model end to end: sample direct and framed answers, cluster
# them, and write results in the released layout under work/.
#
# Usage:
#   bash evaluation/run_model.sh <config> [--limit N]
#
#   <config> is a name in evaluation/configs/ (for example olmo3_7b_sft) or a
#   path to your own .env file. See evaluation/configs/*.env for the fields.
#
# Local models are served with vLLM on GPU $GPU (default 0) and port $PORT
# (default 8000); the server is stopped when the run ends. To use a server
# you already started, set SERVER_URL=http://localhost:8000/v1 instead. API models need
# OPENAI_API_KEY or OPENROUTER_API_KEY in the environment.
#
# Results:
#   work/outputs/<config>.jsonl  and  work/raw/<config>/{direct,framed}.jsonl.gz
# then, for example:
#   python analysis/reproduce_numbers.py --data work
#
# Sampling is resumable. If a run stops, run the same command again.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CFG="${1:?usage: bash evaluation/run_model.sh <config> [--limit N]}"
shift
LIMIT_ARGS=()
if [[ "${1:-}" == "--limit" ]]; then LIMIT_ARGS=(--limit "${2:?--limit needs a number}"); fi

if [[ -f "$CFG" ]]; then CFG_FILE="$CFG"; NAME="$(basename "$CFG" .env)";
else CFG_FILE="evaluation/configs/$CFG.env"; NAME="$CFG"; fi
[[ -f "$CFG_FILE" ]] || { echo "no config $CFG_FILE"; exit 1; }

# defaults (the paper's settings), then the config overrides
SERVE=vllm; PRETRAINED=false; SYSTEM_PROMPT_FILE=""; REASONING_EFFORT=""
TEMPERATURE=0.6; MAX_TOKENS=25; CONCURRENCY=256; BASE_URL=""
# shellcheck disable=SC1090
source "$CFG_FILE"

PY="${PY:-python}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
WORK="${WORK:-work}"
JUDGE_DEVICE="${JUDGE_DEVICE:-}"   # empty = cuda if available, else cpu

# 1) the released prompts and framings, fetched once into work/
if [[ ! -f "$WORK/prompts/train.jsonl" || ! -f "$WORK/framings/train.jsonl" ]]; then
  echo "[run_model] downloading prompts and framings from Hugging Face"
  "$PY" - "$WORK" <<'PYEOF'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
work = Path(sys.argv[1])
for rel in ("prompts/train.jsonl", "framings/train.jsonl"):
    (work / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download("anvo25/deep-bias", rel, repo_type="dataset"), work / rel)
PYEOF
fi

# 2) serve the model with vLLM, unless it is an API model
SERVE_PID=""
cleanup() {
  if [[ -n "$SERVE_PID" ]]; then
    echo "[run_model] stopping vLLM (pid $SERVE_PID)"
    pkill -TERM -P "$SERVE_PID" 2>/dev/null || true
    kill -TERM "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -n "${SERVER_URL:-}" ]]; then
  BASE_URL="$SERVER_URL"
  echo "[run_model] using the running server at $BASE_URL"
elif [[ "$SERVE" == "vllm" ]]; then
  BASE_URL="http://localhost:$PORT/v1"
  mkdir -p "$WORK/logs"
  echo "[run_model] serving $MODEL on GPU $GPU, port $PORT (log: $WORK/logs/vllm_$NAME.log)"
  CUDA_VISIBLE_DEVICES="$GPU" vllm serve "$MODEL" --port "$PORT" \
      --gpu-memory-utilization "${GPU_MEM:-0.90}" --max-model-len 4096 \
      --max-num-seqs 512 --enable-prefix-caching > "$WORK/logs/vllm_$NAME.log" 2>&1 &
  SERVE_PID=$!
  for _ in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SERVE_PID" 2>/dev/null || { echo "vLLM exited, see $WORK/logs/vllm_$NAME.log"; exit 1; }
    sleep 10
  done
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null || { echo "vLLM did not come up in 30 min"; exit 1; }
fi

COMMON=(--target-model "$MODEL" --target-base-url "$BASE_URL"
        --temperature "$TEMPERATURE" --max-tokens "$MAX_TOKENS" --concurrency "$CONCURRENCY")
[[ "$PRETRAINED" == "true" ]] && COMMON+=(--pretrained)
[[ -n "$REASONING_EFFORT" ]] && COMMON+=(--reasoning-effort "$REASONING_EFFORT")
[[ -n "$SYSTEM_PROMPT_FILE" ]] && COMMON+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")

S="$WORK/samples/$NAME"
mkdir -p "$S" "$WORK/clustered"

echo "[run_model] step 1/4: direct samples (30 per prompt family)"
"$PY" evaluation/sample_direct.py --pool "$WORK/prompts/train.jsonl" --out "$S/direct.jsonl" \
    "${COMMON[@]}" "${LIMIT_ARGS[@]}"

echo "[run_model] step 2/4: framed samples (1 per reframing, 30 per family)"
FRAMINGS="$WORK/framings/train.jsonl"
if [[ ${#LIMIT_ARGS[@]} -gt 0 ]]; then
  # keep the framings of the same families that step 1 sampled
  IDS="$("$PY" -c "import json,sys; print(','.join(sorted({json.loads(l)['entry_id'] for l in open(sys.argv[1])})))" "$S/direct.jsonl")"
  FRAMED_LIMIT=(--ids "$IDS")
else
  FRAMED_LIMIT=()
fi
"$PY" evaluation/sample_framed.py --in "$FRAMINGS" --out "$S/framed.jsonl" \
    "${COMMON[@]}" "${FRAMED_LIMIT[@]}"

cleanup; SERVE_PID=""   # free the GPU before clustering

echo "[run_model] step 3/4: cluster answers"
DEV_ARGS=(); [[ -n "$JUDGE_DEVICE" ]] && DEV_ARGS=(--device "$JUDGE_DEVICE")
"$PY" evaluation/cluster.py --responses-raw "$S/direct.jsonl" --framing-raw "$S/framed.jsonl" \
    --out "$WORK/clustered/$NAME.jsonl" "${DEV_ARGS[@]}"

echo "[run_model] step 4/4: write results"
"$PY" evaluation/finalize.py --name "$NAME" --model "$MODEL" --prompts "$WORK/prompts/train.jsonl" \
    --clustered "$WORK/clustered/$NAME.jsonl" --direct "$S/direct.jsonl" --framed "$S/framed.jsonl" \
    --work "$WORK"
