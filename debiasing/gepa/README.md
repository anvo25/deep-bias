# GEPA system prompt (Section 4.4)

We use [GEPA](https://github.com/gepa-ai/gepa) (through [DSPy](https://dspy.ai)) to search for a system prompt that makes Olmo-3-7B-SFT spread its answers over the valid options instead of repeating one favorite.

- **Result:** [`optimized_system_prompt.txt`](optimized_system_prompt.txt). This is the exact prompt evaluated in the paper, as GEPA wrote it.
- **Evaluate it:** `bash evaluation/run_model.sh olmo3_7b_sft_gepa` (from the repo root).

## Rerun the optimization

```bash
pip install dspy==3.2.1
vllm serve tuongvy2603/BITD_baseline --port 8000 \
    --max-model-len 4096 --max-num-seqs 512 --enable-prefix-caching
export OPENAI_API_KEY=...
python debiasing/gepa/run_gepa.py --out-dir work/gepa
```

| Setting | Value |
|---|---|
| Student | Olmo-3-7B-SFT through vLLM, temperature 0.6, at most 25 tokens |
| Reflection LM | `openai/gpt-5.6-sol`, temperature 1.0, reasoning effort medium |
| Train / validation prompts | [`anchors/`](anchors/): 30 Deep, 30 Shallow and 30 Non-bias prompt families of Olmo-3-7B-SFT |
| Budget | DSPy `auto="medium"` (the paper run used 20 iterations and 1,746 metric calls) |
| Seed prompt and feedback | [`gepa_debias/prompts/`](gepa_debias/prompts/) |

After the search, the script also checks the new prompt on 100 held-out prompt families. Both the student and the reflection LM sample at temperature above 0, so a rerun finds a similar but not identical prompt.

Files:
- `run_gepa.py`: entry point.
- `gepa_debias/`: data loading, metric, and GEPA runner. It is named `gepa_debias` so it does not shadow the `gepa` package.
