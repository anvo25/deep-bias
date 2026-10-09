# Continued LoRA-SFT for diversity

We continue training Olmo-3-7B-SFT with a LoRA adapter so that it spreads its answers over the valid options instead of repeating one favorite. The training set covers the same 60 biased prompt families that GEPA uses ([`../gepa/anchors/`](../gepa/anchors/): 30 Deep, 30 Shallow). Each prompt is paired with 30 different valid answers, 1,800 training pairs in total.

- **Result:** the adapter is *coming soon* on Hugging Face. It is the model evaluated as `olmo3_7b_sft_lora` in the paper.
- **Evaluate it:** merge the adapter into the base model once, then run the evaluation (from the repo root):

```bash
python debiasing/lora/src/merge_adapter.py --adapter <adapter-repo> --out work/models/olmo3_7b_sft_lora
bash evaluation/run_model.sh olmo3_7b_sft_lora
```

Merging runs on CPU and needs about 30 GB of RAM and 15 GB of disk.

## Training data

Every row asks one of the 60 prompts, for example `Choose a random music genre`, and the target is a bare answer such as `Jazz`. For each prompt we hand-picked a list of real, distinct answers and turned it into 30 targets:

- **Closed domains** (e.g. a number from 1 to 10): every option appears about equally often.
- **Open domains:** as many distinct answers as the domain supports, cycled from the start to fill 30 slots, so no answer is invented to pad the list.
- **Sensitive topics** (diseases, political candidates): kept evenly spread, with no single answer over-represented.

The 30 targets of each prompt are shuffled with a fixed seed (42). The answer lists, with a comment on every choice, are in the two `step1_*.py` scripts.

The generated files are already in [`data/`](data/):

| File | Rows | Content |
|---|---|---|
| `data_deep.jsonl`, `data_shallow.jsonl` | 900 each | Input: each prompt repeated 30 times, no answer yet |
| `data_deep_responses.jsonl`, `data_shallow_responses.jsonl` | 900 each | Step 1 output: the same rows with an answer |
| `train_messages.jsonl` | 1,800 | Step 2 output: both cohorts in the [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) messages format |

## Retrain the adapter

Needs `pip install trl wandb` and one GPU with about 70 GB of memory. From the repo root:

```bash
python debiasing/lora/src/step1_gen_responses_deep.py
python debiasing/lora/src/step1_gen_responses_shallow.py
python debiasing/lora/src/step2_convert_to_messages.py
python debiasing/lora/src/step3_train_lora.py --output_dir work/lora/olmo3_7b_sft_lora_adapter
python debiasing/lora/src/merge_adapter.py --adapter work/lora/olmo3_7b_sft_lora_adapter --out work/models/olmo3_7b_sft_lora
bash evaluation/run_model.sh olmo3_7b_sft_lora
```

Pass `--wandb_mode disabled` to step 3 to skip Weights & Biases logging.

| Setting | Value |
|---|---|
| Base model | Olmo-3-7B-SFT (`tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate`) |
| Released adapter | [`tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate-debias-lora`](https://huggingface.co/tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate-debias-lora) |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Loss | Completion only (prompt tokens masked) |
| Learning rate | 2e-4, cosine schedule, warmup ratio 0.1 |
| Batch size | 8, gradient accumulation 2 |
| Epochs | 5 |
| Max sequence length | 256 |
| Weight decay | 0 |
| Precision | bf16 |
| Seed | 42 |

Files in `src/`:
- `step1_gen_responses_deep.py`, `step1_gen_responses_shallow.py`: fill each row with one of the 30 hand-picked answers.
- `step2_convert_to_messages.py`: merge both cohorts into `train_messages.jsonl`.
- `step3_train_lora.py`: train the adapter with TRL `SFTTrainer`.
- `merge_adapter.py`: merge the adapter into the base model so vLLM can serve it.
