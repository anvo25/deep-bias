# Continued LoRA-SFT for diversity

We continue training Olmo-3-7B-SFT with a LoRA adapter on the 60 biased prompt families in [`../gepa/anchors/`](../gepa/anchors/) (30 Deep, 30 Shallow). Each prompt is paired with 30 different valid answers, 1,800 training pairs in total, so the adapter learns to spread its answers instead of repeating one.

## Use the trained adapter

*Coming soon.* Once the adapter is on Hugging Face, merge it into the base model and evaluate it:

```bash
python debiasing/lora/merge_adapter.py --adapter <adapter-repo> --out work/models/olmo3_7b_sft_lora
bash evaluation/run_model.sh olmo3_7b_sft_lora
```

## Hyperparameters

| | |
|---|---|
| Base model | `tuongvy2603/BITD_baseline` |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Learning rate | 2e-4, cosine schedule, warmup ratio 0.1 |
| Batch size | 8, gradient accumulation 2 |
| Epochs | 5 |
| Max sequence length | 256 |
| Weight decay | 0 |
| Seed | 42 |

## Training code and data

*Coming soon.* The adapter, the training script and the 1,800 training pairs will be added here.
