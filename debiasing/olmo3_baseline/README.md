# Olmo-3-7B-SFT baseline

This folder rebuilds the Olmo-3-7B-SFT model used in the paper: [`allenai/Olmo-3-1025-7B`](https://huggingface.co/allenai/Olmo-3-1025-7B) fine-tuned on [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) with Ai2's SFT recipe.

You do not need it to use the model. The trained model is released as [`tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate`](https://huggingface.co/tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate), and that is the checkpoint behind the paper's numbers. A rebuild gives a close but not identical model.

## Requirements

- Linux and NVIDIA GPUs with a CUDA 12.8 driver (the original run used 2 × B200)
- About 100 GB of free disk
- A Hugging Face token

## Run

From this folder:

```bash
cp env.example.sh env.sh     # put your HF_TOKEN in env.sh
source env.sh

bash setup.sh                # install the env, download the base model, tokenize the data (several hours)
source env.sh

bash olmo3_sft/run_sft.sh                            # train
bash olmo3_sft/convert_listed_checkpoints_to_hf.sh   # convert to a Hugging Face model
```

The model is saved in `olmo3_sft/repro/olmo3-7b-instruct-sft-repro-ai2data/checkpoints/olmo3-7b-instruct-sft-repro-ai2data/step<N>-hf`.

On other GPUs, set the GPU count and type. For example, on 8 × H100:

```bash
NUM_GPUS=8 GPU_CLUSTER=ai2/jupiter bash olmo3_sft/run_sft.sh   # ai2/titan = B200 (default), ai2/jupiter = H100
```

## Training settings

| Setting | Value |
|---|---|
| Sequence length | 32,768 |
| Global batch size | about 1M tokens |
| Learning rate | 8e-5, linear decay, 3% warmup |
| Epochs | 2 |
| Seed | 33333 |

## Files

- `env.example.sh`: paths and tokens
- `setup.sh`: environment, base model and data
- `olmo3_sft/run_sft.sh`: training
- `olmo3_sft/convert_listed_checkpoints_to_hf.sh`: conversion to Hugging Face format
- `olmo3_sft/src/`: Ai2's [OLMo-core](https://github.com/allenai/OLMo-core) and [open-instruct](https://github.com/allenai/open-instruct). The training recipe is lightly modified to run outside Ai2's cluster.
