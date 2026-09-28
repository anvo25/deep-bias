# Deep and shallow biases in language models

<div align="center">
  <p style="font-size: 20px;">by
    <a href="https://anvo.me">An Vo</a><sup>1,2*</sup>,
    <a href="https://www.linkedin.com/in/dang-thi-tuong-vy-00a357278/">Vy Tuong Dang</a><sup>3*</sup>,
    <a href="https://nkn002.github.io/">Khai-Nguyen Nguyen</a><sup>4</sup>,
    <a href="https://villacu.github.io/">Emilio Villa-Cueva</a><sup>1</sup>,<br>
    <a href="https://mbzuai.ac.ae/study/faculty/thamar-solorio/">Thamar Solorio</a><sup>1†</sup>,
    <a href="https://anhnguyen.me/research/">Anh Totti Nguyen</a><sup>5†</sup>,
    <a href="https://www.resl.kaist.ac.kr/members/director">Daeyoung Kim</a><sup>3†</sup>
  </p>
  <p>
    <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Equal advising<br>
    <sup>1</sup>MBZUAI, <sup>2</sup>University of Michigan, <sup>3</sup>KAIST, <sup>4</sup>University of Virginia, <sup>5</sup>Auburn University
  </p>

  <h3>The 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026)</h3>

[![Project Page](https://img.shields.io/badge/Project_Page-deepbias.github.io-blue.svg)](https://deepbias.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.09901-b31b1b.svg)](https://arxiv.org/abs/2609.09901)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/anvo25/deep-bias)
[![Code License](https://img.shields.io/badge/Code_License-MIT-green.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/Data_License-ODC--BY-green.svg)](https://opendatacommons.org/licenses/by/1-0/)

</div>

---

## Abstract

<p align="center">
  <img src="./figures/main_pipeline.png" alt="Deep and Shallow bias" width="90%"/>
</p>

*Large language models often repeatedly select the same answer even when many alternatives are plausible. Prior work treats this concentration as bias, but it does not distinguish stable model preferences from responses that depend on a particular prompt wording. We introduce a bias depth score that measures both how strongly a model prefers its top answer under direct prompting and whether that answer survives scenario reframing. Across 4,442 opinion prompts and four large language models, only about a quarter of the concentrated preferences survive reframing. We call these persistent cases **Deep** biases, and the remaining prompt-dependent cases **Shallow** biases. Our results show that Deep biases are more often inherited from pretraining and preserved through SFT. Under both continued fine-tuning and prompt-based debiasing for diversity, Deep biases are consistently harder to remove than Shallow biases. Bias depth therefore separates stable learned biases from prompt-wording artifacts that single-prompt metrics conflate.*

---

## 1. How it works

For every **prompt family** (for example *"Choose a random popular butterfly"*) we sample the model 30 times on the direct prompt and once on each of 30 everyday **reframings** of the same choice (for example *"A zoo exhibit adds one butterfly species. Which one?"*).

| Metric | Meaning |
|---|---|
| **DR** (direct rate) | share of the 30 direct answers that give the top answer |
| **FR** (framed rate) | share of the 30 reframings whose answer is that same direct top answer |
| **π = DR × FR** | bias depth score |

A family is **Deep** if π > 0.40, **Shallow** if it is not Deep and DR > 0.65, and **Non-bias** otherwise. Two answers count as the same when they match after lemmatization, derivational stemming, word-set containment, or embedding similarity (the cluster judge, paper Section 2.2 and [`deepbias/match.py`](deepbias/match.py)).

---

## 2. Quick start

### 2.1 Load the dataset

```python
from datasets import load_dataset

prompts  = load_dataset("anvo25/deep-bias", "prompts", split="train")       # 4,442 prompt families
framings = load_dataset("anvo25/deep-bias", "framings", split="train")      # 133,260 reframings (30 per family)
sft      = load_dataset("anvo25/deep-bias", "olmo3_7b_sft", split="train")  # results for one model

row = sft[0]
print(row["prompt"], row["direct"]["top_answer"], row["direct"]["dr"], row["framed"]["fr"], row["pi"], row["bias_type"])
```

### 2.2 Reproduce every number and figure in the paper (CPU, no API keys)

```bash
git clone https://github.com/anvo25/deep-bias.git
cd deep-bias
pip install -r requirements.txt

python analysis/reproduce_numbers.py      # 33 headline numbers, each checked against the paper
bash analysis/reproduce_figures.sh        # data figures of the paper (Figures 2 to 5, Figure 1 panels, appendix), written to figures/
```

Both scripts download the released model outputs from Hugging Face on first use. To work offline, clone the dataset repo and pass `--data <path>`.

---

## 3. Dataset details

The Hugging Face repo [`anvo25/deep-bias`](https://huggingface.co/datasets/anvo25/deep-bias) has one config per file group.

| Config | Rows | Content | Paper |
|---|---:|---|---|
| `prompts` | 4,442 | prompt families built from Dolci-Instruct-SFT | Section 3 |
| `framings` | 133,260 | 30 scenario reframings per family | Section 3 |
| `olmo3_7b_pretrained` | 4,442 | Olmo-3-1025-7B (base model) | Section 4.2, 4.3 |
| `olmo3_7b_sft` | 4,442 | Olmo-3-7B-SFT, our re-SFT on Dolci-Instruct-SFT | Sections 4.1 to 4.4 |
| `olmo3_7b_dpo` | 4,442 | Olmo-3-7B-Instruct-DPO | Section 4.2 |
| `olmo3_7b_rlvr` | 4,442 | Olmo-3-7B-Instruct (after RLVR) | Section 4.2 |
| `tulu3_8b_sft` | 4,442 | Llama-3.1-Tulu-3-8B-SFT | Section 4.1 |
| `claude_sonnet_5` | 4,440 | Claude Sonnet 5, reasoning off | Section 4.1 |
| `gpt_5_6_sol` | 4,442 | GPT-5.6 Sol, reasoning off | Section 4.1 |
| `olmo3_7b_sft_gepa` | 4,442 | Olmo-3-7B-SFT with the GEPA-optimized system prompt | Section 4.4 |
| `olmo3_7b_sft_lora` | 4,442 | Olmo-3-7B-SFT after continued LoRA-SFT for diversity | Section 4.4 |

Each model config has one record per prompt family:

```
id, prompt, model,
direct: {top_answer, dr, n_samples, n_clusters, distribution: [{answer, count, rate, surface_forms, lemma_keys}]},
framed: {top_answer, fr, n_samples, n_clusters, distribution: [...]},
pi, bias_type          # "deep", "shallow" or "non_bias"
```

Every raw model response is also in the repo, under `raw/<config>/direct.jsonl.gz` and `raw/<config>/framed.jsonl.gz` (133,260 each).

Each prompt family also keeps where it came from: `evidence` holds the Dolci-Instruct-SFT rows it was built from, and `merged_row_idxs` lists every Dolci row merged into it. The `id` is the row index in [allenai/Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) at revision `bd3c8f3a`.

---

## 4. Models

| Model | Hugging Face |
|---|---|
| Olmo-3-7B-SFT (Olmo-3-1025-7B trained on Dolci-Instruct-SFT only) | [`tuongvy2603/BITD_baseline`](https://huggingface.co/tuongvy2603/BITD_baseline) |
| Continued LoRA-SFT adapter for diversity (Section 4.4) | *coming soon* |
| GEPA-optimized system prompt (Section 4.4) | [`debiasing/gepa/optimized_system_prompt.txt`](debiasing/gepa/optimized_system_prompt.txt) |

The public Olmo-3-7B-SFT release is trained on both Dolci-Instruct and Dolci-Thinking. To study the effect of instruction data alone, we trained Olmo-3-1025-7B on Dolci-Instruct-SFT with AllenAI's [OLMo-core](https://github.com/allenai/OLMo-core) recipe. It closely matches the official no-thinking SFT checkpoint on the OLMES benchmarks (paper appendix).

---

## 5. Evaluate a model

Requires a GPU with [vLLM](https://github.com/vllm-project/vllm) for open models, or an API key for API models.

```bash
pip install -r requirements.txt vllm

# one of the paper's models (configs in evaluation/configs/)
bash evaluation/run_model.sh olmo3_7b_sft

# a quick test on 3 prompt families
bash evaluation/run_model.sh olmo3_7b_sft --limit 3
```

`run_model.sh` serves the model with vLLM, samples 30 direct answers and 30 framed answers per family (temperature 0.6, at most 25 tokens), clusters the answers, and writes `work/outputs/<config>.jsonl` in the same format as the released data. Sampling can be resumed: if a run stops, run the same command again.

**Your own model.** Copy a config and change `MODEL`:

```bash
cat > my_model.env <<'EOF'
MODEL="your-org/your-model"
SERVE=vllm            # or SERVE=api with BASE_URL=... for an OpenAI-compatible API
EOF
bash evaluation/run_model.sh my_model.env
```

Then load the results like the released ones:

```python
from deepbias.data import load_outputs
from deepbias.metrics import breakdown

rows = load_outputs("my_model", data_dir="work")
print(breakdown(rows.values()))   # percent Deep, Shallow, Non-bias
```

API models read their key from `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. A full run is 266,520 calls per model (4,442 families × 60).

The four steps can also be run one by one: [`sample_direct.py`](evaluation/sample_direct.py), [`sample_framed.py`](evaluation/sample_framed.py), [`cluster.py`](evaluation/cluster.py), [`finalize.py`](evaluation/finalize.py).

**Note on sampling.** Answers are sampled at temperature 0.6, so a rerun gives slightly different answer counts than the released outputs, while the bias types stay close. The released outputs are the ones behind the paper.

---

## 6. Debiasing (Section 4.4)

### 6.1 GEPA system prompt

The optimized prompt is [`debiasing/gepa/optimized_system_prompt.txt`](debiasing/gepa/optimized_system_prompt.txt). Evaluate it with `bash evaluation/run_model.sh olmo3_7b_sft_gepa`. To rerun the optimization, see [`debiasing/gepa/`](debiasing/gepa/).

### 6.2 Continued LoRA-SFT

The released outputs of this model are in the `olmo3_7b_sft_lora` config. The adapter, its training code and training data will be added to [`debiasing/lora/`](debiasing/lora/) soon.

---

## 7. Build the dataset from scratch

The six steps in [`dataset/`](dataset/) follow paper Section 3. Steps 4 to 6 call the OpenAI API (`OPENAI_API_KEY`).

| Step | Script | Output |
|---|---|---|
| 1. Extract | [`step1_extract.py`](dataset/step1_extract.py) | first user and assistant turn of every Dolci-Instruct-SFT row (1.94M) |
| 2. Filter | [`step2_filter.py`](dataset/step2_filter.py) | rows asking for something random, code noise removed, ranked by TF-IDF (11,859) |
| 3. Select | [`step3_select.py`](dataset/step3_select.py) | diverse candidates (6,845) |
| 4. Rewrite | [`step4_rewrite.py`](dataset/step4_rewrite.py) | one "Choose a random ..." prompt per candidate |
| 5. Deduplicate | [`step5_deduplicate.py`](dataset/step5_deduplicate.py) | embedding merge plus an LLM judge for borderline pairs (4,442) |
| 6. Reframe | [`step6_reframe.py`](dataset/step6_reframe.py) | 30 reframings per prompt family (133,260) |

```bash
python dataset/step1_extract.py
python dataset/step2_filter.py
python dataset/step3_select.py
python dataset/step4_rewrite.py
python dataset/step5_deduplicate.py
python dataset/step6_reframe.py
```

Steps 4 to 6 use an LLM, so a fresh run gives a similar but not identical set of prompt families. Use the released dataset to compare with the paper.

**Table 1 (SFT evidence).** After step 1, `python analysis/tab1_sft_evidence.py` searches Dolci-Instruct-SFT for training rows that explain a bias and prints the 5 matches reported in the paper.

---

## 8. Repository structure

```
deepbias/      shared library: answer matching, metrics, data loading, API client
dataset/       steps 1 to 6 that build the prompt families and reframings
evaluation/    sample, cluster and score a model; configs for every model in the paper
analysis/      reproduce the paper's numbers, tables and figures
debiasing/     GEPA system prompt and continued LoRA-SFT (Section 4.4)
figures/       generated figures
scripts/       maintainer script that exported the released data
```

---

## Citation

```bibtex
@inproceedings{vo2026deepbias,
  title     = {Deep and shallow biases in language models},
  author    = {Vo, An and Dang, Vy Tuong and Nguyen, Khai-Nguyen and Villa-Cueva, Emilio and Solorio, Thamar and Nguyen, Anh Totti and Kim, Daeyoung},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2609.09901}
}
```
