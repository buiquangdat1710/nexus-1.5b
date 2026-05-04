<div align="center">

# Nexus-1.5B

### Length-Penalized Reward Optimization for Mathematical Reasoning

[![Model](https://img.shields.io/badge/🤗%20HuggingFace-Nexus--1.5B-blue?style=flat-square)](https://huggingface.co/Dat1710/nexus-1.5b)
[![Base](https://img.shields.io/badge/Base-Qwen2.5--Math--1.5B--Instruct-orange?style=flat-square)](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![MATH](https://img.shields.io/badge/MATH%20500-80.2%25-blueviolet?style=flat-square)]()
[![TIR](https://img.shields.io/badge/MATH%20%2B%20TIR-84%25-red?style=flat-square)]()

</div>

---

## Table of Contents

- [Overview](#overview)
- [Method: LPRO](#method-lpro)
- [Project Structure](#project-structure)
- [Training Configuration](#training-configuration)
- [Benchmark Results](#benchmark-results)
- [Quick Start](#quick-start)
- [Evaluation Framework](#evaluation-framework)
- [Web Demo Application](#web-demo-application)
- [Citation](#citation)
- [License](#license)

---

## Overview

**Nexus-1.5B** is a 1.54B-parameter language model fine-tuned for mathematical reasoning on top of [Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct).

The model is trained with **Length-Penalized Reward Optimization (LPRO)** — a reinforcement learning technique that combines:
- **Group-wise rank advantages**
- **Token-level clipped importance sampling**
- **Explicit penalty on response verbosity**

The training pipeline flexibly supports two modes:
- **Full Fine-Tuning**: For high-end multi-GPU systems (A100/H100)
- **LoRA Adapters**: For resource-constrained environments (Google Colab, T4/L4/A10G)

### Key Results

| Benchmark | Base Instruct | **Nexus-1.5B** | Δ |
|-----------|:---:|:---:|:---:|
| MATH-500 | 75.8 | **80.2** | +4.4 |
| MMLU STEM | 57.5 | **60.3** | +2.8 |
| QA Chinese | 54.1 | **56.9** | +2.8 |
| MATH + TIR | 80.0 | **84.0** | +4.0 |

---

## Method: LPRO

LPRO introduces two core modifications over vanilla GRPO (Group Relative Policy Optimization) to prevent reward hacking and encourage concise reasoning.

### 1 · Length-Penalized Advantage

Instead of normalizing rewards with a plain z-score, the advantage for output $i$ in a group is penalized based on its length:

$$
A_i = \underbrace{\frac{r_i - \mu_r}{\sigma_r + \varepsilon_r}}_{\text{Correctness signal}} \;-\; \lambda \cdot \underbrace{\frac{L_i - \mu_L}{\sigma_L + \varepsilon_L}}_{\text{Length penalty}}
$$

Where:
- $r_i$: reward for the $i$-th response
- $L_i$: response length
- $\lambda$: verbosity penalty coefficient (default: 0.1)
- $\varepsilon_r, \varepsilon_L$: numerical stability constants (default: 1e-8)

**Intuition:** The model is penalized for producing unnecessarily long chains-of-thought while still being rewarded for correctness.

### 2 · DAPO Objective with Asymmetric Clipping

The policy update uses a token-level surrogate loss with asymmetric clip bounds $(\varepsilon_\text{low}, \varepsilon_\text{high})$:

$$
\mathcal{J}_\text{LPRO}(\theta) = \mathbb{E}\!\left[\frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \min\!\Big(r_{i,t}(\theta)\,\hat{A}_{i,t},\;\text{clip}\big(r_{i,t}(\theta),\,1-\varepsilon_\text{low},\,1+\varepsilon_\text{high}\big)\hat{A}_{i,t}\Big)\right]
$$

Key design choices:
- **Asymmetric clip**: $\varepsilon_\text{high} \geq \varepsilon_\text{low}$ allows larger policy updates in the positive direction, improving sample efficiency.
- **Token-level normalization**: Dividing by $\sum|o_i|$ rather than $G$ ensures that shorter, correct responses are not under-weighted relative to longer ones.

---

## Project Structure

```
nexus-1.5b/
├── train.py                    # Training entry point
├── config.py                   # Configuration & CLI argument parsing
├── eval.py                     # Evaluation entry point
├── app.py                      # Streamlit frontend (Math Solver AI)
├── server.py                   # FastAPI backend + ngrok tunnel (for Google Colab)
├── requirements.txt            # Dependencies
│
├── nexus/                      # Core package
│   ├── nexus_trainer/
│   │   └── trainer.py          # NexusTrainer class
│   ├── rl/
│   │   ├── advantages.py       # LPRO advantage computation
│   │   └── loss.py             # DAPO token-level loss
│   ├── models/
│   │   ├── policy.py           # Policy model loader
│   │   └── reward.py           # Reward model (Neural RM + Rule-based scorer)
│   └── data/
│       └── builder.py          # Dataset builder & prompt formatting
│
└── evaluation/                 # Evaluation framework
    ├── evaluator.py            # MathEvaluator
    ├── metrics.py              # Metric functions
    └── tasks/                  # Benchmark tasks
        ├── base_task.py        # Abstract base class
        ├── math_task.py        # MATH-500
        ├── gsm8k_task.py       # GSM8K
        ├── mmlu_stem_task.py   # MMLU STEM
        ├── cmath_task.py       # CMATH (Chinese)
        ├── gaokao_cloze_task.py # GaoKao Math Cloze
        ├── gaokao_qa_task.py   # GaoKao QA (Chinese)
        ├── minerva_math_task.py # Minerva Math (STEM)
        ├── gaokao_2023_en_task.py # GaoKao 2023 EN
        ├── olympiad_bench_task.py # OlympiadBench
        └── college_math_task.py # College Math (TheoremQA)
```
---

## Training Configuration

### Option A: High-End Hardware (Full Fine-Tuning + Neural RM)

Best for clusters with A100/H100 GPUs.

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Math-1.5B-Instruct` |
| Reward model | `Qwen/Qwen2.5-Math-RM-72B` |
| Mode | Full Fine-Tuning (`use_lora=False`) |
| Reward | Neural RM (`use_rule_based_rm=False`) |
| Dataset | MATH-500 |
| Group size $G$ | 32 |
| Learning rate | 5e-7 ~ 1e-6 |
| $\varepsilon_\text{low}$ / $\varepsilon_\text{high}$ | 0.20 / 0.28 |
| $\lambda$ (length penalty) | 0.1 |
| Max generation tokens | 2048 |
| Gradient accumulation | 8 |

### Option B: Resource-Constrained / Google Colab (LoRA + Rule-Based RM)

Best for single GPU T4/L4/A10G (16GB–24GB VRAM).

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Math-1.5B-Instruct` |
| Mode | LoRA Adapter (`use_lora=True`) |
| Reward | Rule-Based Scorer (`use_rule_based_rm=True`) |
| LoRA rank / alpha | 16 / 32 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Group size $G$ | 4 |
| Learning rate | 1e-4 ~ 2e-4 |
| Max generation tokens | 2048 |

---

## Benchmark Results

### Standard Benchmarks (Chain-of-Thought)

| Model | GSM8K | MATH | MMLU STEM | CMATH | GaoKao Cloze | QA (zh) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-3.1-8B | 56.7 | 20.3 | 53.1 | 51.5 | 8.5 | 28.5 |
| Llama-3.1-70B | 85.5 | 41.4 | 78.1 | 75.5 | 11.9 | 43.3 |
| Qwen2-7B | 79.9 | 44.2 | 67.6 | 76.7 | 37.3 | 51.6 |
| DeepSeekMath-7B | 64.2 | 36.2 | 56.5 | 71.7 | 20.3 | 40.7 |
| Qwen2-Math-1.5B-Instruct | 84.2 | 69.4 | 54.9 | 79.6 | 59.7 | 50.7 |
| Qwen2.5-Math-1.5B-Instruct | 84.8 | 75.8 | 57.5 | 83.0 | 65.5 | 54.1 |
| **Nexus-1.5B (Ours)** | **85.2** | **80.2** | **60.3** | **83.5** | 49.2 | **56.9** |


### Tool-Integrated Reasoning (TIR)

| Benchmark | Instruct (CoT) | Nexus (CoT) | Instruct (TIR) | **Nexus (TIR)** |
|---|:---:|:---:|:---:|:---:|
| MATH-500 | 78 | 83 | 80 | **84** |
| Minerva Math | 30 | 33 | 33 | **40** |
| GaoKao 2023 EN | 66 | 72 | 69 | **73** |
| Olympiad Bench | 40 | 52 | 41 | **53** |
| College Math | 49 | 54 | 50 | **55** |

Nexus-1.5B + TIR achieves **84% on MATH-500** and **53% on Olympiad Bench** (+12pp over the instruct TIR baseline).

---

## Quick Start

### Installation

```bash
git clone https://huggingface.co/Dat1710/nexus-1.5b
cd nexus-1.5b
pip install -r requirements.txt
```

### Inference: Full Fine-Tuned Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Dat1710/Nexus-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

SYSTEM = (
    "You are a mathematics expert. "
    "Solve the problem step by step and enclose your final answer in \\boxed{}."
)

def solve(problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, temperature=0.0, do_sample=False)

    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(solve("If $x^2 - 5x + 6 = 0$, find the value of $x^2 + \\frac{1}{x^2}$."))
```

### Inference: LoRA Adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
adapter_id = "Dat1710/Nexus-1.5B-LoRA"  # Replace with your adapter path

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_id)
```

### Tool-Integrated Reasoning (TIR)

For TIR evaluation with a Python interpreter, use the official [Qwen2.5-Math evaluation pipeline](https://github.com/QwenLM/Qwen2.5-Math).

### Reproducing Training

**LoRA + Rule-Based RM (single GPU):**

```bash
export HF_TOKEN="hf_your_token_here"

python train.py \
  --model_name   "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --hub_repo_id  "your-username/nexus-1.5b"   \
  --G            4      \
  --lambda_len   0.10   \
  --eps_low      0.20   \
  --eps_high     0.28   \
  --lr           2e-4   \
  --num_epochs   3
```

**Full Fine-Tuning + Neural RM (multi-GPU):**

```bash
python source.py \
  --model_name   "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --hub_repo_id  "your-username/nexus-1.5b-lpro"   \
  --G            32      \
  --lambda_len   0.10   \
  --eps_low      0.20   \
  --eps_high     0.28   \
  --lr           5e-7   \
  --num_epochs   5
```

---

## Evaluation Framework

The evaluation framework supports **10 math benchmarks**:

| Task | Dataset | Language | Format |
|---|---|---|---|
| `math` | MATH-500 | English | Open-ended |
| `gsm8k` | GSM8K | English | Open-ended |
| `mmlu_stem` | MMLU STEM | English | Multiple-choice |
| `cmath` | CMATH | Chinese | Open-ended |
| `gaokao_cloze` | GaoKao Math Cloze | Chinese | Fill-in-the-blank |
| `gaokao_qa` | GaoKao QA | Chinese | Open-ended |
| `minerva_math` | Minerva Math (`math-ai/minervamath`) | English | Open-ended (STEM) |
| `gaokao_2023_en` | GaoKao 2023 EN (`MARIO-Math-Reasoning/Gaokao2023-Math-En`) | English | Open-ended |
| `olympiad_bench` | OlympiadBench (`Hothan/OlympiadBench`) | English | Open-ended (Olympiad) |
| `college_math` | College Math (`TIGER-Lab/TheoremQA`) | English | Open-ended (College) |

### Running Evaluation

```bash
python eval.py \
  --model_path "Dat1710/Nexus-1.5B" \
  --task math \
  --tp 1 \
  --temperature 0.0
```

Supported tasks: `math`, `gsm8k`, `mmlu_stem`, `cmath`, `gaokao_cloze`, `gaokao_qa`, `minerva_math`, `gaokao_2023_en`, `olympiad_bench`, `college_math`.

---

## Web Demo Application

This project includes a **Math Solver AI** web application with a client-server architecture:

### Backend (Google Colab)

`server.py` implements a FastAPI server with:
- `/generate` endpoint: Generates math solutions
- `/health` endpoint: Health check
- Supports both **Chain-of-Thought (CoT)** and **Tool-Integrated Reasoning (TIR)**
- Built-in ngrok tunnel for exposing from Colab

### Frontend (Local)

`app.py` is a Streamlit application featuring:
- Chat interface with full LaTeX rendering support
- Configurable parameters: temperature, top-p, max tokens
- Reasoning method selection (CoT / TIR)
- Custom system message
- Example math problems

### Getting Started

```bash
# 1. Run the server on Google Colab (server.py)
# Visit https://dashboard.ngrok.com/get-started/your-authtoken to get your NGROK_AUTH_TOKEN
# 2. Copy the ngrok URL
# 3. Run the frontend locally
streamlit run app.py
# 4. Paste the ngrok URL into the sidebar
```

---

## Citation

If you use Nexus-1.5B or this training pipeline in your research, please cite:

```bibtex
@misc{nexus1.5b2025,
  title  = {Nexus-1.5B: Length-Penalized Reward Optimization for Mathematical Reasoning},
  author = {Dat1710},
  year   = {2025},
  url    = {https://huggingface.co/Dat1710/nexus-1.5b}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
The base model [Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) is subject to its own license terms.
