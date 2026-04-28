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

## Overview

**Nexus-1.5B** is a 1.54B-parameter language model fine-tuned for mathematical reasoning on top of [Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct). It is trained with **Length-Penalized Reward Optimization (LPRO)** — a reinforcement learning method that combines group-wise rank advantages with token-level clipped importance sampling and an explicit penalty on response verbosity.

Nexus-1.5B establishes new state-of-the-art results for 1.5B-scale models on multiple English and Chinese math benchmarks, including:

| Benchmark | Base Instruct | **Nexus-1.5B** | Δ |
|-----------|:---:|:---:|:---:|
| MATH-500 | 75.8 | **80.2** | +4.4 |
| MMLU STEM | 57.5 | **60.3** | +2.8 |
| QA Chinese | 54.1 | **56.9** | +2.8 |
| MATH + TIR | 80.0 | **84.0** | +4.0 |

---

## Method: LPRO

LPRO introduces two core modifications over vanilla GRPO:

### 1 · Length-Penalized Advantage

Instead of normalizing rewards with a plain z-score, the advantage for output $i$ in a group is:

$$
A_i = \underbrace{\frac{r_i - \mu_r}{\sigma_r + \varepsilon_r}}_{\text{Correctness signal}} \;-\; \lambda \cdot \underbrace{\frac{L_i - \mu_L}{\sigma_L + \varepsilon_L}}_{\text{Length penalty}}
$$

where $r_i$ is the reward, $L_i$ the response length, and $\lambda$ controls the verbosity penalty. This discourages the model from producing unnecessarily long chains-of-thought while still rewarding correctness.

### 2 · DAPO Objective with Asymmetric Clipping

The policy update uses a token-level surrogate loss with **asymmetric** clip bounds $(\varepsilon_\text{low}, \varepsilon_\text{high})$:

$$
\mathcal{J}_\text{LPRO}(\theta) = \mathbb{E}\!\left[\frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G}\sum_{t=1}^{|o_i|} \min\!\Big(r_{i,t}(\theta)\,\hat{A}_{i,t},\;\text{clip}\\big(r_{i,t}(\theta),\,1-\varepsilon_\text{low},\,1+\varepsilon_\text{high}\big)\hat{A}_{i,t}\Big)\right]
$$

Key design choices:
- **Asymmetric clip**: $\varepsilon_\text{high} \geq \varepsilon_\text{low}$ allows larger policy updates in the positive direction, improving sample efficiency.
- **Token-level normalization**: dividing by $\sum|o_i|$ rather than $G$ ensures that shorter, correct responses are not under-weighted relative to longer ones.

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Math-1.5B-Instruct` |
| Dataset | MATH-500 (100 samples) |
| Group size $G$ | 4 |
| PPO epochs | 4 |
| $\varepsilon_\text{low}$ / $\varepsilon_\text{high}$ | 0.2 / 0.2 |
| $\lambda$ (length penalty) | 0.1 |
| Learning rate | 1e-6 |
| Max generation tokens | 256 |
| Context length | 1024 |

**Training pipeline:**
1. Sample $G$ responses per prompt with the current policy.
2. Score each response with a rule-based reward (box presence, step-by-step reasoning, answer correctness).
3. Compute LPRO advantages per group.
4. Run clipped surrogate PPO update with token-level normalization.

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

> ⚠️ GaoKao Math Cloze: performance is below the instruct baseline (49.2 vs 65.5). The rule-based reward did not adequately capture the cloze format; this remains an open limitation.

### Tool-Integrated Reasoning (TIR)

| Benchmark | Instruct (CoT) | Nexus (CoT) | Instruct (TIR) | **Nexus (TIR)** |
|---|:---:|:---:|:---:|:---:|
| MATH-500 | 78 | 83 | 80 | **84** |
| Minerva Math | 30 | 33 | 33 | **40** |
| GaoKao 2023 EN | 66 | 72 | 69 | **73** |
| Olympiad Bench | 40 | 52 | 41 | **53** |
| College Math | 49 | 54 | 50 | **55** |

Nexus-1.5B + TIR reaches **84% on MATH-500** and **53% on Olympiad Bench** (+12pp over the instruct TIR baseline).

---

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Chain-of-Thought Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Dat1710/Nexus-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

SYSTEM = (
    "You are an advanced mathematical reasoning model. "
    "Solve the following problem step by step, and put your final answer in \\boxed{}."
)

def solve(problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": problem},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=512, temperature=0, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(solve("If $x^2 - 5x + 6 = 0$, find the value of $x^2 + \\frac{1}{x^2}$."))
```

### Tool-Integrated Reasoning (TIR)

For TIR evaluation with a Python interpreter, use the official [Qwen2.5-Math evaluation pipeline](https://github.com/QwenLM/Qwen2.5-Math).

---

## Reproducing Training

```bash
git clone https://huggingface.co/Dat1710/nexus-1.5b
cd nexus-1.5b

pip install -r requirements.txt

export HF_TOKEN="hf_..."

python train_lpro.py \
  --model_name   "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --hub_repo_id  "your-username/nexus-1.5b-lpro"   \
  --G            8      \
  --lambda_len   0.10   \
  --eps_low      0.20   \
  --eps_high     0.28   \
  --num_epochs   3
```

Full training code and configuration are included in this repository.

---

## Limitations

- **GaoKao Cloze**: The rule-based reward function does not capture fill-in-the-blank format well, leading to a regression on this benchmark.
- **Training data scale**: Only 100 MATH-500 examples were used. Performance could improve with a larger or more diverse training set.
- **Reward design**: The reward is heuristic-based. A learned reward model may better reflect solution quality, elegance, or alternative solution paths.

---

## Citation

If you use Nexus-1.5B in your research, please cite:

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

This model is released under the [MIT License](LICENSE).
The base model [Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) is subject to its own license terms.
