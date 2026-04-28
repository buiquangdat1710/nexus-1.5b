#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Nexus-1.5B  —  DAPO + Length-Penalized Reward Optimization (LPRO)      ║
║  Base : Qwen2.5-Math-1.5B-Instruct                                       ║
║  Data : HuggingFaceH4/MATH-500                                           ║
╚══════════════════════════════════════════════════════════════════════════╝

Key differences vs vanilla GRPO:
  • DAPO  : asymmetric clipping  (ε_low ≠ ε_high) + token-level normalization
  • LPRO  : A_i = z-score(r_i) − λ · z-score(L_i)   (reward − length penalty)
  • No KL penalty by default (DAPO style, stable with ref-model clipping)
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os, re, math, json, random, logging, argparse
from dataclasses  import dataclass, field
from typing       import List, Dict, Tuple, Optional

# ── third-party ─────────────────────────────────────────────────────────────
import torch
import numpy as np
from datasets           import load_dataset
from transformers       import (AutoTokenizer, AutoModelForCausalLM,
                                get_cosine_schedule_with_warmup)
from torch.optim        import AdamW
from huggingface_hub    import HfApi

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. CONFIGURATION                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
@dataclass
class Config:
    # ── Model ───────────────────────────────────────────────────────────────
    model_name      : str   = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_dir      : str   = "./nexus-1.5b-lpro"
    hub_repo_id     : str   = "YOUR_HF_USERNAME/nexus-1.5b-lpro"   # ← đổi cái này

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset_name    : str   = "HuggingFaceH4/MATH-500"
    max_prompt_len  : int   = 512

    # ── Group sampling (GRPO style) ──────────────────────────────────────────
    G               : int   = 8      # số output được sample cho mỗi prompt
    max_new_tokens  : int   = 1024
    temperature     : float = 0.7
    top_p           : float = 0.95

    # ── DAPO asymmetric clipping ─────────────────────────────────────────────
    #   clip ratio ∈ [1 − ε_low,  1 + ε_high]
    eps_low         : float = 0.20   # standard PPO lower bound
    eps_high        : float = 0.28   # larger upper bound (DAPO trick)

    # ── LPRO Length Penalty ──────────────────────────────────────────────────
    #   A_i = (r_i − μ_r)/(σ_r + ε_r)  −  λ · (L_i − μ_L)/(σ_L + ε_L)
    lambda_len      : float = 0.10
    eps_r           : float = 1e-8
    eps_l           : float = 1e-8

    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs      : int   = 3
    lr              : float = 5e-7
    weight_decay    : float = 1e-2
    warmup_ratio    : float = 0.05
    grad_clip       : float = 1.0
    grad_accum      : int   = 8      # effective batch = grad_accum prompts
    bf16            : bool  = True

    # ── Logging / Saving ─────────────────────────────────────────────────────
    log_steps       : int   = 10
    save_steps      : int   = 100
    seed            : int   = 42

    # ── HuggingFace Hub ──────────────────────────────────────────────────────
    push_to_hub     : bool  = True
    hf_token        : str   = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. MATH REWARD  (correctness binary signal)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_boxed(text: str) -> Optional[str]:
    """Lấy nội dung cuối cùng trong \\boxed{...}."""
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else None


def normalize_expr(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    # bỏ dollar, dấu phẩy, ngoặc nhọn cosmetic
    for ch in ("$", "\\,", "\\!", "{", "}"):
        s = s.replace(ch, "")
    return s


def sympy_equal(pred: str, gold: str) -> bool:
    """Symbolic equality qua sympy (fallback)."""
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, N
        p = parse_latex(pred)
        g = parse_latex(gold)
        return simplify(p - g) == 0
    except Exception:
        return False


def math_reward(response: str, gold_answer: str) -> float:
    """
    Trả 1.0 nếu đúng, 0.0 nếu sai.
    Thử string match trước, sau đó sympy.
    """
    pred = extract_boxed(response)
    if pred is None:
        return 0.0
    if normalize_expr(pred) == normalize_expr(gold_answer):
        return 1.0
    if sympy_equal(pred, gold_answer):
        return 1.0
    return 0.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. LENGTH-PENALIZED ADVANTAGE  (LPRO core)                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def lpro_advantages(
    rewards    : List[float],
    lengths    : List[int],
    lambda_len : float,
    eps_r      : float,
    eps_l      : float,
) -> np.ndarray:
    """
    A_i = (r_i − μ_r) / (σ_r + ε_r)   ← tín hiệu Đúng/Sai (z-score)
        − λ · (L_i − μ_L) / (σ_L + ε_L)  ← tín hiệu Phạt độ dài

    Khớp với công thức trong hình ảnh.
    """
    r = np.array(rewards, dtype=np.float64)
    L = np.array(lengths, dtype=np.float64)

    z_r = (r - r.mean()) / (r.std() + eps_r)
    z_L = (L - L.mean()) / (L.std() + eps_l)

    return z_r - lambda_len * z_L


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. DAPO LOSS  (asymmetric clip + token-level normalisation)              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def dapo_token_loss(
    new_lp   : torch.Tensor,    # [resp_len]  log π_θ(a_t | s_t)
    old_lp   : torch.Tensor,    # [resp_len]  log π_old(a_t | s_t)   (no grad)
    adv      : float,           # scalar advantage A_i
    eps_low  : float,
    eps_high : float,
) -> Tuple[torch.Tensor, int]:
    """
    Trả (sum_loss, n_tokens) cho một output trong group.

    J_DAPO = Σ_t  min(ρ_t · Â,  clip(ρ_t, 1−ε_low, 1+ε_high) · Â)

    Loss được cộng dồn; chia tổng token ở bên ngoài (token-level normalisation).
    """
    ratio         = torch.exp(new_lp - old_lp)                          # [resp_len]
    ratio_clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)

    adv_t  = torch.full_like(new_lp, adv)
    surr1  = ratio         * adv_t
    surr2  = ratio_clipped * adv_t

    # Pessimistic (min) → negative vì ta maximize
    loss_sum = -torch.min(surr1, surr2).sum()
    return loss_sum, new_lp.shape[0]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. TOKEN LOG-PROB HELPER                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_resp_log_probs(
    model       ,
    full_ids    : torch.Tensor,   # [1, full_len]
    prompt_len  : int,
    no_grad     : bool = False,
) -> torch.Tensor:
    """
    Trả log-prob của từng token trong phần response.
    logits[t] → dự đoán token[t+1], nên ta shift:
        logits[prompt_len-1 : full_len-1]  ↔  ids[prompt_len : full_len]
    """
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        out    = model(input_ids=full_ids)
        logits = out.logits[0]                                   # [full_len, vocab]
        lp     = torch.log_softmax(logits, dim=-1)

        resp_ids = full_ids[0, prompt_len:]                      # [resp_len]
        resp_lp  = lp[prompt_len - 1 : full_ids.shape[1] - 1]   # [resp_len, vocab]

        token_lp = resp_lp.gather(1, resp_ids.unsqueeze(-1)).squeeze(-1)  # [resp_len]
    return token_lp


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. DATASET  (MATH-500 → chat prompt)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SYSTEM_PROMPT = (
    "You are a mathematics expert. "
    "Solve the problem step by step and enclose your final answer in \\boxed{}."
)


def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def load_math500(tokenizer, max_prompt_len: int) -> List[Dict]:
    ds  = load_dataset("HuggingFaceH4/MATH-500", split="test")
    out = []
    for item in ds:
        txt = build_prompt(item["problem"], tokenizer)
        ids = tokenizer.encode(txt, add_special_tokens=False)
        if len(ids) <= max_prompt_len:
            out.append({"prompt": txt, "prompt_ids": ids, "answer": item["answer"]})
    log.info(f"Loaded {len(out)} examples  (max_prompt_len={max_prompt_len})")
    return out


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. TRAINING LOOP                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train(cfg: Config):
    # ── reproducibility ──
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = (torch.bfloat16
              if cfg.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
              else torch.float32)
    log.info(f"Device: {device}  |  dtype: {dtype}")

    # ── tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── trainable policy ───────────────────────────────────────────────────
    log.info(f"Loading policy model: {cfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype  = dtype,
        device_map   = "auto",
        trust_remote_code = True,
    )
    model.gradient_checkpointing_enable()   # tiết kiệm bộ nhớ
    model.train()

    # ── frozen reference model (π_ref / π_old) ────────────────────────────
    log.info("Loading reference model (frozen) …")
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype  = dtype,
        device_map   = "auto",
        trust_remote_code = True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── dataset ────────────────────────────────────────────────────────────
    dataset = load_math500(tokenizer, cfg.max_prompt_len)

    # ── optimizer + scheduler ──────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_ep = math.ceil(len(dataset) / cfg.grad_accum)
    total_steps  = cfg.num_epochs * steps_per_ep
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    log.info(f"Total steps: {total_steps}  |  Warmup: {warmup_steps}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    global_step  = 0
    acc_loss     = 0.0
    acc_reward   = 0.0
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        random.shuffle(dataset)
        log.info(f"═══ Epoch {epoch + 1}/{cfg.num_epochs} ═══")

        for idx, example in enumerate(dataset):
            prompt_ids = example["prompt_ids"]
            gold       = example["answer"]
            prompt_len = len(prompt_ids)

            prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # ── 7a. Sample G outputs ────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                gen_out = model.generate(
                    prompt_t.expand(cfg.G, -1),
                    max_new_tokens = cfg.max_new_tokens,
                    temperature    = cfg.temperature,
                    top_p          = cfg.top_p,
                    do_sample      = True,
                    pad_token_id   = tokenizer.eos_token_id,
                )
            model.train()

            # ── 7b. Compute rewards & lengths ───────────────────────────────
            rewards, lengths, full_seqs = [], [], []
            for i in range(cfg.G):
                resp_ids  = gen_out[i, prompt_len:]
                resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)

                r = math_reward(resp_text, gold)
                l = max(int(resp_ids.shape[0]), 1)

                rewards.append(r)
                lengths.append(l)
                full_seqs.append(gen_out[i])   # [full_len]

            # Skip nếu toàn bộ reward giống nhau (không có gradient signal)
            if len(set(rewards)) <= 1:
                continue

            # ── 7c. LPRO advantages ─────────────────────────────────────────
            advs = lpro_advantages(rewards, lengths,
                                   cfg.lambda_len, cfg.eps_r, cfg.eps_l)

            # ── 7d. DAPO loss tổng hợp cho G outputs ───────────────────────
            total_loss_sum = torch.tensor(0.0, device=device)
            total_n_tokens = 0

            for i in range(cfg.G):
                seq      = full_seqs[i].unsqueeze(0).to(device)  # [1, full_len]
                resp_len = lengths[i]
                adv_i    = float(advs[i])

                # Trim nếu quá dài (tránh OOM)
                max_len = prompt_len + cfg.max_new_tokens
                if seq.shape[1] > max_len:
                    seq      = seq[:, :max_len]
                    resp_len = max_len - prompt_len

                # --- log probs mới (có grad) ---
                with torch.cuda.amp.autocast(enabled=cfg.bf16):
                    new_lp = get_resp_log_probs(model, seq, prompt_len, no_grad=False)

                # --- log probs cũ từ ref model (không grad) ---
                old_lp = get_resp_log_probs(ref_model, seq, prompt_len, no_grad=True)

                loss_sum, n = dapo_token_loss(
                    new_lp, old_lp.detach(), adv_i,
                    cfg.eps_low, cfg.eps_high,
                )
                total_loss_sum += loss_sum
                total_n_tokens += n

            if total_n_tokens == 0:
                continue

            # Token-level normalization (DAPO / LPRO style)
            loss = total_loss_sum / total_n_tokens / cfg.grad_accum
            loss.backward()

            acc_loss   += loss.item() * cfg.grad_accum
            acc_reward += float(np.mean(rewards))

            # ── 7e. Optimizer step ──────────────────────────────────────────
            if (idx + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_steps == 0:
                    avg_loss   = acc_loss   / cfg.log_steps
                    avg_reward = acc_reward / (cfg.log_steps * cfg.grad_accum)
                    log.info(
                        f"Step {global_step:5d} | "
                        f"Loss {avg_loss:.4f} | "
                        f"AvgReward {avg_reward:.3f} | "
                        f"LR {scheduler.get_last_lr()[0]:.2e}"
                    )
                    acc_loss = acc_reward = 0.0

                if global_step % cfg.save_steps == 0:
                    ckpt = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    log.info(f"Checkpoint saved → {ckpt}")

    # ── 8. Save final model ────────────────────────────────────────────────
    log.info("Saving final model …")
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    _save_config(cfg, os.path.join(cfg.output_dir, "lpro_config.json"))
    log.info(f"Model saved to {cfg.output_dir}")

    # ── 9. Push to HuggingFace Hub ─────────────────────────────────────────
    if cfg.push_to_hub:
        if not cfg.hf_token:
            log.warning("HF_TOKEN not set — skipping Hub push.")
        else:
            log.info(f"Pushing to Hub: {cfg.hub_repo_id} …")
            model.push_to_hub(cfg.hub_repo_id, token=cfg.hf_token, private=False)
            tokenizer.push_to_hub(cfg.hub_repo_id, token=cfg.hf_token, private=False)
            _push_readme(cfg)
            log.info(f"✅ Model pushed → https://huggingface.co/{cfg.hub_repo_id}")


# ── helpers ────────────────────────────────────────────────────────────────

def _save_config(cfg: Config, path: str):
    import dataclasses
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)


def _push_readme(cfg: Config):
    """Push một model card tối giản lên Hub."""
    from huggingface_hub import HfApi
    readme = f"""---
language: en
license: apache-2.0
base_model: {cfg.model_name}
tags:
  - math
  - reinforcement-learning
  - dapo
  - lpro
---

# Nexus-1.5B (DAPO + LPRO)

Fine-tune của **{cfg.model_name}** trên [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
sử dụng **DAPO** (asymmetric clipping ε_low={cfg.eps_low}, ε_high={cfg.eps_high})
và **LPRO** (Length-Penalized Reward Optimization, λ={cfg.lambda_len}).

## Advantage formula

```
A_i = (r_i − μ_r) / (σ_r + ε)  −  λ · (L_i − μ_L) / (σ_L + ε)
```

## DAPO objective

```
J = (1/Σ|o_i|) Σ_i Σ_t  min( ρ_t · Â_i,  clip(ρ_t, 1−ε_low, 1+ε_high) · Â_i )
```
"""
    api = HfApi(token=cfg.hf_token)
    api.upload_file(
        path_or_fileobj = readme.encode(),
        path_in_repo    = "README.md",
        repo_id         = cfg.hub_repo_id,
        repo_type       = "model",
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="DAPO + LPRO training for Qwen2.5-Math")
    p.add_argument("--model_name",    default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--output_dir",    default="./nexus-1.5b-lpro")
    p.add_argument("--hub_repo_id",   default="YOUR_HF_USERNAME/nexus-1.5b-lpro")
    p.add_argument("--G",             type=int,   default=8)
    p.add_argument("--num_epochs",    type=int,   default=3)
    p.add_argument("--lr",            type=float, default=5e-7)
    p.add_argument("--lambda_len",    type=float, default=0.10)
    p.add_argument("--eps_low",       type=float, default=0.20)
    p.add_argument("--eps_high",      type=float, default=0.28)
    p.add_argument("--max_new_tokens",type=int,   default=1024)
    p.add_argument("--temperature",   type=float, default=0.70)
    p.add_argument("--grad_accum",    type=int,   default=8)
    p.add_argument("--save_steps",    type=int,   default=100)
    p.add_argument("--no_push",       action="store_true")
    p.add_argument("--hf_token",      default=os.getenv("HF_TOKEN", ""))
    args = p.parse_args()

    cfg = Config()
    for k, v in vars(args).items():
        if k == "no_push":
            cfg.push_to_hub = not v
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    log.info("Config:\n" + json.dumps(cfg.__dict__, indent=2, default=str))
    train(cfg)
