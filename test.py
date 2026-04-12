# ============================================================
#  DAPO Fine-tuning: Qwen2.5-Math-1.5B-Instruct on MATH-500
#  Kaggle single-file script (2x T4 / P100 / A100)
#  Baseline: 327/500 → Target: 400+/500
# ============================================================
# !pip install -q peft bitsandbytes flash-attn --no-build-isolation
# !pip install -q sympy accelerate datasets transformers

import os, re, json, logging, warnings
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


# ════════════════════════════════════════════
#  CONFIG  —  chỉnh tại đây
# ════════════════════════════════════════════

@dataclass
class Config:
    model_name:   str   = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_dir:   str   = "/kaggle/working/dapo_out"

    # --- GPU ---
    use_flash_attn: bool  = True   # True nếu có A100; False cho T4/P100
    use_4bit:       bool  = False    # QLoRA 4-bit cho T4 (16GB)
    use_8bit:       bool  = False

    # --- LoRA ---
    lora_r:         int   = 64
    lora_alpha:     int   = 64
    lora_dropout:   float = 0.05

    # --- DAPO ---
    group_size:     int   = 8       # số completions / prompt (giảm xuống 6 cho T4)
    max_new_tokens: int   = 512
    epsilon_low:    float = 0.20    # clip thấp
    epsilon_high:   float = 0.28    # clip cao (DAPO đặc trưng)
    entropy_coef:   float = 0.001
    temperature:    float = 0.9
    top_p:          float = 0.95
    filter_uniform: bool  = True    # bỏ group all-correct/all-wrong

    # --- Training ---
    batch_size:     int   = 4       # prompt / step
    lr:             float = 5e-6
    weight_decay:   float = 0.01
    num_epochs:     int   = 2
    warmup_ratio:   float = 0.05
    eval_every:     int   = 40
    save_every:     int   = 100
    max_prompt_len: int   = 384

CFG = Config()


# ════════════════════════════════════════════
#  REWARD FUNCTION
# ════════════════════════════════════════════

def _extract_boxed(text: str) -> str:
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1].strip() if matches else ""

def _normalize(expr: str) -> str:
    expr = re.sub(r'\\left|\\right', '', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    return expr.replace('\\,','').replace('\\;','').replace('\\!','')

def _to_sympy_str(s: str) -> str:
    s = s.replace('\\frac{', '((').replace('}{', ')/(').replace('}', '))')
    s = s.replace('^', '**').replace('\\cdot','*').replace('\\times','*')
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
    return s.replace('\\pi','pi').replace('\\infty','oo')

def _numeric_eq(pred: str, gold: str, tol: float = 1e-6) -> bool:
    try:
        from sympy import N as SN
        from sympy.parsing.sympy_parser import parse_expr
        pv = complex(SN(parse_expr(_to_sympy_str(pred)), 20))
        gv = complex(SN(parse_expr(_to_sympy_str(gold)), 20))
        return abs(pv - gv) < tol * (1 + abs(gv))
    except Exception:
        return False

def _symbolic_eq(pred: str, gold: str) -> bool:
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr
        return simplify(parse_expr(_to_sympy_str(pred)) - parse_expr(_to_sympy_str(gold))) == 0
    except Exception:
        return False

def is_correct(prediction: str, ground_truth: str) -> bool:
    pred_box = _extract_boxed(prediction)
    gold_box = _extract_boxed(ground_truth) if '\\boxed' in ground_truth else ground_truth.strip()
    if not pred_box:
        return False
    if _normalize(pred_box) == _normalize(gold_box):
        return True
    if _numeric_eq(pred_box, gold_box):
        return True
    if _symbolic_eq(pred_box, gold_box):
        return True
    return False

def reward_fn(predictions: List[str], ground_truths: List[str]) -> List[float]:
    """Correctness (0/1) + nhỏ format bonus."""
    rewards = []
    for pred, gt in zip(predictions, ground_truths):
        r = 1.0 if is_correct(pred, gt) else 0.0
        # format bonus
        if _extract_boxed(pred):               r += 0.10
        if 300 < len(pred) < 2500:             r += 0.05
        if re.search(r'(therefore|thus|so|=)', pred, re.I):  r += 0.05
        rewards.append(r)
    return rewards


# ════════════════════════════════════════════
#  MODEL LOADING
# ════════════════════════════════════════════

def load_model_tokenizer(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if cfg.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif cfg.use_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    attn_impl = "flash_attention_2" if cfg.use_flash_attn else "eager"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )

    if cfg.use_4bit or cfg.use_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ════════════════════════════════════════════
#  DATASET
# ════════════════════════════════════════════

SYSTEM_PROMPT = """You are an advanced mathematical reasoning model.
Follow these rules carefully for every problem:

1. Think step-by-step and show complete reasoning.
2. Give a short plan before solving.
3. No hand-waving. Be precise.
4. Final answer must be in the form \\boxed{answer}.
5. No text after the boxed result.

Solve the following problem:
"""

def make_prompt(problem: str, tokenizer) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": problem}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

class Math500Dataset(Dataset):
    def __init__(self, tokenizer, max_prompt_len=384):
        self.data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        self.tok  = tokenizer
        self.max_len = max_prompt_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"prompt": make_prompt(item["problem"], self.tok),
                "answer": item["answer"],
                "subject": item.get("subject",""),
                "level": item.get("level", 0)}

    def collate(self, batch):
        prompts = [b["prompt"] for b in batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_len)
        return {**enc,
                "answers":  [b["answer"]  for b in batch],
                "subjects": [b["subject"] for b in batch],
                "levels":   [b["level"]   for b in batch]}


# ════════════════════════════════════════════
#  DAPO TRAINER
# ════════════════════════════════════════════

class DAPOTrainer:
    """
    DAPO: Decoupled Clip + Token-level PG + Dynamic Sampling + Entropy Bonus
    """
    def __init__(self, model, tokenizer, cfg: Config):
        self.model = model
        self.tok   = tokenizer
        self.cfg   = cfg

    # ── token log-probs (chỉ phần completion) ──
    def _logprobs(self, input_ids, attn_mask, prompt_len, no_grad=False):
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1]                         # (B, L-1, V)
        labels = input_ids[:, 1:]                           # (B, L-1)
        logp   = F.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
        mask = attn_mask[:, 1:].clone().float()
        mask[:, :prompt_len-1] = 0.0                        # zero prompt
        return tok_logp, mask

    # ── DAPO loss (clip-higher, token-level) ──
    def _loss(self, old_logp, new_logp, adv, mask):
        log_r = new_logp - old_logp.detach()
        r     = log_r.exp()
        a     = adv.unsqueeze(1).expand_as(r)
        el, eh = self.cfg.epsilon_low, self.cfg.epsilon_high
        clipped = torch.where(
            a >= 0,
            r.clamp(1 - el, 1 + eh),
            r.clamp(1 - eh, 1 + el),
        )
        pg = -torch.min(r * a, clipped * a)
        return (pg * mask).sum() / mask.sum().clamp(min=1)

    # ── entropy bonus ──
    def _entropy(self, input_ids, attn_mask, mask):
        out   = self.model(input_ids=input_ids, attention_mask=attn_mask)
        logits = out.logits[:, :-1]
        p  = F.softmax(logits,     dim=-1)
        lp = F.log_softmax(logits, dim=-1)
        ent = -(p * lp).sum(-1)                             # (B, L-1)
        return -(ent * mask).sum() / mask.sum().clamp(min=1)

    # ── main step ──
    @torch.no_grad()
    def _generate(self, input_ids, attn_mask):
        G = self.cfg.group_size
        B = input_ids.shape[0]
        seqs, masks = [], []
        for i in range(B):
            out = self.model.generate(
                input_ids     = input_ids[i:i+1].expand(G, -1),
                attention_mask= attn_mask[i:i+1].expand(G, -1),
                max_new_tokens= self.cfg.max_new_tokens,
                do_sample=True, temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                pad_token_id=self.tok.pad_token_id,
            )
            seqs.append(out)
            masks.append((out != self.tok.pad_token_id).long())
        return seqs, masks

    def step(self, batch, optimizer):
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        answers   = batch["answers"]
        P = input_ids.shape[1]
        B = input_ids.shape[0]
        G = self.cfg.group_size

        # ── 1. Generate ──
        self.model.eval()
        seqs, att_masks = self._generate(input_ids, attn_mask)

        # ── 2. Decode & reward ──
        comp_texts, gt_exp = [], []
        for i in range(B):
            for j in range(G):
                comp_texts.append(self.tok.decode(seqs[i][j, P:], skip_special_tokens=True))
                gt_exp.append(answers[i])
        rewards_flat = reward_fn(comp_texts, gt_exp)
        rg = [rewards_flat[i*G:(i+1)*G] for i in range(B)]

        # ── 3. Dynamic sampling (lọc group đồng nhất) ──
        keep = [not all(r == rg[i][0] for r in rg[i]) for i in range(B)]

        # ── 4. Advantages (within-group normalize) ──
        advs = []
        for i in range(B):
            if not keep[i]:
                advs.extend([0.0] * G); continue
            mu  = sum(rg[i]) / G
            std = max((sum((r-mu)**2 for r in rg[i])/G)**0.5, 1e-8)
            advs.extend([(r - mu) / std for r in rg[i]])

        # ── 5. Old log-probs ──
        self.model.train()
        old_logps, comp_masks = [], []
        all_seqs, all_atts = [], []
        for i in range(B):
            for j in range(G):
                s = seqs[i][j:j+1]
                a = att_masks[i][j:j+1]
                lp, cm = self._logprobs(s, a, P, no_grad=True)
                old_logps.append(lp); comp_masks.append(cm)
                all_seqs.append(s);  all_atts.append(a)

        # ── 6. Policy gradient ──
        total_loss = torch.tensor(0.0, device=device, requires_grad=False)
        n_active = 0
        losses = []
        for i in range(B):
            if not keep[i]: continue
            n_active += 1
            for j in range(G):
                idx = i * G + j
                new_lp, cm = self._logprobs(all_seqs[idx], all_atts[idx], P)
                adv_t = torch.tensor([advs[idx]], device=device)
                pg  = self._loss(old_logps[idx], new_lp, adv_t, comp_masks[idx])
                ent = self._entropy(all_seqs[idx], all_atts[idx], comp_masks[idx])
                losses.append(pg + self.cfg.entropy_coef * ent)

        if losses:
            total_loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

        return {
            "loss":        total_loss.item() if losses else 0.0,
            "avg_reward":  sum(rewards_flat) / len(rewards_flat),
            "n_active":    n_active,
        }


# ════════════════════════════════════════════
#  EVALUATION
# ════════════════════════════════════════════

def evaluate(model, tokenizer, cfg: Config, device: str) -> Dict:
    ds   = Math500Dataset(tokenizer, cfg.max_prompt_len)
    dl   = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=ds.collate)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dl:
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            P    = ids.shape[1]
            outs = model.generate(
                input_ids=ids, attention_mask=attn,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            for i, ans in enumerate(batch["answers"]):
                pred = tokenizer.decode(outs[i, P:], skip_special_tokens=True)
                results.append({
                    "correct": is_correct(pred, ans),
                    "subject": batch["subjects"][i],
                    "level":   batch["levels"][i],
                })
    total   = len(results)
    correct = sum(r["correct"] for r in results)
    by_subj = {}
    for r in results:
        by_subj.setdefault(r["subject"], []).append(r["correct"])
    model.train()
    return {
        "accuracy":  correct / total,
        "n_correct": correct,
        "total":     total,
        "by_subject": {s: f"{sum(v)}/{len(v)} ({sum(v)/len(v):.2%})"
                       for s, v in sorted(by_subj.items())},
    }


# ════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device} | Config: 4bit={CFG.use_4bit} LoRA-r={CFG.lora_r} G={CFG.group_size}")

    out = Path(CFG.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    log.info("Loading model & tokenizer...")
    model, tokenizer = load_model_tokenizer(CFG)

    # Dataset
    log.info("Loading MATH-500...")
    ds = Math500Dataset(tokenizer, CFG.max_prompt_len)
    dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, collate_fn=ds.collate)

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr, weight_decay=CFG.weight_decay, betas=(0.9, 0.95),
    )
    total_steps   = len(dl) * CFG.num_epochs
    warmup_steps  = int(total_steps * CFG.warmup_ratio)
    scheduler     = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    trainer       = DAPOTrainer(model, tokenizer, CFG)

    # Baseline
    log.info("Evaluating baseline...")
    base = evaluate(model, tokenizer, CFG, device)
    log.info(f"Baseline: {base['n_correct']}/{base['total']} = {base['accuracy']:.4f}")
    log.info("By subject:\n" + json.dumps(base["by_subject"], indent=2, ensure_ascii=False))

    best_acc  = base["accuracy"]
    global_step = 0

    for epoch in range(CFG.num_epochs):
        log.info(f"\n{'─'*50}\nEpoch {epoch+1}/{CFG.num_epochs}\n{'─'*50}")
        for batch in dl:
            metrics = trainer.step(batch, optimizer)
            scheduler.step()
            global_step += 1

            if global_step % 5 == 0:
                log.info(f"[{global_step:4d}] loss={metrics['loss']:.4f}  "
                         f"reward={metrics['avg_reward']:.4f}  "
                         f"active={metrics['n_active']}  "
                         f"lr={scheduler.get_last_lr()[0]:.2e}")

            # Eval
            if global_step % CFG.eval_every == 0:
                log.info("Evaluating...")
                ev = evaluate(model, tokenizer, CFG, device)
                log.info(f"✦ Step {global_step}: {ev['n_correct']}/500 = {ev['accuracy']:.4f}")
                if ev["accuracy"] > best_acc:
                    best_acc = ev["accuracy"]
                    model.save_pretrained(str(out / "best"))
                    tokenizer.save_pretrained(str(out / "best"))
                    log.info(f"  → New best! Saved to {out/'best'}")
                    log.info("  By subject:\n" +
                             json.dumps(ev["by_subject"], indent=2, ensure_ascii=False))

            # Checkpoint
            if global_step % CFG.save_every == 0:
                ckpt = out / f"ckpt-{global_step}"
                model.save_pretrained(str(ckpt))
                tokenizer.save_pretrained(str(ckpt))

    # Final
    log.info("\nFinal evaluation...")
    final = evaluate(model, tokenizer, CFG, device)
    log.info(f"Final:    {final['n_correct']}/500 = {final['accuracy']:.4f}")
    log.info(f"Baseline: {base['n_correct']}/500 = {base['accuracy']:.4f}")
    log.info(f"Best:     {best_acc:.4f}")
    log.info("By subject:\n" + json.dumps(final["by_subject"], indent=2, ensure_ascii=False))

    model.save_pretrained(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    log.info("Done ✓")


if __name__ == "__main__":
    main()
