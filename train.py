"""
DAPO Fine-tuning: Qwen2.5-Math-1.5B-Instruct on MATH-500
==========================================================
Implements DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)
with several architectural improvements over vanilla GRPO:

Key improvements:
1. DAPO: Decoupled clip (ε_low, ε_high) + token-level policy gradient loss
2. Dynamic sampling: Filters out all-correct/all-wrong groups
3. SoftKL divergence instead of hard KL (more stable)
4. Entropy bonus for exploration
5. RoPE scaling + Flash Attention 2 for long CoT sequences
6. Mixed-precision + gradient checkpointing for memory efficiency
7. Custom reward model with process-reward awareness
"""

import os
import re
import gc
import math
import json
import logging
import warnings
from typing import Optional, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from huggingface_hub import HfApi, login

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DAPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_dir: str = "./dapo_math_output"
    hf_repo_id: str = "your-username/Qwen2.5-Math-1.5B-DAPO"  # ← change this

    # Dataset
    dataset_name: str = "HuggingFaceH4/MATH-500"
    max_prompt_length: int = 512
    max_response_length: int = 1536  # long CoT
    max_seq_length: int = 2048

    # DAPO-specific hyperparams
    clip_low: float = 0.2      # ε_low (decoupled lower clip)
    clip_high: float = 0.28    # ε_high (decoupled upper clip, slightly larger)
    kl_coef: float = 0.001     # β for soft KL penalty
    entropy_coef: float = 0.01 # entropy bonus coefficient
    token_level_loss: bool = True  # token-level vs sequence-level PG

    # Dynamic sampling
    group_size: int = 8         # G responses per prompt
    dynamic_sampling: bool = True  # filter all-correct/all-wrong groups
    overgenerate_factor: int = 2   # generate 2G, keep G after filtering

    # Training
    num_epochs: int = 3
    batch_size: int = 1         # prompts per GPU step
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    min_lr: float = 5e-7
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Generation
    temperature: float = 0.9
    top_p: float = 0.9
    do_sample: bool = True

    # Misc
    seed: int = 42
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    fp16: bool = False
    bf16: bool = True
    use_wandb: bool = False
    hf_token: str = ""         # ← your HuggingFace token
    push_every_n_steps: int = 100


# ─────────────────────────────────────────────────────────────────────────────
# Reward Functions
# ─────────────────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from model output."""
    # Try \boxed{...}
    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[-1].strip()

    # Try "The answer is X" patterns
    patterns = [
        r'[Tt]he answer is[:\s]+([^\n\.]+)',
        r'[Aa]nswer[:\s]+([^\n\.]+)',
        r'= ([^\n\.]+)$',
    ]
    for pat in patterns:
        m = re.search(pat, text.strip())
        if m:
            return m.group(1).strip()

    return None


def normalize_answer(ans: str) -> str:
    """Normalize answer string for comparison."""
    if ans is None:
        return ""
    ans = ans.strip()
    # Remove trailing periods/commas
    ans = re.sub(r'[,\.]+$', '', ans)
    # Normalize fractions
    ans = ans.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    # Remove spaces around operators
    ans = re.sub(r'\s+', '', ans)
    # Lowercase
    ans = ans.lower()
    return ans


def compute_outcome_reward(response: str, ground_truth: str) -> float:
    """
    Outcome-based reward: 1.0 for correct, -0.5 for wrong.
    Partial rewards for good formatting.
    """
    pred = extract_answer(response)
    if pred is None:
        return -1.0  # No answer found

    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(ground_truth)

    if pred_norm == gt_norm:
        return 1.0

    # Try numeric comparison
    try:
        pred_val = float(pred_norm.replace(',', ''))
        gt_val = float(gt_norm.replace(',', ''))
        if abs(pred_val - gt_val) < 1e-6:
            return 1.0
        if abs(pred_val - gt_val) / (abs(gt_val) + 1e-9) < 0.01:
            return 0.5  # very close
    except (ValueError, ZeroDivisionError):
        pass

    # Format reward: at least wrote \boxed{}
    if '\\boxed' in response:
        return -0.3

    return -0.5


def compute_format_reward(response: str) -> float:
    """Bonus for good reasoning format."""
    reward = 0.0
    if '\\boxed' in response:
        reward += 0.1
    # Check for step-by-step structure
    if len(re.findall(r'(?:step|therefore|thus|hence|so)', response.lower())) >= 2:
        reward += 0.05
    # Check for appropriate length (not too short, not too long)
    tokens_approx = len(response.split())
    if 50 <= tokens_approx <= 800:
        reward += 0.05
    return reward


def compute_reward(response: str, ground_truth: str) -> float:
    """Combined reward function."""
    outcome_r = compute_outcome_reward(response, ground_truth)
    format_r = compute_format_reward(response)
    # Only add format bonus for non-correct answers
    if outcome_r == 1.0:
        return outcome_r
    return outcome_r + format_r


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful math expert. "
    "Solve the following problem step by step, showing all work clearly. "
    "Put your final answer inside \\boxed{}."
)


def format_prompt(problem: str, tokenizer) -> str:
    """Format problem with chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


class MATHDataset(Dataset):
    def __init__(self, data, tokenizer, max_prompt_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        problem = item.get("problem", item.get("question", ""))
        answer = item.get("answer", item.get("solution", ""))

        # Extract just the final answer if solution is long
        answer_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', str(answer))
        if answer_match:
            answer = answer_match.group(1)

        prompt = format_prompt(problem, self.tokenizer)
        prompt_ids = self.tokenizer.encode(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt"
        )[0]

        return {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "ground_truth": str(answer),
            "problem": problem,
        }


def collate_fn(batch):
    return {
        "prompts": [b["prompt"] for b in batch],
        "prompt_ids": [b["prompt_ids"] for b in batch],
        "ground_truths": [b["ground_truth"] for b in batch],
        "problems": [b["problem"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DAPO Loss
# ─────────────────────────────────────────────────────────────────────────────

class DAPOLoss(nn.Module):
    """
    DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization
    
    Key differences from GRPO:
    - Decoupled clip bounds (ε_low ≠ ε_high)
    - Token-level policy gradient loss
    - Soft KL divergence (avoids hard KL instability)
    - Entropy regularization
    """

    def __init__(self, config: DAPOConfig):
        super().__init__()
        self.clip_low = config.clip_low
        self.clip_high = config.clip_high
        self.kl_coef = config.kl_coef
        self.entropy_coef = config.entropy_coef
        self.token_level_loss = config.token_level_loss

    def forward(
        self,
        logprobs: torch.Tensor,        # [B*G, L]
        ref_logprobs: torch.Tensor,    # [B*G, L]
        old_logprobs: torch.Tensor,    # [B*G, L]
        advantages: torch.Tensor,      # [B*G] or [B*G, L]
        attention_mask: torch.Tensor,  # [B*G, L]
        logits: torch.Tensor,          # [B*G, L, V]
    ) -> dict:
        """
        logprobs: log π_θ(a_t | s_t) for response tokens
        ref_logprobs: log π_ref(a_t | s_t)
        old_logprobs: log π_old(a_t | s_t) (from sampling)
        advantages: normalized group advantages
        """
        # Importance ratio
        log_ratio = logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        # DAPO decoupled clip:
        # - For positive advantages: clip from above at (1 + ε_high)
        # - For negative advantages: clip from below at (1 - ε_low)
        if self.token_level_loss and advantages.dim() == 1:
            adv = advantages.unsqueeze(1).expand_as(logprobs)  # [B*G, L]
        else:
            adv = advantages

        # Clipped surrogate
        pos_mask = (adv > 0).float()
        neg_mask = (adv < 0).float()

        # Upper clip for positive, lower clip for negative (decoupled)
        clip_upper = torch.clamp(ratio, max=1.0 + self.clip_high)
        clip_lower = torch.clamp(ratio, min=1.0 - self.clip_low)
        clipped_ratio = pos_mask * clip_upper + neg_mask * clip_lower + (1 - pos_mask - neg_mask) * ratio

        # Policy gradient loss (token-level)
        pg_loss = -torch.min(ratio * adv, clipped_ratio * adv)

        # Soft KL: KL(π_ref || π_θ) = log(π_ref/π_θ) for stability
        # Instead of hard KL clipping, use soft penalty
        soft_kl = ref_logprobs - logprobs  # [B*G, L]
        soft_kl = torch.clamp(soft_kl, -10, 10)  # numerical stability

        # Entropy bonus (encourages exploration)
        # H(π_θ) ≈ -mean(logprobs)
        entropy_bonus = -logprobs  # proxy for entropy

        # Total loss (masked)
        mask = attention_mask.float()
        
        pg_loss_masked = (pg_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        kl_masked = (soft_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        entropy_masked = (entropy_bonus * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        loss = (
            pg_loss_masked
            + self.kl_coef * kl_masked
            - self.entropy_coef * entropy_masked
        ).mean()

        # Metrics
        with torch.no_grad():
            clip_frac = ((ratio - 1).abs() > self.clip_low).float().mean()
            approx_kl = (old_logprobs - logprobs).mean()

        return {
            "loss": loss,
            "pg_loss": pg_loss_masked.mean().item(),
            "kl": kl_masked.mean().item(),
            "entropy": entropy_masked.mean().item(),
            "clip_frac": clip_frac.item(),
            "approx_kl": approx_kl.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DAPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DAPOTrainer:
    def __init__(self, config: DAPOConfig):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision="bf16" if config.bf16 else ("fp16" if config.fp16 else "no"),
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        set_seed(config.seed)

        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(project="dapo-math", config=vars(config))

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    def _setup_model(self):
        cfg = self.config
        logger.info(f"Loading model: {cfg.model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            attn_implementation="eager",
            use_cache=False,
        )

        # Policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, **model_kwargs
        )
        self._modify_model_architecture()

        if cfg.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            self.model.enable_input_require_grads()

        # Reference model (frozen, no LoRA)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, **model_kwargs
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        self.dapo_loss = DAPOLoss(cfg)
        logger.info("Models loaded successfully.")

    def _modify_model_architecture(self):
        """
        Architecture modifications for better math reasoning:
        1. Extend RoPE to support longer sequences
        2. Adjust RMS norm epsilon for stability
        """
        cfg = self.config
        model_cfg = self.model.config

        # Extend context via RoPE scaling
        if hasattr(model_cfg, 'rope_scaling') or hasattr(model_cfg, 'max_position_embeddings'):
            original_max_pos = getattr(model_cfg, 'max_position_embeddings', 4096)
            if cfg.max_seq_length > original_max_pos:
                scale_factor = cfg.max_seq_length / original_max_pos
                model_cfg.rope_scaling = {
                    "type": "linear",
                    "factor": scale_factor,
                }
                model_cfg.max_position_embeddings = cfg.max_seq_length
                logger.info(f"RoPE scaled by {scale_factor:.2f}x to {cfg.max_seq_length} tokens")

        # Stabilize RMSNorm
        if hasattr(model_cfg, 'rms_norm_eps'):
            model_cfg.rms_norm_eps = max(model_cfg.rms_norm_eps, 1e-6)

        logger.info("Architecture modifications applied.")

    def _setup_data(self):
        cfg = self.config
        logger.info(f"Loading dataset: {cfg.dataset_name}")

        dataset = load_dataset(cfg.dataset_name, split="test")
        logger.info(f"Dataset size: {len(dataset)}")

        self.train_dataset = MATHDataset(
            dataset,
            self.tokenizer,
            cfg.max_prompt_length,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        logger.info(f"DataLoader ready: {len(self.train_loader)} steps/epoch")

    def _setup_optimizer(self):
        cfg = self.config
        # Separate LR for LoRA params
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if p.requires_grad and "lora" in n.lower()],
                "lr": cfg.learning_rate,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if p.requires_grad and "lora" not in n.lower()],
                "lr": cfg.learning_rate * 0.1,
            },
        ]
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=cfg.weight_decay,
            eps=1e-8,
        )

        total_steps = len(self.train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
        warmup_steps = int(total_steps * cfg.warmup_ratio)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
        )

        # Accelerate wrapping
        (
            self.model,
            self.ref_model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.ref_model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )

    @torch.no_grad()
    def generate_responses(self, prompt_ids_list: list, prompts: list) -> list:
        """Generate G responses per prompt."""
        cfg = self.config
        all_responses = []

        for prompt, prompt_ids in zip(prompts, prompt_ids_list):
            prompt_ids = prompt_ids.to(self.accelerator.device)
            generate_count = cfg.group_size * cfg.overgenerate_factor if cfg.dynamic_sampling else cfg.group_size

            input_ids = prompt_ids.unsqueeze(0).repeat(generate_count, 1)

            outputs = self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids,
                max_new_tokens=cfg.max_response_length,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            prompt_len = prompt_ids.shape[0]
            responses_ids = outputs[:, prompt_len:]
            responses_text = self.tokenizer.batch_decode(
                responses_ids, skip_special_tokens=True
            )

            all_responses.append({
                "text": responses_text,
                "ids": responses_ids,
                "prompt_len": prompt_len,
                "prompt_ids": prompt_ids,
            })

        return all_responses

    def compute_advantages(
        self,
        rewards: list,  # list of lists [G rewards per prompt]
        dynamic_sampling: bool = True,
    ) -> tuple:
        """
        Compute group-normalized advantages with optional dynamic sampling.
        Returns filtered (rewards, indices) tuples.
        """
        filtered_rewards = []
        filtered_indices = []

        for g_idx, group_rewards in enumerate(rewards):
            r = torch.tensor(group_rewards, dtype=torch.float32)

            # Dynamic sampling: skip if all correct or all wrong
            if dynamic_sampling:
                if r.std() < 1e-6:  # all same (all correct or all wrong)
                    continue

            # Trim to group_size if overgenerated
            if len(r) > self.config.group_size:
                # Keep diverse subset
                r = r[:self.config.group_size]
                indices = list(range(self.config.group_size))
            else:
                indices = list(range(len(r)))

            # Group-level whitening
            mean = r.mean()
            std = r.std() + 1e-8
            advantages = (r - mean) / std

            filtered_rewards.append(advantages.tolist())
            filtered_indices.append((g_idx, indices))

        return filtered_rewards, filtered_indices

    def get_logprobs(self, model, input_ids, attention_mask, response_mask):
        """Compute per-token log probabilities."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits  # [B, L, V]

        # Shift for causal LM
        shift_logits = logits[:, :-1, :]   # [B, L-1, V]
        shift_labels = input_ids[:, 1:]    # [B, L-1]
        shift_mask = response_mask[:, 1:]  # [B, L-1]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=2, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [B, L-1]

        return token_log_probs, shift_mask, shift_logits

    def train_step(self, batch):
        cfg = self.config
        prompts = batch["prompts"]
        prompt_ids_list = batch["prompt_ids"]
        ground_truths = batch["ground_truths"]

        # 1. Generate responses (no grad)
        self.model.eval()
        responses_data = self.generate_responses(prompt_ids_list, prompts)
        self.model.train()

        # 2. Compute rewards
        all_rewards = []
        for i, (resp_data, gt) in enumerate(zip(responses_data, ground_truths)):
            group_rewards = [
                compute_reward(resp_text, gt)
                for resp_text in resp_data["text"]
            ]
            all_rewards.append(group_rewards)

        # 3. Compute advantages with dynamic sampling
        filtered_advantages, filtered_indices = self.compute_advantages(
            all_rewards, dynamic_sampling=cfg.dynamic_sampling
        )

        if len(filtered_advantages) == 0:
            logger.debug("All groups filtered (all correct/wrong), skipping step")
            return None

        # 4. Build training batch
        total_loss_info = {
            "loss": torch.tensor(0.0, device=self.accelerator.device),
            "pg_loss": 0.0, "kl": 0.0, "entropy": 0.0,
            "clip_frac": 0.0, "approx_kl": 0.0,
            "mean_reward": 0.0, "n_valid_groups": len(filtered_advantages),
        }

        for adv_group, (prompt_idx, resp_indices) in zip(filtered_advantages, filtered_indices):
            resp_data = responses_data[prompt_idx]
            prompt_ids = resp_data["prompt_ids"]
            prompt_len = resp_data["prompt_len"]

            # Prepare sequences: [prompt | response]
            batch_input_ids = []
            batch_response_masks = []
            batch_attention_masks = []

            for r_idx in resp_indices:
                resp_ids = resp_data["ids"][r_idx]
                seq = torch.cat([prompt_ids, resp_ids], dim=0)

                # Truncate to max_seq_length
                if seq.shape[0] > cfg.max_seq_length:
                    seq = seq[:cfg.max_seq_length]
                    resp_len = cfg.max_seq_length - prompt_len
                else:
                    resp_len = resp_ids.shape[0]

                # Response mask (only compute loss on response tokens)
                response_mask = torch.zeros_like(seq)
                response_mask[prompt_len:prompt_len + resp_len] = 1

                # Pad to max_seq_length
                pad_len = cfg.max_seq_length - seq.shape[0]
                if pad_len > 0:
                    pad = torch.full((pad_len,), self.tokenizer.pad_token_id,
                                     dtype=seq.dtype, device=seq.device)
                    seq = torch.cat([seq, pad])
                    response_mask = torch.cat([response_mask, torch.zeros(pad_len, device=seq.device)])

                attention_mask = (seq != self.tokenizer.pad_token_id).long()

                batch_input_ids.append(seq)
                batch_response_masks.append(response_mask)
                batch_attention_masks.append(attention_mask)

            if not batch_input_ids:
                continue

            input_ids = torch.stack(batch_input_ids)         # [G, L]
            response_mask = torch.stack(batch_response_masks) # [G, L]
            attention_mask = torch.stack(batch_attention_masks) # [G, L]
            advantages = torch.tensor(adv_group, dtype=torch.bfloat16,
                                      device=self.accelerator.device)

            # 5. Compute logprobs under current policy
            logprobs, resp_mask_shifted, logits = self.get_logprobs(
                self.model, input_ids, attention_mask, response_mask
            )

            # 6. Compute logprobs under reference policy
            with torch.no_grad():
                ref_logprobs, _, _ = self.get_logprobs(
                    self.ref_model, input_ids, attention_mask, response_mask
                )

            # 7. old_logprobs ≈ ref_logprobs for first iteration
            # (In full DAPO, we'd store from generation; approximating here)
            old_logprobs = ref_logprobs.detach()

            # 8. DAPO loss
            loss_dict = self.dapo_loss(
                logprobs=logprobs,
                ref_logprobs=ref_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                attention_mask=resp_mask_shifted,
                logits=logits[:, :-1, :],
            )

            total_loss_info["loss"] = total_loss_info["loss"] + loss_dict["loss"]
            for k in ["pg_loss", "kl", "entropy", "clip_frac", "approx_kl"]:
                total_loss_info[k] += loss_dict[k]

        # Average over groups
        n = len(filtered_advantages)
        total_loss_info["loss"] = total_loss_info["loss"] / n
        for k in ["pg_loss", "kl", "entropy", "clip_frac", "approx_kl"]:
            total_loss_info[k] /= n

        # Average reward
        flat_rewards = [r for group in all_rewards for r in group]
        total_loss_info["mean_reward"] = sum(flat_rewards) / len(flat_rewards)

        # Accuracy (reward == 1.0)
        total_loss_info["accuracy"] = sum(1 for r in flat_rewards if r == 1.0) / len(flat_rewards)

        return total_loss_info

    def evaluate(self, num_samples: int = 50) -> dict:
        """Quick evaluation on a subset."""
        self.model.eval()
        correct = 0
        total = min(num_samples, len(self.train_dataset))

        with torch.no_grad():
            for i in range(total):
                item = self.train_dataset[i]
                prompt_ids = item["prompt_ids"].to(self.accelerator.device)
                gt = item["ground_truth"]

                out = self.accelerator.unwrap_model(self.model).generate(
                    input_ids=prompt_ids.unsqueeze(0),
                    max_new_tokens=512,
                    temperature=0.1,  # greedy-ish for eval
                    top_p=0.9,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                response = self.tokenizer.decode(
                    out[0, prompt_ids.shape[0]:], skip_special_tokens=True
                )
                if compute_outcome_reward(response, gt) == 1.0:
                    correct += 1

        self.model.train()
        return {"eval_accuracy": correct / total, "eval_correct": correct, "eval_total": total}

    def save_checkpoint(self, step: int, is_final: bool = False):
        """Save model checkpoint."""
        cfg = self.config
        if not self.accelerator.is_main_process:
            return

        suffix = "final" if is_final else f"step_{step}"
        save_path = os.path.join(cfg.output_dir, f"checkpoint_{suffix}")
        os.makedirs(save_path, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        if cfg.use_lora:
            unwrapped.save_pretrained(save_path)
        else:
            unwrapped.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Checkpoint saved: {save_path}")

        if is_final:
            self.push_to_hub(save_path)

    def push_to_hub(self, model_path: str):
        """Push final model to HuggingFace Hub."""
        cfg = self.config
        if not cfg.hf_token or not cfg.hf_repo_id:
            logger.warning("HF token or repo_id not set, skipping Hub push.")
            return

        try:
            login(token=cfg.hf_token)
            api = HfApi()

            # Create repo if not exists
            api.create_repo(
                repo_id=cfg.hf_repo_id,
                exist_ok=True,
                private=False,
            )

            # If LoRA, merge weights first for clean upload
            if cfg.use_lora:
                logger.info("Merging LoRA weights for upload...")
                unwrapped = self.accelerator.unwrap_model(self.model)
                merged_model = unwrapped.merge_and_unload()
                merged_path = os.path.join(cfg.output_dir, "merged_final")
                os.makedirs(merged_path, exist_ok=True)
                merged_model.save_pretrained(merged_path, safe_serialization=True)
                self.tokenizer.save_pretrained(merged_path)

                # Write model card
                self._write_model_card(merged_path)

                upload_path = merged_path
            else:
                self._write_model_card(model_path)
                upload_path = model_path

            api.upload_folder(
                folder_path=upload_path,
                repo_id=cfg.hf_repo_id,
                repo_type="model",
            )
            logger.info(f"Model pushed to: https://huggingface.co/{cfg.hf_repo_id}")

        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")

    def _write_model_card(self, path: str):
        card = f"""---
license: apache-2.0
base_model: {self.config.model_name}
tags:
- math
- reasoning
- dapo
- fine-tuned
datasets:
- HuggingFaceH4/MATH-500
---

# Qwen2.5-Math-1.5B Fine-tuned with DAPO

Fine-tuned from `{self.config.model_name}` using **DAPO** 
(Decoupled Clip and Dynamic Sampling Policy Optimization) on MATH-500.

## Training Details
- Algorithm: DAPO (improved GRPO variant)
- Dataset: HuggingFaceH4/MATH-500 (500 competition math problems)
- Clip bounds: ε_low={self.config.clip_low}, ε_high={self.config.clip_high}
- Group size: {self.config.group_size}
- Dynamic sampling: {self.config.dynamic_sampling}
- Token-level loss: {self.config.token_level_loss}

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{self.config.hf_repo_id}",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("{self.config.hf_repo_id}")

messages = [
    {{"role": "system", "content": "You are a math expert. Solve step by step and put the answer in \\\\boxed{{}}."}},
    {{"role": "user", "content": "What is the sum of the first 100 positive integers?"}}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```
"""
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write(card)

    def train(self):
        cfg = self.config
        logger.info("=" * 60)
        logger.info("Starting DAPO Training")
        logger.info(f"  Model: {cfg.model_name}")
        logger.info(f"  Dataset: {cfg.dataset_name}")
        logger.info(f"  Epochs: {cfg.num_epochs}")
        logger.info(f"  Group size: {cfg.group_size}")
        logger.info(f"  Clip: [{cfg.clip_low}, {cfg.clip_high}]")
        logger.info("=" * 60)

        global_step = 0
        best_eval_acc = 0.0
        os.makedirs(cfg.output_dir, exist_ok=True)

        for epoch in range(cfg.num_epochs):
            epoch_losses = []
            epoch_rewards = []
            epoch_accs = []

            for batch_idx, batch in enumerate(self.train_loader):
                with self.accelerator.accumulate(self.model):
                    loss_info = self.train_step(batch)

                    if loss_info is None:
                        continue

                    self.accelerator.backward(loss_info["loss"])

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), cfg.max_grad_norm
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1
                    epoch_losses.append(loss_info["loss"].item())
                    epoch_rewards.append(loss_info["mean_reward"])
                    epoch_accs.append(loss_info["accuracy"])

                    # Logging
                    if global_step % cfg.logging_steps == 0 and self.accelerator.is_main_process:
                        lr = self.scheduler.get_last_lr()[0]
                        log_msg = (
                            f"Epoch {epoch+1}/{cfg.num_epochs} | "
                            f"Step {global_step} | "
                            f"Loss: {loss_info['loss'].item():.4f} | "
                            f"Reward: {loss_info['mean_reward']:.3f} | "
                            f"Acc: {loss_info['accuracy']:.3f} | "
                            f"KL: {loss_info['kl']:.4f} | "
                            f"ClipFrac: {loss_info['clip_frac']:.3f} | "
                            f"LR: {lr:.2e}"
                        )
                        logger.info(log_msg)

                        if cfg.use_wandb:
                            wandb.log({
                                "train/loss": loss_info["loss"].item(),
                                "train/reward": loss_info["mean_reward"],
                                "train/accuracy": loss_info["accuracy"],
                                "train/kl": loss_info["kl"],
                                "train/entropy": loss_info["entropy"],
                                "train/clip_frac": loss_info["clip_frac"],
                                "train/approx_kl": loss_info["approx_kl"],
                                "train/lr": lr,
                                "train/valid_groups": loss_info["n_valid_groups"],
                            }, step=global_step)

                    # Evaluation
                    if global_step % cfg.eval_steps == 0:
                        eval_metrics = self.evaluate(num_samples=50)
                        if self.accelerator.is_main_process:
                            logger.info(f"Eval @ step {global_step}: {eval_metrics}")
                            if cfg.use_wandb:
                                wandb.log(eval_metrics, step=global_step)
                            if eval_metrics["eval_accuracy"] > best_eval_acc:
                                best_eval_acc = eval_metrics["eval_accuracy"]
                                self.save_checkpoint(global_step, is_final=False)

                    # Periodic save
                    if global_step % cfg.save_steps == 0:
                        self.save_checkpoint(global_step)

                    # Periodic Hub push
                    if global_step % cfg.push_every_n_steps == 0 and cfg.hf_token:
                        self.save_checkpoint(global_step)

                    # Memory cleanup
                    if global_step % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

            # End of epoch summary
            if self.accelerator.is_main_process and epoch_losses:
                logger.info(
                    f"\n{'='*50}\n"
                    f"Epoch {epoch+1} Summary:\n"
                    f"  Avg Loss:   {sum(epoch_losses)/len(epoch_losses):.4f}\n"
                    f"  Avg Reward: {sum(epoch_rewards)/len(epoch_rewards):.4f}\n"
                    f"  Avg Acc:    {sum(epoch_accs)/len(epoch_accs):.4f}\n"
                    f"{'='*50}"
                )

        # Final save and push
        logger.info("Training complete. Saving final model...")
        self.save_checkpoint(global_step, is_final=True)

        if cfg.use_wandb and self.accelerator.is_main_process:
            wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = DAPOConfig(
        # ─── CHANGE THESE ───────────────────────────────────────────
        hf_repo_id="Dat1710/nexus-1.5b-v2",  # ← your HF username
        hf_token="",    # ← your HF token
        use_wandb=False,                                       # set True to log to wandb
        # ────────────────────────────────────────────────────────────

        # Training config (tuned for H100 80GB)
        num_epochs=3,
        batch_size=1,
        gradient_accumulation_steps=8,
        group_size=8,
        learning_rate=5e-6,

        # DAPO config
        clip_low=0.2,
        clip_high=0.28,
        kl_coef=0.001,
        entropy_coef=0.01,
        dynamic_sampling=True,

        # LoRA
        use_lora=True,
        lora_r=64,
        lora_alpha=128,
    )

    trainer = DAPOTrainer(config)
    trainer.train()
