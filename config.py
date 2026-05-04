import os
import argparse
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # model
    model_name      : str   = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    rm_model_name   : str   = "Qwen/Qwen2.5-Math-RM-72B"
    output_dir      : str   = "./nexus-1.5b"
    hub_repo_id     : str   = "YOUR_HF_USERNAME/nexus-1.5b"

    # dataset
    dataset_name    : str   = "HuggingFaceH4/MATH-500"
    max_prompt_len  : int   = 1024

    # group sampling
    G               : int   = 32
    max_new_tokens  : int   = 2048
    temperature     : float = 0.7
    top_p           : float = 0.95

    # LPRO params
    eps_low         : float = 0.20   
    eps_high        : float = 0.28   
    lambda_len      : float = 0.1
    eps_r           : float = 1e-8
    eps_l           : float = 1e-8

    # training
    use_lora          : bool  = True
    use_rule_based_rm : bool  = True
    
    lora_r            : int   = 16
    lora_alpha        : int   = 32
    lora_dropout      : float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    num_epochs      : int   = 5
    lr              : float = 2e-5   # full finetune: 1e-5 - 5e-5, LoRA: 1e-4 - 5e-4
    weight_decay    : float = 1e-2
    warmup_ratio    : float = 0.05
    grad_clip       : float = 1.0
    grad_accum      : int   = 8      
    bf16            : bool  = True

    # logging
    log_steps       : int   = 10
    save_steps      : int   = 100
    seed            : int   = 42

    # huggingface hub
    push_to_hub     : bool  = True
    hf_token        : str   = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="DAPO + LPRO training for Qwen2.5-Math")
    p.add_argument("--model_name",    default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--output_dir",    default="./nexus-1.5b")
    p.add_argument("--hub_repo_id",   default="YOUR_HF_USERNAME/nexus-1.5b")
    p.add_argument("--G",             type=int,   default=32)
    p.add_argument("--num_epochs",    type=int,   default=5)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--lambda_len",    type=float, default=0.1)
    p.add_argument("--eps_low",       type=float, default=0.20)
    p.add_argument("--eps_high",      type=float, default=0.28)
    p.add_argument("--max_new_tokens",type=int,   default=1024)
    p.add_argument("--temperature",   type=float, default=0.70)
    p.add_argument("--grad_accum",    type=int,   default=8)
    p.add_argument("--save_steps",    type=int,   default=100)
    p.add_argument("--no_push",       action="store_true")
    p.add_argument("--hf_token",      default=os.getenv("HF_TOKEN", ""))
    
    args, _ = p.parse_known_args()
    cfg = Config()
    for k, v in vars(args).items():
        if k == "no_push":
            cfg.push_to_hub = not v
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg