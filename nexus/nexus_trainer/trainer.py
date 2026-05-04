import os
import math
import random
import logging
import torch
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from nexus.models.policy import load_policy_and_ref_models
from nexus.models.reward import RewardModelScorer
from nexus.rl.advantages import compute_lpro_advantages
from nexus.rl.loss import compute_dapo_token_loss
from nexus.data.builder import MathDatasetBuilder

log = logging.getLogger(__name__)

class NexusTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.bf16 and torch.cuda.is_bf16_supported() else torch.float32
        
        self._set_seed()
        self._setup_components()

    def _set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _setup_components(self):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Models
        self.model, self.ref_model = load_policy_and_ref_models(self.cfg.model_name, self.dtype)
        self.rm_scorer = RewardModelScorer(self.cfg.rm_model_name, self.dtype, self.device)

    @staticmethod
    def get_resp_log_probs(model, full_ids: torch.Tensor, prompt_len: int, no_grad: bool = False) -> torch.Tensor:
        """Hàm helper để lấy log probabilities của token sinh ra."""
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            out = model(input_ids=full_ids)
            lp = torch.log_softmax(out.logits[0], dim=-1)
            resp_ids = full_ids[0, prompt_len:]
            resp_lp = lp[prompt_len - 1 : full_ids.shape[1] - 1]
            return resp_lp.gather(1, resp_ids.unsqueeze(-1)).squeeze(-1)

    def train(self):
        # Prepare Data & Optimizer
        dataset_builder = MathDatasetBuilder(self.cfg.dataset_name, self.cfg.max_prompt_len)
        dataset = dataset_builder.load_train_data(self.tokenizer)

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        total_steps = self.cfg.num_epochs * math.ceil(len(dataset) / self.cfg.grad_accum)
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(self.cfg.warmup_ratio * total_steps), total_steps)
        
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        global_step, acc_loss, acc_reward = 0, 0.0, 0.0

        # Training Loop
        for epoch in range(self.cfg.num_epochs):
            random.shuffle(dataset)
            log.info(f"========== EPOCH {epoch + 1}/{self.cfg.num_epochs} ==========")

            for idx, example in enumerate(dataset):
                prompt_ids = example["prompt_ids"]
                prompt_txt = example["prompt"]
                prompt_len = len(prompt_ids)
                prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

                # Generate Responses
                self.model.eval()
                with torch.no_grad():
                    gen_out = self.model.generate(
                        prompt_t.expand(self.cfg.G, -1),
                        max_new_tokens=self.cfg.max_new_tokens,
                        temperature=self.cfg.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                self.model.train()

                # Parse Lengths & Clean Texts
                lengths, masks, full_seqs, resp_texts = [], [], [], []
                for i in range(self.cfg.G):
                    resp_ids = gen_out[i, prompt_len:]
                    pad_mask = (resp_ids != self.tokenizer.eos_token_id) & (resp_ids != self.tokenizer.pad_token_id)
                    actual_len = max(pad_mask.sum().item(), 1)
                    
                    lengths.append(actual_len)
                    masks.append(pad_mask.to(self.device))
                    full_seqs.append(gen_out[i])
                    resp_texts.append(self.tokenizer.decode(resp_ids[:actual_len], skip_special_tokens=True))

                # Reward & LPRO Advantages
                rewards = self.rm_scorer.get_scores([prompt_txt]*self.cfg.G, resp_texts)
                if len(set(rewards)) <= 1: continue # Skip if all rewards are the same
                
                advs = compute_lpro_advantages(rewards, lengths, self.cfg.lambda_len)

                # Loss Computation
                total_loss_sum = torch.tensor(0.0, device=self.device)
                total_n_tokens = 0

                for i in range(self.cfg.G):
                    seq = full_seqs[i].unsqueeze(0).to(self.device)[:, :prompt_len + lengths[i]]
                    mask_i = masks[i][:lengths[i]]

                    with torch.cuda.amp.autocast(enabled=self.cfg.bf16):
                        new_lp = self.get_resp_log_probs(self.model, seq, prompt_len, no_grad=False)
                    old_lp = self.get_resp_log_probs(self.ref_model, seq, prompt_len, no_grad=True)

                    loss_sum, n_valid = compute_dapo_token_loss(
                        new_lp, old_lp.detach(), float(advs[i]), mask_i, self.cfg.eps_low, self.cfg.eps_high
                    )
                    total_loss_sum += loss_sum
                    total_n_tokens += n_valid

                if total_n_tokens == 0: continue

                # Token-level Normalization
                loss = total_loss_sum / total_n_tokens / self.cfg.grad_accum
                loss.backward()

                acc_loss += loss.item() * self.cfg.grad_accum
                acc_reward += float(np.mean(rewards))

                # Gradient Update
                if (idx + 1) % self.cfg.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.cfg.log_steps == 0:
                        log.info(f"Step {global_step} | Loss: {acc_loss/self.cfg.log_steps:.4f} | RM Score: {acc_reward/(self.cfg.log_steps*self.cfg.grad_accum):.3f}")
                        acc_loss, acc_reward = 0.0, 0.0

                    if global_step % self.cfg.save_steps == 0:
                        self.model.save_pretrained(os.path.join(self.cfg.output_dir, f"ckpt-{global_step}"))
                        self.tokenizer.save_pretrained(os.path.join(self.cfg.output_dir, f"ckpt-{global_step}"))

        log.info("Training complete. Saving final model...")
        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)