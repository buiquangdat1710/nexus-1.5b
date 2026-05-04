import os
import math
import random
import logging
import torch
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from nexus.models.policy import load_policy_and_ref_models
from nexus.models.reward import RewardModelScorer, RuleBasedRewardScorer
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
        if getattr(self.cfg, "use_lora", False):
            log.info("Finetune LoRA: Đang cấu hình PEFT/LoRA adapter...")
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                target_modules=self.cfg.lora_target_modules
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        else:
            log.info("Full Finetune: Đang cấu hình để fine-tune toàn bộ model...")

        # Setup Reward Scorer
        if getattr(self.cfg, "use_rule_based_rm", True):
            self.rm_scorer = RuleBasedRewardScorer(self.device)
        else:
            self.rm_scorer = RewardModelScorer(self.cfg.rm_model_name, self.dtype, self.device)

    @staticmethod
    def get_resp_log_probs(model, full_ids: torch.Tensor, prompt_len: int, no_grad: bool = False) -> torch.Tensor:
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            out = model(input_ids=full_ids)
            lp = torch.log_softmax(out.logits[0], dim=-1)
            resp_ids = full_ids[0, prompt_len:]
            resp_lp = lp[prompt_len - 1 : full_ids.shape[1] - 1]
            return resp_lp.gather(1, resp_ids.unsqueeze(-1)).squeeze(-1)

    # eval per epoch
    def evaluate(self, val_dataset):
        self.model.eval()
        correct = 0
        total = len(val_dataset)
        
        log.info(f"Đang evaluation trên {total} samples...")
        
        scorer = self.rm_scorer if isinstance(self.rm_scorer, RuleBasedRewardScorer) else RuleBasedRewardScorer(self.device)
        
        pbar = tqdm(val_dataset, desc="Evaluating", leave=False)
        
        for example in pbar:
            prompt_ids = example["prompt_ids"]
            gold_answer = example.get("gold_answer", "")
            
            prompt_len = len(prompt_ids)
            prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.cfg.bf16):
                    gen_out = self.model.generate(
                        prompt_t,
                        max_new_tokens=self.cfg.max_new_tokens,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            
            resp_ids = gen_out[0, prompt_len:]
            pad_mask = (resp_ids != self.tokenizer.eos_token_id) & (resp_ids != self.tokenizer.pad_token_id)
            actual_len = max(pad_mask.sum().item(), 1)
            resp_text = self.tokenizer.decode(resp_ids[:actual_len], skip_special_tokens=True)
            
            # score reward
            pred_ans = scorer.extract_boxed_answer(resp_text)
            if scorer.normalize_answer(pred_ans) == scorer.normalize_answer(gold_answer) and gold_answer:
                correct += 1
            
            pbar.set_postfix({"Acc": f"{(correct/(pbar.n+1))*100:.2f}%"})
            
            del prompt_t, gen_out
            torch.cuda.empty_cache()
            
        acc = (correct / total) * 100
        log.info(f"Evaluate xong. Accuracy = {acc:.2f}% ({correct}/{total})")
        
        self.model.train()
        return acc

    # training loop
    def train(self):
        # Prepare Data & Optimizer
        dataset_builder = MathDatasetBuilder(self.cfg.dataset_name, self.cfg.max_prompt_len)
        
        train_dataset = dataset_builder.load_train_data(self.tokenizer)
        val_dataset = dataset_builder.load_val_data(self.tokenizer)

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        total_steps = self.cfg.num_epochs * math.ceil(len(train_dataset) / self.cfg.grad_accum)
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(self.cfg.warmup_ratio * total_steps), total_steps)
        
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        global_step, acc_loss, acc_reward = 0, 0.0, 0.0

        # Training Loop
        for epoch in range(self.cfg.num_epochs):
            random.shuffle(train_dataset)
            log.info(f"Epoch {epoch + 1}/{self.cfg.num_epochs} (TRAIN)")
            
            pbar = tqdm(enumerate(train_dataset), total=len(train_dataset), desc=f"Epoch {epoch + 1}/{self.cfg.num_epochs}")

            for idx, example in pbar:
                prompt_ids = example["prompt_ids"]
                prompt_txt = example["prompt"]
                gold_answer = example.get("gold_answer", "")

                prompt_len = len(prompt_ids)
                prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

                # Generate responses
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

                # process generations
                lengths, masks, full_seqs, resp_texts = [], [], [], []
                for i in range(self.cfg.G):
                    resp_ids = gen_out[i, prompt_len:]
                    pad_mask = (resp_ids != self.tokenizer.eos_token_id) & (resp_ids != self.tokenizer.pad_token_id)
                    actual_len = max(pad_mask.sum().item(), 1)
                    
                    lengths.append(actual_len)
                    masks.append(pad_mask.to(self.device))
                    full_seqs.append(gen_out[i])
                    resp_texts.append(self.tokenizer.decode(resp_ids[:actual_len], skip_special_tokens=True))

                # Reward
                if isinstance(self.rm_scorer, RuleBasedRewardScorer):
                    rewards = self.rm_scorer.get_scores(resp_texts, gold_answer)
                else:
                    rewards = self.rm_scorer.get_reward([prompt_txt]*self.cfg.G, resp_texts)
                
                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

                if torch.all(rewards == rewards[0]): 
                    del gen_out, prompt_t
                    torch.cuda.empty_cache()
                    continue 
                
                advs = compute_lpro_advantages(rewards.tolist(), lengths, self.cfg.lambda_len)

                # Compute loss
                total_loss_sum = torch.tensor(0.0, device=self.device)
                total_n_tokens = 0

                for i in range(self.cfg.G):
                    seq = full_seqs[i].unsqueeze(0).to(self.device)[:, :prompt_len + lengths[i]]
                    mask_i = masks[i][:lengths[i]]

                    with torch.amp.autocast('cuda', enabled=self.cfg.bf16):
                        new_lp = self.get_resp_log_probs(self.model, seq, prompt_len, no_grad=False)
                    old_lp = self.get_resp_log_probs(self.ref_model, seq, prompt_len, no_grad=True)

                    loss_sum, n_valid = compute_dapo_token_loss(
                        new_lp, old_lp.detach(), float(advs[i]), mask_i, self.cfg.eps_low, self.cfg.eps_high
                    )
                    total_loss_sum += loss_sum
                    total_n_tokens += n_valid

                if total_n_tokens == 0: 
                    del gen_out, prompt_t, seq, new_lp, old_lp, loss_sum
                    torch.cuda.empty_cache()
                    continue
                
                # Backprop
                loss = total_loss_sum / total_n_tokens / self.cfg.grad_accum
                loss.backward()

                acc_loss += loss.item() * self.cfg.grad_accum
                acc_reward += rewards.float().mean().item()

                pbar.set_postfix({
                    "loss": f"{loss.item() * self.cfg.grad_accum:.4f}", 
                    "reward": f"{rewards.float().mean().item():.2f}"
                })

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

                del gen_out, prompt_t, seq, new_lp, old_lp, loss_sum, total_loss_sum
                torch.cuda.empty_cache()
            
            log.info(f"Hoàn thành Train Epoch {epoch + 1}. Bắt đầu Eval...")
            self.evaluate(val_dataset)

        log.info("Training hoàn tất. Đang lưu model...")
        
        # save
        self.model.save_pretrained(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)

        # push to hub
        if getattr(self.cfg, "push_to_hub", False):
            if not self.cfg.hf_token:
                log.error("Chưa cung cấp hf_token trong Config.")
            else:
                log.info(f"Đang đẩy model lên Hugging Face Hub ({self.cfg.hub_repo_id})...")
                try:
                    # weights
                    self.model.push_to_hub(
                        self.cfg.hub_repo_id, 
                        token=self.cfg.hf_token,
                        commit_message="Upload Nexus Qwen-Math weights"
                    )
                    # tokenizer
                    self.tokenizer.push_to_hub(
                        self.cfg.hub_repo_id, 
                        token=self.cfg.hf_token,
                        commit_message="Upload tokenizer"
                    )
                    log.info("Đã đẩy model lên Hugging Face Hub thành công!")
                except Exception as e:
                    log.error(f"Có lỗi xảy ra khi đẩy lên Hub: {e}")