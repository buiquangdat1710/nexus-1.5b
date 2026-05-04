import re
import torch
import logging
from transformers import AutoTokenizer, AutoModel

log = logging.getLogger(__name__)

class RuleBasedRewardScorer:
    def __init__(self, device: torch.device):
        self.device = device
        log.info("Đang khởi tạo Rule-Based Reward Scorer...")

    def extract_boxed_answer(self, text: str) -> str:

        # tìm pattern \boxed{}
        matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
        if matches:
            return matches[-1].strip()
        
        # fallback
        fallback = re.findall(r'The answer is:?\s*\$?([^\$\n]+)\$?', text, re.IGNORECASE)
        if fallback:
            return fallback[-1].strip()
            
        return ""

    def normalize_answer(self, ans: str) -> str:
        if not ans: return ""
        ans = ans.replace(" ", "").lower().replace(",", "")
        ans = ans.rstrip('.')
        return ans

    def get_scores(self, responses: list[str], gold_answer: str) -> torch.Tensor:
        """
        - Đúng đáp án: +1.0 (Phần thưởng tối đa)
        - Có \boxed{} nhưng tính sai: -0.5 (Phạt nhẹ)
        - Không đưa ra được kết luận / Lạc đề: -1.0 (Phạt nặng)
        """
        rewards = []
        norm_gold = self.normalize_answer(gold_answer)
        
        for resp in responses:
            pred_ans = self.extract_boxed_answer(resp)
            
            if not pred_ans:
                rewards.append(-1.0)
                continue
                
            norm_pred = self.normalize_answer(pred_ans)
            
            if norm_pred == norm_gold:
                rewards.append(1.0)
            else:
                rewards.append(-0.5)
                
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    

class RewardModelScorer:
    def __init__(self, rm_name: str, dtype: torch.dtype, device: str):
        self.device = device
        log.info(f"Đang khởi tạo Reward Model {rm_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            rm_name, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModel.from_pretrained(
            rm_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device
        )
        self.model.eval()

    def get_reward(self, questions: list[str], responses: list[str]) -> torch.Tensor:

        rewards = []
        for q, r in zip(questions, responses):
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": r}
            ]
            
            # Format
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                if hasattr(outputs, "score"):
                    score = outputs.score[0].item()
                elif hasattr(outputs, "logits"):
                    score = outputs.logits[0, 0].item()
                else:
                    # Fallback
                    score = outputs[0][0].item()
                    
            rewards.append(score)
            
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)