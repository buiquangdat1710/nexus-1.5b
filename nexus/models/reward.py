import torch
import logging
from transformers import AutoTokenizer, AutoModel

log = logging.getLogger(__name__)

class RewardModelScorer:
    def __init__(self, rm_name: str, dtype: torch.dtype, device: str):
        self.device = device
        log.info(f"Initializing Neural Reward Model: {rm_name}")
        
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