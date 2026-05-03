import logging
from typing import List, Dict
from datasets import load_dataset

log = logging.getLogger(__name__)

class MathDatasetBuilder:
    def __init__(self, dataset_name: str, max_prompt_len: int = 512):
        self.dataset_name = dataset_name
        self.max_prompt_len = max_prompt_len
        self.system_prompt = (
            "You are a mathematics expert. "
            "Solve the problem step by step and enclose your final answer in \\boxed{}."
        )

    def format_chat_prompt(self, problem: str, tokenizer) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def load_train_data(self, tokenizer) -> List[Dict]:
        log.info(f"Loading dataset: {self.dataset_name}")
        ds = load_dataset(self.dataset_name, split="test") 
        out = []
        for item in ds:
            problem = item.get("problem", "")
            if not problem: continue
                
            txt = self.format_chat_prompt(problem, tokenizer)
            ids = tokenizer.encode(txt, add_special_tokens=False)
            if len(ids) <= self.max_prompt_len:
                out.append({"prompt": txt, "prompt_ids": ids})
                
        log.info(f"Loaded {len(out)} valid training examples.")
        return out
    
if __name__ == "__main__":

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test") 
    out = []
    for item in ds:
        problem = item.get("problem", "")
        print(problem)
        break
