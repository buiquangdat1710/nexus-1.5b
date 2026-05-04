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

    def _process_dataset(self, dataset_split, tokenizer) -> List[Dict]:
        """Hàm xử lý chung để encode prompt và trích xuất gold_answer."""
        out = []
        for item in dataset_split:
            problem = item.get("problem", "")
            gold_answer = item.get("answer", "") 
            
            if not problem: continue
                
            txt = self.format_chat_prompt(problem, tokenizer)
            ids = tokenizer.encode(txt, add_special_tokens=False)
            
            if len(ids) <= self.max_prompt_len:
                out.append({
                    "prompt": txt, 
                    "prompt_ids": ids,
                    "gold_answer": gold_answer
                })
                
        return out

    def load_train_data(self, tokenizer) -> List[Dict]:
        log.info(f"Đang load data train từ {self.dataset_name}...")

        ds = load_dataset(self.dataset_name, split="test") 
        
        # 100 sample đầu
        train_ds = ds.select(range(100))
        
        out = self._process_dataset(train_ds, tokenizer)
        log.info(f"Đã load {len(out)} training examples.")
        return out

    def load_val_data(self, tokenizer) -> List[Dict]:
        log.info(f"Đang load data validation từ {self.dataset_name}...")
        ds = load_dataset(self.dataset_name, split="test") 
        
        # 400 sample 
        val_ds = ds.select(range(100, len(ds)))
        
        out = self._process_dataset(val_ds, tokenizer)
        log.info(f"Đã load {len(out)} validation examples.")
        return out

if __name__ == "__main__":
    # test
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    
    builder = MathDatasetBuilder("HuggingFaceH4/MATH-500")
    
    train_data = builder.load_train_data(tokenizer)
    val_data = builder.load_val_data(tokenizer)
    
    print("Example:")
    print("Prompt:", train_data[0]["prompt"][:100], "...")
    print("Gold Answer:", train_data[0]["gold_answer"])