import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from tqdm import tqdm

class MathEvaluator:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        """Khởi tạo HuggingFace Transformers engine."""
        self.log = logging.getLogger(__name__)
        self.log.info(f"Initializing HuggingFace Transformers from {model_path}...")
        
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        self.log.info(f"Using device: {self.device}, dtype: {dtype}")

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        self.model.eval()

    def generate_answers(self, prompts: list[str], temperature: float = 0.0) -> list[str]:
        """Sinh câu trả lời cho danh sách prompts."""
        predictions = []
        
        print("Đang sinh câu trả lời...")
        for prompt in tqdm(prompts, desc="Generating"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # params
            gen_kwargs = {
                "max_new_tokens": 2048,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                 gen_kwargs["temperature"] = temperature
                 gen_kwargs["top_p"] = 0.9

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            input_length = inputs.input_ids.shape[1]
            generated_ids = outputs[0][input_length:]
            
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # remove tag <|im_end|>
            response = response.replace("<|im_end|>", "").strip()
            predictions.append(response)
            
        return predictions
