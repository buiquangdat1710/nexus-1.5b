import logging
import torch
from transformers import AutoModelForCausalLM

log = logging.getLogger(__name__)

def load_policy_and_ref_models(model_name: str, dtype: torch.dtype, device_map: str = "auto"):
    """Loads the trainable Actor (Policy) and frozen Reference models."""
    
    log.info(f"Loading Actor model (Policy): {model_name}")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype, 
        device_map=device_map, 
        trust_remote_code=True
    )
    policy_model.gradient_checkpointing_enable()
    policy_model.train()

    log.info(f"Loading Reference model (Frozen): {model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype, 
        device_map=device_map, 
        trust_remote_code=True
    )
    ref_model.eval()
    
    # freeze reference model parameters
    for param in ref_model.parameters(): 
        param.requires_grad_(False)

    return policy_model, ref_model