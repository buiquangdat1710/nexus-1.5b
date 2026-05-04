import torch
from typing import Tuple

def compute_dapo_token_loss(
    new_logprobs: torch.Tensor, 
    old_logprobs: torch.Tensor, 
    advantage: float, 
    mask: torch.Tensor, 
    eps_low: float = 0.20, 
    eps_high: float = 0.28
) -> Tuple[torch.Tensor, int]:

    ratio = torch.exp(new_logprobs - old_logprobs)
    ratio_clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)

    adv_tensor = torch.full_like(new_logprobs, advantage)
    surrogate1 = ratio * adv_tensor
    surrogate2 = ratio_clipped * adv_tensor

    # Pessimistic bound (min) và đổi dấu vì PyTorch dùng Gradient Descent (minimize)
    token_losses = -torch.min(surrogate1, surrogate2)
    
    # Chỉ tính loss trên các token không phải padding
    masked_loss = token_losses * mask
    
    return masked_loss.sum(), mask.sum().item()