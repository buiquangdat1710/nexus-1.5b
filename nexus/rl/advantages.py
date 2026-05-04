import numpy as np
from typing import List

def compute_lpro_advantages(
    rewards: List[float], 
    lengths: List[int], 
    lambda_len: float = 0.10, 
    eps: float = 1e-8
) -> np.ndarray:

    r = np.array(rewards, dtype=np.float64)
    L = np.array(lengths, dtype=np.float64)

    # Z-score
    z_r = (r - r.mean()) / (r.std() + eps)
    z_L = (L - L.mean()) / (L.std() + eps)

    advantages = z_r - lambda_len * z_L
    return advantages