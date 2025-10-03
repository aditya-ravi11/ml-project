# ultralytics/utils/repro.py
"""Reproducibility utilities for deterministic training and inference."""

def seed_all(seed: int = 42):
    """
    Set random seeds across all libraries for reproducibility.

    Args:
        seed (int): Random seed to use (default: 42)
    """
    import os
    import random
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
