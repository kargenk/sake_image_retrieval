import os
import random

import numpy as np
import torch


def fix_seed(seed: int = 3407) -> None:
    """
    再現性の担保のために乱数のシード値を固定する関数.

    Args:
        seed (int): シード値
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)     # random
    np.random.seed(seed)  # NumPy
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False