import os
import random

import numpy as np
import torch
from accelerate import Accelerator


def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accel_break(bad_process_index=0):
    accelerator = Accelerator()
    if accelerator.process_index == bad_process_index:
        breakpoint()
    accelerator.wait_for_everyone()
