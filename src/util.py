import os
import random

import numpy as np
from pathlib import Path
import torch


def get_root_dir() -> Path:
    return Path(__file__).parent.parent.resolve()


def mkdir(dir_path: Path):
    if dir_path.is_dir():
        dir_path.mkdir()


def seed_everything(seed: int):
    """
    Set seed of the entire environment.
    """
    # Standard Python3 library-related
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy-related
    np.random.seed(seed)

    # Torch-related
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


