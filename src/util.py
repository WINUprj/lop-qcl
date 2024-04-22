import argparse
import copy
import itertools
import json
import os
import random

import numpy as np
from pathlib import Path
import torch
from torchvision.transforms import v2 as T


def get_root_dir() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).parent.parent.resolve()


def mkdir(dir_path: Path):
    """Make the given directory."""
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


def create_hyperparam_dirname(search_keys: list, cfg: dict):
    dirname = ""
    for i, key in search_keys:
        dirname += f"{key}_{cfg[key]}{'_' if i < len(search_keys) - 1 else ''}"
    
    return dirname


def seed_everything(seed: int):
    """Set seed of the entire environment."""
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


def moving_avg(data: np.ndarray, n: int):
    cs = np.cumsum(data)
    diff = cs[n:] - cs[:-n]
    return diff / n


def accuracy(pred, label):
    return np.sum(pred.argmax(axis=1) == label) / pred.shape[0]


def parse_and_process_cfg():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, default="./configs/default.json", help="Path to the config file for the experiments")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        orig_cfg = json.load(f)

    # Prepare directories
    orig_cfg["root_dir"] = get_root_dir()
    orig_cfg["exp_dir"] = orig_cfg["root_dir"] / f"results/{orig_cfg['exp_name']}"
    mkdir(orig_cfg["exp_dir"])
    orig_cfg["task_params"]["dataset_root"] = orig_cfg["root_dir"] / orig_cfg["task_params"]["dataset_root"]
    
    # Image transforms
    orig_cfg["task_params"]["img_transform"] = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(orig_cfg["task_params"]["in_dim"]),
        T.Normalize((0.5, ), (0.5, )),
        T.Lambda(lambda x: x.flatten()),
    ])

    return orig_cfg


class HyperParamSearcher():
    def __init__(self, cfg, search_keys):
        self.cfg = cfg
        self.search_keys = search_keys
        
        if len(self.search_keys) > 0:
            value_lists = []
            for k in self.search_keys:
                assert isinstance(self.cfg[k], list), "Searchable parameters must be in the list format"
                value_lists.append(self.cfg[k])
            self.search_values = itertools.product(*value_lists)
        else:
            self.search_values = [None]

        self.idx = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        try:
            cp_cfg = copy.deepcopy(self.cfg)
            h_values = self.search_values[self.idx]
            self.idx += 1
            if h_values == None:
                return cp_cfg
            else:
                for i, k in enumerate(self.search_keys):
                    cp_cfg[k] = h_values[i]
                return cp_cfg
        except IndexError:
            raise StopIteration()
