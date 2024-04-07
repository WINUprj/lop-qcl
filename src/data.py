from abc import ABC
from pathlib import Path
from pennylane import numpy as np
import torchvision


class LabelPermutedEMNIST:
    def __init__(self, root: Path):
        self.root = root
        super(LabelPermutedEMNIST, self).__init__()