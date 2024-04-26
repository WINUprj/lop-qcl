from torch.optim import SGD, Adam, RMSprop
from torch.nn import functional as F

from src.data import LabelPermutedEMNIST
from src.model import TorchHybridModel, ClassicalReLUFCNN


TASKS = {"LabelPermutedEMNIST": LabelPermutedEMNIST}

MODELS = {"TorchHybridModel": TorchHybridModel, "ClassicalReLUFCNN": ClassicalReLUFCNN}

OPTIMS = {"SGD": SGD, "Adam": Adam, "RMSProp": RMSprop}

LOSSES = {"mse_loss": F.mse_loss, "cross_entropy": F.cross_entropy, "binary_cross_entropy_with_logits": F.binary_cross_entropy_with_logits}

