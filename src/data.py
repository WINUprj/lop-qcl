import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as T

from src.util import get_root_dir


class LabelPermutedEMNIST:
    """
    Label permuted EMNIST task.
    """
    def __init__(
        self,
        dataset_root,
        in_dim=(16, 16),
        out_dim=26,
        train=True,
        shuffle=True,
        update_freq=1000,
        batch_size=1,
    ):
        self.dataset_root = dataset_root
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.train = train
        self.shuffle = shuffle
        self.update_freq = update_freq
        self.batch_size = batch_size

        self.step = 0
        self.permutation = torch.randperm(self.out_dim)
        self._get_dataset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.step % self.update_freq == 0:
            self._permute()
        self.step += 1

        try:
            return next(self.data_loader)
        except StopIteration:
            self.data_loader = self._get_data_loader()
            return next(self.iterator)
    
    def _get_data_loader(self):
        return iter(DataLoader(
            self.dataset,
            self.batch_size,
            shuffle=self.shuffle
        ))
    
    def _get_dataset(self):
        self.dataset = torchvision.datasets.EMNIST(
            self.dataset_root,
            split="letters",
            train=self.train,
            transform=T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(self.in_dim),
                T.Normalize((0.5,), (0.5,)),
                T.Lambda(lambda x: x.flatten()),
            ]),
        )

    def _apply_permutation(self, x):
        return self.permutation[x-1]    # Subtracting 1 since target index starts from 1
    
    def _permute(self):
        self.permutation = torch.randperm(self.out_dim)
        self._get_dataset()
        self.dataset.targets = self._apply_permutation(self.dataset.targets)
        self.data_loader = self._get_data_loader()
    