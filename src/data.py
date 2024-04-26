from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2 as T

from src.util import get_root_dir


class CustomLetterEMNIST(Dataset):
    """Customized letter EMNIST dataset."""
    def __init__(
        self,
        data_root_dir,
        train,
        transforms,
        label_subset = None,
    ):
        # Get the original letter EMNIST data
        original_emnist = torchvision.datasets.EMNIST(
            data_root_dir,
            split="letters",
            train=train,
        )

        self.transforms = transforms

        if label_subset is None:
            self.images = original_emnist.data
            self.targets = original_emnist.targets
        else:
            # Collect subset index of the data
            original_targets = original_emnist.targets
            subset_idx = (original_targets == label_subset[0])
            for l in label_subset[1:]:
                subset_idx |= (original_targets == l)
                
            subset_dict = {label_subset[i]: i for i in range(len(label_subset))}

            self.images = original_emnist.data[subset_idx, :]
            self.targets = original_targets[subset_idx]

            # Only take half of the overall data
            half_idx = torch.randperm(self.targets.size()[0] // 2)
            self.images = self.images[half_idx]
            self.targets = self.targets[half_idx]

            # Convert the value of label indices to allow training
            fn = lambda x: subset_dict[x]
            self.targets = self.targets.apply_(fn)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        images = self.images[idx, :]
        if images.ndim <= 2:
            images = images.unsqueeze(0)
        if self.transforms is not None:
            images = self.transforms(images)
            
        labels = self.targets[idx]
        
        return images, labels
        

class LabelPermutedEMNIST:
    """Label permuted EMNIST task."""
    def __init__(
        self,
        dataset_root,
        in_dim=(16, 16),
        out_dim=26,
        train=True,
        shuffle=True,
        update_freq=1000,
        batch_size=1,
        img_transform=None,
        label_subset=None,
    ):
        self.dataset_root = dataset_root
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.train = train
        self.shuffle = shuffle
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.img_transform=img_transform
        self.label_subset=label_subset

        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Update label permutation after one task ends
        if self.step % self.update_freq == 0:
            self.permute()
        self.step += 1

        try:
            return next(self.data_loader)
        except StopIteration:
            # Regenerate the data loader if current loader exhaust the data
            self.data_loader = self._get_data_loader()
            return next(self.data_loader)
    
    def _get_data_loader(self):
        # Generate new dataloader
        return iter(DataLoader(
            self.dataset,
            self.batch_size,
            shuffle=self.shuffle
        ))
    
    def _get_dataset(self):
        # Redefine the dataset
        self.dataset = CustomLetterEMNIST(self.dataset_root,
                                          self.train,
                                          self.img_transform,
                                          self.label_subset)

    def _apply_permutation(self, x):
        return self.permutation[x-1]    # Subtracting 1 since target index follows 1-based indexing
    
    def permute(self):
        # Permute the labels and redefine the dataset/dataloader
        self.permutation = torch.randperm(self.out_dim)
        self._get_dataset()
        self.dataset.targets = self._apply_permutation(self.dataset.targets)
        self.data_loader = self._get_data_loader()

