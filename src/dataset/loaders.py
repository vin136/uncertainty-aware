import torch
import os
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms


class Splitter(Dataset):
    # Gives the appropriate dataset-split from torch Dataset.
    def __init__(self, dset, frac=0.75, split="train"):
        self.dset = dset
        self.frac = frac
        self.split = split
        self.perm_ind = np.random.permutation(np.arange(0, len(self.dset)))
        if split == "train":
            self.ind = np.arange(0, int(len(self.dset) * (self.frac)))
        else:
            self.ind = np.arange(int(len(self.dset) * (self.frac)), len(self.dset))

    def __getitem__(self, i):
        return self.dset[self.perm_ind[self.ind[i]]]

    def __len__(self):
        return len(self.ind)


class AmbiguousMNIST(Dataset):
    def __init__(self, root=None, train=True, device=None):
        # Scale data to [0,1]
        if not root:
            root_dir = "/".join(os.getcwd().split("/")[:-2])
            root = root_dir + "/data-store"

        self.data = torch.load(os.path.join(root, "amnist_samples.pt")).to(device)
        self.targets = torch.load(os.path.join(root, "amnist_labels.pt")).to(device)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        num_multi_labels = self.targets.shape[1]

        self.data = self.data.expand(-1, num_multi_labels, 28, 28).reshape(
            -1, 1, 28, 28
        )
        self.targets = self.targets.reshape(-1)

        data_range = slice(None, 60000) if train else slice(60000, None)
        self.data = self.data[data_range]
        self.targets = self.targets[data_range]

    # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


def get_loaders(
    root,
    mode,
    batch_size,
    frac=0.75,
    data_type="fmnist",
    num_workers=4,
    pin_memory=True,
    **kwargs,
):
    print(f"num_workers:{num_workers}")
    train = True
    loader = None
    if mode == "test":
        train = False

    # define transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),
                (0.3081,),
            ),
        ]
    )

    # load the dataset
    # data_dir = "./data"
    if data_type == "fmnist":
        dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=True,
            transform=transform,
        )
    elif data_type == "mnist":
        dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform,
        )
    elif data_type == "ambiguousmnist":
        # This dataset should be pre-downloaded
        device = "cpu"
        dataset = AmbiguousMNIST(
            root=root,
            train=train,
            device=device,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=train, num_workers=0
        )
    else:
        raise NotImplementedError

    if mode == "test" and not loader:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif not loader:
        dset = Splitter(dataset, frac=frac, split=mode)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        pass
    return loader
