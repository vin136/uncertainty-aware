"""
Fashion-MNIST used as an OOD dataset.
"""

import torch

from torchvision import datasets
from torchvision import transforms


def get_fashionMNIST_loaders(
    root, train, batch_size, num_workers=4, pin_memory=True, **kwargs
):
    print(f"num_workers:{num_workers}")
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

    dataset = datasets.FashionMNIST(
        root=root,
        train=train,
        download=True,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader
