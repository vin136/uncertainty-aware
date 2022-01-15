"""
FastMNIST taken from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
"""
import torch
import torchvision

# from torchvision.datasets import MNIST

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def get_MNIST_loaders(root, train, batch_size):
    # import pdb;pdb.set_trace()
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root,
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=train,
    )
    return loader
