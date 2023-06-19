"""
This module provides utility functions for data processing.

It includes functions for image processing, building data loaders, loading and preparing datasets,
and getting loaders for conditional ARDM.
"""

# Standard library imports
import copy
from typing import Tuple

# Third-party library imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore
import torchvision
from omegaconf import DictConfig

# Local imports
from utils import image_float_to_int


def get_loaders(
    dataset: str, config: DictConfig
) -> Tuple[
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
]:
    """
    Builds train, validation, and test loaders for a given dataset.

    Args:
        dataset (str): Name of the dataset.
        config (DictConfig): Configuration settings.

    Returns:
        Tuple[torch.utils.data.dataloader.DataLoader]: A tuple containing trainloader, validloader,
        and testloader.
    """
    tr_data, va_data, te_data = load(dataset, config.data_dir, config.Reduction)

    # Imagenet has already been permuted
    if dataset == "CIFAR10":
        tr_tensor = torch.Tensor(np.transpose(tr_data, (0, 3, 1, 2)))
        va_tensor = torch.Tensor(np.transpose(va_data, (0, 3, 1, 2)))
        te_tensor = torch.Tensor(np.transpose(te_data, (0, 3, 1, 2)))

    trainloader = build_loader(torch.utils.data.TensorDataset(tr_tensor), config)
    validloader = build_loader(torch.utils.data.TensorDataset(va_tensor), config)
    testloader = build_loader(torch.utils.data.TensorDataset(te_tensor), config)

    return trainloader, validloader, testloader


def get_cond_ardm_loaders(
    dataset: str, config: DictConfig
) -> Tuple[
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
    torch.utils.data.dataloader.DataLoader,
]:
    """
    Builds a train, valid and test loader for a given dataset for conditional ARDM.

    Args:
        dataset (str): Name of the dataset.
        config (DictConfig): Configuration settings.

    Returns:
        Tuple[torch.utils.data.dataloader.DataLoader]: A tuple containing trainloader, validloader,
        and testloader.
    """
    if dataset == "CIFAR10":
        tr_data, va_data, te_data = load(dataset, config.data_dir, config.Reduction)
        tr_tensor = torch.Tensor(np.transpose(tr_data, (0, 3, 1, 2)))
        va_tensor = torch.Tensor(np.transpose(va_data, (0, 3, 1, 2)))
        te_tensor = torch.Tensor(np.transpose(te_data, (0, 3, 1, 2)))
        if config.x_hat == "x_given_x":
            trainloader = build_loader(
                torch.utils.data.TensorDataset(tr_tensor, tr_tensor), config
            )
            validloader = build_loader(
                torch.utils.data.TensorDataset(va_tensor, va_tensor), config
            )
            testloader = build_loader(
                torch.utils.data.TensorDataset(te_tensor, te_tensor), config
            )
        elif config.x_hat == "x_given_gentle_blur_x":
            blurrer = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5))

            trainloader = build_loader(
                torch.utils.data.TensorDataset(tr_tensor, blurrer(tr_tensor)), config
            )
            validloader = build_loader(
                torch.utils.data.TensorDataset(va_tensor, blurrer(va_tensor)), config
            )
            testloader = build_loader(
                torch.utils.data.TensorDataset(te_tensor, blurrer(te_tensor)), config
            )
        elif config.x_hat == "x_given_medium_blur_x":
            blurrer = transforms.GaussianBlur(kernel_size=(13, 13), sigma=(0.1, 5))

            trainloader = build_loader(
                torch.utils.data.TensorDataset(tr_tensor, blurrer(tr_tensor)), config
            )
            validloader = build_loader(
                torch.utils.data.TensorDataset(va_tensor, blurrer(va_tensor)), config
            )
            testloader = build_loader(
                torch.utils.data.TensorDataset(te_tensor, blurrer(te_tensor)), config
            )
        elif config.x_hat == "x_given_strong_blur_x":
            blurrer = transforms.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 5))

            trainloader = build_loader(
                torch.utils.data.TensorDataset(tr_tensor, blurrer(tr_tensor)), config
            )
            validloader = build_loader(
                torch.utils.data.TensorDataset(va_tensor, blurrer(va_tensor)), config
            )
            testloader = build_loader(
                torch.utils.data.TensorDataset(te_tensor, blurrer(te_tensor)), config
            )
    else:
        raise NotImplementedError("Conditional ARDM is only implemented for CIFAR10")

    return trainloader, validloader, testloader


def build_loader(
    dataset: torch.utils.data.dataset.TensorDataset, config: DictConfig
) -> torch.utils.data.dataloader.DataLoader:
    """
    Builds a data loader for a given dataset.

    Args:
        dataset (torch.utils.data.dataset.TensorDataset): The dataset to be loaded.
        config: (DictConfig) The configuration settings.

    Returns:
        torch.utils.data.dataloader.DataLoader: The data loader object.
    """
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )


def load(
    dataset_name: str, path: str, reduction=None, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads and prepares a data for training and testing.

    Args:
        dataset_name (str): Name of the dataset.
        path (str): Path to the dataset.
        Reduction (Optional): Subset reduction type. Defaults to None.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[torch.Tensor]: A tuple containing the train, validation, and test datasets.
    """
    np.random.seed(seed)

    # Load the dataset
    if dataset_name == "CIFAR10":
        image_augmented_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                lambda x: image_float_to_int(x),
            ]
        )

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                lambda x: image_float_to_int(x),
            ]
        )

        te_dataset = torchvision.datasets.CIFAR10(
            root=path, train=False, download=True, transform=transform
        )
        tr_dataset = torchvision.datasets.CIFAR10(
            root=path, train=True, download=True, transform=image_augmented_transforms
        )

    # Subset the dataset if requested
    if reduction == "single_class":
        # Reduce dataset to only class 0 -- To speed up testing during model building
        tr_idx = np.where(np.array(tr_dataset.targets) == 0)[0]
        te_idx = np.where(np.array(te_dataset.targets) == 0)[0]

        tr_dataset.data = tr_dataset.data[tr_idx]
        te_dataset.data = te_dataset.data[te_idx]

    elif reduction is not None:
        indices = np.arange(len(tr_dataset))
        np.random.shuffle(indices)
        indices = indices[: int((reduction / 100) * len(tr_dataset))]
        tr_dataset.data = tr_dataset.data[indices]

    # Create the validation set (10% of train set)
    va_size = int(0.1 * len(tr_dataset))
    indices = np.arange(len(tr_dataset))

    # Shuffle the indices given the seed
    np.random.shuffle(indices)

    va_indices, tr_indices = indices[:va_size], indices[va_size:]

    va_dataset = copy.deepcopy(tr_dataset)

    tr_dataset.data = tr_dataset.data[tr_indices]
    va_dataset.data = va_dataset.data[va_indices]

    return tr_dataset.data, va_dataset.data, te_dataset.data
