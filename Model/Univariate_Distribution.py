"""
    This file contains functions that reshape thetas to allow for estimation of univariate distributions.
"""
# Standard library imports
from typing import Optional, Union

# Third-party library imports
import torch


def gaussian_mixture_model(
    thetas: torch.Tensor, n_features: int
) -> Union[torch.distributions.Normal, torch.distributions.MixtureSameFamily]:
    """
    Reshapes input parameters to enable Gaussian Mixture Model estimation.

    Parameters:
    thetas (torch.Tensor): Input tensor with shape [batch_size, n_params].
    n_features (int): The number of features in the dataset.

    Returns:
    torch.distributions.MixtureSameFamily: A Gaussian mixture model distribution.
    """
    # Get shape
    batch_size, n_params = thetas.shape

    # If there is only a single gaussian (non-mixture model), i.e. only mus and log(var)s
    if n_params == 2 * n_features:
        # We estimate alpha log(var) to avoid issues with estimated negative var, var = exp(alpha)
        mus, alpha = torch.chunk(thetas, 2, dim=1)

        return torch.distributions.Normal(mus, torch.exp(alpha))

    # If there is a mixture of guassians
    # Count the number in the mixture (n_features*3 since there are 3 params per gaussian)
    k = n_params // (n_features * 3)

    # We estimate alpha log(var) to avoid issues with estimated negative var, var = exp(alpha)
    mus, alpha, mix_logit = torch.chunk(
        thetas.reshape((batch_size, n_params // k, k)), 3, dim=1
    )

    mix = torch.distributions.Categorical(logits=mix_logit)
    comp = torch.distributions.Normal(mus, torch.exp(alpha))
    return torch.distributions.MixtureSameFamily(mix, comp)


def cifar10(
    thetas: torch.Tensor, n_features: Optional[int] = None
) -> torch.distributions.Categorical:
    """
    Reshapes input parameters to enable estimation of a categorical distribution over the CIFAR10 dataset.

    Parameters:
    thetas (torch.Tensor): Input tensor with shape [batch_size, n_params].
    n_features (int): The number of features in the dataset, not used here but kept for function signature consistency.

    Returns:
    torch.distributions.Categorical: A categorical distribution over the CIFAR10 dataset.
    """
    del n_features

    reshaped_thetas = torch.permute(
        thetas.reshape(thetas.shape[0], 256, 3, 32, 32), (0, 2, 3, 4, 1)
    )
    distribution = torch.distributions.categorical.Categorical(logits=reshaped_thetas)

    return distribution


def mnist(
    thetas: torch.Tensor, n_features: Optional[int] = None
) -> torch.distributions.Categorical:
    """
    Reshapes input parameters to enable estimation of a categorical distribution over the CIFAR10 dataset.

    Parameters:
    thetas (torch.Tensor): Input tensor with shape [batch_size, n_params].
    n_features (int): The number of features in the dataset, not used here but kept for function signature consistency.

    Returns:
    torch.distributions.Categorical: A categorical distribution over the CIFAR10 dataset.
    """
    del n_features

    reshaped_thetas = torch.permute(
        thetas.reshape(thetas.shape[0], 256, 1, 28, 28), (0, 2, 3, 4, 1)
    )
    distribution = torch.distributions.categorical.Categorical(logits=reshaped_thetas)

    return distribution


def binary_mnist(
    thetas: torch.Tensor, n_features: Optional[int] = None
) -> torch.distributions.Categorical:
    """
    Reshapes input parameters to enable estimation of a categorical distribution over the Binary MNIST dataset.

    Parameters:
    thetas (torch.Tensor): Input tensor with shape [batch_size, n_params].
    n_features (int): The number of features in the dataset, not used here but kept for function signature consistency.

    Returns:
    torch.distributions.Categorical: A categorical distribution over the Binary MNIST dataset.
    """
    del n_features

    reshaped_thetas = torch.permute(
        thetas.reshape(thetas.shape[0], 2, 1, 28, 28), (0, 2, 3, 4, 1)
    )
    distribution = torch.distributions.categorical.Categorical(logits=reshaped_thetas)

    return distribution
