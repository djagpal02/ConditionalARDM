"""
Model Class for Conditional Autoregressive Distribution Model (ARDM) https://arxiv.org/pdf/2110.02037.pdf.

This version can be used as the original authors intended or used to model conditoinal distributions.
"""

# Standard library imports
from typing import List, Optional

# Third-party library imports
import torch
from torch import nn
import numpy as np

# Local imports
from Model.ordering import Ordering
from utils import sum_over_all_but_batch


class ARDM(nn.Module):
    r"""
    Autoregressive Distribution Model (ARDM) for implementing the ARDM concept as
    described in the paper https://arxiv.org/pdf/2110.02037.pdf.

    This model can be used in two ways:
        1. As the original authors intended.
        2. To model conditional distributions.

    Args:
        net (nn.Module): The model architecture.
        univariate_distributions (torch.distributions.Distribution): Each univariate distribution for each dimension.
        data_shape (list of int): Shape of the data.
        num_params_per_dist (int): Number of parameters required to describe univariate distribution.
        conditioned_on_x_hat (bool, optional): If True, the model is P(X|\hat{X}). Defaults to False.
        ordering (Optional[Ordering], optional): An instance of the Ordering class. Defaults to None.

    Attributes:
        epoch (int): Number of epochs trained count, set to 1, increased by trainer.
    """

    def __init__(
        self,
        net,
        univariate_distributions,
        data_shape: List[int],
        num_params_per_dist: int,
        conditioned_on_x_hat: bool = False,
    ):
        super().__init__()
        # Model Architecture
        self.net = net

        # Wheter model is P(X) or P(X|\hat{X})
        self.conditioned_on_x_hat = conditioned_on_x_hat

        # Each univariate Distribution for each Dim
        self.univariate_distributions = univariate_distributions
        # Number of parameters required to describe univariate distribution
        self.num_params_per_dist = num_params_per_dist

        # Ordering, timestep and Mask generating class for given data shape
        self.ordering = Ordering(data_shape=data_shape)

        self.data_shape = data_shape
        self.nin = np.prod(self.data_shape)

        # Number of Epochs trained count, set to 1, increased by trainer
        self.epoch = 0

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, x_hat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            x_hat (Optional[torch.Tensor], optional): Optional tensor for conditioned model. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Get Timesteps
        timesteps = sum_over_all_but_batch(mask)

        # Mask x
        masked_x = mask * x

        if self.conditioned_on_x_hat:
            inp = [masked_x, x_hat]
        else:
            inp = [masked_x]

        thetas = self.net(inp, timesteps, mask)

        # Check for any nan parameter values
        assert not torch.isnan(thetas).any()
        return thetas

    def nll(
        self, x: torch.Tensor, mask: torch.Tensor, x_hat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the negative log likelihood (NLL).

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            x_hat (Optional[torch.Tensor], optional): Optional tensor for conditioned model.
            Defaults to None.

        Returns:
            torch.Tensor: The negative log likelihood.
        """
        return -self.univariate_distributions(self(x, mask, x_hat), self.nin).log_prob(
            x
        )

    def sample(
        self, x: torch.Tensor, mask: torch.Tensor, x_hat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Samples from the distribution model.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            x_hat (Optional[torch.Tensor], optional): Optional tensor for conditioned model.
            Defaults to None.

        Returns:
            torch.Tensor: Sampled tensor.
        """
        return self.univariate_distributions(self(x, mask, x_hat), self.nin).sample()
