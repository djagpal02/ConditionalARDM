"""
    Module for generating orderings for autoregressive models.
    Generates random orderings and masks for given orderings and timesteps.
"""

# Standard library imports
from typing import List

# Third-party library imports
import torch
import numpy as np


class Ordering:
    """
    The Ordering class is used to generate orderings for autoregressive models.
    It provides methods to generate timestep tensors, sample random orderings and masks,
    and sample masks for given ordering and timesteps.
    """

    def __init__(self, data_shape: List[int]):
        """
        Initialize an instance of the Ordering class.

        Args:
            data_shape (List[int]): Shape of the data to be ordered.
        """

        self.data_shape = data_shape
        self.dims = np.prod(
            self.data_shape
        )  # Treats each R-G-B value as a separate dimension

        # Dummy values to reshape dims of timestep tensor to allow timestep comparison ( need list of 1s size of data_shape)
        # E.G. for data shape [3, 32, 32] we need dummy_dims = [1, 1, 1]
        self.dummy_dims = [1 for i in range(len(data_shape))]

    def generate_timestep_tensor(
        self, batch_size: int, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates a timestep tensor with all t the same.

        Args:
            batch_size (int): Batch size.
            t (torch.Tensor): Timestep to be used for all samples.

        Returns:
            torch.Tensor: Timestep tensor of shape (batch_size, *dummy_dims).
        """
        return (torch.ones(size=(batch_size,)) * timesteps).reshape(
            batch_size, *self.dummy_dims
        )

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Returns a tensor containing timesteps in [0, dims) for each sample in the batch.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Timestep tensor of shape (batch_size, *dummy_dims).
        """
        return torch.randint(high=int(self.dims), size=(batch_size,)).reshape(
            batch_size, *self.dummy_dims
        )

    def sample_random_orderings(self, batch_size: int) -> torch.Tensor:
        """
        Samples random orderings for each sample in the batch.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor: Random ordering tensor of shape (batch_size, *data_shape).
        """
        sigma = torch.rand(size=(batch_size, int(self.dims)))

        sigma = torch.argsort(sigma, dim=-1).reshape(batch_size, *self.data_shape)

        return sigma

    def sample_random_masks(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples random orderings (strategic or otherwise) and then compares with timesteps to generate random masks.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor, torch.Tensor: Two random masks of shape (batch_size, *data_shape). The first tensor is for previous selection and the second is for current selection.
        """

        sigma = self.sample_random_orderings(batch_size)

        timesteps = self.sample_timesteps(batch_size)

        previous_selection = (sigma < timesteps).int()
        current_selection = (sigma == timesteps).int()

        return previous_selection, current_selection

    def sample_masks(
        self, batch_size: int, sigma: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples random orderings (strategic or otherwise) and then compares with timesteps to generate random masks.

        Args:
            batch_size (int): Batch size.

        Returns:
            torch.Tensor, torch.Tensor: Two random masks of shape (batch_size, *data_shape). The first tensor is for previous selection and the second is for current selection.
        """
        del batch_size

        previous_selection = (sigma < timesteps).int()
        current_selection = (sigma == timesteps).int()

        return previous_selection, current_selection
