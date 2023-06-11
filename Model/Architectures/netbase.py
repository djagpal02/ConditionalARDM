"""
Base class for network architectures to be used in ARDM model.
"""
# Standard library imports
from typing import List

# Third-party library imports
import torch
from torch import nn


class Net(nn.Module):
    """
    Base class for Net architecture to be used in ARDM model.
    """

    def __init__(
        self,
        max_time: int = 3072,
        conditional_model: bool = False,
    ):
        """ """
        super().__init__()

        # Save key parameters
        self.max_time = max_time
        self.conditional_model = conditional_model

    def forward(
        self, x: List[torch.Tensor], t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Net architecture.
        """
        pass
