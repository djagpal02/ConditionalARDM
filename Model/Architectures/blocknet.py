"""
BlockNet architecture.
Simply a residual attention block with a modified input embedding.
"""
# Standard library imports
from typing import List

# Third-party library imports
import torch
from torch import nn

# Local imports
from Model.Architectures.layers import TimeEmbedding, conv3x3_ddpm_init, MiddleBlock
from Model.Architectures.x_hat_mod import ModifiedInputEmbedding
from Model.Architectures.netbase import Net


class BlockNet(Net):
    """
    Class for blocknet architecture to be used in ARDM model.
    """

    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 256,
        param_channels: int = 3 * 256,
        dropout: float = 0.1,
        max_time: int = 3072,
        group_norm_n: int = 32,
        conditional_model: bool = False,
    ):
        """
            Initialize BlockNet architecture.

        Args:
            image_channels (int, optional): Number of channels in the input image. Defaults to 3.
            n_channels (int, optional): Number of channels in the hidden layers. Defaults to 256.
            param_channels (int, optional): Number of channels in the output. Defaults to 3*256.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            max_time (int, optional): Maximum time step. Defaults to 3072.
            group_norm_n (int, optional): Number of groups for group normalization. Defaults to 32.
            conditional_model (bool, optional): Whether to use conditional model. Defaults to False.
        """
        super().__init__(max_time=max_time, conditional_model=conditional_model)

        n_classes = (
            param_channels // image_channels
        )  # Number of classes for output (possible outputs)

        # Project image into feature map using our modified InputEmbedding that allows for conditional models,
        # this is the key difference between our model and the one used in ARDM model
        #####################################################################################################
        self.image_proj = ModifiedInputEmbedding(
            n_classes, n_channels, image_channels, conditional_model
        )
        #####################################################################################################

        # Time embedding layer.
        self.time_emb = TimeEmbedding(n_channels, max_time=max_time)

        self.block = MiddleBlock(
            n_channels, n_channels * 4, dropout=dropout, group_norm_n=group_norm_n
        )

        self.norm = nn.GroupNorm(group_norm_n // 4, n_channels)
        self.act = nn.SiLU()
        self.final = conv3x3_ddpm_init(n_channels, param_channels, padding=(1, 1))

    def forward(
        self, x: List[torch.Tensor], t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the BlockNet architecture.

        Args:
            x (List[torch.Tensor]): List of input tensors.
            t (torch.Tensor): Tensor of timestamps.
            mask (torch.Tensor): Tensor of masks.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x, mask)

        x = self.block(x, t)

        # Final normalization and convolution
        x = self.final(self.act(self.norm(x)))

        return x
