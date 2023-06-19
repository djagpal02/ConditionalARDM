"""
Simple convolutional network with 3x3 convolutions for ARDM model.
"""
# Standard library imports
from typing import List

# Third-party library imports
import torch
from torch import nn

# Local imports
from Model.Architectures.layers import (
    conv3x3_ddpm_init,
    linear_ddpm_init,
    TimeEmbedding,
)
from Model.Architectures.x_hat_mod import ModifiedInputEmbedding
from Model.Architectures.netbase import Net


class ConvBlock(nn.Module):
    """
    Simple block with 3x3 convolutions
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 32,
        dropout: float = 0.1,
    ):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        self.in_channels = in_channels

        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels, eps=1e-6)
        self.act1 = nn.SiLU()
        self.conv1 = conv3x3_ddpm_init(in_channels, out_channels)

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels, eps=1e-6)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv3x3_ddpm_init(out_channels, out_channels)

        # Linear layer for time embeddings
        self.time_emb = linear_ddpm_init(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        assert x.shape[1] == self.in_channels

        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add bias to each feature map condtioned on time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h


class ConvNet(Net):
    """
    Class for network of simple covoultional blocks.
    """

    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 256,
        param_channels: int = 3 * 256,
        n_blocks: int = 5,
        dropout: float = 0.1,
        max_time: int = 3072,
        group_norm_n: int = 32,
        conditional_model: bool = False,
    ):
        """
            Initialize the ConvNet.

        Args:
            image_channels: Number of channels in the input image.
            n_channels: Number of channels in the intermediate feature maps.
            param_channels: Number of channels in the output feature maps.
            n_blocks: Number of convolutional blocks.
            dropout: Dropout rate.
            max_time: Maximum time step.
            group_norm_n: Number of groups for group normalization.
            conditional_model: Whether to use conditional model.
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

        # Convolutional blocks
        self.blocks = nn.ModuleList()
        in_channels = n_channels
        for i in range(n_blocks - 1):
            out_channels = in_channels * 2
            self.blocks.append(
                ConvBlock(
                    in_channels, out_channels, n_channels * 4, group_norm_n, dropout
                )
            )
            in_channels = out_channels

        # The final needs to output 'param_channels' number of channels
        self.norm = nn.GroupNorm(group_norm_n // 4, n_channels)
        self.act = nn.SiLU()
        self.final = conv3x3_ddpm_init(n_channels, param_channels, padding=(1, 1))

    def forward(
        self, x: List[torch.Tensor], t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the ConvNet.

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

        # Apply the blocks
        for block in self.blocks:
            x = block(x, t)

        # Final normalization and convolution
        x = self.final(self.act(self.norm(x)))

        return x
