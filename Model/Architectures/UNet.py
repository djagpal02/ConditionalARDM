"""
U-Net architecture used in DDPM paper: https://arxiv.org/abs/2006.11239 for image reconstruction and in ARDM paper: https://arxiv.org/abs/2110.02037 for density modeling and generation.
Used for image reconstruction, and for the image decoder in the our model.
"""
# Standard library imports
from typing import Tuple, Union, List

# Third-party library imports
import torch
from torch import nn

# Local imports
from Model.Architectures.layers import *
from Model.Architectures.x_hat_mod import ModifiedInputEmbedding
from Model.Architectures.netbase import Net

class UNet(Net):
    """
    Class for UNet architecture.
    """

    def __init__(
        self,
        image_channels: int = 3,
        n_channels: int = 256,
        param_channels: int = 3 * 256,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
        n_blocks: int = 2,
        dropout: float = 0.1,
        max_time: int = 3072,
        group_norm_n: int = 32,
        conditional_model: bool = False,
    ):
        """
        Initialize the UNet model.

        Args:
            image_channels (int, optional): The number of channels in the input image. Default is 3.
            n_channels (int, optional): Base number of channels in convolutional layers. Default is 256.
            param_channels (int, optional): The number of channels in the parameter tensor. Default is 3 * 256.
            ch_mults (tuple or list, optional): Channel multipliers for each resolution level in the U-Net. Default is (1, 2, 2, 4).
            is_attn (tuple or list, optional): Whether to use attention in each resolution level. Default is (False, False, True, True).
            n_blocks (int, optional): The number of blocks in each resolution level. Default is 2.
            dropout (float, optional): Dropout rate. Default is 0.1.
            max_time (int, optional): Maximum sequence length for time embedding. Default is 3072.
            group_norm_n (int, optional): Number of groups for Group Normalization. Default is 32.
            conditional_model (bool, optional): Whether to use conditional model. Default is False.
        """
        super().__init__(max_time=max_time, conditional_model=conditional_model)



        n_resolutions = len(ch_mults)
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

        #####################################################################################################
        # First half of U-Net - decreasing resolution
        #####################################################################################################
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels,
                        out_channels,
                        n_channels * 4,
                        is_attn[i],
                        dropout=dropout,
                        group_norm_n=group_norm_n,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        #####################################################################################################
        # Middle block
        #####################################################################################################
        self.middle = MiddleBlock(
            out_channels, n_channels * 4, dropout=dropout, group_norm_n=group_norm_n
        )

        #####################################################################################################
        # Second half of U-Net - increasing resolution
        #####################################################################################################
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels,
                        out_channels,
                        n_channels * 4,
                        is_attn[i],
                        dropout=dropout,
                        group_norm_n=group_norm_n,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels,
                    out_channels,
                    n_channels * 4,
                    is_attn[i],
                    dropout=dropout,
                    group_norm_n=group_norm_n,
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        #####################################################################################################
        # Final normalization and convolution layer to output distribution parameters
        #####################################################################################################
        self.norm = nn.GroupNorm(group_norm_n // 4, n_channels)
        self.act = nn.SiLU()
        self.final = conv3x3_ddpm_init(in_channels, param_channels, padding=(1, 1))

    def forward(
        self, x: List[torch.Tensor], t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the UNet model.

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

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            x = m(x, t)
            h.append(x)

        # Middle (bottom)
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        x = self.final(self.act(self.norm(x)))

        return x
