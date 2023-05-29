from typing import  Tuple, Union, List
import torch
from torch import nn
from Model.Architectures.layers import *
from Model.Architectures.x_hat_mod import modified_InputEmbedding

class UNet(nn.Module):
    """
    U-Net architecture used in DDPM paper: https://arxiv.org/abs/2006.11239 for image reconstruction and in ARDM paper: https://arxiv.org/abs/2110.02037 for density modeling and generation.
    Used for image reconstruction, and for the image decoder in the our model.
    """

    def __init__(self, image_channels: int = 3, n_channels: int = 256, param_channels: int = 3*256,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2, dropout: float = 0.1, max_time: int =3072, group_norm_n: int = 32, conditional_model: bool=False):
        """
        :param image_channels: Number of channels in the input image
        :param n_channels: Number of channels in the first layer
        :param param_channels: Number of channels in the output, which is the number of parameters for the distribution
        :param ch_mults: Number of channels at each resolution is multiplied by this number
        :param is_attn: Whether to use attention at each resolution
        :param n_blocks: Number of blocks at each resolution
        :param dropout: Dropout rate
        :param max_time: Maximum time for the time embedding
        :param group_norm_n: Number of groups for group normalization
        :param conditional_model: Whether to use the conditional model (model P(x|other params) instead of P(x))
        """
        super().__init__()

        # Save key parameters
        self.max_time = max_time
        self.conditional_model = conditional_model
        n_resolutions = len(ch_mults)
        n_classes = param_channels // image_channels # Number of classes for output (possible outputs)

        
        # Project image into feature map using our modified InputEmbedding that allows for conditional models, this is the key difference between our model and the one used in ARDM model
        #####################################################################################################
        self.image_proj = modified_InputEmbedding(n_classes, n_channels, image_channels, conditional_model)
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
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i], dropout=dropout,group_norm_n=group_norm_n))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)




        #####################################################################################################
        # Middle block
        #####################################################################################################
        self.middle = MiddleBlock(out_channels, n_channels * 4, dropout=dropout,group_norm_n=group_norm_n)




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
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i], dropout=dropout,group_norm_n=group_norm_n))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i], dropout=dropout,group_norm_n=group_norm_n))
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

    def forward(self, x: List[torch.Tensor], t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forwards pass of U-Net

        :param x: List of one or two things - input data alone or with x_hat (what it is conditioned on) both of size[batch_size, in_channels, height, width]
        :param t: Contains the timsteps for each image in the batch, of size [batch_size]
        :param mask: Contains the mask for each image in the batch, of size [batch_size, in_channels, height, width]

        :return: Distribution parameters of size [batch_size, param_channels, height, width]
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x,mask)

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