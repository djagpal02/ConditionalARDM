import torch
import torch.nn as nn
from Model.Architectures.layers import conv3x3_ddpm_init, linear_ddpm_init

class modified_InputEmbedding(nn.Module):
    """
    Modifed input embedding, modified fork the U-Net architecture used in DDPM paper: https://arxiv.org/abs/2006.11239 for image reconstruction and in ARDM paper: https://arxiv.org/abs/2110.02037 for density modeling and generation.

    Modified to allow for conditional models, which take in two inputs (x and xhat) instead of just one (x), where we are modeling p(x|xhat).
    """
    def __init__(self, n_classes: int, n_channels: int, input_channels: int, conditional_model: bool):
        """
        :param n_classes: Number of classes for output (possible outputs)
        :param n_channels: Number of channels in the first layer
        :param input_channels: Number of channels in the input image
        :param conditional_model: Boolean, whether or not the model is conditional (i.e. takes in two inputs)
        """
        super().__init__()
        # Save key parameters
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.act = nn.SiLU()
        self.input_channels = input_channels
        self.conditional_model = conditional_model

        # Input Processing
        assert(self.n_channels % 4 == 0)
        
        if conditional_model:
            multiplier = 3
        else:
            multiplier = 2

        # Convolutions over mask and input
        self.conv_1 = conv3x3_ddpm_init(self.input_channels * multiplier, self.n_channels * 3 // 4)


        # Class embedding
        self.class_embed = nn.Embedding(self.n_classes, self.n_channels // 4)
        self.linear_1 = linear_ddpm_init(self.input_channels * self.n_channels // 4, self.n_channels // 4)



    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input images (batch_size, input_channels, height, width)
        :param mask: Mask for input images (batch_size, input_channels, height, width)
 
        :return: Output of the input embedding (batch_size, n_channels, height, width)
        
        ** Module assumes data (x) is in 0-256 bit format 
        """
        assert(self.n_classes >= 1)


        # Assign 3/4 of channels to standard input.
        if self.conditional_model:
            xint, xhat_int = x[0].long(), x[1].long()
            x = torch.cat([xint, xhat_int, mask], dim=1)
        else:
            xint = x[0].long()
            x = torch.cat([xint,  mask], dim=1)
        
        h_first = self.conv_1(x.float())


        # # Assign 1/4 of channels to class embeddings.
        xint_permute = xint.permute(0, 2, 3, 1)
        emb_x = self.class_embed(xint_permute)
        emb_x = emb_x.reshape( *xint_permute.shape[:-1], self.input_channels * self.n_channels // 4 )
        h_emb_x = self.linear_1(emb_x)
        h_emb_x = h_emb_x.permute(0, 3, 1, 2)
        
        # Concat input representation to embedding
        h_first = torch.cat( [ h_first, h_emb_x ], dim=1)

        return h_first