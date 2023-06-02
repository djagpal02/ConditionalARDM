"""
This module provides all component layers of the DDPM UNet. 

This code is a slightly modified version of the one presented in: https://github.com/AndyShih12/mac .

"""
# Standard library imports
import math
import string
from typing import Optional, Union

# Third-party library imports
import numpy as np
import torch  # type: ignore
from torch import nn
import torch.nn.functional as F


def variance_scaling(
    scale, mode, distribution, in_axis=1, out_axis=0, dtype=torch.float32, device="cpu"
):
    """Ported from JAX."""

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode)
            )
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (
                torch.rand(*shape, dtype=dtype, device=device) * 2.0 - 1.0
            ) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale: float = 1.0):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, "fan_avg", "uniform")


def linear_ddpm_init(inp: int, out: int):
    linear = nn.Linear(inp, out)
    linear.weight.data = default_init()(linear.weight.data.shape)
    nn.init.zeros_(linear.bias)

    return linear


def conv1x1_ddpm_init(
    in_channels: int,
    out_channels: int,
    stride=1,
    bias: bool = True,
    init_scale: float = 1.0,
    padding=0,
):
    """1x1 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=stride,
        padding=padding,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    assert conv.bias is not None
    nn.init.zeros_(conv.bias)

    return conv


def conv3x3_ddpm_init(
    in_channels: int,
    out_channels: int,
    stride=1,
    bias: bool = True,
    dilation=1,
    init_scale: float = 1.0,
    padding=1,
):
    """3x3 convolution with DDPM initialization."""
    conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )
    conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
    assert conv.bias is not None
    nn.init.zeros_(conv.bias)

    return conv


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int, max_time: int = 10000):
        """
        * `n_channels` is the number of dimensions in the embedding. Time embedding has  model `n_channels * 4` channels
        """
        super().__init__()
        self.n_channels = n_channels
        self.max_time = max_time

        # First linear layer
        self.lin1 = linear_ddpm_init(self.n_channels, self.n_channels * 4)

        # Activation
        self.act = nn.SiLU()

        # Second linear layer
        self.lin2 = linear_ddpm_init(self.n_channels * 4, self.n_channels * 4)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)

        half_dim = self.n_channels // 2
        emb = torch.Tensor(math.log(self.max_time) / (half_dim - 1))
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        assert emb.shape == (t.shape[0], self.n_channels)
        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        assert emb.shape == (t.shape[0], self.n_channels * 4)

        return emb


class InputEmbedding(nn.Module):
    """
    Input Processing and class embeddings
    """

    def __init__(self, n_classes, n_channels, input_channels):
        super().__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.act = nn.SiLU()
        self.input_channels = input_channels

        # Input Processing
        assert self.n_channels % 4 == 0

        # Convolutions over mask and input
        self.conv_1 = conv3x3_ddpm_init(
            self.input_channels * 2, self.n_channels * 3 // 4
        )

        # Class embedding
        self.class_embed = nn.Embedding(self.n_classes, self.n_channels // 4)
        self.linear_1 = linear_ddpm_init(
            self.input_channels * self.n_channels // 4, self.n_channels // 4
        )

    def forward(self, x, mask):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        * 'mask' has shape '[batch_size, in_channels, height, width]'

        ** Module assumes data (x) is in 0-256 bit format
        """
        assert self.n_classes >= 1

        xint = x.long()
        x = torch.cat([xint, mask], dim=1)

        # Assign 3/4 of channels to standard input.
        h_first = self.conv_1(x.float())

        # # Assign 1/4 of channels to class embeddings.
        xint_permute = xint.permute(0, 2, 3, 1)
        emb_x = self.class_embed(xint_permute)
        emb_x = emb_x.reshape(
            *xint_permute.shape[:-1], self.input_channels * self.n_channels // 4
        )
        h_emb_x = self.linear_1(emb_x)
        h_emb_x = h_emb_x.permute(0, 3, 1, 2)

        # Concat input representation to embedding
        h_first = torch.cat([h_first, h_emb_x], dim=1)

        return h_first


class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
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

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = conv1x1_ddpm_init(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = linear_ddpm_init(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        B, C, H, W = x.shape
        assert C == self.in_channels

        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add bias to each feature map condtioned on time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    """
    Linear function over channel dim.
    Permutes Channel Dim to end, computes linear transformation and then permutes channels back.
    """

    def __init__(self, in_dim: int, num_units: int, init_scale: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(
            default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True
        )
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


class AttentionBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(
        self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 32
    ):
        super().__init__()
        # Default `d_k`
        if d_k is None:
            d_k = n_channels

        # Normalization layer
        self.GroupNorm_0 = nn.GroupNorm(
            num_groups=n_groups, num_channels=n_channels, eps=1e-6
        )

        self.NIN_0 = NIN(n_channels, n_heads * d_k)
        self.NIN_1 = NIN(n_channels, n_heads * d_k)
        self.NIN_2 = NIN(n_channels, n_heads * d_k)
        self.NIN_3 = NIN(n_heads * d_k, n_channels, init_scale=0.0)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t

        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        return x + h


class DownBlock(nn.Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        dropout: float,
        group_norm_n: int,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            time_channels,
            dropout=dropout,
            n_groups=group_norm_n,
        )

        self.attn: Union[AttentionBlock, nn.Identity]
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=group_norm_n)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        dropout: float,
        group_norm_n: int,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            time_channels,
            dropout=dropout,
            n_groups=group_norm_n,
        )
        self.attn: Union[AttentionBlock, nn.Identity]
        if has_attn:
            self.attn = AttentionBlock(out_channels, n_groups=group_norm_n)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(
        self, n_channels: int, time_channels: int, dropout: float, group_norm_n: int
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            dropout=dropout,
            n_groups=group_norm_n,
        )
        self.attn = AttentionBlock(n_channels, n_groups=group_norm_n)
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            dropout=dropout,
            n_groups=group_norm_n,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))
        self.conv.weight.data = default_init(1.0)(self.conv.weight.data.shape)
        assert self.conv.bias is not None
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
        self.conv.weight.data = default_init(1.0)(self.conv.weight.data.shape)
        assert self.conv.bias is not None
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)
