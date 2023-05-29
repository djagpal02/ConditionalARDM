from Model.Architectures.UNet import UNet
from Model.ARDM import ARDM
from Model.Univariate_Distribution import CIFAR10
from Model.Ordering import Ordering





def get_ARDM(config, conditional_model=True):
    """
    Loads and returns model

    :param config: Config
    :return: Model
    """
    if config.dataset == "CIFAR10":
        ud = CIFAR10

    ardm_unet = UNet(
                    image_channels= config.image_channels,
                    n_channels=config.n_channels,
                    param_channels=config.param_channels,
                    ch_mults=config.ch_mults,
                    is_attn=config.is_attn,
                    n_blocks=config.n_blocks,
                    dropout=config.dropout,
                    max_time=config.n_dims,
                    group_norm_n = config.group_norm_n,
                    conditional_model=conditional_model)
        
    ardm = ARDM(
                Net=ardm_unet,
                univariate_distributions=ud,
                data_shape=config.data_shape,
                num_params_per_dist=config.param_channels,
                conditioned_on_x_hat=conditional_model, 
                ordering=Ordering(config.data_shape, config.RGB_channel_grouping))

    return ardm