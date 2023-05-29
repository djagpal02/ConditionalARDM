from Model.Architectures.UNet import UNet
from Model.ARDM import ARDM
from Model.Univariate_Distribution import CIFAR10
from Model.Ordering import Ordering





def get_ARDM(config, conditional_model):
    """
    Loads and returns model

    :param config: Config
    :return: Model
    """
    if config.dataset == "CIFAR10":
        # Set up univariate distributions
        ud = CIFAR10

    # Set up Architecture
    # UNet used in orignial ARDM paper
    if config.architecture == "UNet":
        ardm_net = UNet(
                        image_channels= 3,
                        n_channels=256,
                        param_channels=768,
                        ch_mults=[1],
                        is_attn=True,
                        n_blocks=32,
                        dropout=0.,
                        max_time=3072,
                        group_norm_n = 32,
                        conditional_model=conditional_model)
        


    # Build Model    
    ardm = ARDM(
                Net=ardm_net,
                univariate_distributions=ud,
                data_shape=config.data_shape,
                num_params_per_dist=768,
                conditioned_on_x_hat=conditional_model, 
                ordering=Ordering(config.data_shape))

    return ardm