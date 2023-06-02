"""
Module for loading ARDM models.
"""
# Third-party library imports
from omegaconf import DictConfig

# Local imports
from Model.Architectures.UNet import UNet
from Model.ARDM import ARDM
from Model.univariate_distribution import cifar10


def get_ardm(config: DictConfig, conditional_model: bool):
    """
    Creates an ARDM model with a specific architecture and univariate distribution.
    The architecture and univariate distribution are specified in the configuration.

    Args:
        config (DictConfig): A dictionary-like configuration object,
                             typically an instance of OmegaConf DictConfig.
        conditional_model (bool): A flag that indicates whether to condition on x_hat.

    Returns:
        ARDM: An instance of the ARDM model with the specified architecture
              and univariate distribution.

    """
    if config.dataset == "CIFAR10":
        # Set up univariate distributions
        univariate_dist = cifar10

    # Set up architechture
    # UNet used in orignial ARDM paper
    if config.architecture == "UNet":
        ardm_net = UNet(
            image_channels=3,
            n_channels=256,
            param_channels=768,
            ch_mults=[1],
            is_attn=[True],
            n_blocks=32,
            dropout=0.0,
            max_time=3072,
            group_norm_n=32,
            conditional_model=conditional_model,
        )

    # Build Model
    ardm = ARDM(
        net=ardm_net,
        univariate_distributions=univariate_dist,
        data_shape=config.data_shape,
        num_params_per_dist=768,
        conditioned_on_x_hat=conditional_model,
    )

    return ardm
