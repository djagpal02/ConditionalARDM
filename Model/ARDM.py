import torch
from torch import nn
import numpy as np
from Model.Ordering import Ordering
from typing import List
from utils import sum_over_all_but_batch


class ARDM(nn.Module):
    """
    Autoregressive Distribution Model (ARDM) 

    Models either P(X)= \prod_{i=1}^{N} P(X_i | X_{<i})   or   P(X|\hat{X}) = \prod_{i=1}^{N} P(X_i | X_{<i}, \hat{X})  
    """
    def __init__(self, Net, univariate_distributions, data_shape: List[int], num_params_per_dist: int, conditioned_on_x_hat: bool = False, ordering: Ordering = None):
            """
            :param Net: Autoregressive Distribution Model Network
            :param univariate_distributions: Distribution Class for each univariate distribution
            :param data_shape: Shape of data
            :param num_params_per_dist: Number of parameters required to describe univariate distribution
            :param conditioned_on_x_hat: Whether model is P(X) or P(X|\hat{X})
            """
            super().__init__()
            # Model Architecture
            self.Net = Net
            
            # Wheter model is P(X) or P(X|\hat{X})
            self.conditioned_on_x_hat = conditioned_on_x_hat

            # Each univariate Distribution for each Dim
            self.univariate_distributions = univariate_distributions
            # Number of parameters required to describe univariate distribution
            self.num_params_per_dist = num_params_per_dist

            # Ordering, timestep and Mask generating class for given data shape
            if ordering is not None:
                self.ordering = ordering
            else:
                self.ordering = Ordering(data_shape=data_shape)

            self.data_shape = data_shape
            self.nin = np.prod(self.data_shape)

            # Number of Epochs trained count, set to 1, increased by trainer
            self.epoch = 0


    def forward(self, x: torch.Tensor, Mask: torch.Tensor, x_hat: torch.Tensor=None) -> torch.Tensor:
        """
        Forward Model over data X to generate theta, parameters describing univariate distributions

        :param x: Data
        :param Mask: Mask for data
        :param x_hat: Data to condition on
        :return: theta, parameters describing univariate distributions
        """
        # Get Timesteps
        timesteps =  sum_over_all_but_batch(Mask)

        # Mask x
        Masked_x =  Mask * x

        if self.conditioned_on_x_hat:
            inp = [Masked_x, x_hat]
        else:
            inp = [Masked_x]
            
        thetas = self.Net(inp, timesteps, Mask)

        # Check for any nan parameter values
        assert not torch.isnan(thetas).any()
        return thetas
        
    
    def NLL(self, x: torch.Tensor, Mask: torch.Tensor, x_hat: torch.Tensor=None ) -> torch.Tensor:
        """
        Negative Log Likelihood of data X under model P(X) or P(X|\hat{X})

        :param x: Data
        :param Mask: Mask for data
        :param x_hat: Data to condition on
        :return: NLL
        """
        return - self.univariate_distributions(self(x, Mask, x_hat), self.nin).log_prob(x)

    def sample(self, x: torch.Tensor, Mask: torch.Tensor, x_hat: torch.Tensor=None ) -> torch.Tensor:
        """
        Sample from model P(X) or P(X|\hat{X})

        :param x: Data
        :param Mask: Mask for data
        :param x_hat: Data to condition on
        :return: Sampled Data
        """
        return self.univariate_distributions(self(x, Mask, x_hat), self.nin).sample()