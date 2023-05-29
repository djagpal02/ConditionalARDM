import torch
import numpy as np
from typing import List

class Ordering:
    """
    Class for generating random and strategic orderings for AR models 
    """
    def __init__(self, data_shape: List[int]):
        """
        :param data_shape: (list) Shape of the data to be ordered
        """

        self.data_shape = data_shape
        self.dims = np.prod(self.data_shape) # Treats each R-G-B value as a separate dimension


        # Dummy values to reshape dims of timestep tensor to allow timestep comparison ( need list of 1s size of data_shape)
        # E.G. for data shape [3, 32, 32] we need dummy_dims = [1, 1, 1]
        self.dummy_dims = [1 for i in range(len(data_shape))]
    


    def generate_timestep_tensor(self, batch_size: int, t: torch.Tensor) -> torch.Tensor:
        """
        Generate timestep tensor with all t the same

        :param batch_size: (int) Batch size
        :param t: (torch.Tensor) Timestep to be used for all samples
        :return: (torch.Tensor) Timestep tensor of shape (batch_size, *dummy_dims)
        """
        return (torch.ones(size=(batch_size,))*t).reshape(batch_size,*self.dummy_dims)



    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Returns a tensor containing timesteps in [0,dims) for each sample in batch

        :param batch_size: (int) Batch size
        :return: (torch.Tensor) Timestep tensor of shape (batch_size, *dummy_dims)
        """
        return torch.randint(high=self.dims, size=(batch_size,)).reshape(batch_size,*self.dummy_dims)

        

    def sample_random_orderings(self, batch_size: int) -> torch.Tensor:
        """
        Sample random orderings for each sample in batch

        :param batch_size: (int) Batch size
        :return: (torch.Tensor) Random ordering tensor of shape (batch_size, *data_shape) 
        """
        sigma = torch.rand(size=(batch_size, self.dims))
        

        sigma = torch.argsort(sigma, dim=-1).reshape(batch_size, *self.data_shape)


        return sigma
    


    def sample_random_masks(self, batch_size: int) -> torch.Tensor:
        """
        Samples random orderings (strategic or otherwise) and then compares with timesteps to generate random masks

        :param batch_size: (int) Batch size
        :return: (torch.Tensor, torch.Tensor) Random masks of shape (batch_size, *data_shape)
        """

        sigma = self.sample_random_orderings(batch_size)
            
        t = self.sample_timesteps(batch_size)
        
        previous_selection = (sigma < t).int()
        current_selection = (sigma == t).int()

        return previous_selection, current_selection
    

    
    def sample_masks(self, batch_size: int, sigma: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples Masks for given ordering and timesteps

        :param batch_size: (int) Batch size -- not used but kept for consistency with sample_random_masks
        :param sigma: (torch.Tensor) Ordering tensor of shape (batch_size, *data_shape)
        :param t: (torch.Tensor) Timestep tensor of shape (batch_size, *dummy_dims)
        :return: (torch.Tensor, torch.Tensor) Random masks of shape (batch_size, *data_shape)
        """
        previous_selection = (sigma < t).int()
        current_selection = (sigma == t).int()

        return previous_selection, current_selection
