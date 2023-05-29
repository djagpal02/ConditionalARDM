import torch
import numpy as np
from typing import List

class Ordering:
    """
    Class for generating random and strategic orderings for AR models 
    """
    def __init__(self, data_shape: List[int], RGB_channel_grouping=True):
        """
        :param data_shape: (list) Shape of the data to be ordered
        :param RGB_channel_grouping: (bool) Whether to treat RGB channels as one dimension
        """

        self.data_shape = data_shape


        # RGB channel grouping is only used for image data with three channels. As such we only turn this on if the data has 3 channels
        if RGB_channel_grouping and data_shape[0] == 3:
            self.RGB_channel_grouping = True
            self.dims = np.prod(self.data_shape) // 3 # Since we are treating RGB channels as one dimension
            self.group_channel_data_shape = (data_shape[1],data_shape[2])
        else:
            self.RGB_channel_grouping = False
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
        
        if self.RGB_channel_grouping:
            sigma = torch.argsort(sigma, dim=-1).reshape(batch_size, *self.group_channel_data_shape).unsqueeze(1).expand(-1, 3, -1, -1)
        else:
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
