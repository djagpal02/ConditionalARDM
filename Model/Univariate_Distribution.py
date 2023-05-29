import torch

def Gaussian_Mixture_Model(thetas: torch.Tensor, n_features: int):
    """
    Reshapes parameters to allow estimation of gaussian mixture model
    
    : param thetas: torch.Tensor of shape [batch_size, n_params]
    : param n_features: int, number of features in the dataset
    : return distribution: torch.distributions.MixtureSameFamily
    """
    # Get shape
    batch_size, n_params = thetas.shape

    # If there is only a single gaussian (non-mixture model), i.e. only mus and log(var)s
    if n_params == 2*n_features:
        # We estimate alpha log(var) to avoid issues with estimated negative var, var = exp(alpha)
        mu, alpha = torch.chunk(thetas, 2, dim=1)
        
        distribution = torch.distributions.Normal(mu, torch.exp(alpha))
        
    # If there is a mixture of guassians    
    else:
        # Count the number in the mixture (n_features*3 since there are 3 params per gaussian)
        k = n_params//(n_features*3)

        # We estimate alpha log(var) to avoid issues with estimated negative var, var = exp(alpha)
        mu, alpha, mix_logit = torch.chunk(thetas.reshape((batch_size, n_params// k ,k)), 3, dim=1)


        mix = torch.distributions.Categorical(logits=mix_logit)
        comp = torch.distributions.Normal(mu, torch.exp(alpha))
        distribution = torch.distributions.MixtureSameFamily(mix, comp)
    
    
    return distribution


def CIFAR10(thetas: torch.Tensor, n_features: int=None):
    """
    Reshapes parameters to allow estimation of categorical distribution over CIFAR10 dataset

    : param thetas: torch.Tensor of shape [batch_size, n_params]
    : param n_features: int, number of features in the dataset -- not used but included for consistency between functions
    : return distribution: torch.distributions.Categorical
    """
    reshaped_thetas = torch.permute(thetas.reshape(thetas.shape[0], 256, 3, 32,32),(0,2,3,4,1))
    distribution = torch.distributions.categorical.Categorical(logits=reshaped_thetas)
        
    return distribution
    


def MNIST(thetas: torch.Tensor, n_features: int=None):
    """
    Reshapes parameters to allow estimation of categorical distribution over MNIST dataset

    : param thetas: torch.Tensor of shape [batch_size, n_params]
    : param n_features: int, number of features in the dataset -- not used but included for consistency between functions
    : return distribution: torch.distributions.Categorical
    """
    reshaped_thetas = torch.permute(thetas.reshape(thetas.shape[0], 256, 1, 28,28),(0,2,3,4,1))
    distribution = torch.distributions.categorical.Categorical(logits=reshaped_thetas)
        
    return distribution
    


def Binary_MNIST(thetas: torch.Tensor, n_features: int=None):
    """
    Reshapes parameters to allow estimation of categorical distribution over Binary MNIST dataset

    : param thetas: torch.Tensor of shape [batch_size, n_params]
    : param n_features: int, number of features in the dataset -- not used but included for consistency between functions
    : return distribution: torch.distributions.Categorical
    """
    reshaped_thetas = torch.permute(thetas.reshape(thetas.shape[0], 2, 1, 28,28),(0,2,3,4,1))
    distribution = torch.distributions.categorical.Categorical(logits=reshaped_thetas)
        
    return distribution