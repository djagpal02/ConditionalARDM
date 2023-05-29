import copy
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

import numpy as np


def image_float_to_int(x: torch.Tensor) -> torch.Tensor:
    """
    Converts torch image in [-1,1] to discrete [0,256]

    :param x: Image in [-1,1]
    :return: Image in [0,256]
    """
    return torch.round( (x+1) * 127.5 ).long()




def get_loaders(dataset, config) -> Tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader]:
    """
    Builds a train and test loader for a given dataset
    :param dataset: str name of dataset
    :param config: DictConfig
    :return: trainloader, testloader
    """
    tr, va, te = load(dataset, config.data_dir, config.Reduction)

    # Imagenet has already been permuted
    if dataset == 'CIFAR10':
        tr = torch.Tensor(np.transpose(tr.data, (0, 3, 1, 2)))
        va = torch.Tensor(np.transpose(va.data, (0, 3, 1, 2)))
        te = torch.Tensor(np.transpose(te.data, (0, 3, 1, 2)))
    
    trainloader = build_loader(torch.utils.data.TensorDataset(tr),  config)
    validloader = build_loader(torch.utils.data.TensorDataset(va), config)
    testloader = build_loader(torch.utils.data.TensorDataset(te), config)

    return trainloader, validloader, testloader


def get_cond_ARDM_loaders(dataset, config):
    """
    Builds a train, valid and test loader for a given dataset for conditional ARDM
    :param dataset: str name of dataset
    :param config: DictConfig
    :return: trainloader, testloader
    """
    if dataset == 'CIFAR10':
        tr, va, te = load(dataset, config.data_dir, config.Reduction)
        tr = torch.Tensor(np.transpose(tr.data, (0, 3, 1, 2)))
        va = torch.Tensor(np.transpose(va.data, (0, 3, 1, 2)))
        te = torch.Tensor(np.transpose(te.data, (0, 3, 1, 2)))
        if config.x_hat == 'x_given_x':
            trainloader = build_loader(torch.utils.data.TensorDataset(tr, tr),  config)
            validloader = build_loader(torch.utils.data.TensorDataset(va, va), config)
            testloader = build_loader(torch.utils.data.TensorDataset(te, te), config)
        elif config.x_hat == 'x_given_blur_x':
            blurrer = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))

            trainloader = build_loader(torch.utils.data.TensorDataset(tr, blurrer(tr)),  config)
            validloader = build_loader(torch.utils.data.TensorDataset(va, blurrer(va)), config)
            testloader = build_loader(torch.utils.data.TensorDataset(te, blurrer(te)), config)
    else:
        NotImplementedError
    return trainloader, validloader, testloader


def build_loader(dataset: torch.utils.data.dataset.TensorDataset, config) -> torch.utils.data.dataloader.DataLoader:
    """
    Builds a Pytorch Dataloader with given batch size and Tensor dataset
    
    :param dataset: TensorDataset
    :param config: DictConfig
    :return: Dataloader
    """
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True,pin_memory=True,drop_last= True)


def load(dataset_name: str, path: str, Reduction=None, seed: int = 42) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads a dataset and returns the train, validation, and test set (Validation is set to 0.5* size of test set)
    :param dataset_name: str name of dataset
    :param path: str path to dataset
    :param single_class: bool if True, only one class is loaded, reduces size of dataset for testing purposes
    :param seed: int seed for random number generator
    :return: train, validation, test
    """
    np.random.seed(seed)

    # Load the dataset
    if dataset_name == 'CIFAR10':
        image_augmented_transforms = transforms.Compose([
                                                        transforms.RandomHorizontalFlip(p=0.5),
                                                        transforms.RandomVerticalFlip(p=0.5),
                                                        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                                        transforms.ToTensor(), 
                                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                                        lambda x: image_float_to_int(x)])

        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                    lambda x: image_float_to_int(x)])
 

        te = torchvision.datasets.CIFAR10(root = path, train = False, download=True, transform=transform)
        tr = torchvision.datasets.CIFAR10(root = path, train = True, download=True, transform=image_augmented_transforms)



    
    # Subset the dataset if requested
    if Reduction == 'single_class':
        # Reduce dataset to only class 0 -- To speed up testing during model building
        tr_idx = np.where(np.array(tr.targets) == 0)[0]
        te_idx = np.where(np.array(te.targets) == 0)[0]
            
        tr.data = tr.data[tr_idx]
        te.data = te.data[te_idx]

    elif Reduction != None:
        indices = np.arange(len(tr))
        np.random.shuffle(indices)
        indices = indices[:int((Reduction/100)*len(tr))]
        tr.data = tr.data[indices]



    # Create the validation set (10% of train set)
    va_size = int(0.1*len(tr))
    indices = np.arange(len(tr))

    # Shuffle the indices given the seed
    np.random.shuffle(indices)

    va_indices, tr_indices = indices[:va_size], indices[va_size:]

    va = copy.deepcopy(tr)
    
    tr.data = tr.data[tr_indices]
    va.data = va.data[va_indices]

    

    return tr, va, te
