import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

def loss_array_to_loss(loss_array: torch.Tensor, selection: torch.Tensor) -> torch.Tensor:
    """
    Calculates loss from loss array and selection array, computes average based on selection

    :param loss_array: Array of loss values
    :param selection: Array of selection values
    :return: Loss
    """
    return (sum_over_all_but_batch(loss_array * (selection)) / sum_over_all_but_batch(selection)).mean()


def sum_over_all_but_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Sums over all dimensions except batch dimension

    :param x: Tensor
    :return: Summed tensor
    """
    return x.reshape(x.shape[0], -1).sum(-1)


def image_float_to_int(x: torch.Tensor) -> torch.Tensor:
    """
    Converts torch image in [-1,1] to discrete [0,256]

    :param x: Image in [-1,1]
    :return: Image in [0,256]
    """
    return torch.round( (x+1) * 127.5 ).long()


def image_int_to_float(x: torch.Tensor) -> torch.Tensor:
    """
    Converts torch image in [0,256] to [-1,1]

    :param x: Image in [0,256]
    :return: Image in [-1,1]
    """
    return x/127.5 - 1



def add_random_noise(image: torch.Tensor, mean: float, var: float, clip: bool=True) -> torch.Tensor:
    """
    Adds random noise to image

    :param image: Image
    :param mean: Mean of noise
    :param var: Variance of noise
    :param clip: Whether to clip image to [-1,1]
    :return: Image with added noise
    """
    # Detect if a signed image was input
    if image.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    image = image_int_to_float(image)

    noise = torch.normal(mean, var ** 0.5, size=image.shape)


    out = image + noise


    # Clip back to original range, if necessary
    if clip:
        out = np.clip(out, -1, 1.0)
    
    image = image_float_to_int(out)

    return image


def save_samples(samples: torch.Tensor, dataset = 'CIFAR10', save_dir = "./Samples/", save_name = "samples", ncols=10, nrows=10) -> None:
    """
    Save samples to pdf

    :param samples: Samples
    :param dataset: Dataset
    :param save_path: Path to save samples
    :param ncols: Number of columns
    :param nrows: Number of rows
    :return: None
    """
    # Makes directory for samples if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Add file extension
    PATH = save_dir + dataset + "_" + save_name + '.png'


    # Move samples to CPU, detach gradient and turn into numpy array
    samples = samples.cpu().detach().numpy()

    n_samples = ncols*nrows
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
    ax = axes.ravel()
    for i in range(n_samples):
        if dataset == 'CIFAR10':
            ax[i].imshow(samples[i])
        elif dataset == 'MNIST':
            ax[i].imshow(samples[i], cmap="gray")
        else:
            print('Dataset not recognised')
        ax[i].axis('off')

    fig.subplots_adjust(wspace=-0.35, hspace=0.065)
    plt.gca().set_axis_off()
    plt.savefig(PATH, dpi=300, bbox_inches="tight", pad_inches=0,)
    plt.close()

    print('Samples saved to: ' + PATH)



def save_checkpoint(save_name: str, states: dict, model_dir: str) -> None:
    """
    Save checkpoint to model directory with given name

    :param save_name: Name of checkpoint
    :param states: Dictionary of states
    :param model_dir: Directory to save checkpoint
    :return: None
    """
    # Makes directory for checkpoints if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    PATH =  os.path.join(model_dir, '{}.pth'.format(save_name)) 
    torch.save(states, PATH)
    print(f"Epoch {states['epoch']} | Training checkpoint saved at {PATH}")


def load_checkpoint(save_name, model_dir, map_location=torch.device('cpu')) -> dict:
    """
    Load checkpoint from model directory with given name

    :param save_name: Name of checkpoint
    :param model_dir: Directory to load checkpoint
    :param map_location: Device to load checkpoint to
    :return: Dictionary of states
    """
    PATH =  os.path.join(model_dir, '{}.pth'.format(save_name)) 
    checkpoint = torch.load(PATH, map_location=map_location)
    return checkpoint



class timer:
    """
    Timer class for timing code
    """
    def __init__(self):
        self.time_logs = []
        self.start_time = time.time()
    
    def set_start_time(self):
        """
        Sets start time to current time
        """
        self.start_time = time.time()
    
    def log_time(self):
        """
        Logs time since start time
        """
        self.time_logs.append(time.time() - self.start_time)

    def average_run_time(self):
        """
        Returns average run time
        """
        return np.mean(self.time_logs)


def print_config(dictionary):
    print(' ## CONFIG ## ')
    for key, value in dictionary.items():
        print(f"{key}: {value}")
    
    print("\n \n")


def log_stats_to_wandb(runner, test_bpd):
    if runner.config.active_log:
        import wandb
        wandb.init(project=runner.config.project_name, name=runner.config.run_name)

        # For prelim results, test_bpd is actually the validation bpd (to avoid test set leakage)
        num_gpus = runner.num_gpus
        train_epoch_time = runner.train_timer.average_run_time() # Average time per epoch in seconds - (per gpu, if using multiple gpus - * num_gpus to get total time for batch_size )) 
        approx_test_epoch_time = runner.approx_test_timer.average_run_time()
        test_epoch_time = runner.test_timer.average_run_time()
        sample_gen_time = runner.sample_timer.average_run_time()

        # Run your training or evaluation loop and obtain final output values
        output_dict = { 'Full_test_bpd': test_bpd,
                        'num_gpus': num_gpus,
                        'Train_epoch_time': train_epoch_time,
                        'Approx_test_epoch_time': approx_test_epoch_time,
                        'Test_epoch_time': test_epoch_time,
                        'Sample_gen_time': sample_gen_time}

        # Log final output values to Wandb
        wandb.log(output_dict)

        # Finish logging
        wandb.finish()
    
