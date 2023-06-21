"""
This file is used to run the model
"""
# Standard library imports
import os

# Third-party library imports
import torch
from omegaconf import OmegaConf
from hydra import utils
import hydra

# Local imports
from utils import save_samples, print_config, log_stats_to_wandb
from Runners.runnerardm import RunnerARDM
from Model.get_models import get_ardm


def verify_samples(runner):
    """
    This function is used to verify the samples of the conditional model. Saves true, blurry and generated samples

    Args:
        runner (RunnerARDM): Runner object
    """
    x = runner.loaders["valid"].dataset.tensors[0][: runner.config.num_samples]
    x_hat = runner.loaders["valid"].dataset.tensors[1][: runner.config.num_samples]
    samples = runner.sample(
        runner.config.num_samples, runner.config.num_forward_passes, primary_x=x_hat
    )

    # save samples
    save_samples(
        x.long().permute(0, 2, 3, 1),
        dataset=runner.config.dataset,
        save_dir=runner.config.sample_dir,
        save_name=runner.config.run_name + "_x",
    )
    save_samples(
        x_hat.long().permute(0, 2, 3, 1),
        dataset=runner.config.dataset,
        save_dir=runner.config.sample_dir,
        save_name=runner.config.run_name + "_blurry_x_hat",
    )
    save_samples(
        samples[-1].long().permute(0, 2, 3, 1),
        dataset=runner.config.dataset,
        save_dir=runner.config.sample_dir,
        save_name=runner.config.run_name + "_corrected",
    )


def update_configs(deafult_config):
    """
    This function is used to update the configs with the default configs - dummy config is used for initial loading

    Args:
        deafult_config (dict): Default config

    Returns:
        config (dict): Updated config
    """
    original_cwd = utils.get_original_cwd()
    config_path = f"{original_cwd}/Configs/{deafult_config.new_cfg}"
    config = OmegaConf.load(config_path)

    # Loop over dict1 and update matching keys in dict2
    for key, value in deafult_config.items():
        if key in config:
            config[key] = value
        elif key == "new_cfg":
            pass
        else:
            raise ValueError(f"Unknown Config: {key}")

    ## Update Configs ##
    config.model_dir += f"{config.project_name}/{config.model_name}/"
    config.sample_dir += f"{config.project_name}/{config.model_name}/"

    return config


@hydra.main(config_path="./Configs/", config_name="deafult", version_base=None)
def run_test(config):
    """
    This function is used to run the model

    Args:
        config (dict): Configs

    Returns:
        runner (RunnerARDM): Runner object
    """
    config = update_configs(config)

    ## Print Config File ##
    print_config(config)

    ## Address wandb logging and errors ##
    if not config.log_online:
        os.environ["WANDB_MODE"] = "offline"
    else:
        if os.name == "posix":
            from signal import (
                signal,
                SIGPIPE,
                SIG_DFL,
            )  # to fix signal broken pipe error on linux

            signal(SIGPIPE, SIG_DFL)

    ## Set Multiprocessing Method ##
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.multiprocessing.set_start_method("spawn")

    print(f"Running {config.run_name}")

    ## Get Model and Runner ##
    ardm = get_ardm(config, config.conditioned_on_x_hat)
    runner = RunnerARDM(config, config.dataset, ardm)
    print("Model Loaded")

    ## Train Model ##
    if config.Train:
        print("Training Model")
        runner.train(config.max_epochs)

    ## Test Model ##
    if config.Test:
        if config.final_test:
            loader = runner.loaders["test"]
        else:
            loader = runner.loaders["valid"]

        print("Testing Model")
        test_bpd = runner.test(
            loader, approx=config.approx_test, print_stats=config.print_stats
        )
        print(f"Test BPD: {test_bpd:.4f}")
    else:
        test_bpd = None

    ## Sample from Model ##
    if config.Sample:
        print("Sampling from Model")
        samples = runner.sample(config.num_samples, config.num_forward_passes)

        # Save samples
        for i, sample in enumerate(samples):
            save_samples(
                sample.long().permute(0, 2, 3, 1),
                dataset=config.dataset,
                save_dir=config.sample_dir,
                save_name=f"{config.run_name}_{i}",
            )

    ## Log Stats to Wandb if logging active##
    log_stats_to_wandb(runner, test_bpd)

    ## Verify Samples ## - only for a particular run
    if config.x_hat in [
        "x_given_gentle_blur_x",
        "x_given_medium_blur_x",
        "x_given_strong_blur_x",
    ]:
        print("Generating before and after samples")
        verify_samples(runner)

    return runner


if __name__ == "__main__":
    run_test()
