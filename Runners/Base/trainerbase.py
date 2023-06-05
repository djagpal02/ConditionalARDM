"""
    Training class for the model
"""

# Standard library imports
import os
import math
from typing import Optional, Tuple, Dict
import glob
from abc import ABC, abstractmethod


# Third-party library imports
import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import wandb
from omegaconf import DictConfig

# Local imports
from Data.data import get_loaders
from utils import save_checkpoint, load_checkpoint, Timer


class TrainerBase(ABC):
    """
    Trainer class to train the model.
    """

    def __init__(self, config: DictConfig, dataset: str, model):
        """
        Initialize the Trainer class.

        Args:
            dataset (str): Name of the dataset to use.
            config (DictConfig): Configuration file.
            model (torch.nn.Module): Model instance.
        """
        # Set up config and model
        self.config = config
        self.model = model

        # Set up GPUS
        self.num_gpus = len(config.gpu_ids)

        # Set up GPU IDs - sets desired gpus to be only ones visible to torch
        gpu_str = ", ".join(str(config.gpu_ids[i]) for i in range(self.num_gpus))

        # Set visible GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

        # Set up distributed training
        self.distributed = self.num_gpus > 1

        # Load data
        print("Loading Data")
        trainloader, validloader, testloader = self.get_data_loaders(config, dataset)
        self.loaders = {"train": trainloader, "valid": validloader, "test": testloader}

        # If loading model parameters from checkpoint, load checkpoint
        if self.config.save_name != "None":
            print("Checking for Checkpoint")
            try:
                self._load_checkpoint(self.config.save_name)
            except FileNotFoundError as exc:
                raise FileNotFoundError("No checkpoint found by given name") from exc
        else:
            print("No checkpoint to load, building new model")

        # Setup Timers
        self.timers = {
            "train": Timer(),
            "approx_test": Timer(),
        }

    def get_data_loaders(
        self, config: DictConfig, dataset: str
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for the dataset.

        Args:
            config (DictConfig): Config file
            dataset (str): Name of dataset to use
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
        """
        return get_loaders(dataset, config)

    ##########################################
    # Training
    ##########################################

    def train(self, max_epochs: int) -> None:
        """
        Train the model for a given number of epochs.

        Args:
            max_epochs (int): Number of epochs to train for
        """
        # If using multiple GPUs, use DistributedDataParallel
        if self.distributed:
            print("Using Distributed Training")
            mp.spawn(
                self.run_train,
                nprocs=self.num_gpus,
                join=True,
                args=(self.loaders["train"], self.loaders["valid"], max_epochs),
            )
        else:
            print("Using Single GPU Training")
            self.run_train(0, self.loaders["train"], self.loaders["valid"], max_epochs)

        self._load_checkpoint("latest_train")

        print(f"{self.model.epoch} Epochs of Training Completed")

    def run_train(
        self,
        rank: int,
        dataloader: DataLoader,
        validloader: DataLoader,
        max_epochs: int,
    ) -> None:
        """
        Internal training function.

        Args:
            rank (int): Rank of the GPU
            max_epochs (int): Number of epochs to train for
            dataloader (DataLoader): DataLoader for training data
            validloader (DataLoader): DataLoader for validation data
        """
        (
            dataloader,
            model_module,
            early_stopping,
            optimizer,
            master_node,
        ) = self._train_setup(dataloader, rank)

        for epoch in range(max_epochs):
            if self.distributed:
                self._set_sampler_epoch(dataloader, epoch)

            losses = self._run_epoch_training(
                dataloader,
                model_module,
                optimizer,
                (epoch, master_node, max_epochs, rank),
            )

            valid_loss, early_stopping = self._validation(
                validloader, model_module, (early_stopping, epoch, master_node, rank)
            )

            self._print_and_log(model_module, losses, master_node, valid_loss)

            if early_stopping["epochs_without_improvement"] >= self.config.patience:
                print(
                    f"[GPU{rank}] Early stopping triggered. Training stopped after {epoch + 1} epochs."
                )
                break

        self._clean_up(model_module, early_stopping, master_node)

    ##########################################
    # Hidden Training Outer Functions
    ##########################################
    def _train_setup(self, dataloader: DataLoader, rank: int) -> Tuple:
        """
        Set up training.

        Args:
            dataloader (DataLoader): DataLoader for training data
            rank (int): Rank of the GPU
        Returns:
            Tuple: dataloader, model_module, early_stopping, optimizer, master_node
        """
        # To ensure that only one GPU is used for logging
        master_node = True

        # Set device to GPU
        model = self.model.to(device=rank)

        early_stopping = {
            "best_valid_loss": math.inf,
            "epochs_without_improvement": 0,
        }

        # If using multiple GPUs, set up distributed training
        if self.distributed:
            (
                dataloader,
                model_module,
                master_node,
                optimizer,
            ) = self._distributed_train_setup(dataloader, model, rank)
        else:
            model_module = model
            # Set up optimizer
            optimizer = torch.optim.Adam(model_module.parameters(), lr=self.config.lr)

        # Initialize wandb
        if master_node and self.config.active_log:
            wandb.init(project=self.config.project_name, name=self.config.run_name)

        if self.config.epochs_without_improvement is not None:
            early_stopping = {
                "best_valid_loss": self.config.best_valid_loss,
                "epochs_without_improvement": self.config.epochs_without_improvement,
            }

        return dataloader, model_module, early_stopping, optimizer, master_node

    def _distributed_train_setup(
        self, dataloader: DataLoader, model, rank: int
    ) -> Tuple:
        """
        Set up distributed training.

        Args:
            dataloader (DataLoader): DataLoader for training data
            model (torch.nn.Module): Model to train
            rank (int): Rank of the GPU
        Returns:
            Tuple: dataloader, model_module, master_node, optimizer
        """
        # Set up the distributed environment
        os.environ["MASTER_ADDR"] = "localhost"
        try:
            os.environ["MASTER_PORT"] = self.config.port
        except OSError as exc:
            raise OSError(
                "No port specified for distributed training or specified port in use."
            ) from exc

        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=self.num_gpus)

        # Wrap the model
        model = DDP(model, device_ids=[rank])
        model_module = model.module

        # Set the dataloader to use the distributed sampler
        dataloader = self.create_distributed_dataloader(dataloader, rank, self.num_gpus)

        # Check if master node
        if rank != 0:
            master_node = False

        # Set up optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        return dataloader, model_module, master_node, optimizer

    def _expected_train_time(self, max_epochs: int):
        """
        Calculates the expected train time based on the average run times of the 'train' and 'approx_test' timers.

        Args:
            max_epochs (int): The maximum number of epochs for training.

        Prints:
            The expected train time in seconds and hours.
            The average run time for the 'train' timer.
            The average run time for the 'approx_test' timer.
        """
        expected_train_time = self.timers[
            "train"
        ].average_run_time() * max_epochs + self.timers[
            "approx_test"
        ].average_run_time() * (
            max_epochs / self.config.test_every
        )
        ett_hours = expected_train_time / 3600
        print(
            f" \n \n  Expected Train time: {expected_train_time} seconds or {ett_hours} hours \n \n",
            flush=True,
        )
        print(
            f"Train average run time: {self.timers['train'].average_run_time()} \n \n",
            flush=True,
        )
        print(
            f"approx test average run time: {self.timers['approx_test'].average_run_time()} \n \n",
            flush=True,
        )

    def _run_epoch_training(
        self,
        dataloader: DataLoader,
        model_module,
        optimizer: torch.optim.Optimizer,
        args: Tuple,
    ) -> dict:
        """
        Runs a single epoch of training.

        Args:
            dataloader (DataLoader): DataLoader for training data
            model_module (torch.nn.Module): Model to train
            optimizer (torch.optim.Optimizer): Optimizer to use for training
            args (Tuple): Tuple containing epoch, master_node, max_epochs, and rank
        Returns:
            dict: Dictionary containing the losses for the epoch
        """
        epoch, master_node, max_epochs, rank = args

        if master_node and epoch == self.config.test_every:
            self._expected_train_time(max_epochs)

        self.timers["train"].set_start_time()

        losses = self._run_epoch(
            dataloader=dataloader, model=model_module, gpu_id=rank, split="train", optimizer=optimizer
        )

        self.timers["train"].log_time()

        model_module.epoch += 1

        self._train_save(model_module, epoch, master_node, rank)

        return losses

    def _set_sampler_epoch(self, dataloader: DataLoader, epoch: int):
        """
        Sets the epoch for the dsitributed sampler.

        Args:
            dataloader (DataLoader): DataLoader for training data
            epoch (int): The epoch number
        """
        assert isinstance(dataloader.sampler, DistributedSampler)
        dataloader.sampler.set_epoch(epoch)

    def _validation(
        self,
        validloader: DataLoader,
        model_module,
        args: Tuple,
    ) -> Tuple[Optional[Dict], Dict]:
        """
        Runs validation on the model.

        Args:
            validloader (DataLoader): DataLoader for validation data
            model_module (torch.nn.Module): Model to validate
            args (Tuple): Tuple containing early_stopping, epoch, master_node, and rank
        Returns:
            Tuple: Tuple containing the validation loss and the early stopping dictionary
        """
        early_stopping, epoch, master_node, rank = args

        testing_epoch = master_node and (model_module.epoch) % self.config.test_every == 0

        valid_loss, early_stopping = self._validate_if_testing_epoch(
            validloader, model_module, early_stopping, testing_epoch
        )

        epochs_without_improvement_tensor = (
            self._create_epochs_without_improvement_tensor(early_stopping, rank)
        )

        if self.distributed:
            self._broadcast_and_barrier(epochs_without_improvement_tensor)
            early_stopping["epochs_without_improvement"] = int(
                epochs_without_improvement_tensor.item()
            )

        return valid_loss, early_stopping

    def _create_epochs_without_improvement_tensor(
        self, early_stopping: Dict, rank: int
    ) -> torch.Tensor:
        """
        Creates a tensor containing the number of epochs without improvement.

        Args:
            early_stopping (Dict): Dictionary containing the number of epochs without improvement
            rank (int): The rank of the process
        Returns:
            torch.Tensor: Tensor containing the number of epochs without improvement
        """
        return torch.tensor(early_stopping["epochs_without_improvement"]).to(
            torch.device(f"cuda:{rank}")
        )

    def _validate_if_testing_epoch(
        self,
        validloader: DataLoader,
        model_module,
        early_stopping: Dict,
        testing_epoch: bool,
    ) -> Tuple[Optional[Dict], Dict]:
        """
        Validates the model if the epoch is a testing epoch.

        Args:
            validloader (DataLoader): DataLoader for validation data
            model_module (torch.nn.Module): Model to validate
            early_stopping (Dict): Dictionary containing the number of epochs without improvement
            testing_epoch (bool): Whether or not the epoch is a testing epoch
        Returns:
            Tuple: Tuple containing the validation loss and the early stopping dictionary
        """
        valid_loss = None
        if testing_epoch:
            valid_loss = self.during_train_test(validloader, model_module)[
                "loss"
            ].item()

            if early_stopping["best_valid_loss"] - valid_loss > self.config.tolerance:
                early_stopping["best_valid_loss"] = valid_loss
                early_stopping["epochs_without_improvement"] = 0
            else:
                early_stopping["epochs_without_improvement"] += self.config.test_every

        return valid_loss, early_stopping

    def _broadcast_and_barrier(self, epochs_without_improvement_tensor: torch.Tensor):
        """
        Broadcasts the number of epochs without improvement and then performs a barrier.

        Args:
            epochs_without_improvement_tensor (torch.Tensor): Tensor containing the number of epochs without improvement
        """
        dist.broadcast(epochs_without_improvement_tensor, src=0)
        dist.barrier()

    def _train_save(self, model_module, epoch: int, master_node: bool, rank: int):
        """
        Saves the model if the epoch is a save epoch.

        Args:
            model_module (torch.nn.Module): Model to save
            epoch (int): The epoch number
            master_node (bool): Whether or not the process is the master node
            rank (int): The rank of the process
        """
        # Save model if save_every is reached
        if master_node and (epoch + 1) % self.config.save_every == 0:
            self._save_checkpoint(
                model_module,
                f"{self.config.project_name}_{self.config.run_name}_Epoch_{model_module.epoch}",
            )
            print(f"[GPU{rank}] Epoch {model_module.epoch} Model Parameters Saved")

    def _clean_up(self, model_module, early_stopping: Dict, master_node: bool):
        """
        Performs clean up after training.

        Args:
            model_module (torch.nn.Module): Model to save
            early_stopping (Dict): Dictionary containing the number of epochs without improvement
            master_node (bool): Whether or not the process is the master node
        """
        # Save model after training
        if master_node:
            print(
                f"best_valid_loss: {early_stopping['best_valid_loss']} \n epochs_without_improvement: {early_stopping['epochs_without_improvement']} \n ",
                flush=True,
            )

            self._save_checkpoint(
                model_module,
                f"{self.config.project_name}_{self.config.run_name}_Epoch_{model_module.epoch}",
            )

            if self.config.active_log:
                wandb.finish()

        if self.distributed:
            # Destroy process group
            dist.destroy_process_group()

    ##########################################
    # Train time Testing
    ##########################################
    def during_train_test(self, dataloader: DataLoader, model) -> dict:
        """
        Evaluate the model during training.

        Args:
            model (torch.nn.Module): Model instance
            dataloader (DataLoader): DataLoader for evaluation data

        Returns:
            torch.Tensor: Negative log-likelihood (NLL) of the model's predictions
        """
        # Start timer
        self.timers["approx_test"].set_start_time()

        #######################
        nll_dict = self._during_train_test(dataloader, model)
        #######################

        # Stop timer
        self.timers["approx_test"].log_time()

        return nll_dict

    ##########################################
    # Data loading and distributing
    ##########################################
    def _load_to_gpu(
        self, gpu_id: int, *Tensors: Optional[torch.Tensor]
    ):
        """
        Load tensors to a specific GPU.

        Args:
            gpu_id (int): GPU ID to load tensors to
            *Tensors (torch.Tensor): Tensors to load

        Returns:
            Tuple[torch.Tensor, ...]: Loaded tensors
        """
        loaded_tensors = []
        for tensor in Tensors:
            if isinstance(tensor, torch.Tensor):
                loaded_tensors.append(
                    tensor.to(gpu_id, non_blocking=True)
                )
            else:
                loaded_tensors.append(tensor)
        return tuple(loaded_tensors)

    def create_distributed_dataloader(
        self, dataloader: DataLoader, rank: int, world_size: int
    ) -> DataLoader:
        """
        Create a distributed data loader.

        Args:
            dataloader (DataLoader): DataLoader for distributed training data
            rank (int): Rank of the GPU
            world_size (int): Number of GPUs used for training

        Returns:
            DataLoader: Distributed data loader
        """
        # Create a new DataLoader with the DistributedSampler
        distributed_dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            sampler=DistributedSampler(
                dataloader.dataset, num_replicas=world_size, rank=rank
            ),
            drop_last=dataloader.drop_last,
        )

        return distributed_dataloader

    ##########################################
    # Core Training/Running Methods (Abstract)
    ##########################################
    @abstractmethod
    def _run_train_batch(self):
        pass

    @abstractmethod
    def _run_batch(self):
        pass

    @abstractmethod
    def _run_epoch(
        self,
        dataloader: DataLoader,
        model,
        gpu_id: int,
        split: str,
        optimizer=Optional[None],
    ) -> dict:
        pass

    @abstractmethod
    def _print_and_log(self, model_module, losses, master_node, valid_loss):
        pass

    @abstractmethod
    def _during_train_test(self, dataloader: DataLoader, model):
        pass

    ##########################################
    # Saving and Loading
    ##########################################
    def _save_checkpoint(self, model, save_name: str) -> None:
        """
        Save a checkpoint of the model.

        Args:
            model (torch.nn.Module): Model instance
            save_name (str): Name of the checkpoint file
        """
        states = {
            "net": model.state_dict(),
            "epoch": model.epoch,
        }

        save_checkpoint(
            save_name=save_name, states=states, model_dir=self.config.model_dir
        )
        # Remove states to clean up memory
        del states

    def _load_checkpoint(self, save_name: str) -> None:
        """
        Load a checkpoint for the model.

        Args:
            save_name (str): Name of the checkpoint file
        """
        if save_name == "latest_train":
            # Get the latest saved checkpoint after training
            save_name = f"{self.config.project_name}_{self.config.model_name}_Epoch_*"
            path = os.path.join(self.config.model_dir, f"{save_name}.pth")
            latest_checkpoint = torch.load(max(glob.glob(path), key=os.path.getctime))
            self.model.load_state_dict(latest_checkpoint["net"])
            self.model.epoch = latest_checkpoint["epoch"]
        else:
            checkpoint = load_checkpoint(
                save_name=save_name, model_dir=self.config.model_dir
            )
            self.model.load_state_dict(checkpoint["net"])
            self.model.epoch = checkpoint["epoch"]
            print(f"Loaded state {save_name}, trained for {self.model.epoch} epochs")
            del checkpoint
