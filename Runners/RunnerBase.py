"""
    Base class to run training, testing, and sampling
"""

# Standard library imports
import os
import math
from typing import Optional, Tuple, List, Dict
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
from utils import save_checkpoint, load_checkpoint, timer


class RunnerBase(ABC):
    """
    Base class to run training, testing, and sampling
    """

    def __init__(self, dataset: str, config: DictConfig, model: torch.nn.Module):
        """
        Initialize Runner class

        Args:
            dataset (str): Name of dataset to use
            config (DictConfig): Config file
            model (torch.nn.Module): Model instance
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
        trainloader, validloader, testloader = self.get_data_loaders(dataset, config)
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
            "train": timer(),
            "approx_test": timer(),
            "test": timer(),
            "sample": timer(),
            "total": timer(),
        }

    def get_data_loaders(
        self, dataset: str, config: DictConfig
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for the dataset.

        Args:
            dataset (str): Name of dataset to use
            config (DictConfig): Config file

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders
        """
        return get_loaders(dataset, config)

    ################################################################################################################################################################################
    #                                                                Key Functions                                                                                                 #
    ################################################################################################################################################################################
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
                self._train,
                nprocs=self.num_gpus,
                join=True,
                args=(max_epochs, self.loaders["train"], self.loaders["valid"]),
            )
        else:
            print("Using Single GPU Training")
            self._train(0, max_epochs, self.loaders["train"], self.loaders["valid"])

        self._load_checkpoint("latest_train")

        print(f"{self.model.epoch} Epochs of Training Completed")

    def _distributed_train_setup(
        self, rank: int, model: torch.nn.Module, dataloader: DataLoader
    ) -> Tuple:
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

        return model_module, optimizer, dataloader, master_node

    def _train_setup(self, rank: int, dataloader: DataLoader) -> Tuple:
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
                model_module,
                optimizer,
                dataloader,
                master_node,
            ) = self._distributed_train_setup(rank, model, dataloader)

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

        return model_module, optimizer, dataloader, master_node, early_stopping

    def _validation(
        self, early_stopping, epoch, model_module, validloader, rank, master_node
    ):
        # Create a tensor to store epochs_without_improvement
        epochs_without_improvement_tensor = torch.tensor(
            early_stopping["epochs_without_improvement"]
        ).to(torch.device(f"cuda:{rank}"))

        testing_epoch = (
            master_node
            and (epoch + 1) % self.config.test_every == 0
            and validloader is not None
        )

        # Test model if test_every is reached
        valid_loss = None
        if testing_epoch:
            valid_loss = self.during_train_test(model_module, validloader)

            # Update best_valid_loss and reset patience counter if there's improvement
            if early_stopping["best_valid_loss"] - valid_loss > self.config.tolerance:
                early_stopping["best_valid_loss"] = valid_loss.item()
                early_stopping["epochs_without_improvement"] = 0
            else:
                early_stopping["epochs_without_improvement"] += self.config.test_every

            # Update the tensor with the new value of epochs_without_improvement
            epochs_without_improvement_tensor = torch.tensor(
                early_stopping["epochs_without_improvement"]
            ).to(torch.device(f"cuda:{rank}"))

        if self.distributed:
            # broadcast the value of epochs_without_improvement from GPU 0
            dist.broadcast(epochs_without_improvement_tensor, src=0)
            dist.barrier()
            early_stopping["epochs_without_improvement"] = int(
                epochs_without_improvement_tensor.item()
            )

        return valid_loss, early_stopping

    def _train_save(self, master_node, epoch, model_module, rank):
        # Save model if save_every is reached
        if master_node and (epoch + 1) % self.config.save_every == 0:
            self._save_checkpoint(
                model_module,
                f"{self.config.project_name}_{self.config.run_name}_Epoch_{model_module.epoch}",
            )
            print(f"[GPU{rank}] Epoch {model_module.epoch} Model Parameters Saved")

    def _clean_up(self, master_node, early_stopping, model_module):
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

    def _train(
        self,
        rank: int,
        max_epochs: int,
        dataloader: DataLoader,
        validloader: DataLoader,
    ) -> None:
        """
        Internal training function.

        Args:
            rank (int): Rank of the GPU
            max_epochs (int): Number of epochs to train for
        dataloader (DataLoader): DataLoader for training data
        validloader (Optional[DataLoader]): DataLoader for validation data (default: None)
        """
        (
            model_module,
            optimizer,
            dataloader,
            master_node,
            early_stopping,
        ) = self._train_setup(rank, dataloader)

        # Looping over number of epochs
        for epoch in range(max_epochs):
            # Print Expected Training Time using first few epochs and first validation test epoch
            if master_node and epoch == self.config.test_every:
                self.expected_train_time(max_epochs)

            # Start train timer
            self.timers["train"].set_start_time()

            # Set epoch for distributed training
            if self.distributed and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)

            ############################
            # Get losses
            losses = self._run_train(model_module, rank, dataloader, optimizer)
            ####################################

            # Stop train timer
            self.timers["train"].log_time()

            # Increase models internal epoch count
            model_module.epoch += 1

            self._train_save(master_node, epoch, model_module, rank)
            valid_loss, early_stopping = self._validation(
                early_stopping, epoch, model_module, validloader, rank, master_node
            )

            self._print_and_log(losses, valid_loss, model_module, master_node, rank)

            # Break the loop if early stopping criterion is met
            if early_stopping["epochs_without_improvement"] >= self.config.patience:
                print(
                    f"[GPU{rank}] Early stopping triggered. Training stopped after {epoch + 1} epochs."
                )
                break

        # Clean up
        self._clean_up(master_node, early_stopping, model_module)

    def during_train_test(
        self, model: torch.nn.Module, dataloader: DataLoader
    ) -> torch.Tensor:
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
        nll = self._during_test_train(model, dataloader)
        #######################

        # Stop timer
        self.timers["approx_test"].log_time()

        return nll

    def test(
        self, dataloader: DataLoader, approx: bool = True, print_stats: bool = False
    ) -> torch.Tensor:
        """
        Evaluate the model on the test set.

        Args:
            dataloader (DataLoader): DataLoader for test data
            approx (bool): Whether to use an approximate test or full test (default: True)
            print_stats (bool): Whether to print evaluation statistics (default: False)

        Returns:
            torch.Tensor: Negative log-likelihood (NLL) of the model's predictions
        """
        if approx:
            model = self.model.to(0)
            nll = self.during_train_test(model, dataloader)
        else:
            # Start timer
            self.timers["test"].set_start_time()
            ######################
            nll = self._full_test(dataloader, print_stats)
            ######################
            # Stop timer
            self.timers["test"].log_time()

        return nll

    ##################
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the model.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Generated samples
        """

    ##################

    ################################################################################################################################################################################
    #                                                                Parts of key functions that need redefing                                                                                                #
    ################################################################################################################################################################################
    @abstractmethod
    def _run_train(
        self,
        model_module: torch.nn.Module,
        rank: int,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Run the training loop for a single epoch.

        Args:
            model_module (torch.nn.Module): Model instance
            rank (int): Rank of the GPU
            dataloader (DataLoader): DataLoader for training data
            optimizer (torch.optim.Optimizer): Optimizer for model parameters
        """

    @abstractmethod
    def _print_and_log(
        self, losses, valid_losses, model_module, master_node, rank, testing_epoch
    ):
        """
        Print and log training and validation losses.

        Args:
            losses: Training losses
            valid_losses: Validation losses
            model_module: Model instance
            master_node: Whether the current process is the master node
            rank: Rank of the GPU
            testing_epoch: Whether it is a testing epoch
        """

    @abstractmethod
    def _during_test_train(self, model: torch.nn.Module, dataloader: DataLoader):
        """
        Evaluate the model during training.

        Args:
            model (torch.nn.Module): Model instance
            dataloader (DataLoader): DataLoader for evaluation data

        Returns:
            torch.Tensor: Negative log-likelihood (NLL) of the model's predictions
        """

    @abstractmethod
    def _full_test(self, dataloader: DataLoader, print_stats: bool):
        """
        Run a full evaluation on the test set.

        Args:
            dataloader (DataLoader): DataLoader for test data
            print_stats (bool): Whether to print evaluation statistics
        """

    ################################################################################################################################################################################
    #                                                                 Parallelisation                                                                                            #
    ################################################################################################################################################################################
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

    ################################################################################################################################################################################
    #                                                                 Private Functions                                                                                            #
    ################################################################################################################################################################################
    def _load_to_gpu(
        self, gpu_id: int, *Tensors: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
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
            try:
                loaded_tensors.append(
                    tensor.to(torch.device(f"cuda:{gpu_id}"), non_blocking=True)
                )
            except RuntimeError:
                loaded_tensors.append(tensor)
        return tuple(loaded_tensors)

    @abstractmethod
    def _run_train_batch(
        self,
        model: torch.nn.Module,
        gpu_id: int,
        x: torch.Tensor,
        mask: torch.Tensor,
        selection: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        primary_reconstruction: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a single training batch.

        Args:
            model (torch.nn.Module): Model instance
            gpu_id (int): GPU ID
            x (torch.Tensor): Input tensor
            mask (torch.Tensor): Mask tensor
            selection (torch.Tensor): Selection tensor
            optimizer (torch.optim.Optimizer): Optimizer for model parameters
            primary_reconstruction (Optional[torch.Tensor]): Primary reconstruction tensor (default: None)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of losses
        """

    @abstractmethod
    def _run_batch(
        self,
        model: torch.nn.Module,
        gpu_id: int,
        x: torch.Tensor,
        mask: torch.Tensor,
        selection: torch.Tensor,
        primary_reconstruction: Optional[torch.Tensor] = None,
        mus: Optional[torch.Tensor] = None,
        log_vars: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run a single batch.

        Args:
            model (torch.nn.Module): Model instance
            gpu_id (int): GPU ID
            x (torch.Tensor): Input tensor
            mask (torch.Tensor): Mask tensor
            selection (torch.Tensor): Selection tensor
            primary_reconstruction (Optional[torch.Tensor]): Primary reconstruction tensor (default: None)
            mus (Optional[torch.Tensor]): Latent means (default: None)
            log_vars (Optional[torch.Tensor]): Latent log variances (default: None)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of losses
        """

    @abstractmethod
    def _run_epoch(
        self,
        model: torch.nn.Module,
        gpu_id: int,
        dataloader: DataLoader,
        split: str,
        sigma: List[Optional[torch.Tensor]] = [None],
        timestep: Optional[torch.Tensor] = None,
        optimizer=Optional[None],
    ) -> Tuple[float, float]:
        """
        Run a single epoch of training or evaluation.

        Args:
            model (torch.nn.Module): Model instance
            gpu_id (int): GPU ID
            dataloader (DataLoader): DataLoader for the dataset
            split (str): Split name ('train', 'valid', or 'test')
            sigma (List[Optional[torch.Tensor]]): List of optional sigma tensors (default: [None])
            timestep (Optional[torch.Tensor]): Timestep tensor (default: None)
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for model parameters (default: None)

        Returns:
            Tuple[float, float]: Tuple containing the average loss and average NLL
        """
        if sigma is None:
            sigma = [None]

        return (0.0, 0.0)

    ################################################################################################################################################################################
    #                                                                 Save and Load                                                                                                #
    ################################################################################################################################################################################
    def expected_train_time(self, max_epochs: int):
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

    def _save_checkpoint(self, model: torch.nn.Module, save_name: str) -> None:
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
