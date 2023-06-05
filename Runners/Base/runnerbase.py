"""
    Base class to run training, testing, and sampling
"""

# Standard library imports
from abc import abstractmethod


# Third-party library imports
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

# Local imports
from utils import Timer
from Runners.Base.trainerbase import TrainerBase


class RunnerBase(TrainerBase):
    """
    Base class to run training, testing, and sampling
    """

    def __init__(self, config: DictConfig, dataset: str, model):
        super().__init__(config, dataset, model)

        self.timers["test"] = Timer()
        self.timers["sample"] = Timer()

    def test(
        self, dataloader: DataLoader, approx: bool = True, print_stats: bool = False
    ) -> float:
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
            nll = self.during_train_test(dataloader, model)["loss"]
        else:
            # Start timer
            self.timers["test"].set_start_time()
            ######################
            nll = self._test(dataloader, print_stats)
            ######################
            # Stop timer
            self.timers["test"].log_time()

        return nll

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Evaluate the model on the test set.

        Args:
            dataloader (DataLoader): DataLoader for test data
            approx (bool): Whether to use an approximate test or full test (default: True)
            print_stats (bool): Whether to print evaluation statistics (default: False)

        Returns:
            torch.Tensor: Negative log-likelihood (NLL) of the model's predictions
        """

        # Start timer
        self.timers["sample"].set_start_time()
        ######################
        samples = self._sample(num_samples)
        ######################
        # Stop timer
        self.timers["sample"].log_time()

        return samples

    ##################
    @abstractmethod
    def _test(
        self, dataloader: DataLoader, print_stats: bool
    ) -> float:
        """
        Run a full evaluation on the test set.

        Args:
            dataloader (DataLoader): DataLoader for test data
            print_stats (bool): Whether to print evaluation statistics
        """

    @abstractmethod
    def _sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the model.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Generated samples
        """
