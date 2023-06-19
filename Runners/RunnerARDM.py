"""
    Child class of RunnerBase for ARDM model

    Supports training, testing, and sampling of conditional and standard ARDM model
"""
# Standard library imports
from typing import Optional, Tuple, Dict, List
import time

# Third-party library imports
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import numpy as np
import wandb
from omegaconf import DictConfig

# Local imports
from Runners.Base.runnerbase import RunnerBase
from Data.data import get_loaders, get_cond_ardm_loaders
from utils import loss_array_to_loss


class RunnerARDM(RunnerBase):
    """
    Class to run training, testing, and sampling of ARDM
    """

    def __init__(self, config: DictConfig, dataset: str, model):
        super().__init__(config, dataset, model)

        self.num_dims = self.config.n_dims

    def get_data_loaders(
        self, config: DictConfig, dataset: str
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get data loaders for training, validation, and testing for ARDM

        Args:
            config: configuration file
            dataset: dataset to use

        Returns:
            Tuple of training, validation, and testing data loaders
        """
        if self.model.conditioned_on_x_hat and config.x_hat is not None:
            return get_cond_ardm_loaders(dataset, config)
        if not self.model.conditioned_on_x_hat:
            return get_loaders(dataset, config)

        raise NameError(
            "Please check config file for errors in x_hat or conditioned_on_x_hat"
        )

    def sample(self, num_samples: int, num_forward_passes:int, primary_x: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        samples = self._sample(num_samples, num_forward_passes, primary_x=primary_x)
        ######################
        # Stop timer
        self.timers["sample"].log_time()

        return samples


    def _run_train_batch(
        self,
        x: torch.Tensor,
        model,
        gpu_id: int,
        mask: torch.Tensor,
        optimizer,
        selection: torch.Tensor,
        primary_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run a single training batch - computes gradients and updates model

        Args:
            x: batch of data
            model: model to train
            gpu_id: gpu to train on
            mask: mask to apply to data
            optimizer: optimizer to use
            selection: selection mask
            primary_x: primary data

        Returns:
            nll: negative log likelihood of batch
        """
        optimizer.zero_grad()
        nll = self._run_batch(x=x, model=model, gpu_id=gpu_id, mask=mask, selection=selection, primary_x=primary_x)
        nll.backward()

        if self.config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad)
        optimizer.step()

        return nll

    def _run_batch(
        self,
        x: torch.Tensor,
        model,
        gpu_id: int,
        mask: torch.Tensor,
        selection: torch.Tensor,
        primary_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run a single batch - computes negative log likelihood

        Args:
            x: batch of data
            model: model to train
            gpu_id: gpu to train on
            mask: mask to apply to data
            selection: selection mask
            primary_x: primary data

        Returns:
            nll: negative log likelihood of batch
        """
        x, mask, selection, primary_x = self._load_to_gpu(
            gpu_id, x, mask, selection, primary_x
        )

        return loss_array_to_loss(model.nll(x, mask, primary_x), selection)

    def _run_epoch(
        self,
        dataloader: DataLoader,
        model,
        gpu_id: int,
        split: str,
        optimizer=Optional[None],
        sigma: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self._epoch_setup(model, split, sigma, timestep)

        batch_losses = []

        for i, x in enumerate(dataloader):
            # If conditional model then grab x_hat, otherwise just x
            if model.conditioned_on_x_hat:
                x, primary_x = x[0], x[1]
            else:
                x = x[0]
                primary_x = None

            if split == "train":
                mask, _ = model.ordering.sample_random_masks(self.config.batch_size)

                # Multiply AVG loss from unmasked features by num of features to obtain total loss
                future_selection = 1 - mask
                loss = self._run_train_batch(x=x, model=model, gpu_id=gpu_id, mask=mask, optimizer=optimizer, selection=future_selection, primary_x=primary_x)

                batch_losses.append(loss.detach().cpu().numpy())

                if self.config.print_batch_loss:
                    print(f"[GPU{gpu_id}] Batch {i} | Train Loss {loss}")

                if gpu_id == 0 and self.config.active_log:
                    wandb.log({"train_batch_nll": loss})

            elif split == "test":
                mask, current_selection = model.ordering.sample_masks(
                    self.config.batch_size, sigma, timestep
                )

                if self.config.approx_test:
                    selection = 1 - mask
                else:
                    selection = current_selection

                loss = self._run_batch(x=x, model=model, gpu_id=gpu_id, mask=mask, selection=selection, primary_x=primary_x)
                batch_losses.append(loss.detach().cpu().numpy())

        return {"loss": np.mean(batch_losses)}

    def _print_and_log(
        self,
        model_module,
        losses,
        master_node,
        valid_losses,
    ):
        """
        Print and log losses

        Args:
            model_module: model to train
            losses: training losses
            master_node: whether node is master node
            valid_losses: validation losses
        """
        rank = next(model_module.parameters()).device.index

        # Log to wandb if active
        if master_node and self.config.active_log:
            wandb.log({"Epoch": model_module.epoch})
            wandb.log({"train_epoch_nll": losses["loss"]})

        if master_node and (model_module.epoch) % self.config.test_every == 0:
            if self.config.active_log:
                wandb.log({"valid_epoch_nll": valid_losses})

            print(
                f"[GPU{rank}] Epoch {model_module.epoch} | Train Loss {losses['loss']} | Valid Loss {valid_losses} ",
                flush=True,
            )
        else:
            print(
                f"[GPU{rank}] Epoch {model_module.epoch} | Train Loss {losses['loss']} | ",
                flush=True,
            )

    def _during_train_test(self, dataloader: DataLoader, model) -> dict:
        """
        Evaluate the model during training.

        Args:
            model (torch.nn.Module): Model instance
            dataloader (DataLoader): DataLoader for evaluation data

        Returns:
            torch.Tensor: Negative log-likelihood (NLL) of the model's predictions
        """
        # Sample orderings to be used for this estimate ( add choice for single ordering?)
        sigma = model.ordering.sample_random_orderings(dataloader.batch_size)
        timesteps = model.ordering.sample_timesteps(dataloader.batch_size)

        return self._run_epoch(
            dataloader, model, 0, "test", sigma=sigma, timestep=timesteps
        )

    def _test(
        self,
        dataloader: DataLoader,
        print_stats: bool,
    ) -> float:
        """
        Run a full evaluation on the test set.

        Args:
            dataloader (DataLoader): DataLoader for test data
            print_stats (bool): Whether to print evaluation statistics
        """
        sigma = self.model.ordering.sample_random_orderings(dataloader.batch_size)
        num_dims = self.num_dims

        with mp.Pool(processes=self.num_gpus) as pool:
            results = []
            for i in range(self.num_gpus):
                start = i * num_dims // self.num_gpus
                end = (i + 1) * num_dims // self.num_gpus
                if i == self.num_gpus - 1:
                    end = num_dims
                result = pool.apply_async(self._compute_nll_for_timestep_on_gpu, args= (dataloader, i, start, end, sigma, print_stats))
                results.append((start, result))

            nll = [0] * num_dims
            for start, result in results:
                end = start + len(result.get())
                nll[start:end] = result.get()

        for i in nll:
            if i == 0:
                raise ValueError("NLL is zero. Something went wrong.")

        return np.mean(nll).item()

    def _sample(
        self,
        num_samples: int,
        num_forward_passes: Optional[int],
        primary_x: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Generate samples from the model.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Generated samples
        """
        if self.model.conditioned_on_x_hat and primary_x is None:
            raise NotImplementedError(
                "Sampling not supported for conditional models (Requires x_hat distribution, which we do not have and is not learnt(not implemented)). Please either provide a x_hat, build a unconditional model or avoid sampling."
            )

        # Move model to gpu
        model = self.model.to(0)

        # Turn of gradients as unnessesary
        torch.set_grad_enabled(False)
        model.eval()

        # Get output data shape
        data_shape = tuple([num_samples] + list(self.config.data_shape))

        # Empty sample output tensor
        x_out = torch.zeros(data_shape).to(torch.device(f"cuda:{0}"), non_blocking=True)

        sigma = self.model.ordering.sample_random_orderings(num_samples)

        num_dims_per_forward_pass, max_forward_pass, num_forward_passes = self._num_dims_sampling(
            num_forward_passes
        )

        # Stores snapshots of the generated samples at every return_every forward passes
        savepoints = []

        # Loop over forward passes
        for i in range(num_forward_passes):
            if max_forward_pass:
                mask, current_selection = self.model.ordering.sample_masks(
                    num_samples,
                    sigma,
                    self.model.ordering.generate_timestep_tensor(num_samples, i),
                )
            else:
                # sample mask for end state, then sample for start state and subtract to get target selection
                start_timestep = i * num_dims_per_forward_pass

                # For last forward pass, sample all remaining dimensions
                if i == num_forward_passes - 1:
                    end_timestep = self.num_dims
                else:
                    end_timestep = (i + 1) * num_dims_per_forward_pass

                start_mask, _ = self.model.ordering.sample_masks(
                    num_samples,
                    sigma,
                    self.model.ordering.generate_timestep_tensor(
                        num_samples, start_timestep
                    ),
                )
                end_mask, _ = self.model.ordering.sample_masks(
                    num_samples,
                    sigma,
                    self.model.ordering.generate_timestep_tensor(
                        num_samples, end_timestep
                    ),
                )

                current_selection = end_mask - start_mask
                mask = start_mask

            # load to gpu
            x, mask, current_selection, _primary_x = self._load_to_gpu(
                0, x_out, mask, current_selection, primary_x
            )

            # Sample from model
            sample = model.sample(x, mask, _primary_x)

            # Add current selection of generated sample to x_out
            x_out += current_selection * sample

            print(f"Forward pass {i + 1} of {num_forward_passes} complete")

            savepoints.append(x_out.clone())

        return savepoints

    def _epoch_setup(self, model, split, sigma, timestep):
        torch.set_grad_enabled(split == "train")
        if split == "train":
            model.train()
        else:
            model.eval()
            # For testing sigma and timstep must be provided
            assert sigma[0] is not None
            assert timestep is not None

    def _num_dims_sampling(self, num_forward_passes):
        # Check if sampling multiple dimensions per forward pass
        if num_forward_passes is None:
            num_forward_passes = self.num_dims
            max_forward_pass = True
            print(
                f"Using Standard Auto-regressive Sampling, will take {num_forward_passes} forward passes to sample all dimensions"
            )
        else:
            max_forward_pass = False

        # Calculate number of dimensions to sample per forward pass
        num_dims_per_forward_pass = self.num_dims // num_forward_passes

        return num_dims_per_forward_pass, max_forward_pass, num_forward_passes

    def _compute_nll_for_timestep_on_gpu(
        self,
        dataloader: DataLoader,
        gpu_id: int,
        start: int,
        end: int,
        sigma: torch.Tensor,
        print_stats: bool = False,
    ):
        """
        Function to compute the NLL for a given timestep range on a given GPU

        : param gpu_id: (int) GPU to use for computation
        : param start: (int) Start timestep
        : param end: (int) End timestep
        : param dataloader: (DataLoader) Dataloader to use for computation
        : param sigma: (torch.Tensor) Orderings to use for computation
        : param results_queue: (Queue) Queue to store the results in
        : param print_stats: (bool) If True, prints the NLL for each timestep
        """
        # Set the device and move the model to the selected GPU
        device = torch.device(f"cuda:{gpu_id}")
        model = self.model.to(device)

        losses = []
        # Perform the NLL computation for the given timestep range on the selected GPU
        for i in range(start, end):
            time1 = time.time()
            timesteps = model.ordering.generate_timestep_tensor(
                dataloader.batch_size, i
            )
            timestep_loss = self._run_epoch(
                dataloader=dataloader,
                model=model,
                gpu_id=gpu_id,
                split="test",
                sigma=sigma,
                timestep=timesteps,
            )["loss"]
            losses.append(timestep_loss)

            time2 = time.time()

            if i == start:
                seconds = (time2 - time1) * (end - start)
                print(
                    f"Expected approx test time: {seconds} seconds or {seconds/3600} hours",
                    flush=True,
                )

            if print_stats:
                print(
                    f"GPU: {gpu_id} | Timestep: { i} | Loss: {timestep_loss}",
                    flush=True,
                )

        print(f"GPU {gpu_id} finished")
        return losses
