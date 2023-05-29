from Runners.RunnerBase import RunnerBase

from typing import Optional, Tuple, Union, List, Dict
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import numpy as np
import wandb
import time
from Data.Data import get_loaders, get_cond_ARDM_loaders
from utils import loss_array_to_loss





class RunnerARDM(RunnerBase):
    """
    Class to run training, testing, and sampling of ARDM
    """
    def __init__(self, dataset: str, config, model):
        super().__init__(dataset, config, model)

        self.num_dims = self.config.n_dims


    ################################################################################################################################################################################
    def get_data_loaders(self, dataset: str, config):
        if self.model.conditioned_on_x_hat and config.x_hat is not None:
            return get_cond_ARDM_loaders(dataset, config)
        elif not self.model.conditioned_on_x_hat:
            return get_loaders(dataset, config)
        else:
            raise Exception('Please check config file for errors in x_hat or conditioned_on_x_hat')

    ################################################################################################################################################################################
    #                                                                Core Function                                                                                                 #
    ################################################################################################################################################################################
    def _run_train(self,model_module, rank, dataloader, optimizer):
        # Get losses
        return self._run_epoch(model_module, rank, dataloader, 'train', optimizer = optimizer)





    def _print_and_log(self, losses, valid_loss, model_module, Master_Node, rank, Testing_epoch):
        # Log to wandb if active
        if Master_Node and self.config.active_log:
            wandb.log({'Epoch': model_module.epoch})
            wandb.log({'train_epoch_nll': losses})

        if Testing_epoch:
            if self.config.active_log:
                wandb.log({'valid_epoch_nll': valid_loss})
            
            print(f"[GPU{rank}] Epoch {model_module.epoch} | Train Loss {losses} | Valid Loss {valid_loss} ", flush=True)
        else:
            print(f"[GPU{rank}] Epoch {model_module.epoch} | Train Loss {losses} | ", flush=True)





    def _during_test_train(self, model, dataloader):
        # Sample orderings to be used for this estimate ( add choice for single ordering?)
        sigma = model.ordering.sample_random_orderings(dataloader.batch_size)
        timesteps = model.ordering.sample_timesteps(dataloader.batch_size)

        return self._run_epoch(model, self.gpu_ids[0], dataloader, 'test', sigma, timesteps)





    def _full_test(self, dataloader, print_stats):
        sigma = self.model.ordering.sample_random_orderings(dataloader.batch_size)
        num_dims = self.num_dims

        with mp.Pool(processes=self.num_gpus) as pool:
            results = []
            for i, gpu_id in enumerate(self.gpu_ids):
                start = i * num_dims // self.num_gpus
                end = (i + 1) * num_dims // self.num_gpus
                if i == self.num_gpus - 1:
                    end = num_dims

                result = pool.apply_async(self.compute_nll_for_timestep_on_gpu, args=(gpu_id, start, end, dataloader, sigma, print_stats))
                results.append((start, result))

            nll = [0] * num_dims
            for start, result in results:
                end = start + len(result.get())
                nll[start:end] = result.get()

        for i in nll:
            if i == 0:
                raise ValueError("NLL is zero. Something went wrong.")

        return np.mean(nll)




    def Sample(self, num_samples: int, num_forward_passes: int= None, random_every:int = None, ADS: bool=False, gpu_id: int = None, return_every: int = None, x_hat: torch.Tensor = None) -> torch.Tensor:
        """
        Function to sample data from modelled distribution, Generate new samples x' ~ P(X) 

        : param num_samples: (int) Number of samples to generate
        : param num_forward_passes: (int) Number of forward passes to use for sampling (if None, uses number of dimensions)
        : param random_every: (int) If using ADS sampling, how often to randomly sample 
        : param ADS: (bool) Whether to use ADS sampling (samples based on neighbouring dimensions)
        : param gpu_id: (int) GPU to use for sampling
        : param return_every: (int) If not None, provides snapshots of generated sample at every return_every samples
        : return: (torch.Tensor) Samples from model
        """
        if self.model.conditioned_on_x_hat and x_hat is None:
            raise NotImplementedError('Sampling not supported for conditional models (Requires x_hat distribution, which we do not have and is not learnt(not implemented)). Please either provide a x_hat, build a unconditional model or avoid sampling.')
        
        # Start timer
        self.sample_timer.set_start_time()

        # If no gpu_id is specified, use the first gpu
        if gpu_id is None:
            gpu_id = self.gpu_ids[0]

        # Move model to gpu
        model = self.model.to(gpu_id)

        # Turn of gradients as unnessesary 
        torch.set_grad_enabled(False)
        model.eval()
        
        # Get output data shape
        data_shape = tuple([num_samples] + [x for x in self.config.data_shape])

        # Empty sample output tensor
        x_out = torch.zeros(data_shape).to(gpu_id, non_blocking=True)



        sigma = self.model.ordering.sample_random_orderings(num_samples)


        # Check if sampling multiple dimensions per forward pass
        if num_forward_passes == None:
            num_forward_passes = self.num_dims
            Max_forward_pass = True
            print(f'Using Standard Auto-regressive Sampling, will take {num_forward_passes} forward passes to sample all dimensions')
        else:
            Max_forward_pass = False

        # Calculate number of dimensions to sample per forward pass
        num_dims_per_forward_pass = self.num_dims // num_forward_passes

        # Stores snapshots of the generated samples at every return_every forward passes
        savepoints = []

        
        # Loop over forward passes
        for i in range(num_forward_passes):
            if Max_forward_pass:
                Mask, current_selection = self.model.ordering.sample_masks(num_samples, sigma, self.model.ordering.generate_timestep_tensor(num_samples, i))
            else:
                # sample mask for end state, then sample for start state and subtract to get target selection
                Start_timestep = i * num_dims_per_forward_pass

                # For last forward pass, sample all remaining dimensions
                if i == num_forward_passes - 1:
                    End_timestep = self.num_dims
                else:
                    End_timestep = (i + 1) * num_dims_per_forward_pass


                start_mask, _ = self.model.ordering.sample_masks(num_samples, sigma, self.model.ordering.generate_timestep_tensor(num_samples, Start_timestep))
                end_mask, _ = self.model.ordering.sample_masks(num_samples, sigma, self.model.ordering.generate_timestep_tensor(num_samples, End_timestep))

                current_selection = end_mask - start_mask
                Mask = start_mask

            
            #load to gpu
            x, Mask, current_selection, x_hat_ = self._load_to_gpu(gpu_id, x_out, Mask, current_selection, x_hat)

            # Sample from model
            Sample = model.sample(x, Mask, x_hat_)

            # Add current selection of generated sample to x_out
            x_out += current_selection * Sample
            
            print(f'Forward pass {i + 1} of {num_forward_passes} complete')

            if return_every is not None:
                if i % return_every == 0:
                    savepoints.append(x_out.clone())

        # Add final sample to savepoints
        savepoints.append(x_out.clone())
        
        # Stop timer
        self.sample_timer.log_time()

        return savepoints

    ################################################################################################################################################################################
    #                                                                 Parallelisation                                                                                            #
    ################################################################################################################################################################################
    def compute_nll_for_timestep_on_gpu(self, gpu_id:int , start:int , end:int , dataloader: DataLoader, sigma: torch.Tensor, print_stats:bool = False):
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
            timesteps = model.ordering.generate_timestep_tensor(dataloader.batch_size, i)
            timestep_loss = self._run_epoch(model, gpu_id, dataloader=dataloader, split='test', sigma=sigma, timestep=timesteps)
            losses.append(timestep_loss)

            time2 = time.time()

            if i == start:
                seconds = (time2-time1)*(end-start)
                print(f'Expected approx test time: {seconds} seconds or {seconds/3600} hours', flush=True)


            if print_stats:
                print('GPU: {} | Timestep: {} | Loss: {}'.format(gpu_id, i, timestep_loss), flush=True)

        print("GPU {} finished".format(gpu_id))
        return losses
    

    ################################################################################################################################################################################
    #                                                                 Forward Model                                                                                                #
    ################################################################################################################################################################################
    def _run_train_batch(self, model, gpu_id: int, x: torch.Tensor, Mask: torch.Tensor, selection: torch.Tensor, optimizer, x_hat: torch.tensor) -> Dict[str, torch.Tensor]:
        """
        Function to run a single training batch

        : param model: (nn.Module) Model to train
        : param gpu_id: (int) GPU to use for training
        :param x: (torch.Tensor) Input data
        :param Mask: (torch.Tensor) Mask for AR model
        :param selection: (torch.Tensor) Selection for AR model
        :return: (torch.Tensor) NLL
        """
        optimizer.zero_grad()
        nll = self._run_batch(model, gpu_id, x, Mask, selection, x_hat)

        
        nll.backward()

        if self.config.clip_grad != None:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad)
        optimizer.step()

        return nll



    def _run_batch(self, model, gpu_id:int ,  x: torch.Tensor, Mask: torch.Tensor, selection: torch.Tensor, x_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Function to run a single batch

        : param model: (nn.Module) Model to train
        : param gpu_id: (int) GPU to use for training
        :param x: (torch.Tensor) Input data
        :param Mask: (torch.Tensor) Mask for AR model
        :param selection: (torch.Tensor) Selection for AR model
        :return: (torch.Tensor) NLL
        """
        x, Mask, selection, x_hat= self._load_to_gpu(gpu_id, x, Mask, selection, x_hat)         

        return loss_array_to_loss(model.NLL(x, Mask, x_hat), selection)




    def _run_epoch(self,model, gpu_id: int, dataloader: DataLoader, split: str, sigma=[None], timestep: torch.Tensor = None, optimizer=None) -> Tuple[float, float]:
        """
        Runs an epoch

        :param dataloader: (torch.utils.data.DataLoader) Dataloader to use
        :param split: (str) Split to run
        :param sigma: (list) Orderings to use
        :param timestep: (torch.Tensor) Timestep to use
        :return: (float) NLL
        """
        torch.set_grad_enabled(split=='train')
        if split == 'train':
            model.train()
        else:
            model.eval()
            # For testing sigma and timstep must be provided
            assert(sigma[0] != None)
            assert(timestep != None)

        batch_losses = []

        for i,x in enumerate(dataloader):
            # If conditional model then grab x_hat, otherwise just x
            if model.conditioned_on_x_hat:
                x, x_hat = x[0], x[1]
            else:
                x = x[0]
                x_hat = None
                

            if split == "train":
                Mask, _ = model.ordering.sample_random_masks(self.config.batch_size)

                # Multiply AVG loss from unmasked features by num of features to obtain total loss 
                future_selection = 1-Mask 
                loss = self._run_train_batch(model, gpu_id, x, Mask, future_selection, optimizer, x_hat)
                batch_losses.append(loss.detach().cpu().numpy())

                if self.config.print_batch_loss:
                    print(f"[GPU{gpu_id}] Batch {i} | Train Loss {loss}")
                    
                if gpu_id == 0 and self.config.active_log:
                    wandb.log({'train_batch_nll': loss})

            elif split == "test":
                    Mask, current_selection = model.ordering.sample_masks(self.config.batch_size, sigma, timestep)

                    if self.config.approx_test:
                        selection = 1-Mask
                    else:
                        selection = current_selection

                    loss = self._run_batch(model, gpu_id, x, Mask, selection, x_hat)
                    batch_losses.append(loss.detach().cpu().numpy())
                
        
        return np.mean(batch_losses)
    
