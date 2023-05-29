import os
import math
from typing import Optional, Tuple, Union, List, Dict
import glob
from abc import ABC, abstractmethod

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import wandb

from Data.Data import get_loaders
from utils import save_checkpoint, load_checkpoint, timer


class RunnerBase(ABC):
    """
    Base class to run training, testing, and sampling
    """

    def __init__(self, dataset: str, config, model):
        """
        Initialise Runner class

        : param dataset: (str) Name of dataset to use
        : param gpu_id: (int) ID of GPU to use
        : param config: (DictConfig) Config file
        : param model:  model
        """
        # Set up config and model
        self.config = config
        self.model = model

        # Set up GPUS
        self.gpu_ids = config.gpu_ids
        self.num_gpus = len(config.gpu_ids)

        # Set up GPU IDs - sets desired gpus to be only ones visible to torch
        gpu_str = ""
        for i in range(self.num_gpus):
            gpu_str += str(config.gpu_ids[i])
            if i != self.num_gpus - 1:
                gpu_str += ", "

        # Set visible GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

        # Set up gpu ids in natural ordering as visible to torch
        self.gpu_ids = [i for i in range(len(config.gpu_ids))]

        # Set up distributed training
        self.distributed = (self.num_gpus > 1)

        # Load data
        print("Loading Data")
        self.trainloader, self.validloader, self.testloader = self.get_data_loaders(dataset, config)

        # If loading model parameters from checkpoint, load checkpoint
        if self.config.save_name != "None":
            print("Checking for Checkpoint")
            try:
                self._load_checkpoint(self.config.save_name)
            except:
                print("No checkpoint found by given name")
        else:
            print("No checkpoint to load, building new model")

        # Setup Timers
        self.train_timer = timer()
        self.approx_test_timer = timer()
        self.test_timer = timer()
        self.sample_timer = timer()

    def get_data_loaders(self, dataset: str, config):
        return get_loaders(dataset, config)


    ################################################################################################################################################################################
    #                                                                Key Functions                                                                                                 #
    ################################################################################################################################################################################
    def Train(self, max_epochs:int ) -> None:
        """
        Function to train model for a given number of epochs

        : param max_epochs: (int) Number of epochs to train for
        """

        # If using multiple GPUs, use DistributedDataParallel
        if self.distributed:
            print("Using Distributed Training")
            mp.spawn(self._train, nprocs=self.num_gpus, join=True, args=(max_epochs, self.trainloader, self.validloader))
        else:
            print("Using Single GPU Training")
            self._train(self.gpu_ids[0], max_epochs, self.trainloader, self.validloader)

        self._load_checkpoint('latest_train')

        print(f'{self.model.epoch} Epochs of Training Completed')




    def _train(self, rank:int , max_epochs: int, dataloader: DataLoader, validloader: DataLoader=None) -> None:
        """
        Function to train model for a given number of epochs

        : param rank: (int) Rank of GPU
        : param max_epochs: (int) Number of epochs to train for
        : param dataloader: (DataLoader) Dataloader for training data
        : param validloader: (DataLoader) Dataloader for validation data
        """
        # To ensure that only one GPU is used for logging
        Master_Node = True


        # Set device to GPU
        model = self.model.to(device=rank)

        # If using multiple GPUs, set up distributed training
        if self.distributed:
            # Set up the distributed environment
            os.environ['MASTER_ADDR'] = 'localhost'
            try:
                os.environ['MASTER_PORT'] = self.config.port
            except:
                os.environ['MASTER_PORT'] = '12355'
            # Initialize the process group
            dist.init_process_group("nccl", rank=rank, world_size=self.num_gpus)

            # Wrap the model
            model = DDP(model, device_ids=[rank])
            model_module = model.module
            
            # Set the dataloader to use the distributed sampler
            dataloader = self.create_distributed_dataloader(dataloader, rank, self.num_gpus)

            # Check if master node
            if rank != 0:
                Master_Node = False
    
            # Set up optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

    
        else:
            model_module = model

            # Set up optimizer
            optimizer = torch.optim.Adam(model_module.parameters(), lr=self.config.lr)


        if Master_Node and self.config.active_log:
            wandb.init(project=self.config.project_name, name=self.config.run_name)



        if validloader != None:
            if self.config.epochs_without_improvement == None:
                # Initialize variables for early stopping
                best_valid_loss = math.inf
                epochs_without_improvement = 0
            else:
                epochs_without_improvement: self.config.epochs_without_improvement
                best_valid_loss: self.config.best_valid_loss


        # Looping over number of epochs
        for epoch in range(max_epochs):
            if Master_Node and epoch == self.config.test_every:
                expected_train_time = self.train_timer.average_run_time()*max_epochs + self.approx_test_timer.average_run_time() * (max_epochs/self.config.test_every)
                ett_hours = expected_train_time / 3600
                print(f' \n \n  Expected Train time: {expected_train_time} seconds or {ett_hours} hours \n \n', flush=True)
                print(f'self.train_timer.average_run_time(): {self.train_timer.average_run_time()} \n \n', flush=True)
                print(f'self.approx_test_timer.average_run_time(): {self.approx_test_timer.average_run_time()} \n \n', flush=True)

            # Start train timer
            self.train_timer.set_start_time()

            # Set epoch for distributed training
            if self.distributed:
                dataloader.sampler.set_epoch(epoch)

############################
            # Get losses
            losses = self._run_train(model_module, rank, dataloader, optimizer)
####################################

            # Stop train timer
            self.train_timer.log_time()

            # Increase models internal epoch count
            model_module.epoch += 1
            

            # Create a tensor to store epochs_without_improvement
            epochs_without_improvement_tensor = torch.tensor(epochs_without_improvement).to(rank)




            Testing_epoch = Master_Node and  (epoch+1) % self.config.test_every == 0 and validloader != None


            # Test model if test_every is reached
            valid_loss = None
            if Testing_epoch:
                valid_loss = self.during_train_test(model_module, validloader)

                # Update best_valid_loss and reset patience counter if there's improvement
                if best_valid_loss - valid_loss > self.config.tolerance:
                    best_valid_loss = valid_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += self.config.test_every


                # Update the tensor with the new value of epochs_without_improvement
                epochs_without_improvement_tensor = torch.tensor(epochs_without_improvement).to(rank)

######################                
            self._print_and_log(losses, valid_loss, model_module, Master_Node, rank, Testing_epoch)
######################


            # Save model if save_every is reached
            if Master_Node and (epoch+1) % self.config.save_every == 0:
                self._save_checkpoint(model_module, f'{self.config.project_name}_{self.config.run_name}_Epoch_{model_module.epoch}')

                print(f"[GPU{rank}] Epoch {model_module.epoch} Model Parameters Saved")


            if self.distributed:
                # broadcast the value of epochs_without_improvement from GPU 0
                dist.broadcast(epochs_without_improvement_tensor, src=0)
                dist.barrier()
                epochs_without_improvement = epochs_without_improvement_tensor.item()

            # Break the loop if early stopping criterion is met
            if epochs_without_improvement >= self.config.patience:
                print(f"[GPU{rank}] Early stopping triggered. Training stopped after {epoch + 1} epochs.")
                break

            



        # Save model after training
        if Master_Node:
            print(f'best_valid_loss: {best_valid_loss} \n epochs_without_improvement: {epochs_without_improvement} \n ' , flush=True)

            self._save_checkpoint(model_module, f'{self.config.project_name}_{self.config.run_name}_Epoch_{model_module.epoch}')

            if self.config.active_log:
                wandb.finish()

        if self.distributed:
            # Destroy process group
            dist.destroy_process_group()




    def during_train_test(self, model, dataloader: DataLoader) -> torch.Tensor:
        """
        Function to test model during training (approx test)

        : param model: (nn.Module) Model to test
        : param dataloader: (DataLoader) Dataloader for test data
        : return: (torch.Tensor) Test loss / NLL
        """
        # Start timer
        self.approx_test_timer.set_start_time()

#######################
        nll = self._during_test_train(model, dataloader)
#######################

        # Stop timer
        self.approx_test_timer.log_time()

        return nll
    

    def Test(self, dataloader: DataLoader, approx: bool=True, print_stats: bool = False) -> torch.Tensor:
        if approx: 
            model = self.model.to(self.gpu_ids[0])
            nll = self.during_train_test(model, dataloader)
        else:
            # Start timer
            self.test_timer.set_start_time()
######################
            nll = self._full_test(dataloader, print_stats)
######################
            # Stop timer
            self.test_timer.log_time()

        return nll


##################
    @abstractmethod
    def Sample(self, num_samples: int) -> torch.Tensor:
        pass
##################

    ################################################################################################################################################################################
    #                                                                Parts of key functions that need redefing                                                                                                #
    ################################################################################################################################################################################
    @abstractmethod
    def _run_train(self,model_module, rank, dataloader, optimizer):
        pass

    @abstractmethod
    def _print_and_log(self, losses, valid_losses, model_module, Master_Node, rank, Testing_epoch):
        pass
    
    @abstractmethod
    def _during_test_train(self, model, dataloader):
        pass

    @abstractmethod
    def _full_test(self, dataloader, print_stats):
        pass 


    ################################################################################################################################################################################
    #                                                                 Parallelisation                                                                                            #
    ################################################################################################################################################################################   
    def create_distributed_dataloader(self, dataloader: DataLoader, rank: int, world_size: int):
        """
        Function to create a DistributedSampler for a given DataLoader

        : param dataloader: (DataLoader) DataLoader to create a DistributedSampler for
        : param rank: (int) Rank of the current process
        : param world_size: (int) Number of processes in the current DDP setup
        : return: (DataLoader) Distributed DataLoader
        """
        # Create a DistributedSampler using the original dataset
        distributed_sampler = DistributedSampler(dataloader.dataset, num_replicas=world_size, rank=rank)

        # Create a new DataLoader with the DistributedSampler
        distributed_dataloader = DataLoader(dataset=dataloader.dataset, batch_size=dataloader.batch_size, num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory, sampler=distributed_sampler, drop_last=dataloader.drop_last)

        return distributed_dataloader
    

    ################################################################################################################################################################################
    #                                                                 Private Functions                                                                                            #
    ################################################################################################################################################################################
    def _load_to_gpu(self, gpu_id: int, *Tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loaded_tensors = []
        for tensor in Tensors:
            try:
               loaded_tensors.append(tensor.to(gpu_id, non_blocking=True))
            except:
                loaded_tensors.append(tensor)
        return loaded_tensors
    


    @abstractmethod
    def _run_train_batch(self, model, gpu_id: int, x: torch.Tensor, Mask: torch.Tensor, selection: torch.Tensor, optimizer, primary_reconstruction: torch.Tensor=None) -> Dict[str, torch.Tensor]:
       pass

    @abstractmethod
    def _run_batch(self, model, gpu_id:int ,  x: torch.Tensor, Mask: torch.Tensor, selection: torch.Tensor, primary_reconstruction: torch.Tensor=None, mus: torch.Tensor=None, log_vars: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        pass



    @abstractmethod
    def _run_epoch(self,model, gpu_id: int, dataloader: DataLoader, split: str, sigma=[None], timestep: torch.Tensor = None, optimizer=None) -> Tuple[float, float]:
        pass



    ################################################################################################################################################################################
    #                                                                 Save and Load                                                                                                #
    ################################################################################################################################################################################


    def _save_checkpoint(self, model,  save_name: str) -> None:
        """
        Function to save a checkpoint

        : param save_name: (str) Name to save checkpoint as
        """
        states = {
                    'net': model.state_dict(),
                    'epoch': model.epoch,
                }
        
        save_checkpoint(save_name=save_name, states=states, model_dir=self.config.model_dir)
        # Remove states to clean up memory
        del states
        
        
        
        
    def _load_checkpoint(self, save_name: str) -> None:
        """
        Function to load a checkpoint

        : param save_name: (str) Name of checkpoint to load
        """
        if save_name == "latest_train":
            # Get the latest saved checkpoint after training
            save_name = f'{self.config.project_name}_{self.config.model_name}_Epoch_*'
            PATH =  os.path.join(self.config.model_dir, '{}.pth'.format(save_name)) 
            latest_checkpoint = torch.load(max(glob.glob(PATH), key=os.path.getctime))
            self.model.load_state_dict(latest_checkpoint['net'])
            self.model.epoch = latest_checkpoint['epoch']
        else:
            checkpoint = load_checkpoint(save_name=save_name, model_dir=self.config.model_dir)
            self.model.load_state_dict(checkpoint['net'])         
            self.model.epoch = checkpoint['epoch']
            print(f"Loaded state {save_name}, trained for {self.model.epoch} epochs")
            del checkpoint