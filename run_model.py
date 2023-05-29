from utils import *
from Runners.RunnerARDM import RunnerARDM
from Model.get_models import get_ARDM
from omegaconf import DictConfig, OmegaConf
from hydra import utils
import hydra

def verify_samples(runner):

    x = runner.validloader.dataset.tensors[0][:runner.config.num_samples]
    x_hat = runner.validloader.dataset.tensors[1][:runner.config.num_samples]
    samples = runner.Sample(runner.config.num_samples, runner.config.num_forward_passes, runner.config.random_every, runner.config.ADS, return_every=runner.config.return_every, x_hat=x_hat)

    # save samples
    save_samples(x.long().permute(0,2,3,1), dataset = runner.config.dataset, save_dir = runner.config.sample_dir, save_name = "x")
    save_samples(x_hat.long().permute(0,2,3,1), dataset = runner.config.dataset, save_dir = runner.config.sample_dir, save_name = "blurry_x_hat")
    save_samples(samples[-1].long().permute(0,2,3,1), dataset = runner.config.dataset, save_dir = runner.config.sample_dir, save_name = runner.config.run_name)
    






def update_configs(deafult_config):
    original_cwd = utils.get_original_cwd()
    config_path = f"{original_cwd}/Configs/{deafult_config.new_cfg}"
    config = OmegaConf.load(config_path)

    # Loop over dict1 and update matching keys in dict2
    for key, value in deafult_config.items():
        if key in config:
            config[key] = value
        elif key == 'new_cfg':
            pass
        else:
            raise ValueError(f"Unknown Config: {key}")
        
    ## Update Configs ##
    config.model_dir += f"/{config.model_name}/"
    config.sample_dir += f"/{config.model_name}/"

    return config






@hydra.main(config_path="./Configs/", config_name="deafult", version_base=None)
def run_test(config):

    config = update_configs(config)

    ## Print Config File ##
    print_config(config)

    ## Address wandb logging and errors ##
    if not config.log_online: 
        os.environ['WANDB_MODE'] = 'offline'
    else:
        if os.name == 'posix':
            # to fix signal broken pipe error on linux
            from signal import signal, SIGPIPE, SIG_DFL
            signal(SIGPIPE,SIG_DFL)
        

    ## Set Multiprocessing Method ##
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    

    print(f"Running {config.run_name}")


    ## Get Model and Runner ##
    ardm = get_ARDM(config, config.conditioned_on_x_hat)
    runner = RunnerARDM(config.dataset, config, ardm)
    
    print("Model Loaded")

    ## Train Model ##
    if config.Train:
        print("Training Model")
        runner.Train(config.max_epochs)


    ## Test Model ##    
    if config.Test:
        if config.final_test:
            loader = runner.testloader
        else:
            loader = runner.validloader

        print("Testing Model")
        test_bpd = runner.Test(loader,approx=config.approx_test, print_stats=config.print_stats)
    else:
        test_bpd = None


    ## Sample from Model ##
    if config.Sample:
        print("Sampling from Model")
        samples = runner.Sample(config.num_samples, config.num_forward_passes, config.random_every, config.ADS, return_every=config.return_every)
        if config.save_all_samples:
            for i in range(len(samples)):
                save_samples(samples[i].long().permute(0,2,3,1), dataset = config.dataset, save_dir = config.sample_dir, save_name = f"{config.run_name}_{i}")
        else:
            save_samples(samples[-1].long().permute(0,2,3,1), dataset = config.dataset, save_dir = config.sample_dir, save_name = config.run_name)


    ## Log Stats to Wandb if logging active##
    log_stats_to_wandb(runner, test_bpd)

    ## Verify Samples ## - only for a particular run
    if config.run_name == "x_given_blur_x":
        verify_samples(runner)

    return runner



if __name__ == "__main__":
    run_test()