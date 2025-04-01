import sys
import copy
import torch
import wandb
import numpy as np
import torch.nn as nn

sys.path.append('../')
sys.path.append('../data')
sys.path.append('../losses')
sys.path.append('../models')
from training import model_train
from preprocessing import data_loader
from models.fno import fno2d
from losses.physics_loss import PhysicsLoss

def wandb_sweep(project_name = 'fno_reshuffle_PILoss_sweep', count=100):
    
        wandb.login()
        
        sweep_config = {
            'method'    : 'random',
            'metric'    : { 'name': 'valid_loss', 
                            'goal': 'minimize'},
            'parameters': { 'learning_rate' : {"values": [0.001, 0.0005, 0.0001]},
                            'PIloss_weight' : {"max": 0.45, "min": 0.0001},
                            # 'epochs'        : {'values': [50, 100, 200]},
                            'batch_size'    : {"values": [16, 32, 64, 128]},
                            # 'n_layers'      : {"values": [4, 6, 8, 10]},
                            }       
                        }
        
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        wandb.agent(sweep_id, sweep_fno, count=count)
        wandb.finish()
        
        return

def sweep_fno():
    
    # Initialize a new wandb run
    with wandb.init() as run:
        # If called by wandb.agent, as below,
        
        # this config will be set by Sweep Controller
        config = run.config
        
        # retrieving sweeping hyperparameter
        # epochs          = config.epochs
        batch_size      = config.batch_size
        learning_rate   = config.learning_rate
        # n_layers        = config.n_layers
        PIloss_weight   = config.PIloss_weight
        
        epochs          = 150
        
        # model setup
        model_class = fno2d
        param_dict  = {'modes'          : 32, 
                       'width'          : 64, 
                       'n_layers'       : 4, 
                       'padding_frac'   : 0.25}
        
        casename     = 'data2_20241116_normalised'
        casename     = 'data_20241125_alpha_normalised'
        casename     = 'data_20241125_alpha_reshuffle_normalised'
        
        modelname    = ''
        in_channels  = 1 
        out_channels = 3 
        dim          = (128, 128)
        
        restart_model   = ''
        load_checkpoint = False
        check_step      = 10
        
        scheduler_step  = 10
        scheduler_gamma = 0.5
        optimizer_class = torch.optim.Adam
        scheduler_class = torch.optim.lr_scheduler.StepLR
        trainloss_class = PhysicsLoss(loss_param_dict = {'loss' : nn.MSELoss(reduction = 'sum'),
                                                         'a_div': PIloss_weight,
                                                         'a_ns' : PIloss_weight,
                                                         'scale_base': 0.3e6,
                                                         'scale_div' : 1e-5,
                                                         'scale_ns'  : 0.01,
                                                         'grad_dim': [2 , 1], # gradient will be given in (x,y)
                                                         'spacing' : [1.2/(dim[1]-1) , 0.8/(dim[0]-1)],
                                                         'rho'     : 1,
                                                         'mu'      : 0.01,
                                                         'norm_info' : '../data/processed/%s/norm_info.csv'%(casename),
                                                         'dim'       : dim
                                                         })
        validloss_class = nn.MSELoss(reduction='sum')
        
        path_model = model_train(model_class  = model_class, 
                                 param_dict   = param_dict,
                                 casename     = casename,
                                 modelname    = modelname,
                                 in_channels  = in_channels,
                                 out_channels = out_channels, 
                                 dim          = dim,
                                 restart_model   = restart_model,
                                 load_checkpoint = load_checkpoint,
                                 check_step      = check_step,
                                 epochs          = epochs,
                                 batch_size      = batch_size,
                                 learning_rate   = learning_rate,
                                 scheduler_step  = scheduler_step,
                                 scheduler_gamma = scheduler_gamma,
                                 optimizer_class = optimizer_class,
                                 scheduler_class = scheduler_class,
                                 trainloss_class = trainloss_class,
                                 validloss_class = validloss_class,
                                 save_model      = False,
                                 wandb_flag      = False,
                                 wandb_sweep     = True)
    
    return path_model

if __name__ == "__main__":
    
    # Random Seeds
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    np.random.seed(0)
    
    wandb_sweep()
    