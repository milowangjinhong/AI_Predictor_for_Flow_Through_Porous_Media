import sys
import copy
import torch
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

if __name__ ==  '__main__':
    
    # Random Seeds
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    np.random.seed(0)
    
    # model setup
    model_class = fno2d
    param_dict  = {'modes'          : 30, 
                   'width'          : 64, 
                   'n_layers'       : 4, 
                   'padding_frac'   : 0.25} # current paper
    
    # param_dict  = {'modes'          : 22, 
    #                'width'          : 64, 
    #                'n_layers'       : 8, 
    #                'padding_frac'   : 0.25}
    
    casename     = 'data_check_128_normalised'
    casename     = 'data2_20241116_normalised'
    casename     = 'data_20241125_alpha_normalised'
    casename     = 'data_20241125_alpha_reshuffle_3_normalised'
    
    modelname    = 'trial'
    modelname    = 'with_L2_loss_22-8'
    
    in_channels  = 1 # only gamma
    out_channels = 3 # 1 for p, 2 for u
    dim          = (128, 128)
    
    wandb_flag      = 1
    
    restart_model   = ''
    load_checkpoint = False
    check_step      = 10
    epochs          = 200
    batch_size      = 16
    learning_rate   = 0.001
    scheduler_step  = 10
    scheduler_gamma = 0.5
    optimizer_class = torch.optim.Adam
    scheduler_class = torch.optim.lr_scheduler.StepLR
    trainloss_class = nn.MSELoss(reduction='sum')
    
    modelname       = 'PILoss_scaled_optimised_30_mode_bc3'
    trainloss_class = PhysicsLoss(loss_param_dict = {'loss'         : nn.MSELoss(reduction = 'sum'),
                                                    #  'a_div'        : 0.03319342705152602,
                                                    #  'a_ns'         : 0.03319342705152602,
                                                    #  'a_bc'         : 0.03319342705152602,
                                                     'a_div'        : 0.015,
                                                     'a_ns'         : 0.015,
                                                     'a_bc'         : 0.015,
                                                     'scale_base'   : 0.3e6,
                                                     'scale_div'    : 1e-5,
                                                     'scale_ns'     : 0.01,
                                                     'scale_bc'     : 1e-8, # 1, 
                                                     'grad_dim'     : [2 , 1], # gradient will be given in (x,y)
                                                     'spacing'      : [1.2/(dim[1]-1) , 0.8/(dim[0]-1)],
                                                     'rho'          : 1,
                                                     'mu'           : 0.01,
                                                     'norm_info'    : '../data/processed/%s/norm_info.csv'%(casename),
                                                     'dim'          : dim
                                                     })
    
    validloss_class = copy.deepcopy(trainloss_class)
    
    # # for debugging
    # casename        = 'test'
    # modelname       = 'debug'
    # epochs          = 1
    # batch_size      = 5
    
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
                             wandb_flag      = wandb_flag)
    
    