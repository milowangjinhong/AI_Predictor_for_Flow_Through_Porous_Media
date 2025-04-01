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
from models.ae import conv_ae
from losses.physics_loss import PhysicsLoss

if __name__ ==  '__main__':
    
    # Random Seeds
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    np.random.seed(0)
    
    # model setup
    model_class = conv_ae
    # param_dict  = {'n_layers'           : 4,
    #                'init_channel'       : 32,
    #                'latent_dim'         : 1024, 
    #                'activation_function': nn.ReLU() # nn.LeakyReLU()
    #                }
    
    # param_dict  = {'n_layers'           : 6,
    #                'init_channel'       : 32,
    #                'latent_dim'         : 4096, 
    #                'activation_function': nn.ReLU() # nn.LeakyReLU()
    #                }
    
    param_dict  = {'n_layers'           : 6,
                   'init_channel'       : 32,
                   'latent_n_layers'    : 3,  # used to be n-1 due to different class set up, equivalent
                   'latent_dim'         : 2048, 
                   'activation_function': nn.ReLU() # nn.LeakyReLU()
                  } # current paper
    
    # param_dict  = {'n_layers'           : 4,
    #                'init_channel'       : 64,
    #                'latent_n_layers'    : 3, 
    #                'latent_dim'         : 2048, 
    #                'activation_function': nn.ReLU() # nn.LeakyReLU()
    #               }
    
    # matching the initial channel number
    # param_dict  = {'n_layers'           : 4,
    #                'init_channel'       : 64,
    #                'latent_n_layers'    : 3, 
    #                'latent_dim'         : 2048, 
    #                'activation_function': nn.ReLU() # nn.LeakyReLU()
    #                } # weird
    
    casename     = 'data_check_128_normalised' # 'data_check_128'
    casename     = 'data2_20241116_normalised'
    casename     = 'data_20241125_alpha_normalised'
    casename     = 'data_20241125_alpha_reshuffle_3_normalised'
    
    modelname    = 'trial'
    modelname    = 'with_L2_loss_with_conv_64_proper'
    
    in_channels  = 1 # only gamma
    out_channels = 3 # 1 for p, 2 for u
    dim          = (128, 128)
    
    wandb_flag      = 1
    
    restart_model   = ''
    load_checkpoint = False
    check_step      = 10
    epochs          = 200
    batch_size      = 16
    learning_rate   = 0.001 # 0.0005
    scheduler_step  = 10
    scheduler_gamma = 0.5
    optimizer_class = torch.optim.Adam
    scheduler_class = torch.optim.lr_scheduler.StepLR
    trainloss_class = nn.MSELoss(reduction='sum')
    
    
    modelname       = 'PILoss_scaled_optimised_with_conv_multi-latent-layer_bc'
    trainloss_class = PhysicsLoss(loss_param_dict = {'loss'         : nn.MSELoss(reduction = 'sum'),
                                                     'a_div'        : 0.06200565548956508,
                                                     'a_ns'         : 0.06200565548956508,
                                                     'a_bc'         : 0.06200565548956508,
                                                     'scale_base'   : 0.3e6,
                                                     'scale_div'    : 1e-5,
                                                     'scale_ns'     : 0.01,
                                                     'scale_bc'     : 1,
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
    
    