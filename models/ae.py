import os
import torch
import operator
import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
from functools import reduce
# from vae_base import *

class conv_ae(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=3, device = 'cpu', 
                 param_dict = {'n_layers'           : 4,
                               'init_channel'       : 32,
                               'latent_n_layers'    : 3, 
                               'latent_dim'         : 1024, 
                               'activation_function': nn.SELU() }
                 ):
        
        super(conv_ae, self).__init__()
        
        self.model_name   = 'conv_ae'
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.param_dict   = param_dict
        
        # parameter extraction
        n_layers            = param_dict["n_layers"]
        n_channel           = param_dict["init_channel"]
        latent_dim          = param_dict["latent_dim"]
        latent_n_layers     = param_dict["latent_n_layers"]
        activation_function = param_dict["activation_function"]
        
        # encoder
        encoder_layers = []
        for _ in range(n_layers):
            encoder_layers.append(nn.Conv2d(in_channels, n_channel, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(activation_function)
            
            # setting for next depth - doubling channels
            in_channels = n_channel
            n_channel *= 2
        # convert into nn.sequential
        self.encoder = nn.Sequential(*encoder_layers)
        
        # latent space
        latent_layers = []
        for i in range(latent_n_layers):
            
            if i == 0:
                latent_in = in_channels
            else:
                latent_in = latent_dim
            
            if i == latent_n_layers-1:
                latent_out = in_channels
            else:
                latent_out = latent_dim
                
            latent_layers.append(nn.Conv2d(in_channels=latent_in, out_channels=latent_out, kernel_size=1))
            latent_layers.append(activation_function)

        # convert into nn.sequential
        self.latent_layers = nn.Sequential(*latent_layers)
        
        # decoder
        decoder_layers = []
        for _ in range(n_layers-1):
            n_channel //= 2
            decoder_layers.append(nn.ConvTranspose2d(in_channels, n_channel, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(activation_function)
            in_channels = n_channel
        decoder_layers += [nn.ConvTranspose2d(n_channel, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                           # activation_function,
                           ]
        
        # convert into nn.sequential
        self.decoder = nn.Sequential(*decoder_layers)
     
    def forward(self, x):
        
        # Encoder
        x = self.encoder(x)
        
        # latent space
        x = self.latent_layers(x)
        
        # Decoder
        x = self.decoder(x)
        
        return x
    
    def get_grid(self, S, batchsize, device):
        gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridx = gridx.reshape(1, 1, S, 1).repeat([batchsize, 1, 1, S])
        gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, S).repeat([batchsize, 1, S, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)
    
    def count_params(self): 
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
    
    def load_state(self, state_path):
        if os.path.exists(state_path):
            self.load_state_dict(torch.load(state_path))
            print(f"Model state loaded from {state_path}")
        else:
            print(f"No model found at {state_path}")
        return
    
    def save_model(self, path_model, label=''):
        if not os.path.exists(path_model):
            os.mkdir(path_model)
        path_model += '/model%s.pt'%(label)
        torch.save(self, path_model)
        print('Model saved at', path_model)
        return path_model
    