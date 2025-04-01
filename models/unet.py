import torch
import operator
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

class unet(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=3, device = 'cpu', 
                 param_dict = {'depth' : 4,
                               'init_filters' : 64,
                               'conv_param_dict' : {'batch_norm' : True}}
                 ):
        
        super(unet, self).__init__()
        
        self.model_name  = 'unet'
        self.in_channels  = in_channels
        self.out_channels = out_channels
        
        # parameter extraction
        depth           = param_dict['depth']
        filters         = param_dict['init_filters']
        conv_param_dict = param_dict['conv_param_dict']
        
        # Pooling layer setup
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2) # shirnking dim by 2  
        
        # Encoder: Downsampling path
        self.encoder = nn.ModuleList()
        for _ in range(depth):
            self.encoder.append(conv_block(in_channels, filters, conv_param_dict))
            
            # setting for next depth - doubling channels
            in_channels = filters
            filters *= 2

        # Bottleneck layer
        self.bottleneck = conv_block(in_channels, filters, conv_param_dict)

        # Decoder: Upsampling path
        self.upconv  = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for _ in range(depth):
            in_channels = filters
            filters //= 2
            self.upconv.append(nn.ConvTranspose2d(in_channels, filters, kernel_size=2, stride=2))
            self.decoder.append(conv_block(in_channels, filters, conv_param_dict))
            

        # Final 1x1 convolution to map to output channels
        self.final_conv = nn.Conv2d(filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc_outputs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outputs.append(x)
            x = self.pool(x) # size dim//2, dim//2

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        for i in range(len(self.decoder)):
            x = self.upconv[i](x)
            x = torch.cat([x, enc_outputs[-i-1]], dim=1)  # Concatenate with encoder output
            x = self.decoder[i](x)

        return self.final_conv(x)
    
    def count_params(self): 
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

class conv_block(nn.Module):
    '''
    input size : (batch_size, in_channels , height, width)
    output size: (batch_size, out_channels, height, width)
    '''
    def __init__(self, in_channels, out_channels, 
                 param_dict = {'batch_norm' : True}
                 ):
        
        super(conv_block, self).__init__()
        
        # constructing basic 
        layers = [nn.Conv2d(in_channels , out_channels, kernel_size=3, padding=1), # batch norm here
                  nn.ReLU(inplace=True),
                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # batch norm here
                  nn.ReLU(inplace=True)]
        
        # adding batch norm
        if param_dict['batch_norm']:
            layers.insert( 1, nn.BatchNorm2d(out_channels))
            layers.insert(-1, nn.BatchNorm2d(out_channels))
        
        # convert into nn.sequential
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

