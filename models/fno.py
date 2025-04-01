import torch
import operator
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

# https://github.com/camlab-ethz/AI_Science_Engineering/blob/main/Tutorial%2005%20-%20Operator%20Learing%20-%20Fourier%20Neural%20Operator.ipynb

class fno2d(nn.Module):
    
    def __init__(self, in_channels = 1, out_channels = 3, device = 'cpu', 
                 param_dict = {'modes'          : 16, 
                               'width'          : 64, 
                               'n_layers'       : 4, 
                               'padding_frac'   : 0.25}, 
                 ):
        
        super(fno2d, self).__init__()

        self.model_name   = 'fno2d'
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.param_dict   = param_dict
        
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, c=1, x=s, y=s)
        output: the solution
        output shape: (batchsize, c=3, x=s, y=s)
        """
        
        # parameter extraction
        self.modes1      = param_dict["modes"]
        self.modes2      = param_dict["modes"]
        self.width       = param_dict["width"] # number of channels inside the FNO layers
        self.n_layers    = param_dict["n_layers"]

        # self.padding = 9 # pad the domain if input is non-periodic probably needed?
        self.padding_frac = param_dict["padding_frac"]
        
        # input/lifting layer
        self.lift = nn.Conv2d(in_channels, self.width, kernel_size=1)

        # FNO block layers
        self.conv_list      = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list  = nn.ModuleList([SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])

        # output/decoding layers
        self.decode = nn.Sequential(#nn.Conv2d(self.width, self.out_channels, kernel_size=1),
                                    nn.Conv2d(self.width, 128, kernel_size=1),
                                    nn.GELU(),
                                    nn.Conv2d(128, self.out_channels, kernel_size=1)
                                    )

    def forward(self, x):
        
        # input/lift layer
        x = self.lift(x)

        x1_padding = int(np.round(x.shape[-1] * self.padding_frac))
        x2_padding = int(np.round(x.shape[-2] * self.padding_frac))
        x = F.pad(x, [0, x1_padding, 0, x2_padding])

        # FNO block layers
        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):
            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = F.gelu(x)
        
        x = x[..., :-x1_padding, :-x2_padding]
        
        # output/decode layer
        x = self.decode(x)
        
        return x
    
    def count_params(self): 
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c

# 2D fourier layer: FFT-linear-FFT
class SpectralConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, modes1, modes2):
        
        super(SpectralConv2d, self).__init__()

        # both equal to the width of the fourier layer
        self.in_channels  = in_channels
        self.out_channels = out_channels
        
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1  
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        
        # Complex-valued weights of shape 
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, a, b):
        # (batch, in_channels, x,y ), (in_channels, out_channels, x,y) -> (batch, out_channels, x,y)
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        
        batchsize = x.shape[0]
        
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x) # , norm="ortho"

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1 , :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1 , :self.modes2], self.weights1) # weighted low frequency modes
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2) # weighted high frequency modes

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) # , norm="ortho"
        
        return x

    