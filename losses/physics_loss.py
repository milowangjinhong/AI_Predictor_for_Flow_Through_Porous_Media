import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

sys.path.append('../data')
from preprocessing import denormalisation

# continuity and ns loss, 
class PhysicsLoss(nn.Module):
    
    def __init__(self, loss_param_dict = {'loss'      : nn.MSELoss(reduction = 'sum'),
                                          'a_div'     : 0.3,
                                          'a_ns'      : 0.3,
                                          'a_bc'      : 0.3,
                                          'scale_base': 0.3e6,
                                          'scale_div' : 1e-5,
                                          'scale_ns'  : 0.01,
                                          'scale_bc'  : 1,
                                          'grad_dim'  : [2 , 1], # gradient will be given in (x,y)
                                          'spacing'   : [1.2/127 , 0.8/127], # ratio of x and y grids
                                          'rho'       : 1,
                                          'mu'        : 1,
                                          'norm_info' : './data/processed/data2_20241116_normalised/norm_info.csv',
                                          'dim'       : (128, 128),
                                        #   'wandb_flag': False
                                          }):
        
        super(PhysicsLoss, self).__init__() 
        
        # base loss type
        self.loss   = loss_param_dict['loss']  
        
        # Relative Importance for the losses
        self.a_div  = loss_param_dict['a_div']  
        self.a_ns   = loss_param_dict['a_ns']  
        self.a_bc   = loss_param_dict['a_bc']
        self.a_loss = 1.0 - self.a_div - self.a_ns - self.a_bc
        
        # Scaling for the losses
        self.scale_base = loss_param_dict['scale_base']  
        self.scale_div  = loss_param_dict['scale_div']  
        self.scale_ns   = loss_param_dict['scale_ns']  
        self.scale_bc   = loss_param_dict['scale_bc']  
        
        assert (self.a_div >= 0 and self.a_ns >= 0 and self.a_loss >= 0), 'Incorrect loss weights.'
        
        # (dP/dy , dP/dx)
        self.grad_dim = loss_param_dict['grad_dim']       # 1 is for y, 2 is for x
        self.spacing  = loss_param_dict['spacing']        # first for y (shorter), second for x (longer)
        
        self.rho = loss_param_dict['rho']
        self.mu  = loss_param_dict['mu']
        
        self.dim  = loss_param_dict['dim']
        
        # data normalisation info retrieval
        try:
            df = pd.read_csv(loss_param_dict['norm_info'])
            self.norm_info = np.array(df)[:,2:].astype(float)
        except:
            print('No normalisation data found, using 0 and 1:')
            self.norm_info = np.array([[0., 0., 0., 0.],
                                       [1., 1., 1., 1.]])
        
        # print(self.norm_info)
        '''
        norm_info is 2D array: 
        - [0,:] being averages amd [1,:] being std
        - [:,0:in_channels] are for inputs and [:,in_channels:] are for outputs
        '''
        
        # self.wandb_flag = loss_param_dict['wandb_flag']

    def forward(self, inputs, outputs, targets):
        
        # outputs shape is [batch, channel, y, x]
        
        # ========================================================
        # numerical loss 
        loss_base = self.loss(outputs, targets) / self.scale_base
        
        # ========================================================
        # physics-informed losses
        
        # retrieving values for physics laws
        self.data_processing(inputs, outputs)
        
        # calculate residuals
        loss_continuity = self.continuity_residual_2D() / self.scale_div
        loss_ns         = self.NS_residual_2D() / self.scale_ns
        
        loss_bc         = self.BC_residual() / self.scale_bc
        
        loss_total      = self.a_loss * loss_base + \
                          self.a_div * loss_continuity + \
                          self.a_ns * loss_ns + \
                          self.a_bc * loss_bc
                          
        # if self.wandb_flag:
        #     wandb.log({"loss_base": loss_base, 
        #                "loss_continuity": loss_continuity,
        #                "loss_ns": loss_ns}, commit=False)
        
        # print('base', loss_base, '\n',
        #       'cont', loss_continuity, '\n',
        #       '  ns', loss_ns)
        
        return loss_total
    
    def data_processing(self, inputs, outputs):
        
        # denormalisation for the physics laws 
        inputs_loc, outputs_loc = denormalisation(inputs, outputs, self.norm_info)
        
        # keeping only the interior points
        self.inputs_loc  = inputs_loc [:, :, 1:self.dim[0]-1, 1:self.dim[1]-1]
        self.outputs_loc = outputs_loc[:, :, 1:self.dim[0]-1, 1:self.dim[1]-1]
        
        # boundary values
        in_top = self.dim[0]//2 + self.dim[0]//16 + 1
        in_bot = self.dim[0]//2 - self.dim[0]//16 - 1
        
        self.boundary = torch.cat((outputs_loc[:,1,in_bot:in_top,0].flatten()-0.01, # velocity inlet
                                #    outputs_loc[:,2,in_bot:in_top,0].flatten(), # v inlet
                                   outputs_loc[:,0,in_bot:in_top,-1].flatten() # pressure outlet
                                   )) 
        for ch in range(1,3):
            for index in [0, -1]:
                self.boundary = torch.cat((self.boundary, # 
                                           outputs_loc[:,ch,index,:].flatten(),  # 
                                           outputs_loc[:,ch,in_top::,index].flatten(),
                                           outputs_loc[:,ch,0:in_bot,index].flatten(),
                                           ))
                                           
        # retrieving gradients
        self.grad_calc()
        
        return None
    
    def grad_calc(self):
        
        # extracting gradient information
        (self.dp_dx, self.dp_dy) = torch.gradient(self.outputs_loc[:,0,:,:], dim = self.grad_dim, spacing = self.spacing)
        (self.du_dx, self.du_dy) = torch.gradient(self.outputs_loc[:,1,:,:], dim = self.grad_dim, spacing = self.spacing)
        (self.dv_dx, self.dv_dy) = torch.gradient(self.outputs_loc[:,2,:,:], dim = self.grad_dim, spacing = self.spacing)

        # second order derivatives
        # print(self.grad_dim[0],self.spacing[0])
        # print(self.grad_dim[1],self.spacing[1])
        (self.du_dxx,)= torch.gradient(self.du_dx, dim = self.grad_dim[0], spacing = self.spacing[0])
        (self.du_dyy,)= torch.gradient(self.du_dy, dim = self.grad_dim[1], spacing = self.spacing[1])
        (self.dv_dxx,)= torch.gradient(self.dv_dx, dim = self.grad_dim[0], spacing = self.spacing[0])
        (self.dv_dyy,)= torch.gradient(self.dv_dy, dim = self.grad_dim[1], spacing = self.spacing[1])
        
        return None
    
    def continuity_residual_2D(self):
        
        div_residual = self.continuity_residual_2D_tensor()
        
        mse_div_residual = torch.mean((div_residual).pow(2))
        
        return mse_div_residual 
    
    def NS_residual_2D(self):
        
        x_res, y_res = self.NS_residual_2D_tensor()
        
        mse_ns_residual = torch.mean(x_res.pow(2), dim=(1, 2)) + torch.mean(y_res.pow(2), dim=(1, 2))
        # print(mse_ns_residual.shape)
        mse_ns_residual = mse_ns_residual.mean() / 2
        
        # mse_ns_residual = (torch.mean(x_res**2) + torch.mean(y_res**2))/2
        
        return mse_ns_residual 

    def BC_residual(self):
        
        bc_residual = torch.mean((self.boundary).pow(2))
        
        return bc_residual
    
    def continuity_residual_2D_tensor(self):
        
        div_residual = self.du_dx + self.dv_dy
        
        # for debugging only
        # val_abs = torch.abs(div_residual)
        # val_max = torch.max(val_abs)
        # val_avg = torch.mean(val_abs)
        # print(val_max, val_avg)

        # keeping only interior points
        # div_residual = div_residual[:,1:127,1:127]
        
        return div_residual
    
    def NS_residual_2D_tensor(self):
        
        alpha = self.inputs_loc[:,0,:,:] 
        u     = self.outputs_loc[:,1,:,:]
        v     = self.outputs_loc[:,2,:,:]
        
        # in x direction
        x_res = u * self.du_dx + v * self.du_dy \
                + self.dp_dx / self.rho\
                - self.mu * (self.du_dxx + self.du_dyy) / self.rho \
                + alpha * u / self.rho
        
        # in y direction
        y_res = u * self.dv_dx + v * self.dv_dy \
                + self.dp_dy / self.rho \
                - self.mu * (self.dv_dxx + self.dv_dyy) / self.rho \
                + alpha * v / self.rho
                
        # keeping only interior points
        # x_res = x_res[:,1:127,1:127]
        # y_res = y_res[:,1:127,1:127]
        
        # for debugging
        # print(x_res)
        # print(y_res)
        
        return x_res, y_res