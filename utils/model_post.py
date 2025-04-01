import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
sys.path.append('../losses')
from physics_loss import PhysicsLoss

sys.path.append('../data')
from preprocessing import denormalisation

def model_loader(model_paths, device, rule_length = 90):
    
    model_list = []
    
    for model_path in model_paths:
        
        if torch.cuda.is_available():
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            
        model.eval()
        model_list.append(model)
    
    # model dimension check
    for i in range(len(model_paths)-1):
        assert (model_list[-1].dim == model_list[i].dim), "Dimension mismatch!"
        assert (model_list[-1].in_channels == model_list[i].in_channels), "in_channels mismatch!"
        assert (model_list[-1].out_channels == model_list[i].out_channels), "out_channels mismatch!"

    print('Models Loaded:')
    print('='*rule_length)
    print ('%20s %15s      %s'%('Model Name', 'N_Params', 'Model Path'))
    print('-'*rule_length)
    for i in range(len(model_paths)):
        print ('%20s %15i      %s'%(model_list[i].model_name, 
                                    model_list[i].count_params(), 
                                    model_list[i].model_path))
    print('='*rule_length)
    
    return model_list

def model_predictions(model_list, a_tensor, u_tensor, device, 
                      error_mode = 'relative', 
                      origin_recover = False, norm_info = None):
    
    model_out = []
    error_out = []
    
    n_in = u_tensor.shape[0]
    for model in model_list:
        model.device = device
        x = a_tensor.to(device).view(n_in, model.in_channels , 
                                     u_tensor.shape[-2], u_tensor.shape[-1])
        y = u_tensor.to(device).view(n_in, model.out_channels, 
                                     u_tensor.shape[-2], u_tensor.shape[-1])

        with torch.no_grad():
            out = model(x)
        if type(out) == tuple: out = out[0]
        if out.shape[-2::] != u_tensor.shape[-2::]:
            out = F.interpolate(out, size=u_tensor.shape[-2::], mode='bilinear', align_corners=True)
        
        pred = out.reshape(n_in, model.out_channels, 
                                     u_tensor.shape[-2], u_tensor.shape[-1]) 
            
        if error_mode == 'PI':
            div_residual, NS_x_res, NS_y_res = physics_error_evaluation(model.casename, 
                                                                        a_tensor, pred, eval_mode='tensor')

            loss_dim = div_residual.shape
            error = torch.cat((div_residual.view(loss_dim[0],1,loss_dim[1],loss_dim[2]),
                                NS_x_res.view(loss_dim[0],1,loss_dim[1],loss_dim[2]),
                                NS_y_res.view(loss_dim[0],1,loss_dim[1],loss_dim[2]),),1)
            
        else: 
            error = prediction_error(a_tensor, pred, y, error_mode=error_mode)
        
        if origin_recover:
            a_origin, pred = denormalisation(a_tensor, pred, norm_info)
            
        model_out.append(pred)  
        error_out.append(error)
    
    model_out = torch.stack(model_out)
    error_out = torch.stack(error_out)
    
    # oputput dimension 
    # (number of models, number of data, out_channels, dim1, dim2)
    
    return model_out, error_out

def prediction_error(input_tensor, predicted, ground_truth, error_mode='relative'):
    
    if error_mode == 'abs_relative':
        error = torch.abs((predicted - ground_truth) / ground_truth)
        
    elif error_mode == 'relative':
        error = (predicted - ground_truth) / ground_truth
        
    elif error_mode == 'abs':
        error = torch.abs((predicted - ground_truth))
        
    elif error_mode == 'err':
        error = (predicted - ground_truth) 

    else:
        raise TypeError("Unrecognised error mode %s"%(error_mode))

    return error 

def prediction_evaluation(predicted, ground_truth, eval_mode='RMSRE'):
    
    # absolute errors
    if eval_mode == 'max_abs':
        max_error = torch.max(torch.abs(predicted - ground_truth))
        return max_error
    
    elif eval_mode == 'MAE':
        mae = torch.mean(torch.abs(predicted - ground_truth))
        return mae
    
    elif eval_mode == 'MSE':
        mse = torch.mean((predicted - ground_truth) ** 2)
        return mse
    
    elif eval_mode == 'RMSE':
        rmse = torch.sqrt(torch.mean((predicted - ground_truth) ** 2))
        return rmse
    
    # relative errors
    elif eval_mode == 'max_rel':
        max_rel = torch.max(torch.abs((predicted - ground_truth) / ground_truth))
        return max_rel
    
    elif eval_mode == 'MARE':
        mare = torch.mean(torch.abs((predicted - ground_truth) / ground_truth))
        return mare
    
    elif eval_mode == 'MSRE':
        msre = torch.mean(((predicted - ground_truth) / ground_truth) ** 2)
        return msre
    
    elif eval_mode == 'RMSRE':
        rmsre = torch.sqrt(torch.mean(((predicted - ground_truth) / ground_truth) ** 2))
        return rmsre
    
    # L2 norm errors
    elif eval_mode == 'L2':
        error_norm = torch.norm(predicted - ground_truth, p=2)  # Compute L2 norm of the error
        return error_norm.item() / (predicted.flatten().numel())
    
    elif eval_mode == 'L2_rel':
        error_norm = torch.norm(predicted - ground_truth, p=2, keepdim=True)  # Compute L2 norm of the error
        true_norm  = torch.norm(ground_truth, p=2, keepdim=True)              # Compute L2 norm of the ground truth
        l2_rel_norm = error_norm / true_norm
        return torch.mean(l2_rel_norm)
    
    elif eval_mode == 'L2_rel_std':
        # print(predicted.shape)
        # print(ground_truth.shape)
        # print((predicted - ground_truth).shape)
        error_norm = torch.norm(predicted - ground_truth, p=2, keepdim=True)  # Compute L2 norm of the error
        # print(error_norm.shape)
        true_norm  = torch.norm(ground_truth, p=2, keepdim=True)              # Compute L2 norm of the ground truth
        l2_rel_norm = error_norm / true_norm
        # print(l2_rel_norm.shape)
        
        return torch.std(l2_rel_norm)
    
    # error fitting
    elif eval_mode == 'R2':
        total_variance = torch.sum((ground_truth - torch.mean(ground_truth)) ** 2)
        explained_variance = torch.sum((predicted - ground_truth) ** 2)
        r2 = 1 - (explained_variance / total_variance)
        return r2
        
    else:
        raise TypeError("Unrecognised error mode %s"%(eval_mode))

    return

def physics_error_evaluation(casename, inputs, outputs, eval_mode='all'):
    
    dim = (inputs.shape[-2], inputs.shape[-1])
    
    loss_class = PhysicsLoss(loss_param_dict = {'loss' : nn.MSELoss(reduction = 'sum'),
                                                'a_div'     : 0.3,
                                                'a_ns'      : 0.3,
                                                'a_bc'      : 0.3,
                                                'scale_base': 0.3e6,
                                                'scale_div' : 1e-5,
                                                'scale_ns'  : 0.01,
                                                'scale_bc'  : 1,
                                                'grad_dim'  : [2 , 1], # gradient will be given in (x,y)
                                                'spacing'   : [1.2/(dim[1]-1) , 0.8/(dim[0]-1)],
                                                'rho'       : 1,
                                                'mu'        : 0.01,
                                                'norm_info' : './data/processed/%s/norm_info.csv'%(casename),
                                                'dim'       : dim
                                                })
    
    # retrieve values 
    _ = loss_class.data_processing(inputs, outputs)
    
    div_loss = loss_class.continuity_residual_2D()
    ns_loss  = loss_class.NS_residual_2D()
    bc_loss  = loss_class.BC_residual()
        
    if eval_mode == 'all':
        return div_loss, ns_loss
    elif eval_mode == 'all_3':
        return div_loss, ns_loss, bc_loss
    elif eval_mode == 'continuity':
        return div_loss
    elif eval_mode == 'NS':
        return ns_loss
    elif eval_mode == 'BC':
        return bc_loss
    elif eval_mode == 'tensor':
        div_residual = loss_class.continuity_residual_2D_tensor()
        NS_x_res, NS_y_res = loss_class.NS_residual_2D_tensor()
        
        div_residual = F.pad(div_residual, (1, 1, 1, 1), mode='constant', value=0)
        NS_x_res     = F.pad(NS_x_res    , (1, 1, 1, 1), mode='constant', value=0)
        NS_y_res     = F.pad(NS_y_res    , (1, 1, 1, 1), mode='constant', value=0)
        
        return div_residual, NS_x_res, NS_y_res
        
    else:
        raise TypeError("Unrecognised error mode %s"%(eval_mode))

    return

def print_error_statistics( casename, model_list, model_out, a_tensor, u_tensor, 
                            out_channel_names = ['P', 'U_x', 'U_y'],
                            eval_mode_list = ['MARE', 'RMSRE', 'R2'], 
                            All_Loss = True, PI_Loss = True,
                            if_print = True, if_save = False, overwrite = False,
                            file_path = 'Model_Performance_Metric/test/',
                            file_name = 'metric_summary',
                            ):
    
    # ======================================================
    
    n_eval  = len(eval_mode_list)
    n_model = len(model_list)
    n_out   = model_list[0].out_channels


    # setting 1st and 2nd columns
    n_additional = 0 # additional columns if needed
    
    channel_list = [' ']*n_additional
    metric_list  = [' ']*n_additional
    for j in range(n_out):
        for k in range(n_eval):
            
            if k == 0:
                channel_list.append(out_channel_names[j])
            else:
                channel_list.append(' ')
            
            metric_list.append(eval_mode_list[k])
     
    # all losses
    if All_Loss:
        channel_list += ['Combined'] + [' ']*(len(eval_mode_list)-1)
        metric_list  += eval_mode_list      
         
    # adding for physics laws
    if PI_Loss:
        # channel_list += ['PI_Loss', ' ']
        # metric_list  += ['Continuity', 'NS']
        channel_list += ['PI_Loss', ' ', ' ']
        metric_list  += ['Continuity', 'NS', 'BC']
        
    data = {'Channel Name'      : channel_list,
            'Metric'            : metric_list,
            }

    # populating performance metrics
    df_arr  = np.zeros((n_model, len(metric_list) + n_additional)) # +2 for physics laws
    for i in range(n_model):
        # df_arr[i, 0] = model_list[i].count_params()
        for j in range(n_out):
            # print('j', j)
            for k in range(n_eval):
                # print('k', k)
                row_number = j*n_eval + k + n_additional
                # print('row_number',row_number)
                # print('channel %i row %i, row no %i'%(j,k,row_number))
                df_arr[i, row_number] = prediction_evaluation(model_out[i, :, j, :, :], 
                                                                u_tensor[:, j, :, :], 
                                                                eval_mode=eval_mode_list[k])
    
        if All_Loss:
            # combined losses
            for k in range(n_eval):
                row_number = n_out*n_eval + k + n_additional
                df_arr[i, row_number] = prediction_evaluation(model_out[i, :, :, :, :], 
                                                                u_tensor[:, :, :, :], 
                                                                eval_mode=eval_mode_list[k])
                
            (df_arr[i, -2], df_arr[i, -1]) = physics_error_evaluation(casename, 
                                                                      a_tensor[:, :, :, :],
                                                                      model_out[i, :, :, :, :], 
                                                                      eval_mode='all')
                    
        if PI_Loss:
            # physics laws
            # (df_arr[i, -2], df_arr[i, -1]) = physics_error_evaluation(casename, 
            #                                                           a_tensor[:, :, :, :],
            #                                                           model_out[i, :, :, :, :], 
            #                                                           eval_mode='all')
            
            (df_arr[i, -3], df_arr[i, -2], df_arr[i, -1]) = physics_error_evaluation(casename, 
                                                                                     a_tensor[:, :, :, :],
                                                                                     model_out[i, :, :, :, :], 
                                                                                     eval_mode='all_3')
            
        
    
    for i in range(n_model):
        
        # filename = model_list[i].model_path
        # parts = filename.replace(".pt", "").split("model_final")
        # col_title_now = model_list[i].model_name + parts[-1]
        
        col_title = model_list[i].model_name
        col_title_now = col_title
        
        tags = 1
        while col_title_now in data:
            col_title_now = col_title + '_%s'%(tags)
            tags += 1
        
        data[col_title_now] = df_arr[i,:]

    # ======================================================        
    # converting to datafram, printing and plotting
    
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    df = pd.DataFrame(data)

    if if_print: 
        pd.options.display.float_format = "{:,.7f}".format
        print(df)

    if if_save:
        
        os.makedirs(file_path, exist_ok=True)
        file_name_now = file_path + file_name + '.csv'
        if not overwrite:
            i = 1
            while os.path.exists(file_name_now):
                file_name_now = file_path + file_name + '%i.csv'%(i)
                i += 1
        
        df.to_csv(file_name_now, index=False)  
    
    return
    
def print_csv(files):
    
    pd.options.display.float_format = "{:,.7f}".format
    
    for file in files:
        
        print('====================================================================')
        print(file)
        df = pd.read_csv(file)
        print(df)
        print()
    
    return None

