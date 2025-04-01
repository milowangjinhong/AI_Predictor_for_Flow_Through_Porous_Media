import os
import sys
import torch
import wandb
import tqdm as tqdm
import numpy as np
import torch.nn as nn

from timeit import default_timer
from datetime import datetime

sys.path.append('../')
sys.path.append('../data')
sys.path.append('../losses')
sys.path.append('../models')
from preprocessing import data_loader

def model_train(model_class, param_dict,
                casename     = 'test',
                modelname    = '',
                in_channels  = 1, 
                out_channels = 3, 
                dim          = (128, 128),
                restart_model   = '',
                load_checkpoint = False,
                check_step      = 10,
                epochs          = 200,
                batch_size      = 200,
                drop_last       = False,
                learning_rate   = 0.0005,
                scheduler_step  = 10,
                scheduler_gamma = 0.5,
                optimizer_class = torch.optim.Adam,
                scheduler_class = torch.optim.lr_scheduler.StepLR,
                trainloss_class = nn.MSELoss(reduction='sum'),
                validloss_class = nn.MSELoss(reduction='sum'),
                save_model      = True,
                wandb_flag      = False,
                wandb_project   = '',
                wandb_sweep     = False
                ):
    
    time_tag = datetime.now().strftime("%d-%m-%Y_%H.%M.%S")
    
    bar = '========================================================================================================================================================'
    
    # ========================================================================================================================================================
    # Case Setup
    processed_data_path = '../data/processed/%s/'%(casename)
    print(bar)
    print('Training Setup:', '\n',
          'epochs           :', epochs, '\n',
          'batch_size       :', batch_size, '\n',
          'learning_rate    :', learning_rate, '\n',
          'scheduler_step   :', scheduler_step, '\n',
          'scheduler_gamma  :', scheduler_gamma)
    print(bar)
    
    # ========================================================================================================================================================
    # Data loading
    
    train_a, train_u, \
        valid_a, valid_u, \
            test_a, test_u = data_loader(processed_save_path = processed_data_path, 
                                         in_channels = 1, out_channels = 3, dim = dim)

    # data loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), 
                                               batch_size=batch_size, shuffle=True , drop_last=drop_last)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_a, valid_u), 
                                               batch_size=batch_size, shuffle=True , drop_last=drop_last)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a , test_u ), 
                                               batch_size=batch_size, shuffle=False, drop_last=drop_last)
    print('Data loaded from '+processed_data_path)
    print(bar)
    
    # ========================================================================================================================================================
    # Initialize the model, loss function, and optimizer
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    
    model  = model_class(in_channels=in_channels, out_channels=out_channels, 
                         device=device, param_dict=param_dict).to(device)
    
    # save path setup
    path_model = '../saved_models/' + casename + '/' + model.model_name + '/' # + '_ep_' + str(epochs)
    # if not os.path.exists(path_model): os.makedirs(path_model, exist_ok=True)
    os.makedirs(path_model, exist_ok=True)
    
    if modelname == '': modelname = time_tag
    model_file_name = 'model_final_%s.pt'%(modelname)
    model_path = path_model + model_file_name
    
    i = 0
    while os.path.exists(model_path):
        i += 1
        model_file_name = 'model_final_%s%i.pt'%(modelname, i)
        model_path = path_model + model_file_name
        
    onnx_path  = path_model + model_file_name[:-2]+'onnx'
    check_path = path_model + "checkpoint.pth"

    print("Model parameters:", model.count_params(), 'Device:', device)
    print(bar)
    print('Dataset shape:'),
    print('train:', train_a.shape, '\n'
          'valid:', valid_a.shape, '\n'
          'test :', test_a.shape, '\n'
          'train_loader:', len(iter(train_loader)), 'train sample shape', next(iter(train_loader))[0].shape, '\n'
          'valid_loader:', len(iter(valid_loader)), 'valid sample shape', next(iter(valid_loader))[0].shape, '\n'
          'test_loader :', len(iter(test_loader)) , 'test  sample shape', next(iter(test_loader ))[0].shape
        )
    print(bar)

    # optimiser and loss setup
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    # trainloss = nn.MSELoss(reduction='sum').to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = scheduler_class(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    trainloss = trainloss_class.to(device)
    validloss = validloss_class.to(device)

    # ========================================================================================================================================================
    # Logging training in wandb
    
    if wandb_flag: 
        wandb.login(force=True) # --relogin
        wandb_project   = casename
        wandb_name      = model.model_name + '_' + modelname + '_' + time_tag
        wandb.init(project  = wandb_project,
                   name     = wandb_name,
                   config   = dict(model_class     = model_class, 
                                   param_dict      = param_dict,
                                   epochs          = epochs,
                                   batch_size      = batch_size,
                                   learning_rate   = learning_rate,
                                   scheduler_step  = scheduler_step,
                                   scheduler_gamma = scheduler_gamma,
                                   optimizer_class = optimizer_class,
                                   scheduler_class = scheduler_class,
                                   trainloss_class = trainloss_class))

    # ========================================================================================================================================================
    # if restarting from model / checkpoint
    
    if restart_model != '':
        if os.path.exists(restart_model):
            model = model.load_model(restart_model)
            print("Model state loaded from %s"%(state_path))
    
    if load_checkpoint:
        if os.path.exists(check_path):
            # Load checkpoint
            checkpoint = torch.load(check_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            epoch = checkpoint["epoch"]
            loss_train = checkpoint["loss_train"]
            loss_valid = checkpoint["loss_valid"]
            print(f"Resuming training from epoch {epoch}, loss {loss_train}")
        else:
            print('No checkpoints available.')
    
    # ========================================================================================================================================================
    # Training loop
    
    train_loss_list = []
    valid_loss_list = []
    for ep in range(1, epochs + 1):
        
        model.train()
        
        t1 = default_timer()
        
        # ======================================
        # training Loop
        train_loss_batch = []
        batch_counter = 1
        
        progress_bar = tqdm.tqdm(train_loader, desc="Epoch %4i"%(ep))
        for x_train, y_train in progress_bar:
            
            loader_size = x_train.shape[0]
            
            x_train = x_train.to(device).view(loader_size, in_channels , dim[0], dim[1])
            y_train = y_train.to(device).view(loader_size, out_channels, dim[0], dim[1])

            out_train = model(x_train)
            # out_train = out_train.reshape(loader_size, out_channels, dim[0], dim[1])

            # loss_train = trainloss(out_train, y_train)
            loss_train = compute_loss(trainloss, out_train, y_train, x_train)
            train_loss_batch.append(loss_train.item())
            # print('loss_train:', loss_train.item())
            progress_bar.set_postfix({"Batch ": '  %i/%i'%(batch_counter, len(iter(train_loader))), 
                                      "Loss_train ": '%7.4f'%(loss_train.item())})

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            batch_counter += 1

        # ======================================
        # Validation loop
        valid_loss_batch = []
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                
                loader_size = x_valid.shape[0]
                
                x_valid = x_valid.to(device).view(loader_size, in_channels , dim[0], dim[1])
                y_valid = y_valid.to(device).view(loader_size, out_channels, dim[0], dim[1])

                out_valid = model(x_valid) # .reshape(loader_size, out_channels, dim[0], dim[1]) # reshape or view?
                # test_l2.append(validloss(out_valid, y_valid).item()) # @@@ rethink about customised losses
                
                loss_valid = compute_loss(validloss, out_valid, y_valid, x_valid)
                valid_loss_batch.append(loss_valid.item()) 
                # print('loss_valid:', loss_valid.item())
        
        # ======================================
        # epoch post-processing
        
        train_loss_list.append(np.mean(train_loss_batch))
        valid_loss_list.append(np.mean(valid_loss_batch))   
        
        t2 = default_timer()
        scheduler.step()
        
        print("Epoch %i completed in %.2f seconds. Train loss: %.3f, Validation loss: %.3f"%(ep, t2-t1, train_loss_list[-1], valid_loss_list[-1]))
        
        if wandb_flag or wandb_sweep:
            wandb.log({"epoch": ep, 
                       "train_loss": train_loss_list[-1],
                       "valid_loss": valid_loss_list[-1]})
        
        if ep%check_step == 0:
            
            # saving whole model
            # model.save_model(path_model,'ep_%i'%(ep))
            
            # saving checkpoint
            checkpoint = {
                          "epoch": ep,
                          "model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(),
                          "loss_train": loss_train,
                          "loss_valid": loss_valid
                         }
            torch.save(checkpoint, check_path)

    # ========================================================================================================================================================
    # End of Training
    
    # additional information for the model
    model.dim        = dim
    model.casename   = casename
    model.model_path = model_path
    model.onnx_path  = onnx_path
    
    if save_model:
        # save whole model
        torch.save(model, model_path)
        print('Model saved in:', model_path)
        
        if model.model_name != 'fno2d':
            torch.onnx.export(model, x_train, onnx_path)
            # torch.onnx.dynamo_export(model, train_a, onnx_path)
            print('Model onnx saved in:', onnx_path)
    
    if wandb_flag:
        
        if save_model and model.model_name != 'fno2d':
            
            # move to mode saved directory
            mycwd = os.getcwd()
            os.chdir(path_model)
            
            # wandb.save('model_final.onnx')
            wandb.save(model_file_name[:-2]+'onnx')
            
            # move back to previous directory
            os.chdir(mycwd)
        
        wandb.finish()

    return path_model


def compute_loss(loss_fn, predicted, target, *args, **kwargs):
    """
    Computes the loss, accommodating both custom and standard loss functions.

    Args:
        loss_fn (callable): The loss function (can be standard or custom).
        predicted (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth values.
        *args: Additional positional arguments (e.g., input tensor).
        **kwargs: Additional keyword arguments for custom loss functions.

    Returns:
        torch.Tensor: The computed loss.
    """
    try:
        # Try passing additional arguments (if the loss_fn supports them)
        loss = loss_fn(*args, predicted, target, **kwargs)
    except TypeError:
        # Fallback to standard loss functions that only accept (predicted, target)
        
        # print('using traditional loss')
        loss = loss_fn(predicted, target)
        
    return loss