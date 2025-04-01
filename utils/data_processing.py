import torch

def get_data_stats(data_tensor):
    
    dim     = data_tensor.shape

    val_min = torch.min (data_tensor)
    val_max = torch.max (data_tensor)
    
    val_avg = torch.mean(data_tensor)
    val_std = torch.std (data_tensor)
    
    return dim, val_min, val_max, val_avg, val_std

def tensor_normalisation(data_tensor):
    
    # means = data_tensor.mean(dim=1, keepdim=True)
    # stds  = data_tensor.std (dim=1, keepdim=True)
    
    # means = torch.mean(data_tensor, dim=1, keepdim=True)
    # stds  = torch.std (data_tensor, dim=1, keepdim=True)
    
    # normalised_data = (data_tensor - means) / stds
    
    n_channel = data_tensor.shape[1]
    for i in range(n_channel):
        means = torch.mean(data_tensor[:,i,:,:])
        stds  = torch.std (data_tensor[:,i,:,:])
        
        print(means, stds)
        
        data_tensor[:,i,:,:] = (data_tensor[:,i,:,:] - means) / stds
    
    print('=======')
    return data_tensor