import os
import sys
import torch
import scipy.io
import numpy as np
import pandas as pd
import torch.nn.functional as F
from timeit import default_timer


def data_loader(processed_save_path = './processed/case1/', 
                in_channels = 1, out_channels = 3, dim = (128,128), if_print = False):
    
    path_train = processed_save_path + 'train.mat'
    path_valid = processed_save_path + 'valid.mat'
    path_test  = processed_save_path + 'test.mat'
    
    
    if not os.path.isfile(path_train) : print(path_train, 'not found!')
    if not os.path.isfile(path_valid) : print(path_valid, 'not found!')
    if not os.path.isfile(path_test)  : print(path_test , 'not found!')
        
    if os.path.isfile(path_train) and os.path.isfile(path_valid) and os.path.isfile(path_test):
        
        print('Loading Data from: '+ processed_save_path)
        # load train and test data from mat file
        # train_a = torch.tensor(scipy.io.loadmat(path_train)['a'], dtype=torch.float32)
        # train_u = torch.tensor(scipy.io.loadmat(path_train)['u'], dtype=torch.float32)
        # valid_a = torch.tensor(scipy.io.loadmat(path_valid)['a'], dtype=torch.float32)
        # valid_u = torch.tensor(scipy.io.loadmat(path_valid)['u'], dtype=torch.float32)
        # test_a  = torch.tensor(scipy.io.loadmat(path_test )['a'], dtype=torch.float32)
        # test_u  = torch.tensor(scipy.io.loadmat(path_test )['u'], dtype=torch.float32)
        
        train_a, train_u = read_from_mat(path_train)
        valid_a, valid_u = read_from_mat(path_valid)
        test_a , test_u  = read_from_mat(path_test)
        
        print('Data loaded from: '+ processed_save_path + '\n')
        
    else:
        
        raise TypeError("No data available.")
        
        # print('Generating training and testing dataset...')
        # train_a, train_u, \
        #     valid_a, valid_u, \
        #     test_a , test_u = data_preprocessing(processed_save_path = processed_save_path, 
        #                                          in_channels = in_channels, out_channels = out_channels, 
        #                                          dim = dim, save = True)
    
    # Data dimension check
    print('Checking Data Dimensions')
    
    path_list = [path_train, path_valid, path_test]
    a_list    = [train_a, valid_a, test_a]
    u_list    = [train_u, valid_u, test_u]
    
    for i in range(len(path_list)):
        
        mat_a = a_list[i]
        mat_u = u_list[i]
        
        message_out = "%s\nModel Dimension : (%i, %i, %i) \nData Dimension  :(%i, %i, %i)"%(path_list[i] ,
                                                                                            in_channels , dim[0], dim[1], 
                                                                                            mat_a.shape[1], mat_a.shape[2], mat_a.shape[3])
        assert ((mat_a.shape[1], mat_a.shape[2], mat_a.shape[3]) == (in_channels , dim[0], dim[1])) , "Input dimension mismatch: " + message_out
    
        message_out = "%s\nModel Dimension : (%i, %i, %i) \nData Dimension  :(%i, %i, %i)"%(path_list[i] ,
                                                                                            out_channels , dim[0], dim[1], 
                                                                                            mat_u.shape[1], mat_u.shape[2], mat_u.shape[3])
        assert ((mat_u.shape[1], mat_u.shape[2], mat_u.shape[3]) == (out_channels, dim[0], dim[1])) , "Output dimension mismatch!" + message_out    

    print('Data Dimension Matches.\n')
    
    # Summary dataset stats
    if if_print:
        print ('\nDataset Information Summary: ')
        data_info(train_a, 'train_a').print_info()
        data_info(train_u, 'train_u').print_info()
        data_info(valid_a, 'valid_a').print_info()
        data_info(valid_u, 'valid_u').print_info()
        data_info(test_a , 'test_a').print_info()
        data_info(test_u , 'test_u').print_info()

    # print('Train in ', train_a.shape)
    # print('Train out', train_u.shape)
    # print('Valid in ', valid_a.shape)
    # print('Valid out', valid_u.shape)
    # print('Test  in ', test_a.shape)
    # print('Test  out', test_u.shape)
    
    return train_a, train_u, valid_a, valid_u, test_a, test_u
    

def read_from_mat(data_path):
    
    a = torch.tensor(scipy.io.loadmat(data_path)['a'], dtype=torch.float32)
    u = torch.tensor(scipy.io.loadmat(data_path)['u'], dtype=torch.float32)
    
    return a, u

class data_info():
    
    def __init__(self, data_tensor, data_name='', channel_index=1):
        
        self.data_tensor    = data_tensor
        self.data_name      = data_name
        
        self.tensor_shape   = data_tensor.shape
        
        self.n_channels     = self.tensor_shape[channel_index]
        self.channel_index  = channel_index

        # for dim in range(len(self.tensor_shape)):
        #     if dim != channel_index:
        #         val_min, _  = torch.min (data_tensor, dim=dim, keepdim=True)
        #         val_max, _  = torch.max (data_tensor, dim=dim, keepdim=True)
        #         val_avg     = torch.mean(data_tensor, dim=dim, keepdim=True)
        #         val_std     = torch.std (data_tensor, dim=dim, keepdim=True)
        
        self.val_min        = self.calc_min()
        self.val_max        = self.calc_max()
        
        self.val_avg        = self.calc_avg()
        self.val_std        = self.calc_std()

    def calc_min(self):
        x = self.data_tensor.clone().detach()
        for dim in range(len(self.tensor_shape)):
            if dim != self.channel_index:
                x, _  = torch.min (x, dim=dim, keepdim=True)
        return x.squeeze().view(-1)
    
    def calc_max(self):
        x = self.data_tensor.clone().detach()
        for dim in range(len(self.tensor_shape)):
            if dim != self.channel_index:
                x, _  = torch.max (x, dim=dim, keepdim=True)
        return x.squeeze().view(-1)
    
    def calc_avg(self):
        x = torch.zeros(self.n_channels)
        for i in range(self.n_channels):
            x[i] = torch.mean (self.data_tensor[:,i,:,:])
        return x
    
    def calc_std(self):
        x = torch.zeros(self.n_channels)
        for i in range(self.n_channels):
            x[i] = torch.std (self.data_tensor[:,i,:,:])
        return x

    def print_info(self):
        
        rule_length = 60
        
        print('='*rule_length)
        print('Data Name  :', self.data_name)
        print('Data Shape :', *self.tensor_shape)
        
        for i in range(self.n_channels):
            print('-'*rule_length)
            print('  Channel %i Stats:'%(i))
            print('    MIN:', self.val_min.tolist()[i])
            print('    MAX:', self.val_max.tolist()[i])
            print('    AVG:', self.val_avg.tolist()[i])
            print('    STD:', self.val_std.tolist()[i]) 
            # print('-'*rule_length)
        
        # print('='*rule_length)
        
        return

def data_normalisation(processed_save_path      = '../data/processed/data2_20241116/',
                       processed_save_path_new  = '../data/processed/data2_20241116_normalised/',
                       in_channels = 1, out_channels = 3, dim = (128,128),
                       norm_mode = 'Train'):
    
    print('Loading Original Data:')
    # reading original data
    train_a, train_u, \
        valid_a, valid_u, \
            test_a, test_u = data_loader(processed_save_path = processed_save_path, 
                                        in_channels = in_channels, out_channels = out_channels, 
                                        dim = dim)

    if norm_mode == 'Train':
        a_info = data_info(train_a)
        u_info = data_info(train_u)
    elif norm_mode == 'All':
        tensor_a = torch.cat((train_a, valid_a, test_a), 0)
        tensor_u = torch.cat((train_u, valid_u, test_u), 0)
        
        a_info = data_info(tensor_a)
        u_info = data_info(tensor_u)
    else:
        raise TypeError("Invalid norm_mode: ", norm_mode)

    # for input channels
    a_avg, a_std = a_info.val_avg, a_info.val_std
    # print('=======')

    # for output channels
    u_avg, u_std = u_info.val_avg, u_info.val_std
    # print('=======')
    a_list    = [train_a, valid_a, test_a]
    u_list    = [train_u, valid_u, test_u]

    # data normalisation (using training dataset avg and std values only)
    for i in range(len(a_list)):
        
        for j in range(in_channels):
            a_list[i][:,j,:,:] = (a_list[i][:,j,:,:] - a_avg[j]) / a_std[j] 
            
        for j in range(out_channels):
            u_list[i][:,j,:,:] = (u_list[i][:,j,:,:] - u_avg[j]) / u_std[j] 
    
    # ======================================================        
    # saving pre-processed data
    print('Saving Normalised Data:')
    
    os.makedirs(processed_save_path_new, exist_ok=True)
    scipy.io.savemat(processed_save_path_new + 'train.mat', mdict={'a': train_a.numpy(), 'u': train_u.numpy()})
    scipy.io.savemat(processed_save_path_new + 'valid.mat', mdict={'a': valid_a.numpy(), 'u': valid_u.numpy()})
    scipy.io.savemat(processed_save_path_new +  'test.mat', mdict={'a': test_a.numpy() , 'u': test_u.numpy()})

    # Storing normalisation data
    
    norm_data = {'Labels': ['avg', 'std']}
    for i in range(in_channels):
        norm_data['in_%i'%(i)] = [a_avg[i].item(), a_std[i].item()]

    for i in range(out_channels):
        norm_data['out_%i'%(i)] = [u_avg[i].item(), u_std[i].item()]

    df = pd.DataFrame(norm_data)
    df.to_csv(processed_save_path_new + 'norm_info.csv', float_format="%.15E", index=True)  
    
    print('Normalised Data saved in : ', processed_save_path_new)
    print(df)
    # _ = data_loader(processed_save_path = processed_save_path_new, 
    #                                      in_channels = 1, out_channels = 3, 
    #                                      dim = (128,128), if_print=True)
    
    return

def denormalisation(inputs, outputs, norm_info):
    
    '''
    norm_info is 2D array: 
     - [0,:] being averages amd [1,:] being std
     - [:,0:in_channels] are for inputs and [:,in_channels:] are for outputs
    '''
    
    inputs_loc  = inputs.clone()
    outputs_loc = outputs.clone()
    
    # inputs
    # inputs_loc = inputs_loc * norm_info[1,0] + norm_info[0,0]
    for i in range(inputs.shape[1]):
        # print(i)
        # print(norm_info[1, i], norm_info[0, i])
        inputs_loc[:,i,:,:] = inputs_loc[:,i,:,:] * norm_info[1, i] + norm_info[0, i]
    
    # outputs
    for i in range(outputs.shape[1]):
        # print(i)
        # print(norm_info[1, inputs.shape[1]+i],  norm_info[0, inputs.shape[1]+i])
        outputs_loc[:,i,:,:] = outputs_loc[:,i,:,:] * norm_info[1, inputs.shape[1]+i] + norm_info[0, inputs.shape[1]+i]
        
    return inputs_loc, outputs_loc

def data_normalisation_single(input_data_file     = '../data/processed/data2_20241116/low_res.mat',
                              processed_save_path = '../data/processed/data2_20241116/',
                              save_name = 'test_low_res.mat',
                              in_channels = 1, out_channels = 3, dim = (128,128),
                              resize = False, mode='bilinear', align_corners=True):
    
    if os.path.isfile(input_data_file):
        # tensor_a = torch.tensor(scipy.io.loadmat(input_data_file)['a'], dtype=torch.float32)
        # tensor_u = torch.tensor(scipy.io.loadmat(input_data_file)['u'], dtype=torch.float32)
        tensor_a, tensor_u = read_from_mat(input_data_file)
    else:
        raise TypeError("No data available: ", input_data_file)
    
    if resize:
        tensor_a = F.interpolate(tensor_a, size=dim, mode=mode, align_corners=align_corners)
        tensor_u = F.interpolate(tensor_u, size=dim, mode=mode, align_corners=align_corners)
    
    
    if os.path.isfile(processed_save_path+'/norm_info.csv'):
        df = pd.read_csv(processed_save_path+'/norm_info.csv')
        norm_info = np.array(df)[:,2:].astype(float)
        assert (norm_info.shape == (2, in_channels+out_channels)), "Dimension mismatch!"
    else:
        raise TypeError("No file available: ", processed_save_path+'/norm_info.csv')
    
    
    # for input channels
    a_avg, a_std = norm_info[0, 0:in_channels], norm_info[1, 0:in_channels]
    # print('=======')

    # for output channels
    u_avg, u_std = norm_info[0, in_channels::], norm_info[1, in_channels::]
    # print('=======')
    
        
    for j in range(in_channels):
        tensor_a[:,j,:,:] = (tensor_a[:,j,:,:] - a_avg[j]) / a_std[j] 
        
    for j in range(out_channels):
        tensor_u[:,j,:,:] = (tensor_u[:,j,:,:] - u_avg[j]) / u_std[j] 
    
    # ======================================================        
    # saving normalsied testing data
    
    scipy.io.savemat(processed_save_path + save_name, mdict={'a': tensor_a.numpy(), 'u': tensor_u.numpy()})
    print('Normalised Data saved as:', processed_save_path + save_name)
    
    return None

def data_reshuffle(processed_save_path     = '../data/processed/data/',
                   processed_save_path_new = '../data/processed/data_1/',
                   in_channels = 1, out_channels = 3, dim = (128,128),
                   train_frac = 0.8):
    
    train_a, train_u, \
        valid_a, valid_u, \
            test_a, test_u = data_loader(processed_save_path = processed_save_path, 
                                        in_channels = in_channels, out_channels = out_channels, 
                                        dim = dim, if_print = False)
    
    all_a = torch.cat((train_a, valid_a, test_a), 0)
    all_u = torch.cat((train_u, valid_u, test_u), 0)
    
    print('All Data')
    print(all_a.shape)
    print(all_u.shape)
    
    # shuffling
    shuffling_index = torch.randperm(all_a.shape[0])
    all_a_shuffle = all_a[shuffling_index,:,:,:]
    all_u_shuffle = all_u[shuffling_index,:,:,:]
    
    # print(all_a_shuffle.shape)
    # print(all_u_shuffle.shape)

    # data splitting
    n_train = round(all_a.shape[0] * train_frac) # 2400
    n_valid = round(all_a.shape[0] * (1-train_frac)/2)
    n_test  = int(all_a.shape[0] - n_train - n_valid)

    train_a = all_a_shuffle[0:n_train, :,:,:]
    train_u = all_u_shuffle[0:n_train, :,:,:]

    valid_a = all_a_shuffle[n_train:n_train+n_valid, :,:,:]
    valid_u = all_u_shuffle[n_train:n_train+n_valid, :,:,:]

    test_a  = all_a_shuffle[n_train+n_valid:n_train+n_valid+n_test, :,:,:]
    test_u  = all_u_shuffle[n_train+n_valid:n_train+n_valid+n_test, :,:,:]

    print('Train')
    print(train_a.shape)
    print(train_u.shape)

    print('Valid')
    print(valid_a.shape)
    print(valid_u.shape)

    print('Test')
    print(test_a.shape)
    print(test_u.shape)

    os.makedirs(processed_save_path_new, exist_ok=True)
    with open(processed_save_path_new+"/shuffling_index.txt", "w") as file:
        for i in range(len(shuffling_index)):
            file.write('%i\n'%(shuffling_index[i].item()))
        # file.write(",".join(map(str, shuffling_index)))
    scipy.io.savemat(processed_save_path_new + 'train.mat', mdict={'a': train_a.numpy(), 'u': train_u.numpy()})
    scipy.io.savemat(processed_save_path_new + 'valid.mat', mdict={'a': valid_a.numpy(), 'u': valid_u.numpy()})
    scipy.io.savemat(processed_save_path_new +  'test.mat', mdict={'a': test_a.numpy() , 'u': test_u.numpy()})
    
    print('Reshuffled data saved in :', processed_save_path_new)
    
    return train_a, train_u, valid_a, valid_u, test_a, test_u

# ==============================================================================
# not using

def data_preprocessing(processed_save_path = './processed/case1/', 
                       in_channels = 1, out_channels = 3, # 1 for p, 2 for u
                       dim = (128,128), save = True):

    # hyperparameters
    sub    = 1 # subsampling
    ntrain = 8 # 900
    ntest  = 2 # 100
    
    frac_train = 0.8 # percentage of training data, validation and test are 1:1
    
    # TODO data interfacing from Matei/Philip input size 1*128*128 of alpha
    
    # original file paths
    origin_data_path = './raw/' # tobe completed
    
    data = np.load(origin_data_path)

    ntrain = int(frac_train * len(data)) # 900
    nvalid = int(len(data) - ntrain) // 2
    ntest  = int(len(data) - ntrain - nvalid) # 100


    # ============
    
    data = torch.tensor(data, dtype=torch.float)[..., ::sub, ::sub]
    episode_samples = 300
    test_samples = int(episode_samples*0.1)

    data_sampled_train = data[torch.randperm(data[:ntrain].size(0))[:episode_samples],...]
    data_sampled_valid = data[torch.randperm(data[ntrain:ntrain+nvalid].size(0))[:test_samples],...]
    data_sampled_test  = data[torch.randperm(data[-ntest:].size(0))[:test_samples],...]

    train_a = data_sampled_train[: , T_in-1:T_out-1].reshape(-1, in_channels , dim[0], dim[1])
    train_u = data_sampled_train[: , T_in  :T_out  ].reshape(-1, out_channels, dim[0], dim[1])
    valid_a = data_sampled_train[: , T_in-1:T_out-1].reshape(-1, in_channels , dim[0], dim[1])
    valid_u = data_sampled_train[: , T_in  :T_out  ].reshape(-1, out_channels, dim[0], dim[1])
    test_a  = data_sampled_test [: , T_in-1:T_out-1].reshape(-1, in_channels , dim[0], dim[1])
    test_u  = data_sampled_test [: , T_in  :T_out  ].reshape(-1, out_channels, dim[0], dim[1])

    if save:
        if (processed_save_path != '') and (processed_save_path[-1] != '/'): 
            processed_save_path = processed_save_path+'/'
        
        if (processed_save_path != '') and (os.path.isdir(processed_save_path) == False):
            os.makedirs(processed_save_path)
            print('Save directory %s is created'%(processed_save_path))
            
        # save train and test data to a mat file to data folder
        scipy.io.savemat(processed_save_path + 'train.mat', mdict={'a': train_a.numpy(), 'u': train_u.numpy()})
        scipy.io.savemat(processed_save_path + 'valid.mat', mdict={'a': valid_a.numpy(), 'u': valid_u.numpy()})
        scipy.io.savemat(processed_save_path +  'test.mat', mdict={'a': test_a.numpy() , 'u': test_u.numpy()})
    
        print('Data saved in ', processed_save_path)
        
    return train_a, train_u, valid_a, valid_u, test_a, test_u

if __name__ ==  '__main__':
    
    # Random Seeds
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    np.random.seed(0)
    
    data = data_preprocessing(processed_save_path = './processed/case1/', save = True)