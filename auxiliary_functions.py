#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:28:25 2023

@author: Max Kerr Winter

Auxiliary functions that are used in other scripts.
"""

import json
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import glob

from NN import NeuralNetwork

def load_model_params(filepath):
    """
    Load the hyperparameter json for a model.
    
    Inputs
    ------
    filepath : str
               The path of the hyperparameter json file.
               
    Outputs
    -------
    params : dict
             A dictionary containing the model hyperparameters.
    """
    
    with open(filepath) as infile:
        params = json.load(infile)

    return params



def save_model_params(params, outpath):
    """
    Save a json of model hyperparameters.
    
    Inputs
    ------
    params  : dict
              A dictionary of model hyperparameters.
    
    outpath : str
              The path where the json is to be saved.
    """
    
    with open(outpath, 'w') as outfile:
        json.dump(params, outfile)
        

def load_models(input_dir, model_name, params, device):
    """
    Load existing models from a directory.
    
    Inputs
    ------
    input_dir  : str
                 The path of the directory where the models are saved.
                
    model_name : str
                 The name of the model. Models are saved with the path 
                 convention {inputdir}{model_name}_epoch_{N}.pth where N is
                 the epoch number.
    
    params     : dict
                 A dictionary of the model hyperparameters.
                 
    device     : str
                 Determines whether pytorch tensors are saved to cpu or gpu.
                 
    Outputs
    -------
    models : list
             The loaded models.
    
    epochs : list
             The epoch of each loaded model.
    """
    
    files = os.listdir(input_dir)
    
    # order files
    epochs = []
    for file in files:
        file2 = file
        try:
            start, end = file2.split('_epoch_')
        except ValueError:
            continue
        
        if start != model_name:
            continue
        
        epoch = int(end[:-4])
        epochs.append(epoch)
    
    epochs.sort()
    
    # load models
    N_inputs = params['N_inputs']
    N_outputs = params['N_outputs']
    h_layer_widths = params['h_layer_widths']
    
    try:
        dropout_p = params['dropout_p']
    except KeyError:
        dropout_p = 0.5
    
    models = []
    for epoch in epochs:
        filename = '{}{}_epoch_{}.pth'.format(input_dir, model_name, epoch)
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                              h_layer_widths=h_layer_widths, 
                              prob=dropout_p).to(device)
        model.load_state_dict(torch.load(filename, 
                                         map_location=torch.device(device)))
        
        # Check device
        if epoch == epochs[0]:
            for l in range(len(model.net)):
                try:
                    l_weight = model.net[l].weight
                    if str(l_weight.device) != device:
                        raise AssertionError('Error: Model saved with' + 
                                             ' different device.')
                except AttributeError:
                    continue
        
        models.append(model)
    
    return models, epochs


def load_dataset(dataset, batch_size, base_path, model_name=None):
    """
    Load a dataset.
    
    Inputs
    ------
    dataset    : str
                 The name of the dataset.
              
    batch_size : int
                 The number of examples in each batch.
                 
    base_path  : str
                 The root directory that this code runs in.
                 
    model_name : str
                 The name of a model, if required.
                 
    Outputs
    -------
    train_dataloader : DataLoader
                       The training set, organised into batches.
                       
    test_dataloader  : DataLoader
                       The test set, organised into batches.
                       
    training_data    : Tensor
                       The training data in a pytorch tensor.
                       
    test_data        : Tensor
                       The test data in a pytorch tensor.
                       
    N_inputs         : int
                       The dimension of the input data.
                       
    N_outputs        : int
                       The dimension of the output data.
    """
    
    if dataset[:6]=='F_to_K':
        noise_str = dataset[7:]
        
        data_output_dir = '{}data/F_to_K_data/noise_{}/'.format(base_path, 
                                                                noise_str)
        
        fname = data_output_dir + 'GLE_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'GLE_test.pt'
        test_data = torch.load(fname)
        
    else:
        print('PROVIDE A DATASET')

    N_inputs = training_data[0][0].shape[1]
    N_outputs = training_data[0][1].shape[0]
    
    # Deal with batch size = training dataset size case
    set_b_for_test = False
    if batch_size==-1:
        batch_size = len(training_data)
        set_b_for_test = True
    elif batch_size>len(test_data):
        set_b_for_test = True
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, 
                                  shuffle=True, drop_last=True)
    
    # Deal with batch size = training dataset size case
    if set_b_for_test:
        batch_size = len(test_data)
        
    test_dataloader = DataLoader(test_data, batch_size=batch_size, 
                                 shuffle=True, drop_last=True)
    
    return (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs)


def load_final_model(input_dir, model_name, params, device):
    """
    Load the final model from a directory containing many models at different
    epochs.
    
    Inputs
    ------
    input_dir  : str
                 The path of the directory where the models are saved.
                
    model_name : str
                 The name of the model. Models are saved with the path 
                 convention {inputdir}{model_name}_epoch_{N}.pth where N is
                 the epoch number.
    
    params     : dict
                 A dictionary of the model hyperparameters.
                 
    device     : str
                 Determines whether pytorch tensors are saved to cpu or gpu.
    
    Outputs
    -------
    model : NeuralNetwork
            The loaded models from the final epoch.
    
    epoch : int
            The final epoch.
    """
    
    files = os.listdir(input_dir)
    
    # order files
    epochs = []
    for file in files:
        file2 = file
        try:
            start, end = file2.split('_epoch_')
        except ValueError:
            continue
        
        if start != '{}_model'.format(model_name):
            continue
        
        epoch = int(end[:-4])
        epochs.append(epoch)
    
    epochs.sort()
    
    N_inputs = params['N_inputs']
    N_outputs = params['N_outputs']
    h_layer_widths = params['h_layer_widths']
    
    try:
        dropout_p = params['dropout_p']
    except KeyError:
        dropout_p = 0.5
    
    # search backwards for first non-NaN network    
    final_idx = len(epochs)-1
    new_idx = final_idx
    init_idx = -final_idx
    print('Loading epoch ', epochs[final_idx])
    epoch = epochs[final_idx]
    filename = '{}/{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                 epoch)
    
    model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                              h_layer_widths=h_layer_widths, 
                              prob=dropout_p).to(device)
    model.load_state_dict(torch.load(filename, 
                                     map_location=torch.device(device)))
    
    NaN_network = check_for_NaN_network(model)
    if not NaN_network:
        init_idx = final_idx
    
    while abs(final_idx-init_idx)>1:
        new_idx = int((final_idx+init_idx)/2.0)
        
        epoch = epochs[new_idx]
        filename = '{}/{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                     epoch)
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                              h_layer_widths=h_layer_widths)
        model.load_state_dict(torch.load(filename, 
                                         map_location=torch.device(device)))
        
        NaN_network = check_for_NaN_network(model)
        
        # Check for epoch 0 NaN
        if NaN_network and new_idx==0:
            raise ValueError('ERROR: First epoch is a NaN network.')
        
        if NaN_network:
            final_idx = new_idx
        else:
            init_idx = new_idx
    
    epoch = epochs[init_idx]
    filename = '{}/{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                     epoch)
    model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                          h_layer_widths=h_layer_widths)
    model.load_state_dict(torch.load(filename, 
                                     map_location=torch.device(device)))
                    
    return model, epoch


def make_weighted_MSELoss():
    """
    Make a weighted mean square error loss function. For model output f, and 
    ground truth K, the weighted mean square error loss function is
    
    L = \frac{1}{P}\sum_{i=1}^P\frac{1}{T}\sum_{j=1}^T\theta_j(f(t_j)-K(t_j))^2
    
    where P is the size of the dataset, and T is the time period over which 
    kernels are defined. The weights \theta_j increase linearly over this time
    period.
    
    Outputs
    -------
    loss : func
           The loss function.
    """

    def loss(output, target):
        N_cols = output.shape[1]
        weight = torch.arange(1, N_cols+1)/N_cols
        return torch.mean(weight*((output - target)**2))
        
    return loss


def evaluate_loss(dataloader, model, loss_fn, device):
    """
    Calculate the value of the loss function with the current state of the 
    neural network.
    
    Inputs
    ------
    dataloader : DataLoader
                 A dataset organised into batches.
                  
    model      : NeuralNetwork
                 A neural network.
                 
    loss_fn    : function
                 The loss function to be minimised.

    device     : str
                 Determines whether pytorch tensors are saved to cpu or gpu.
    """
    
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad(): 
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()

    if num_batches>0:
        loss /= num_batches
    else:
        loss = np.inf
        
    return loss


def plot_losses(models, epochs, model_name, params, device, base_path):
    """
    A function to plot the loss curve for a sequence of models.
    
    Inputs
    ------
    models     : list
                 A list of models at increasing epochs.
             
    epochs     : list
                 A list of epochs corresponding to the models.
             
    model_name : str
                 The name of the models.
                 
    params     : dict
                 A dictionary of model hyperparameters.
                 
    device     : str
                 Determines whether pytorch tensors are saved to cpu or gpu.
                 
    base_path  : str
                 The root directory that this code runs in.
    """
    
    batch_size = params['batch_size']
    dataset = params['dataset']
    
    (train_dataloader, test_dataloader, training_data, test_data, 
        N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path)
        
    # Define loss function 
    loss_function = params['loss_function']
    if loss_function == 'CrossEntropy':
        loss_fn = torch.nn.CrossEntropyLoss()
        
    elif loss_function == 'MSELoss':
        loss_fn = torch.nn.MSELoss()
    
    elif loss_function == 'weighted_MSELoss':
        loss_fn = make_weighted_MSELoss()
        
    elif loss_function == 'Hinge':
        loss_fn = torch.nn.HingeEmbeddingLoss()
        
    else:
        print('PROVIDE A LOSS FUNCTION')
    
    train_loss = []
    test_loss = []
    subsampled_epochs = []
    
    if len(models)>900:
        rate = 60
        print('Subsampling models to speed up loss plotting')
    else:
        rate = 1
    
    for count in range(0, len(epochs), rate):
        epoch = epochs[count]
        model = models[count]
        
        print('Calculating loss at epoch {}'.format(epoch))

        train_l = evaluate_loss(train_dataloader, model, loss_fn, device)
        test_l = evaluate_loss(test_dataloader, model, loss_fn, device)
        
        train_loss.append(train_l)
        test_loss.append(test_l)
        subsampled_epochs.append(epoch)
    
    # Rescale time to be in time steps not epochs
    sub_epoch_array = np.array(subsampled_epochs)
    N_steps = len(train_dataloader)
    timestep_array = sub_epoch_array*N_steps
        
    # Plot result
    plot_dir = '{}plots/'.format(base_path)
    dir_exists = os.path.isdir(plot_dir)
    if not dir_exists:
        os.mkdir(plot_dir)
        
    plot_output_dir = '{}{}/'.format(plot_dir, model_name)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)
    
    NaN_train_x = np.where(np.isnan(train_loss))[0]
    NaN_train_y = np.zeros_like(NaN_train_x)
    
    NaN_test_x = np.where(np.isnan(test_loss))[0]
    NaN_test_y = np.zeros_like(NaN_test_x)

    fig, ax = plt.subplots()
    plt.plot(timestep_array, train_loss)
    plt.scatter(NaN_train_x, NaN_train_y, color='r', label='NaN')
    plt.legend()
    plt.title('Train loss')
    plt.ylabel('Loss')
    plt.xlabel('Time steps')
    plt.yscale('log')
    plt.savefig('{}{}_train_loss.pdf'.format(plot_output_dir, 
                model_name), bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.plot(timestep_array, test_loss)
    plt.scatter(NaN_test_x, NaN_test_y, color='r', label='NaN')
    plt.legend()
    plt.title('Test loss')
    plt.ylabel('Loss')
    plt.xlabel('Time steps')
    plt.yscale('log')
    plt.savefig('{}{}_test_loss.pdf'.format(plot_output_dir, 
                model_name), bbox_inches='tight')
    
    
def find_best_model(model_name_stub, base_path, device):
    """
    Find the best model out of a set of models that all share a model_name.
    In this context, 'best' means the network that achieves the lowest test 
    loss. The results, including a ranked list of the models, is saved to a 
    text file and printed to screen.
    
    Inputs
    ------
    model_name_stub : str
                      The model_name used to save the models.
                      
    base_path       : str
                      The root directory that this code runs in.
                      
    device          : str
                      Determines whether pytorch tensors are saved to cpu or 
                      gpu.
    """
    # Get all models with model_name as their stub
    path_stub = '{}models/{}*'.format(base_path, model_name_stub)
    dirs = glob.glob(path_stub)
    
    model_names = []
    best_losses = []
    best_epochs = []
    
    for my_dir in dirs:
        dir_split = my_dir.split('/')
        model_name = dir_split[-1]
        
        if __name__=='__main__':
            print('Model name: ', model_name)
        
        # See if loss data already exists
        saved_data_dir = '{}measured_data/{}/'.format(base_path, model_name)
        saved_data_path = '{}test_loss.txt'.format(saved_data_dir)
        data_exists = os.path.isfile(saved_data_path)
        
        if data_exists:
            data = np.loadtxt(saved_data_path)
            subsampled_epochs = data[:, 0]
            test_loss = data[:, 1]
            if __name__=='__main__':
                print('Loaded losses from text file.')
        
        else:
            if __name__=='__main__':
                print('Calculating losses')
            
            input_dir = '{}models/{}/'.format(base_path, model_name)
            param_path = '{}/{}_model_params.json'.format(input_dir, model_name)
            params = load_model_params(param_path)
            models, epochs = load_models(input_dir, model_name, params, device)
            
            # Load dataset
            batch_size = params['batch_size']
            dataset = params['dataset']
            
            (train_dataloader, test_dataloader, training_data, test_data, 
                N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path)
                
            # Define loss function 
            loss_function = params['loss_function']
            if loss_function == 'CrossEntropy':
                loss_fn = torch.nn.CrossEntropyLoss()
                
            elif loss_function == 'MSELoss':
                loss_fn = torch.nn.MSELoss()
            
            elif loss_function == 'weighted_MSELoss':
                loss_fn = make_weighted_MSELoss()
                
            elif loss_function == 'Hinge':
                loss_fn = torch.nn.HingeEmbeddingLoss()
   
            else:
                print('PROVIDE A LOSS FUNCTION')
            
            # Evaluate test losses
            test_loss = []
            subsampled_epochs = []
            
            if len(models)>900:
                rate = 60
                print('Subsampling models to speed up training loss plotting')
            else:
                rate = 1
            
            for count in range(0, len(epochs), rate):
                epoch = epochs[count]
                model = models[count]
                    
                test_l = evaluate_loss(test_dataloader, model, loss_fn, device)
                
                test_loss.append(test_l)
                subsampled_epochs.append(epoch)
            
            # Save for re-use
            data_for_saving = np.zeros((len(test_loss), 2))
            data_for_saving[:, 0] = subsampled_epochs
            data_for_saving[:, 1] = test_loss
            
            dir_exists = os.path.isdir(saved_data_dir)
            if not dir_exists:
                os.mkdir(saved_data_dir)
            
            if len(test_loss)>0:
                np.savetxt(saved_data_path, data_for_saving, delimiter=' ') 
            
        # Find best loss
        if len(test_loss)>0:
            min_idx = np.argmin(test_loss)
            best_loss = test_loss[min_idx]
            best_epoch = subsampled_epochs[min_idx]
            
            model_names.append(model_name)
            best_losses.append(best_loss)
            best_epochs.append(best_epoch)
        else:
            model_names.append(model_name)
            best_losses.append(np.nan)
            best_epochs.append(np.nan)
    
    # Order models by best loss
    ordered_indices = np.argsort(best_losses)
    
    ordered_losses = np.array(best_losses)[ordered_indices]
    ordered_epochs = np.array(best_epochs)[ordered_indices]
    ordered_names = np.array(model_names)[ordered_indices]
    
    if __name__=='__main__':
        print('\n Best model is:')
        print('{} at epoch {} with test loss {}\n'.format(ordered_names[0], 
                                                          ordered_epochs[0], 
                                                          ordered_losses[0]))
        
        print('\n Models ordered by test loss are:')
        for k in range(len(ordered_losses)):
            print('{} at epoch {} with loss {}'.format(ordered_names[k], 
                                                       ordered_epochs[k], 
                                                       ordered_losses[k]))
        
        # Save best models to file
        output_dir = '{}measured_data/'.format(base_path)
        dir_exists = os.path.isdir(output_dir)
        if not dir_exists:
            os.mkdir(output_dir)
            
        outpath = '{}ranked_models_{}.txt'.format(output_dir, model_name_stub)
        
        with open(outpath, 'w') as f:
            f.write('Best model is:\n')
            f.write('{} at epoch {}, test loss {}\n'.format(ordered_names[0], 
                                                            ordered_epochs[0], 
                                                            ordered_losses[0]))
            
            f.write('\n Models ordered by test loss are: \n')
            for k in range(len(ordered_losses)):
                f.write('{} at epoch {}, loss {}\n'.format(ordered_names[k], 
                                                           ordered_epochs[k], 
                                                           ordered_losses[k]))
    
    best_model = ordered_names[0]
    best_epoch = int(ordered_epochs[0])
    
    return best_model, best_epoch


def check_for_NaN_network(model):
    """
    A quick check to see if a network contains a NaN parameter.
    
    Inputs
    ------
    model : NeuralNetwork
            A neural network.
            
    Outputs
    -------
    NaN_network : bool
                  True if the network contains a NaN parameter, False 
                  otherwise.
    """
    NaN_network = False
    
    for param in model.parameters():
        param_array = param.detach().numpy()
        bool_array = np.isnan(param_array)
        bool_sum = bool_array.sum()
        
        if bool_sum>0:
            NaN_network = True
            break
    
    return NaN_network


def find_nan_weight(model):
    """
    Find a NaN weight in a model.
    
    Inputs
    ------
    model : NeuralNetwork
            A neural network.
            
    Outputs
    -------
    nan_layers : list
                 A list of layers containing NaN weights.
    
    nan_indices : list
                  A list containing the indices of the NaN weights
                  with the corresponding layers in nan_layers.
    """
    
    nan_layers = []
    nan_indices = []
    
    layer_count = -1
    
    for param in model.parameters():
        layer_count += 1
        np_param = param.detach().numpy()
        nan_test = np_param.sum()
        
        if np.isnan(nan_test):
            nan_layers.append(layer_count)
            nan_idx = np.argwhere(np.isnan(np_param))
            nan_indices.append(nan_idx)
            
    return nan_layers, nan_indices


def print_nan_weights(models, epochs, model_name):
    """
    Check if a network contains an infinite or undefined weight, and if so,
    print where it is.
    
    Inputs
    ------
    models     : list
                 A list of models to check.
    
    epochs     : list
                 A list of epochs corresponding to the models.
    
    model_name : str
                 The name of the models.
    """
    
    print('Starting NaN test')
    
    for idx in range(len(models)):
        model = models[idx]
        epoch = epochs[idx]
        
        nan_test = 0.0
        for param in model.parameters():
            np_param = param.detach().numpy()
            nan_test += np_param.sum()
        
        if np.isnan(nan_test):
            layers, indices = find_nan_weight(model)
            print('Model {} is NaN at epoch {}'.format(model_name, epoch))
            print('NaNs at:')
            for i in range(len(layers)):
                l = layers[i]
                idx_array = indices[i]
                for j in range(idx_array.shape[0]):
                    row = idx_array[j, 0]
                    col = idx_array[j, 1]
                    print('Layer {}, position ({},{})'.format(l, row, col))
    
    print('Ending NaN test')

if __name__=='__main__':
    if torch.cuda.is_available():
        device = "cuda"
        print('Using GPU.')
    else:
        device = "cpu"
        print('Using CPU.')
        
    base_path = './'
    model_name = 'F_to_K_test_model'
    
    # Find best model (i.e. epoch)
    best_model, best_epoch = find_best_model(model_name, base_path, device)
    
            
        
        

