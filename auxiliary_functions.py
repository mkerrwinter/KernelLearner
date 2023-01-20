#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:28:25 2022

@author: mwinter
"""

# Auxiliary functions

import json
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import glob
from collections import defaultdict
import matplotlib

from NN import NeuralNetwork

matplotlib.rcParams.update({'font.size': 24})

# Load the parameters for the model
def load_model_params(filepath):
    with open(filepath) as infile:
        params = json.load(infile)

    return params


# Save json of model params
def save_model_params(params, outpath):
    with open(outpath, 'w') as outfile:
        json.dump(params, outfile)
        

# Load existing models
def load_models(input_dir, model_name, params, device):
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
        filename = '{}{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                     epoch)
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                              h_layer_widths=h_layer_widths, 
                              prob=dropout_p).to(device)
        model.load_state_dict(torch.load(filename, 
                                         map_location=torch.device(device)))
        
        # Check device
        if epoch == epochs[0]:
            print('Current device = ', device)
            for l in range(len(model.net)):
                try:
                    l_weight = model.net[l].weight
                    print('Layer {} device '.format(l), l_weight.device)
                except AttributeError:
                    continue
        
        models.append(model)
    
    return models, epochs


# Load a dataset
def load_dataset(dataset, batch_size, base_path, model_name=None):
    if dataset[:6]=='F_to_M':
        noise_str = dataset[7:]
        
        data_output_dir = '{}data/F_to_M_data/noise_{}/'.format(base_path, 
                                                                noise_str)
        
        fname = data_output_dir + 'MCT_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'MCT_test.pt'
        test_data = torch.load(fname)
        
    else:
        print('PROVIDE A DATASET')

    N_inputs = training_data[0][0].shape[1]
    N_outputs = training_data[0][1].shape[0]
    
    # Deal with batch size = dataset size case
    set_b_for_test = False
    if batch_size==-1:
        batch_size = len(training_data)
        set_b_for_test = True
    elif batch_size>len(test_data):
        set_b_for_test = True
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, 
                                  shuffle=True, drop_last=True)
    
    # Deal with batch size = dataset size case
    if set_b_for_test:
        batch_size = len(test_data)
        
    test_dataloader = DataLoader(test_data, batch_size=batch_size, 
                                 shuffle=True, drop_last=True)
    
    return (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs)


def split_path(ic_path):    
    chunks = ic_path.split('/')
    ic_path += '/'

    return ic_path, chunks[-1]


def check_for_NaN_network(model):
    NaN_network = False
    
    for param in model.parameters():
        param_array = param.detach().numpy()
        bool_array = np.isnan(param_array)
        bool_sum = bool_array.sum()
        
        if bool_sum>0:
            NaN_network = True
            break
    
    return NaN_network


def load_final_model(input_dir, model_name, params, device):
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

    def loss(output, target):
        N_cols = output.shape[1]
        weight = torch.arange(1, N_cols+1)/N_cols
        return torch.mean(weight*((output - target)**2))
        
    return loss


def trapezium_rule_for_batches(y, x):
    deltas = x[1:]-x[:-1]
    average_y = (y[:, 1:]+y[:, :-1])/2.0
    areas = deltas*average_y
    return areas.sum(axis=1)


def make_MaxEntLoss(alpha, times):

    def loss(output, target):
        entropy = -torch.abs(output)*torch.log(torch.abs(output))
        integral = trapezium_rule_for_batches(entropy, times)
        L_output, s_vals = Laplace_transform_for_backprop(times, output)
        L_target, s_vals = Laplace_transform_for_backprop(times, target)
        loss_integrand = 0.5*(L_output-L_target)**2
        loss_MSE = trapezium_rule_for_batches(loss_integrand, s_vals)
        l_by_batch = loss_MSE - alpha*integral
        return torch.mean(l_by_batch)
        
    return loss


def fit_expo(t, y):
    log_y = np.log(y)
    coeffs = np.polyfit(t, log_y, 1)
    B = coeffs[0]
    c = coeffs[1]
    A = np.exp(c)

    return A, B


def forward_Laplace(f, p, tmin=0.0, tmax=np.inf):
    # Calculate Laplace transform for function f
    real_integral = quad(lambda t: (f(t)*np.exp(-t*p)).real, tmin, tmax, limit=200)[0]
    im_integral = quad(lambda t: (f(t)*np.exp(-t*p)).imag, tmin, tmax, limit=200)[0]
    
    integral = real_integral + 1j*im_integral
    
    return integral


def trapezium_rule(y, x):
    deltas = x[1:]-x[:-1]
    average_y = (y[1:]+y[:-1])/2.0
    areas = deltas*average_y
    return areas.sum()


def forward_Laplace_trapz_method(F, times, s):
    integral = trapezium_rule(F*np.exp(-times*s), times)
    
    return integral


def numerical_Laplace_transform(t, f, p, trapz_method=False):
    func = interp1d(t, f, kind='cubic')

    print('Transforming...')
    
    t_max = max(t)
    t_min = min(t)

    transform = []
    
    for idx in range(len(p)):
        p_val = p[idx]
        
        if trapz_method:
            val = forward_Laplace_trapz_method(f, t, p_val)    
        else:
            val = forward_Laplace(func, p_val, tmin=t_min, tmax=t_max)

        transform.append(val)
    
    return np.array(transform)


def Laplace_transform_for_backprop(t, f):
    s_vals = torch.arange(1, 11, 0.1)
    transform = torch.zeros(f.shape[0], s_vals.shape[0])
    
    for s_idx, s in enumerate(s_vals):
        val = forward_Laplace_trapz_method(f, t, s)    
        transform[:, s_idx] = val
        
    return transform, s_vals
    


def make_quadratic_hinge_loss():
    
    def quadratic_hinge(output, target):
        Delta = 1.0-target*output
        zeros = torch.zeros_like(Delta)
        
        max_Delta = torch.max(zeros, Delta)
        
        sq_max_Delta = max_Delta*max_Delta
        
        return 0.5*torch.mean(sq_max_Delta)
    
    return quadratic_hinge


def evaluate_loss(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad(): # Turns off gradient calculations. This saves time.
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
    
    loss /= num_batches
    return loss


def plot_losses(models, epochs, model_name, params, device, base_path):
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
        
    elif loss_function == 'quadratic_hinge':
        loss_fn = make_quadratic_hinge_loss()
        
    else:
        print('PROVIDE A LOSS FUNCTION')
    
    train_loss = []
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
        
        print('Calculating training loss at epoch {}'.format(epoch))

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
    plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
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
    # Get all models with model_name as their stub
    path_stub = '{}models/{}*'.format(base_path, model_name_stub)
    dirs = glob.glob(path_stub)
    
    model_names = []
    best_losses = []
    best_epochs = []
    
    for my_dir in dirs:
        dir_split = my_dir.split('/')
        model_name = dir_split[-1]
        
        print('Model name: ', model_name)
        
        # See if loss data already exists
        saved_data_dir = '{}measured_data/{}/'.format(base_path, model_name)
        saved_data_path = '{}test_loss.txt'.format(saved_data_dir)
        data_exists = os.path.isfile(saved_data_path)
        
        if data_exists:
            data = np.loadtxt(saved_data_path)
            subsampled_epochs = data[:, 0]
            test_loss = data[:, 1]
            print('Loaded losses from text file.')
        
        else:
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
                
            elif loss_function == 'quadratic_hinge':
                loss_fn = make_quadratic_hinge_loss()
                
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
    outpath = '{}measured_data/ranked_models_{}.txt'.format(base_path, 
                                                           model_name_stub)
    print('outpath = ', outpath)
    
    with open(outpath, 'w') as f:
        f.write('Best model is:\n')
        f.write('{} at epoch {} with test loss {}\n'.format(ordered_names[0], 
                                                            ordered_epochs[0], 
                                                            ordered_losses[0]))
        
        f.write('\n Models ordered by test loss are: \n')
        for k in range(len(ordered_losses)):
            f.write('{} at epoch {} with loss {}\n'.format(ordered_names[k], 
                                                         ordered_epochs[k], 
                                                         ordered_losses[k]))


def plot_loss_vs_L2(model_name_stub, base_path, device):
    # Get all models with model_name as their stub
    path_stub = '{}models/{}_ic*_reg*_b2500_w8*'.format(base_path, 
                                                       model_name_stub)
    dirs = glob.glob(path_stub)
    
    losses = []
    L2s = []
    
    print('path_stub = ', path_stub)
    
    for my_dir in dirs:
        print('my_dir = ', my_dir)
        dir_split = my_dir.split('/')
        model_name = dir_split[-1]
        
        print('Model name: ', model_name)
        
        # Extract L2
        L2_name_parts = model_name.split('reg')
        L2_part = L2_name_parts[-1].split('b')
        L2_str = L2_part[0][:-1]
        L2_str = L2_str.replace('_', '.')
        L2 = float(L2_str)
        
        print('L2 = ', L2)
        
        # See if loss data already exists
        saved_data_dir = '{}measured_data/{}/'.format(base_path, model_name)
        saved_data_path = '{}test_loss.txt'.format(saved_data_dir)
        data_exists = os.path.isfile(saved_data_path)
        
        if data_exists:
            data = np.loadtxt(saved_data_path)
            subsampled_epochs = data[:, 0]
            test_loss = data[:, 1]
            print('Loaded losses from text file.')
        
        else:
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
                
            elif loss_function == 'quadratic_hinge':
                loss_fn = make_quadratic_hinge_loss()
                
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
            
            losses.append(best_loss)
            L2s.append(L2)
        else:
            losses.append(np.nan)
            L2s.append(np.nan)
    
    print('losses = ', losses)
    print('L2s = ', L2s)
    
    # Plot best losses
    plot_output_dir = '{}plots/'.format(base_path)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)
        
    L2s = np.array(L2s)
    L2s_for_plotting = L2s.copy()
    idx0 = np.where(L2s==0)[0]
    idx1 = np.where(L2s==0.001)[0]
    idx2 = np.where(L2s==0.01)[0]
    idx3 = np.where(L2s==0.05)[0]
    idx4 = np.where(L2s==0.1)[0]
    
    L2s_for_plotting[idx0] = 0
    L2s_for_plotting[idx1] = 1
    L2s_for_plotting[idx2] = 2
    L2s_for_plotting[idx3] = 3
    L2s_for_plotting[idx4] = 4
    xticks = [0, 1, 2, 3, 4]
    xlabels = ['0', '0.001', '0.01', '0.05', '0.1']
    yticks = [20, 100, 300]
    ylabels = ['20', '100', '300']
        
    fig, ax = plt.subplots()
    plt.scatter(L2s_for_plotting, losses)
    plt.yscale('log')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    plt.ylabel(r'$L_{Test}$')
    plt.xlabel(r'$\lambda$')
    plt.savefig('{}test_loss_vs_L2.pdf'.format(plot_output_dir), 
                bbox_inches='tight')


def find_nan_weight(model):
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


def is_model_nan(models, epochs, model_name):
    print('Starting NaN test')
    
    for idx in range(len(models)):
        model = models[idx]
        epoch = epochs[idx]
        
        nan_test = 0.0
        for param in model.parameters():
            np_param = param.detach().numpy()
            import pdb; pdb.set_trace()
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
    
            
        
        

