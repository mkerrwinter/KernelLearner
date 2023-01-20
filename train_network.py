#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:30:02 2022

@author: mwinter
"""

# A script to train a network for the memory kernel project

import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import os
import sys
import numpy as np

import auxiliary_functions as aux
from NN import NeuralNetwork

start = time.time()
plt.close('all')

# Set number of threads to 1 for cpu jobs
if not torch.cuda.is_available():
    torch.set_num_threads(1)
    

# Define training process
def train(dataloader, model, loss_fn, optimizer, device):

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        for param in model.parameters():
            param.grad = None  
            
        loss.backward() 
        optimizer.step() 
        

# Calculate value of loss function
def evaluate_loss(dataloader, model, loss_fn, device):
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


if __name__ == "__main__":
    # Check whether this is running on my laptop
    current_dir = os.getcwd()
    if current_dir[:6] == '/Users':
        on_laptop = True
    else:
        on_laptop = False
    
    # base_path is defined assuming jobs are run in the output folder
    if on_laptop:
        base_path = './'
    else:
        base_path = '../'
    
    if torch.cuda.is_available():
        device = "cuda"
        print('Training with CUDA GPU.')
    else:
        device = "cpu"
        print('Training with CPU.')
    
    # Load any command line arguments
    if len(sys.argv)==4:
        model_name = sys.argv[3]

        if sys.argv[1]=='True':
            start_from_existing_model = True
        else:
            start_from_existing_model = False
        
        if sys.argv[2]=='True':
            start_from_param_file = True
        else:
            start_from_param_file = False
        
        print('Running with command line arguments')

    elif len(sys.argv)==8:    
        name_stub = sys.argv[7]

        if sys.argv[1]=='True':
            start_from_existing_model = True
        else:
            start_from_existing_model = False
        
        if sys.argv[2]=='True':
            start_from_param_file = True
        else:
            start_from_param_file = False
        
        ic = sys.argv[3]
        reg = sys.argv[4]
        b = sys.argv[5]
        w_fac = sys.argv[6]
        
        model_name = name_stub + '{}_reg{}_b{}_w{}'.format(ic, reg, b, w_fac)
        
        print('Running with command line arguments')
    else:
        model_name = 'MaxEnt_testnet'
        # model_name = 'MSE_testnet'
        start_from_existing_model = True
        start_from_param_file = False
        noise_str = '-2'
        print('Running with default arguments')


    ### Load or create model ###
    start_epoch = -1
    model_output_dir = '{}models/{}/'.format(base_path, model_name)
    
    if start_from_existing_model:
        print('Loading existing model {}'.format(model_name))
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = aux.load_model_params('{}{}'.format(input_dir, model_name) + 
                                       '_model_params.json')
        
        hidden_layers = params['h_layer_widths']
        loss_function = params['loss_function']
        learning_rate = params['learning_rate']
        L2_penalty = params['L2_penalty']
        batch_size = params['batch_size']
        dataset = params['dataset']
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = aux.load_dataset(dataset, batch_size, 
                                                    base_path)
        
        models, epochs = aux.load_models(input_dir, model_name, params, device)
        model = models[-1]
        start_epoch = epochs[-1]
        
        print('Starting from epoch {}'.format(start_epoch+1))
    
    elif start_from_param_file:
        print('Creating model from parameter file')
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = aux.load_model_params('{}{}'.format(input_dir, model_name) + 
                                       '_model_params.json')
        hidden_layers = params['h_layer_widths']
        loss_function = params['loss_function']
        learning_rate = params['learning_rate']
        L2_penalty = params['L2_penalty']
        batch_size = params['batch_size']
        dataset = params['dataset']
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = aux.load_dataset(dataset, batch_size, 
                                                    base_path, model_name)
                
        if 'initial_condition_path' in params:
            ic_path = params['initial_condition_path']
            ic_input_dir, ic_model_name = aux.split_path(ic_path)
            models, _= aux.load_models(ic_input_dir, ic_model_name, params, 
                                       device)
            model = models[-1]
        else:
            model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs,
                              h_layer_widths=hidden_layers).to(device)
        
        # Save initial state
        dir_exists = os.path.isdir(model_output_dir)
        if not dir_exists:
            os.mkdir(model_output_dir)
            
        torch.save(model.state_dict(),'{}{}_model_epoch_0.pth'.format(
            model_output_dir, model_name))
        
        if 'initial_condition_path' not in params:
            ic_path = '{}models/{}_model_epoch_0.pth'.format(base_path, model_name)
            params['initial_condition_path'] = ic_path
            
    else:
        print('Creating new model.')

        loss_function = 'MSELoss'
        learning_rate = 1e-3
        batch_size = -1
        width = 10
        depth = 6
        dataset = 'F_to_M_noisy_-2'
        L2_penalty = 10**(-3)
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = aux.load_dataset(dataset, batch_size, base_path)
        
        hidden_layers = [width]*depth
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs,
                              h_layer_widths=hidden_layers).to(device)
        
        params = {}
        params['batch_size'] = batch_size
        params['dataset'] = dataset
        params['loss_function'] = loss_function
        params['learning_rate'] = learning_rate
        params['L2_penalty'] = L2_penalty
        params['N_inputs'] = N_inputs
        params['N_outputs'] = N_outputs
        params['h_layer_widths'] = hidden_layers
        
        # Save initial state
        model_output_dir = '{}models/{}/'.format(base_path, model_name)
        dir_exists = os.path.isdir(model_output_dir)
        if not dir_exists:
            os.mkdir(model_output_dir)
            
        torch.save(model.state_dict(),'{}{}_model_epoch_0.pth'.format(
            model_output_dir, model_name))
        
        ic_path = '{}models/{}_model_epoch_0.pth'.format(base_path, model_name)
        params['initial_condition_path'] = ic_path
    
    noise_str = dataset[-2:]
        
    # Check data_output_dir exists
    data_output_dir = '{}measured_data/{}/'.format(base_path, model_name)
    dir_exists = os.path.isdir(data_output_dir)
    if not dir_exists:
        os.mkdir(data_output_dir)        
    
    
    ### Define loss function and optimizer ###       
    if loss_function == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
        
    elif loss_function == 'MSELoss':
        loss_fn = nn.MSELoss()
        
    elif loss_function == 'weighted_MSELoss':
        loss_fn = aux.make_weighted_MSELoss()
        
    elif loss_function == 'MaxEnt':
        alpha = params['MaxEnt_alpha']
        data_outpath = '{}data/F_to_M_data/noise_{}/'.format(base_path, 
                                                             noise_str) 
        time_outpath = data_outpath+'times_for_plotting.csv'
        times = np.loadtxt(time_outpath)
        times = torch.Tensor(times)
        loss_fn = aux.make_MaxEntLoss(alpha, times)
        
    else:
        raise NameError('Provide a loss function')
    
    w_decay = params['L2_penalty']
    try:
        adam_betas = params['adam_betas']
    except KeyError:
        adam_betas=(0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 betas=adam_betas, weight_decay=w_decay)
        
    # Save parameters that may have changed
    if (N_inputs != params['N_inputs'] or 
        N_outputs != params['N_outputs'] or
        'initial_condition_path' not in params):
        
        params['N_inputs'] = N_inputs
        params['N_outputs'] = N_outputs
        ic_path = '{}models/{}_model_epoch_0.pth'.format(base_path, model_name)
        params['initial_condition_path'] = ic_path
    
        aux.save_model_params(params,'{}{}_model_params.json'.format(
                model_output_dir, model_name))
       
        
    ### Train ###
    if on_laptop:
        save_freq = 1
        epochs = 10
    else:
        save_freq = 100
        epochs = 10000
    test_loss = []
    train_loss = []
    epoch_list = []

    print('Begin training')
    for t in range(epochs):
        t += start_epoch + 1
        
        # Save every Nth epoch
        time_check = t - start_epoch

        train(train_dataloader, model, loss_fn, optimizer, device)
                
        # Save every Nth epoch
        if (time_check)%save_freq==0:
            print(f"Epoch {t+1}\n-------------------------------")
            torch.save(model.state_dict(), 
                       '{}{}_model_epoch_{}.pth'.format(model_output_dir, 
                                                               model_name, t))
            l = evaluate_loss(train_dataloader, model, loss_fn, device)
            train_loss.append(l)
            
            train_loss_outpath = data_output_dir + 'train_loss.txt'
            out_string = '{} {}\n'.format(t, train_loss[-1])
            try:
                with open(train_loss_outpath, 'a') as f:
                        f.write(out_string)
            except FileNotFoundError:
                with open(train_loss_outpath, 'w') as f:
                        f.write(out_string)
        
            test_loss.append(evaluate_loss(test_dataloader, model, loss_fn, 
                                            device))
            test_loss_outpath = data_output_dir + 'test_loss.txt'
            out_string = '{} {}\n'.format(t, test_loss[-1])
            try:
                with open(test_loss_outpath, 'a') as f:
                        f.write(out_string)
            except FileNotFoundError:
                with open(test_loss_outpath, 'w') as f:
                        f.write(out_string)
                        
            epoch_list.append(t)
        
            if np.isnan(l):
                print('WARNING: Got a NaN loss. Ending training.')
                break
    
    print('Training finished')
    
    
    ### Plot train loss, test loss, and an example ###
    epoch_array = np.array(epoch_list)
    N_steps = len(train_dataloader)
    timestep_array = epoch_array*N_steps
    
    # Plot losses
    plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)
    
    if start_epoch == -1:
        start_epoch = 0 # for plotting

    fig, ax = plt.subplots()
    plt.plot(timestep_array, test_loss)
    plt.yscale('log')
    plt.title('Loss on test set')
    plt.xlabel('Time/steps')
    plt.savefig('{}{}_test_loss_log.pdf'.format(plot_output_dir, model_name), 
                bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.plot(timestep_array, train_loss)
    plt.yscale('log')
    plt.title('Loss on train set')
    plt.xlabel('Time/steps')
    plt.savefig('{}{}_train_loss_log.pdf'.format(plot_output_dir, model_name), 
                bbox_inches='tight')
    
    
    # Load times for plotting
    time_path_stem = base_path + 'data/F_to_M_data/noise_{}/'.format(noise_str)
    time_path = time_path_stem + 'times_for_plotting.csv'
    times_for_plotting = np.loadtxt(time_path)
    
    # Plot an example from the test set
    model.eval()
    with torch.no_grad(): # Turns off gradient calculations. This saves time.
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            break
    
    plt_idx = 0
    fig, ax = plt.subplots()
    plt.plot(times_for_plotting, y[plt_idx, :], label='MCT')
    plt.plot(times_for_plotting, pred[plt_idx, :], label='NN pred')
    plt.legend()
    plt.xscale('log')
    plt.title('An example from the test set')
    plt.ylabel('Memory kernel')
    plt.xlabel('Time/A.U.')
    plt.savefig('{}{}_example_{}_from_testset.pdf'.format(plot_output_dir, 
                model_name, plt_idx), bbox_inches='tight')
    
    print('Running time = ', time.time()-start)







