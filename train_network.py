#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:30:02 2023

@author: Max Kerr Winter

A script to train a neural network from a parameter file.
"""

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
    """
    Perform one step of the network training routine.
    
    Inputs
    ------
    dataloader : DataLoader
                 A dataset organised into batches.
                  
    model      : NeuralNetwork
                 A neural network.
                 
    loss_fn    : function
                 The loss function to be minimised.
                 
    optimizer  : Optimizer
                 An object that performs a gradient descent algorithm.
                 
    device     : str
                 Determines whether pytorch tensors are saved to cpu or gpu.
    """

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


if __name__ == "__main__":
    base_path = './'
    
    if torch.cuda.is_available():
        device = "cuda"
        print('Training with CUDA GPU.')
    else:
        device = "cpu"
        print('Training with CPU.')
    
    # Load any command line arguments
    if len(sys.argv)==8:    
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
        model_name = 'F_to_K_test_model'
        start_from_existing_model = False
        start_from_param_file = True
        print('Running with default arguments')


    # Load or create model
    start_epoch = -1
    model_output_dir = '{}models/{}/'.format(base_path, model_name)
    
    if start_from_existing_model:
        print('Loading existing model {}'.format(model_name))
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = aux.load_model_params('{}{}'.format(input_dir, model_name) + 
                                       '_params.json')
        
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
                                       '_params.json')
        hidden_layers = params['h_layer_widths']
        loss_function = params['loss_function']
        learning_rate = params['learning_rate']
        L2_penalty = params['L2_penalty']
        batch_size = params['batch_size']
        dataset = params['dataset']
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = aux.load_dataset(dataset, batch_size, 
                                                    base_path, model_name)
                
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs,
                              h_layer_widths=hidden_layers).to(device)
        
        # Save initial state
        dir_exists = os.path.isdir(model_output_dir)
        if not dir_exists:
            os.mkdir(model_output_dir)
            
        torch.save(model.state_dict(),'{}{}_model_epoch_0.pth'.format(
                   model_output_dir, model_name))
            
    else:
        raise ValueError('Must provide a parameter file or existing model.')
        
    # Check data_output_dir exists
    data_dir = '{}measured_data/'.format(base_path)
    dir_exists = os.path.isdir(data_dir)
    if not dir_exists:
        os.mkdir(data_dir)
        
    data_output_dir = '{}{}/'.format(data_dir, model_name)
    dir_exists = os.path.isdir(data_output_dir)
    if not dir_exists:
        os.mkdir(data_output_dir)        
    
    # Define loss function and optimizer      
    if loss_function == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
        
    elif loss_function == 'MSELoss':
        loss_fn = nn.MSELoss()
        
    elif loss_function == 'weighted_MSELoss':
        loss_fn = aux.make_weighted_MSELoss()
        
    else:
        raise NameError('Provide a loss function')
    
    w_decay = params['L2_penalty']
    try:
        adam_betas = params['adam_betas']
    except KeyError:
        adam_betas=(0.9, 0.999)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 betas=adam_betas, weight_decay=w_decay)
        
    # Train
    save_freq = 10
    N_epochs = 1000
    test_loss = []
    train_loss = []
    epoch_list = []

    print('Begin training')
    for t in range(N_epochs):
        train(train_dataloader, model, loss_fn, optimizer, device)
        
        t += start_epoch + 1
        time_check = t - start_epoch
        
        # Save every Nth epoch
        if (time_check)%save_freq==0:
            print(f"Epoch {t+1}\n-------------------------------")
            torch.save(model.state_dict(), 
                       '{}{}_epoch_{}.pth'.format(model_output_dir, 
                                                               model_name, t))
            l = aux.evaluate_loss(train_dataloader, model, loss_fn, device)
            train_loss.append(l)
            
            train_loss_outpath = data_output_dir + 'train_loss.txt'
            out_string = '{} {}\n'.format(t, train_loss[-1])
            try:
                with open(train_loss_outpath, 'a') as f:
                        f.write(out_string)
            except FileNotFoundError:
                with open(train_loss_outpath, 'w') as f:
                        f.write(out_string)
        
            test_loss.append(aux.evaluate_loss(test_dataloader, model, loss_fn, 
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
    
    
    # Plot train loss, test loss
    epoch_array = np.array(epoch_list)
    N_steps = len(train_dataloader)
    timestep_array = epoch_array*N_steps
    
    # Plot losses
    plot_dir = '{}plots/'.format(base_path)
    dir_exists = os.path.isdir(plot_dir)
    if not dir_exists:
        os.mkdir(plot_dir)
        
    plot_output_dir = '{}{}/'.format(plot_dir, model_name)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)

    fig, ax = plt.subplots()
    plt.plot(timestep_array, train_loss)
    plt.yscale('log')
    plt.title('Loss on train set')
    plt.xlabel('Time/steps')
    plt.savefig('{}{}_train_loss_log.pdf'.format(plot_output_dir, model_name), 
                bbox_inches='tight')

    fig, ax = plt.subplots()
    plt.plot(timestep_array, test_loss)
    plt.yscale('log')
    plt.title('Loss on test set')
    plt.xlabel('Time/steps')
    plt.savefig('{}{}_test_loss_log.pdf'.format(plot_output_dir, model_name), 
                bbox_inches='tight')
    
    # Plot an example from the test set
    noise_str = dataset[-2:]
    time_path_stem = base_path + 'data/F_to_K_data/noise_{}/'.format(noise_str)
    time_path = time_path_stem + 'times_for_interpolated_K.txt'
    times_for_plotting = np.loadtxt(time_path)
    
    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            break
    
    plt_idx = 0
    fig, ax = plt.subplots()
    plt.plot(times_for_plotting, y[plt_idx, :], label='Ground truth')
    plt.plot(times_for_plotting, pred[plt_idx, :], label='NN pred', 
             linestyle='dashed')
    plt.legend()
    plt.xscale('log')
    plt.title('An example from the test set')
    plt.ylabel('Memory kernel')
    plt.xlabel('Time/A.U.')
    plt.savefig('{}{}_example_{}_from_testset.pdf'.format(plot_output_dir, 
                model_name, plt_idx), bbox_inches='tight')
    
    print('Running time = ', time.time()-start)







