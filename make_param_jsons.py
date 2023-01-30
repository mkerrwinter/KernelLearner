#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:36:31 2023

@author: Max Kerr Winter

A script to make parameter files that define the architecture and 
hyperparameters of neural networks. The boolean variable hyperparam_search
determines whether multiple parameter files are produced across a range of 
hyperparameters, or whether a single parameter file is produced for testing
purposes.
"""



import os
import json
import numpy as np

base_path = './'
hyperparam_search = False

model_dir = '{}models/'.format(base_path)
dir_exists = os.path.isdir(model_dir)
if not dir_exists:
    os.mkdir(model_dir)

if hyperparam_search:    
    # L2 regularisation term \lambda
    regs = [0.1, 0.05, 0.01, 0.001, 0]
    reg_strings = ['0_1', '0_05', '0_01', '0_001', '0']
    
    # Batch size
    batches = [300, 2500, 10000]
    
    # Width factor, used in defining width of a triangular network
    width_factors = [2, 4, 8]
    
    # A counter for different weight initial conditions
    ics = np.arange(5)
    
    for reg_count, reg in enumerate(regs):
        reg_str = reg_strings[reg_count]
        for batch in batches:
            for width_fac in width_factors:
                for ic in ics:
                   
                    model_params = {}
                    model_params['batch_size'] = batch
                    model_params['dataset'] = 'F_to_K_-2'
                    model_params['loss_function'] = 'weighted_MSELoss'
                    model_params['learning_rate'] = 1e-3
                    model_params['L2_penalty'] = reg
                    model_params['N_inputs'] = 17
                    model_params['N_outputs'] = 100
                    model_params['dropout_p'] = 0.5
                    
                    base_w = 50*width_fac            
                    widths = [base_w, 
                              base_w*2, 
                              base_w*3, 
                              base_w*4, 
                              base_w*5, 
                              base_w*6]
                    
                    model_params['h_layer_widths'] = widths
                    
                    model_name = 'F_to_K_ic{}_reg{}_b{}_w{}'.format(ic, 
                                                                    reg_str, 
                                                                    batch, 
                                                                    width_fac)
                    
                    outpath = '{}models/{}/{}_params.json'.format(base_path, 
                                                                  model_name, 
                                                                  model_name)
                    
                    output_dir = '{}models/{}/'.format(base_path, model_name)
                    dir_exists = os.path.isdir(output_dir)
                    if not dir_exists:
                        os.mkdir(output_dir)
                    
                    outpath = '{}{}_params.json'.format(output_dir, model_name)
                    
                    with open(outpath, 'w') as outfile:
                        json.dump(model_params, outfile)  

else:
    # A single network for testing purposes
    model_params = {}
    model_params['batch_size'] = 1000
    model_params['dataset'] = 'F_to_K_-2'
    model_params['loss_function'] = 'weighted_MSELoss'
    model_params['learning_rate'] = 1e-3
    model_params['L2_penalty'] = 1e-2
    model_params['N_inputs'] = 17
    model_params['N_outputs'] = 100
    model_params['dropout_p'] = 0.5
    
    widths = [20, 40, 60, 80, 100, 120]
    model_params['h_layer_widths'] = widths
    
    model_name = 'F_to_K_test_model'
    outpath = '{}models/{}/{}_params.json'.format(base_path, model_name, 
                                                  model_name)                    
    output_dir = '{}models/{}/'.format(base_path, model_name)
    dir_exists = os.path.isdir(output_dir)
    if not dir_exists:
        os.mkdir(output_dir)
    
    outpath = '{}{}_params.json'.format(output_dir, model_name)
    
    with open(outpath, 'w') as outfile:
        json.dump(model_params, outfile) 
    
    







