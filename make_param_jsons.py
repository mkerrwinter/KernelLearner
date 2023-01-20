#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:36:31 2022

@author: mwinter
"""

# A script to make parameter jsons

import os
import json
import numpy as np

base_path = './'

# Coarse L2 grid search
# regs = [0.0, 1e-6, 1e-3, 1e-1, 1, 1e2]
# reg_strings = ['0', 'e-6', 'e-3', 'e-1', 'e0', 'e2']

# for reg_count, reg in enumerate(regs):
#     reg_string = reg_strings[reg_count]
    
#     model_params = {}
#     model_params['batch_size'] = 1024
#     model_params['dataset'] = 'F_to_M_-2'
#     model_params['loss_function'] = 'weighted_MSELoss'
#     model_params['learning_rate'] = 1e-3
#     model_params['L2_penalty'] = reg
#     model_params['N_inputs'] = 10
#     model_params['N_outputs'] = 89
#     model_params['dropout_p'] = 0.5
    
#     model_params['h_layer_widths'] = [50, 100, 150, 200, 250, 300]
    
#     model_name = 'L2_grid_search_{}'.format(reg_string)
    
#     outpath = base_path + 'models/{}/{}_model_params.json'.format(model_name, 
#                                                                     model_name)
    
#     output_dir = '{}models/{}/'.format(base_path, model_name)
#     dir_exists = os.path.isdir(output_dir)
#     if not dir_exists:
#         os.mkdir(output_dir)
    
#     with open(outpath, 'w') as outfile:
#         json.dump(model_params, outfile)
        

# Hyper parameter search
# ic_for_printing = []
# reg_for_printing = []
# b_for_printing = []
# w_for_printing = []
# names_for_printing = []

# regs = [0.1, 0.05, 0.01, 0.001]
# reg_strings = ['0_1', '0_05', '0_01', '0_001']
# batches = [300, 2500, 10000]
# width_factors = [2, 4, 8]
# ics = np.arange(5)

# for reg_count, reg in enumerate(regs):
#     reg_str = reg_strings[reg_count]
#     for batch in batches:
#         for width_fac in width_factors:
#             for ic in ics:
#                 model_params = {}
#                 model_params['batch_size'] = batch
#                 model_params['dataset'] = 'F_to_M_-2'
#                 model_params['loss_function'] = 'weighted_MSELoss'
#                 model_params['learning_rate'] = 1e-3
#                 model_params['L2_penalty'] = reg
#                 model_params['N_inputs'] = 11
#                 model_params['N_outputs'] = 100
#                 model_params['dropout_p'] = 0.5
                
#                 base_w = 50*width_fac            
#                 widths = [base_w, base_w*2, base_w*3, base_w*4, base_w*5, base_w*6]
#                 model_params['h_layer_widths'] = widths
                
#                 model_name = 'balanced_data_param_search_ic{}_reg{}_b{}_w{}'.format(ic, reg_str, batch, width_fac)
                
#                 outpath = base_path + 'models/{}/{}_model_params.json'.format(model_name, 
#                                                                                 model_name)
                
#                 output_dir = '{}models/{}/'.format(base_path, model_name)
#                 dir_exists = os.path.isdir(output_dir)
#                 if not dir_exists:
#                     os.mkdir(output_dir)
                
#                 with open(outpath, 'w') as outfile:
#                     json.dump(model_params, outfile)  
                
#                 ic_for_printing.append(ic)
#                 reg_for_printing.append(reg_str)
#                 b_for_printing.append(batch)
#                 w_for_printing.append(width_fac)
#                 names_for_printing.append(model_name)

# ic_string = '('
# for ic in ic_for_printing:
#     ic_str = str(ic)
#     ic_string += ic_str
#     ic_string += ' '

# ic_string = ic_string[:-1]
# ic_string += ')'

# reg_string = '('
# for reg in reg_for_printing:
#     reg_str = str(reg)
#     reg_string += reg_str
#     reg_string += ' '

# reg_string = reg_string[:-1]
# reg_string += ')'

# batch_string = '('
# for b in b_for_printing:
#     batch_str = str(b)
#     batch_string += batch_str
#     batch_string += ' '

# batch_string = batch_string[:-1]
# batch_string += ')'

# width_string = '('
# for width in w_for_printing:
#     width_str = str(width)
#     width_string += width_str
#     width_string += ' '

# width_string = width_string[:-1]
# width_string += ')'

# name_string = '('
# for name in names_for_printing:
#     name_str = str(name)
#     name_string += name_str
#     name_string += ' '

# name_string = name_string[:-1]
# name_string += ')'




# # MaxEnt test network
# model_params = {}
# model_params['batch_size'] = 1000
# model_params['dataset'] = 'F_to_M_-2'
# model_params['loss_function'] = 'MaxEnt'
# model_params['MaxEnt_alpha'] = 1e-2
# model_params['learning_rate'] = 1e-5
# model_params['L2_penalty'] = 0.0
# model_params['N_inputs'] = 10
# model_params['N_outputs'] = 89
# model_params['dropout_p'] = 0.5

# model_params['h_layer_widths'] = [50, 100, 150, 200, 250, 300]

# model_name = 'MaxEnt_testnet'

# outpath = base_path + 'models/{}/{}_model_params.json'.format(model_name, 
#                                                                 model_name)

# output_dir = '{}models/{}/'.format(base_path, model_name)
# dir_exists = os.path.isdir(output_dir)
# if not dir_exists:
#     os.mkdir(output_dir)

# with open(outpath, 'w') as outfile:
#     json.dump(model_params, outfile)



# Hyper parameter search
ic_for_printing = []
reg_for_printing = []
b_for_printing = []
w_for_printing = []
names_for_printing = []

regs = [0.1, 0.05, 0.01, 0.001, 0]
reg_strings = ['0_1', '0_05', '0_01', '0_001', '0']
batches = [300, 2500, 10000]
width_factors = [2, 4, 8]
ics = np.arange(5)

for reg_count, reg in enumerate(regs):
    reg_str = reg_strings[reg_count]
    for batch in batches:
        for width_fac in width_factors:
            for ic in ics:
                
                if reg!=0:
                    continue
                
                model_params = {}
                model_params['batch_size'] = batch
                model_params['dataset'] = 'F_to_M_-2'
                model_params['loss_function'] = 'weighted_MSELoss'
                model_params['learning_rate'] = 1e-3
                model_params['L2_penalty'] = reg
                model_params['N_inputs'] = 17
                model_params['N_outputs'] = 100
                model_params['dropout_p'] = 0.5
                
                base_w = 50*width_fac            
                widths = [base_w, base_w*2, base_w*3, base_w*4, base_w*5, base_w*6]
                model_params['h_layer_widths'] = widths
                
                model_name = 'F_final_data_param_search_ic{}_reg{}_b{}_w{}'.format(ic, reg_str, batch, width_fac)
                
                outpath = base_path + 'models/{}/{}_model_params.json'.format(model_name, 
                                                                                model_name)
                
                output_dir = '{}models/{}/'.format(base_path, model_name)
                dir_exists = os.path.isdir(output_dir)
                if not dir_exists:
                    os.mkdir(output_dir)
                
                with open(outpath, 'w') as outfile:
                    json.dump(model_params, outfile)  
                
                ic_for_printing.append(ic)
                reg_for_printing.append(reg_str)
                b_for_printing.append(batch)
                w_for_printing.append(width_fac)
                names_for_printing.append(model_name)

ic_string = '('
for ic in ic_for_printing:
    ic_str = str(ic)
    ic_string += ic_str
    ic_string += ' '

ic_string = ic_string[:-1]
ic_string += ')'

reg_string = '('
for reg in reg_for_printing:
    reg_str = str(reg)
    reg_string += reg_str
    reg_string += ' '

reg_string = reg_string[:-1]
reg_string += ')'

batch_string = '('
for b in b_for_printing:
    batch_str = str(b)
    batch_string += batch_str
    batch_string += ' '

batch_string = batch_string[:-1]
batch_string += ')'

width_string = '('
for width in w_for_printing:
    width_str = str(width)
    width_string += width_str
    width_string += ' '

width_string = width_string[:-1]
width_string += ')'

name_string = '('
for name in names_for_printing:
    name_str = str(name)
    name_string += name_str
    name_string += ' '

name_string = name_string[:-1]
name_string += ')'
    
    
    







