#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:41:43 2022

@author: mwinter
"""

# A script to load a trained network and use it to calculate a kernel from
# a correlation function

import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import pickle
from scipy.interpolate import interp1d
import pandas as pd

import auxiliary_functions as aux
from NN import NeuralNetwork
plt.close('all')

# Check whether this is running on my laptop
current_dir = os.getcwd()
if current_dir[:6] == '/Users':
    on_laptop = True
else:
    on_laptop = False

if on_laptop:
    base_path = './'
else:
    base_path = '../'
        
noise_str = '-2'
MCT_path = base_path + 'data/MCT_data/'
novel_MCT_path = base_path + 'data/novel_MCT_data/'
pkl_stem = base_path + 'data/'
F_to_M_path = base_path + 'data/F_to_M_data/noise_{}/'.format(noise_str)

# Get cpu or gpu device for training. 
if torch.cuda.is_available():
    device = "cuda"
    print('Using CUDA GPU.')
else:
    device = "cpu"
    print('Using CPU.')

# Load trained model
# model_name = 'linear_weight_param_search_ic0_reg0_01_b5000_w4'
# model_name = 'linear_weight_param_search_ic0_reg0_05_b5000_w8'
# model_name = 'balanced_data_param_search_ic0_reg0_001_b2500_w8' # Best test loss
# model_name = 'balanced_data_param_search_ic2_reg0_001_b2500_w8' # 2nd best test loss
# model_name = 'balanced_data_param_search_ic3_reg0_1_b10000_w4' # Best non-w8 test loss
# model_name = 'balanced_data_param_search_ic1_reg0_1_b2500_w8' # Best non reg0_001 test loss
model_name = 'F_final_data_param_search_ic4_reg0_001_b2500_w8'

model_path = base_path + 'models/{}/'.format(model_name)

params = aux.load_model_params('{}{}_model_params.json'.format(model_path, 
                                                               model_name))
model, epoch = aux.load_final_model(model_path, model_name, params, device)

# Load time grid of training set
use_MCT_F_curve = False
use_novel_MCT_F_curve = True
use_sim_F_curve = False
use_training_set = False
use_testing_set = False
use_text_testing_set = False

if use_novel_MCT_F_curve:
    MCT_path = novel_MCT_path

numbers = []
files = os.listdir(MCT_path)

for file in files:
    if file[:6] != 'F_PYHS':
        continue
    
    str_number = file[10:-15]
    number = float(str_number)
    numbers.append(number)

numbers.sort()

str_number = '{:.3f}'.format(numbers[0])
t_path = '{}time_PYHS_phi'.format(MCT_path)+str_number+'_Dt1.0_MF11.npy'
MCT_times = np.load(t_path)

if use_MCT_F_curve or use_novel_MCT_F_curve:
    input_data_type = 'MCT'
    
    # Load one set of files
    # number = numbers[0] # Liquid phase
    # number = numbers[-1] # Glass phase
    # number = numbers[25]
    # number = numbers[70]
    # number = numbers[100]
    number = numbers[-4] # Novel MCT curves, highest \phi value
    
    D_str = '1.0'
    str_number = '{:.3f}'.format(number)
    F_path = '{}F_PYHS_phi{}_Dt{}_MF11.npy'.format(MCT_path, str_number, D_str)
    k_path = '{}k_PYHS_phi{}_Dt{}_MF11.npy'.format(MCT_path, str_number, D_str)
    M_path = '{}M_PYHS_phi{}_Dt{}_MF11.npy'.format(MCT_path, str_number, D_str)
    S_path = '{}Sk_PYHS_phi{}_Dt{}_MF11.npy'.format(MCT_path, str_number,D_str)
    
    F = np.load(F_path)
    k = np.load(k_path)
    M = np.load(M_path)
    S = np.load(S_path)
    
    # Pick one wavenumber to study
    k_star_idx = np.argmax(F[0, :])
    k_star = k[k_star_idx]
    S_star = S[k_star_idx]
    F_star = F[:, k_star_idx]
    F_star /= S_star
    M_star = M[:, k_star_idx]
    
    # Add noise to F
    np.random.seed(1234) # Comment out for non-repeatable randomness
    max_sig = F_star.max() - F_star.min()
    noise_str = -2
    exponent = int(noise_str)
    noise_over_signal = 10**exponent
    # noise_over_signal = 0.0
    noise_strength = max_sig*noise_over_signal
    F_star += np.random.randn(len(F_star))*noise_strength
    
    F_interp = F_star.copy()
    t_interp = MCT_times
    sim_name = 'F_PYHS_phi{}_Dt{}_MF11.npy'.format(str_number, D_str)
    input_dir = MCT_path
    D = float(D_str)
    
if use_sim_F_curve:
    input_data_type = 'Simulation'
    # Load simulation data
    input_dir = base_path + 'data/Brownian_data/'
    T_str = '1.0'
    sim_name = 'clean_F_kt_T_{}.txt'.format(T_str)
    F_inpath = input_dir + sim_name
    t_inpath = input_dir + 'clean_t_array.txt'
    k_inpath = input_dir + 'clean_k_array.txt'
    
    F = np.loadtxt(F_inpath)
    t = np.loadtxt(t_inpath)
    k = np.loadtxt(k_inpath)
    
    k_star_idx = np.argmax(F[0, :])
    k_star = k[k_star_idx]
    S_star = F[0, k_star_idx]
    
    F_star = F[:, k_star_idx]
    F_star /= S_star
    
    # Interpolate input to have same number of elements as the training set 
    min_time = MCT_times[0]
    F_min_time = F_star[0] + min_time*(F_star[1]-F_star[0])/(t[1]-t[0])
    F_star[0] = F_min_time
    
    adjusted_t = t
    adjusted_t[0] = min_time
    
    log_t_min = np.log10(min_time)
    log_t_max = np.log10(t[-1])
    t_interp = np.logspace(log_t_min, log_t_max, len(MCT_times))
    t_interp[-1] = t[-1]
    
    F_interp_func = interp1d(adjusted_t, F_star, kind='cubic')
    F_interp = F_interp_func(t_interp)
    noise_str = 'NA'
    
    T = float(T_str)
    kB = 1.0
    m = 1.0
    D = kB*T/m

if use_training_set:
    input_data_type = 'training_set'
    input_dir = '{}models/{}/'.format(base_path, model_name)
    params = aux.load_model_params('{}{}'.format(input_dir, model_name) + 
                                   '_model_params.json')
    batch_size = params['batch_size']
    dataset = params['dataset']
        
    (train_dataloader, 
     test_dataloader, 
     training_data, 
     test_data, 
     N_inputs, 
     N_outputs) = aux.load_dataset(dataset, batch_size, base_path)
    
    # Get an example from the training set
    idx = 0
    data_tuple = training_data[idx]
    input_data = data_tuple[0]
    true_K = data_tuple[1]
    
    sim_name = 'training_set_idx_{}.xyz'.format(idx)

if use_testing_set:
    input_data_type = 'testing_set'
    input_dir = '{}models/{}/'.format(base_path, model_name)
    params = aux.load_model_params('{}{}'.format(input_dir, model_name) + 
                                   '_model_params.json')
    batch_size = params['batch_size']
    dataset = params['dataset']
        
    (train_dataloader, 
     test_dataloader, 
     training_data, 
     test_data, 
     N_inputs, 
     N_outputs) = aux.load_dataset(dataset, batch_size, base_path)
    
    # Get an example from the training set
    idx = 6
    data_tuple = test_data[idx]
    input_data = data_tuple[0]
    true_K = data_tuple[1]
    
    sim_name = 'testing_set_idx_{}.xyz'.format(idx)
    input_data2 = input_data.clone()

if use_text_testing_set:
    input_data_type = 'text_testing_set'
    data_inpath = '{}data/F_to_M_data/noise_{}/'.format(base_path, noise_str)
    F_test_df = pd.read_csv(data_inpath+'F_noisy_test.csv', header=None)
    Omega_test_df = pd.read_csv(data_inpath+'Omega_noisy_test.csv', header=None)
    F_final_test_df = pd.read_csv(data_inpath+'F_final_noisy_test.csv', header=None)
    M_test_df = pd.read_csv(data_inpath+'M_noisy_test.csv', header=None)

    idx = 1
    F_star = F_test_df.loc[idx, :]
    Omega = Omega_test_df.loc[idx, :]
    F_final = F_final_test_df.loc[idx, :]
    true_K = M_test_df.loc[idx, :]
    
    F_interp = np.array(F_star.copy())
    t_interp = MCT_times
    sim_name = 'text_testing_set_idx_{}.xyz'.format(idx)



if use_MCT_F_curve or use_novel_MCT_F_curve or use_sim_F_curve or use_text_testing_set:
    if len(F_interp) != 4352:
        raise ValueError('Input data length {}, not 4352'.format(len(F_interp)))
    
    # Use PCA basis from training set to reduce dimension of F_star
    pca_path = pkl_stem + 'F_to_M_data/noise_-2/pca_object.pkl'
    order = 15
    
    with open(pca_path, 'rb') as infile:
        pca_obj = pickle.load(infile)
    
    F_reshaped = F_interp.reshape(1, -1)
    pca_F = pca_obj.transform(F_reshaped)
    pca_F = pca_F[:, :order]
    
    if use_MCT_F_curve or use_novel_MCT_F_curve or use_sim_F_curve:
        # Add Omega to feature list
        Omega_orig = D*k_star**2/S_star
        Omega_inpath = F_to_M_path+'min_max_Omegas.csv'
        min_max = np.loadtxt(Omega_inpath)
        min_O = min_max[0]
        max_O = min_max[1]
        
        if max_O != min_O:
            Omega = (Omega_orig - min_O)/(max_O - min_O)
        
        # Add F_final to feature list
        F_final = F_interp[-1]/F_interp[0]
    
    input_data = np.zeros((pca_F.shape[0], order+2))
    input_data[:, :-2] = pca_F
    input_data[:, -2] = Omega 
    input_data[:, -1] = F_final

# Make a prediction of the memory kernel
model.eval()
with torch.no_grad():
    input_data = torch.Tensor(input_data)
    input_data = input_data.to(device)
    pred = model(input_data).detach().numpy()[0, :]

# Calculate kernel time grid (same downsampling as generate_F_to_K)
log_min_t = np.log10(min(t_interp))
log_max_t = np.log10(max(t_interp))
dsampled_times = np.logspace(log_min_t, log_max_t, 100)

# Plot prediction
plot_st = base_path + 'plots/extracted_kernels/'
if input_data_type == 'MCT':
    plot_output_dir = plot_st + 'MCT/noise_{}/'.format(noise_str)
elif input_data_type == 'training_set':
    plot_output_dir = plot_st + 'training_set/noise_{}/'.format(noise_str)
elif input_data_type == 'testing_set':
    plot_output_dir = plot_st + 'testing_set/noise_{}/'.format(noise_str)
elif input_data_type == 'text_testing_set':
    plot_output_dir = plot_st + 'text_testing_set/noise_{}/'.format(noise_str)
else:    
    plot_output_dir = plot_st + 'Brownian_data/'
    
dir_exists = os.path.isdir(plot_output_dir)
if not dir_exists:
    os.mkdir(plot_output_dir)

K_output_name = 'kernel_model_{}_sim_{}'.format(model_name, sim_name[:-4])
F_output_name = 'F_model_{}_sim_{}'.format(model_name, sim_name[:-4])

# Plot F
fig, ax = plt.subplots()
plt.plot(t_interp, F_interp)
plt.title('F(k*, t) from {}'.format(input_data_type))
plt.xscale('log')
plt.savefig('{}{}.pdf'.format(plot_output_dir, F_output_name), bbox_inches='tight')
    
# Plot K
fig, ax = plt.subplots()
plt.plot(dsampled_times, pred, label='Net', marker='o')
if input_data_type == 'MCT':
    plt.plot(MCT_times, M_star, label='MCT', color='r')  
if input_data_type in ['training_set', 'testing_set', 'text_testing_set']:
    plt.plot(dsampled_times, true_K, label='Ground truth', color='r')
plt.xscale('log')
# plt.title('Kernel extracted from {}'.format(sim_name))
plt.ylabel(r'$K$')
plt.xlabel(r'$t$')
plt.legend()
ytop, ybottom = ax.get_ylim()
plt.ylim(bottom=-20)
if ytop>1e4:
    plt.ylim(top=500)
plt.savefig('{}{}.pdf'.format(plot_output_dir, K_output_name), bbox_inches='tight')


# Save predicted kernel
name_stub = 'model_{}_sim_{}_noise_{}.txt'.format(model_name, sim_name[:-4], noise_str)
output_dir = base_path + 'extracted_kernels/'
kernel_outpath = output_dir+'kernel_{}'.format(name_stub)
time_outpath = output_dir+'times_for_kernel_{}'.format(name_stub)
k_outpath = output_dir+'wavenumber_for_kernel_{}'.format(name_stub)

np.savetxt(kernel_outpath, pred)
np.savetxt(time_outpath, dsampled_times) 
np.savetxt(k_outpath, np.array([k_star]))













