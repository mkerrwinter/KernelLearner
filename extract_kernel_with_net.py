#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:22:14 2023

@author: Max Kerr Winter

A script to load a trained network and use it to calculate a kernel from
a correlation function.
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import pickle
from scipy.interpolate import interp1d

import auxiliary_functions as aux
from NN import NeuralNetwork
plt.close('all')

base_path = './'      
noise_str = '-2'
pkl_stem = base_path + 'data/'
F_to_K_path = base_path + 'data/F_to_K_data/noise_{}/'.format(noise_str)

if torch.cuda.is_available():
    device = "cuda"
    print('Using CUDA GPU.')
else:
    device = "cpu"
    print('Using CPU.')

# Load best trained model
name_stub = 'F_to_K_test_model'
model_name, best_epoch = aux.find_best_model(name_stub, base_path, device)

model_path = base_path + 'models/{}/'.format(model_name)
params = aux.load_model_params('{}{}_params.json'.format(model_path, 
                                                               model_name))
models, epochs = aux.load_models(model_path, model_name, params, device)
model_idx = np.where(np.array(epochs)==best_epoch)[0][0]
model = models[model_idx]
epoch = epochs[model_idx]

# Load time grid of training set and kernel
GLE_time_path = '{}/data/example_GLE_data/t_GLE_solution.txt'.format(base_path)
GLE_times = np.loadtxt(GLE_time_path)

K_time_stub = '{}/data/F_to_K_data/noise_{}/'.format(base_path, noise_str)
K_time_path = '{}times_for_interpolated_K.txt'.format(K_time_stub)
K_times = np.loadtxt(K_time_path)

# Load correlation function
input_data_type = 'Simulation'
input_dir = base_path + 'data/simulation_data/'
sim_name = 'F.txt'
F_inpath = input_dir + sim_name
Omega_inpath = input_dir + 'Omega.txt'
t_inpath = input_dir + 't_array.txt'

F = np.loadtxt(F_inpath)
F /= F[0]
Omega_original = float(np.loadtxt(Omega_inpath))
t = np.loadtxt(t_inpath)

# Interpolate input to have same number of elements as the training set, 
# assuming both t and GLE_times start at t=0, and GLE_times[1]<t[1]
F_orig = F.copy()
GLE_times_no_zero = GLE_times[GLE_times>0]
min_time = GLE_times_no_zero[0]
F_min_time = F[0] + min_time*(F[1]-F[0])/(t[1]-t[0])
F[0] = F_min_time
F = np.concatenate((np.array([F_orig[0]]), F))

adjusted_t = t.copy()
adjusted_t[0] = min_time
adjusted_t = np.concatenate((np.array([0.0]), adjusted_t))

log_t_min = np.log10(min_time)
log_t_max = np.log10(t[-1])
t_interp = np.logspace(log_t_min, log_t_max, len(GLE_times)-1)
t_interp[-1] = t[-1]
t_interp = np.concatenate((np.array([0.0]), t_interp))

F_interp_func = interp1d(adjusted_t, F, kind='cubic')
F_interp = F_interp_func(t_interp)

# Use PCA basis from training set to reduce dimension of F
pca_path = pkl_stem + 'F_to_K_data/noise_-2/pca_object.pkl'
order = 15

with open(pca_path, 'rb') as infile:
    pca_obj = pickle.load(infile)

F_reshaped = F_interp.reshape(1, -1)
pca_F = pca_obj.transform(F_reshaped)
pca_F = pca_F[:, :order]

# Add Omega to feature list
Omega_inpath = F_to_K_path+'min_max_Omegas.txt'
min_max = np.loadtxt(Omega_inpath)
min_O = min_max[0]
max_O = min_max[1]

if max_O != min_O:
    Omega = (Omega_original - min_O)/(max_O - min_O)

# Add F_final to feature list
F_final = F_interp[-1]
    
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

# Plot prediction
plot_stub = base_path + 'plots/extracted_kernels/'
plot_output_dir = plot_stub + 'simulation_data/'

parent_dir_exists = os.path.isdir(plot_stub)
if not parent_dir_exists:
    os.mkdir(plot_stub)

dir_exists = os.path.isdir(plot_output_dir)
if not dir_exists:
    os.mkdir(plot_output_dir)

K_output_name = 'kernel_{}_sim_{}'.format(model_name, sim_name[:-4])
F_output_name = 'F_{}_sim_{}'.format(model_name, sim_name[:-4])

# Plot F
fig, ax = plt.subplots()
plt.plot(t_interp, F_interp)
plt.title('F from {}'.format(input_data_type))
plt.xscale('log')
plt.savefig('{}{}.pdf'.format(plot_output_dir, F_output_name), 
            bbox_inches='tight')
    
# Plot K
fig, ax = plt.subplots()
plt.plot(K_times, pred) 
plt.xscale('log')
plt.title('Kernel extracted from {}'.format(sim_name))
plt.ylabel(r'$K$')
plt.xlabel(r'$t$')
plt.savefig('{}{}.pdf'.format(plot_output_dir, K_output_name), 
            bbox_inches='tight')


# Save predicted kernel
name_stub = 'model_{}_sim_{}_noise_{}.txt'.format(model_name, sim_name[:-4], 
                                                  noise_str)
output_dir = base_path + 'extracted_kernels/'
dir_exists = os.path.isdir(output_dir)
if not dir_exists:
    os.mkdir(output_dir)

kernel_outpath = output_dir+'kernel_{}'.format(name_stub)
time_outpath = output_dir+'times_for_kernel_{}'.format(name_stub)

np.savetxt(kernel_outpath, pred)
np.savetxt(time_outpath, K_times) 













