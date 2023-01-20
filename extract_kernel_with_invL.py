#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:16:19 2022

@author: mwinter
"""

# A script to calculate a kernel from a correlation function using a direct
# inverse Laplace transform

import numpy as np
import os
from matplotlib import pyplot as plt
from invlap import invlap
from scipy.signal import savgol_filter
import time
import pandas as pd

import auxiliary_functions as aux

plt.close('all')
start = time.time()


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
# noise_str = '-5'
# noise_str = '-8'
# noise_str = '0'
MCT_path = base_path + 'data/MCT_data/'
F_to_M_path = base_path + 'data/F_to_M_data/noise_{}/'.format(noise_str)

# get a single file to start with
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

use_MCT_F_curve = True
use_sim_F_curve = False

if use_MCT_F_curve:
    input_data_type = 'MCT'
    
    # Load one set of files
    # number = numbers[0] # Liquid phase
    # number = numbers[-1] # Glass phase
    number = numbers[25]
    # number = numbers[70]
    # number = numbers[100]
    
    str_number = '{:.3f}'.format(number)
    F_path = '{}F_PYHS_phi'.format(MCT_path)+str_number+'_Dt1.0_MF11.npy'
    k_path = '{}k_PYHS_phi'.format(MCT_path)+str_number+'_Dt1.0_MF11.npy'
    M_path = '{}M_PYHS_phi'.format(MCT_path)+str_number+'_Dt1.0_MF11.npy'
    S_path = '{}Sk_PYHS_phi'.format(MCT_path)+str_number+'_Dt1.0_MF11.npy'
    
    F = np.load(F_path)
    k = np.load(k_path)
    M = np.load(M_path)
    S = np.load(S_path)
    
    # Pick one wavenumber to study
    k_star_idx = np.argmax(F[0, :])
    k_star = k[k_star_idx]
    
    F_star = F[:, k_star_idx]
    F_star /= S[k_star_idx]
    
    M_star = M[:, k_star_idx]
    
    # Add noise to F
    # np.random.seed(1234) # Comment out for non-repeatable randomness
    max_sig = F_star.max() - F_star.min()
    if noise_str=='0':
        noise_over_signal = 0.0
    else:
        exponent = int(noise_str)
        noise_over_signal = 10**exponent
    # noise_over_signal = 0.0
    noise_strength = max_sig*noise_over_signal
    F_star += np.random.randn(len(F_star))*noise_strength
    
    times = MCT_times
    sim_name = 'F_PYHS_phi'+str_number+'_Dt1.0_MF11.npy'
    input_dir = MCT_path
    kT = 1.0
    
if use_sim_F_curve:
    input_data_type = 'Simulation'
    # Load simulation data
    input_dir = base_path + 'data/Brownian_data/'
    kT_str = '1.5'
    sim_name = 'F_kt_T_{}.txt'.format(kT_str)
    F_inpath = input_dir + sim_name
    t_inpath = input_dir + 't_array.txt'
    k_inpath = input_dir + 'k_array.txt'
    
    F = np.loadtxt(F_inpath)
    k = np.loadtxt(k_inpath)
    times = np.loadtxt(t_inpath)
    S = F[0, :]
    
    k_star_idx = np.argmax(S)
    k_star = k[k_star_idx]
    
    F_star = F[:, k_star_idx]
    F_star /= F_star[0]
    kT = float(kT_str)
    noise_str = 'NA'

# Calculate derivative
F_prime_array = np.diff(F_star)/np.diff(times)

# Smooth with savgol. Factor of 8 chosen to give good results with both clean 
# and noisy signal
window = int(len(F_star)/8)
if window%2==0:
    window += 1
savgol_F = savgol_filter(F_star, window, 3)
savgol_F_prime_array = np.diff(savgol_F)/np.diff(times)

# Do inverse transform
degree = 9 # Selected by trial and error
NP = invlap.deHoog()

calculated_K = []
calculated_SG_K = []
N_t_values = len(times)
de_Hoog_min_time = times[1]/10.0
de_Hoog_times = times.copy()
de_Hoog_times[0] = de_Hoog_min_time
F_0 = S[k_star_idx]
m = 1.0

for idx, t in enumerate(de_Hoog_times):
    print('Inversion step {} of {}'.format(idx, N_t_values))
    # Calculate points on complex plane that will be used for inverse Laplace.
    # They are stored in NP.p
    NP.calc_laplace_parameter(t, degree=degree)
    
    # First transform raw data curve with no smoothing
    # Forward transform F and F' on the points NP.p
    Laplace_F = aux.numerical_Laplace_transform(times, F_star, NP.p, 
                                                trapz_method=True)
    F_prime_array = np.diff(F_star)/np.diff(times)
    Laplace_F_prime = aux.numerical_Laplace_transform(times[1:], F_prime_array, 
                                                      NP.p, trapz_method=True)
    A = (kT * k_star**2)/(m*F_0)
    numerator = -Laplace_F_prime-A*Laplace_F
    Laplace_K = numerator/Laplace_F_prime
    
    # Do inverse Laplace_K for particular timepoint t
    calculated_K.append(NP.calc_time_domain_solution(Laplace_K, t))
    
    
    # Now transform smoothed F
    Laplace_SG_F = aux.numerical_Laplace_transform(times, savgol_F, NP.p, 
                                                trapz_method=True)
    SG_F_prime_array = np.diff(savgol_F)/np.diff(times)
    Laplace_SG_F_prime = aux.numerical_Laplace_transform(times[1:], 
                                                      SG_F_prime_array, 
                                                      NP.p, trapz_method=True)
    SG_F_0 = F_0
    A = (kT * k_star**2)/(m*SG_F_0)
    better_numerator = -Laplace_SG_F_prime-A*Laplace_SG_F
    Laplace_SG_K = better_numerator/Laplace_SG_F_prime
    
    # Do inverse Laplace_K for particular timepoint t
    calculated_SG_K.append(NP.calc_time_domain_solution(Laplace_SG_K, t))


# Plot prediction
plot_st = base_path + 'plots/extracted_kernels/'
if input_data_type == 'MCT':
    plot_output_dir = plot_st + 'MCT/noise_{}/'.format(noise_str)
else:    
    plot_output_dir = plot_st + 'Brownian_data/'
    
dir_exists = os.path.isdir(plot_output_dir)
if not dir_exists:
    os.mkdir(plot_output_dir)

model_name = 'De_Hoog'
K_output_name = 'kernel_model_{}_sim_{}'.format(model_name, sim_name[:-4])
F_output_name = 'F_model_{}_sim_{}'.format(model_name, sim_name[:-4])

# Plot F
fig, ax = plt.subplots()
plt.plot(times, F_star)
plt.plot(times, savgol_F, label='SG filter', color='orange', linestyle='dashed')
plt.legend()
plt.xscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$F$')
# plt.ylim(bottom=0.86) # Glass phase
# plt.title('Intermediate scattering function')
filename = '{}{}.pdf'.format(plot_output_dir, F_output_name)
plt.savefig(filename, bbox_inches='tight')

# Plot K
fig, ax = plt.subplots()
if use_MCT_F_curve:
    plt.plot(de_Hoog_times, calculated_SG_K, label='SG De Hoog', color='orange', alpha=0.5)
    plt.plot(de_Hoog_times, calculated_K, label='De Hoog', linestyle='dashed', color='green')
    plt.plot(times, M_star, label='MCT', color='r', linestyle='dotted')
else:
    plt.plot(de_Hoog_times, calculated_SG_K, label='Smooth')
    plt.plot(de_Hoog_times, calculated_K, label='Raw', linestyle='dashed')
    
plt.xscale('log')
plt.xlabel(r'$t$')
plt.ylabel(r'$K$')
plt.legend()
# plt.title('Memory kernel, MCT vs calculated')
ybottom, ytop = ax.get_ylim()
if ytop>1e4:
    # plt.ylim([-20, 1800])
    # plt.ylim([-20.0, 949.2634547488931]) # Glass phase
    plt.ylim([-20.0, 592.9518884450197]) # Liquid phase
plt.xlim(left=10**(-6))
ax.set_xticks([10**(-3), 10**(1), 10**5])
filename = '{}{}.pdf'.format(plot_output_dir, K_output_name)
plt.savefig(filename, bbox_inches='tight')

# Save predicted kernel
name_stub = 'model_{}_sim_{}_noise_{}.txt'.format(model_name, sim_name[:-4], noise_str)
output_dir = base_path + 'extracted_kernels/'
kernel_outpath = output_dir+'kernel_{}'.format(name_stub)
time_outpath = output_dir+'times_for_kernel_{}'.format(name_stub)
k_outpath = output_dir+'wavenumber_for_kernel_{}'.format(name_stub)

np.savetxt(kernel_outpath, calculated_SG_K)
np.savetxt(time_outpath, de_Hoog_times)
np.savetxt(k_outpath, np.array([k_star]))

print('Running time = ', time.time()-start)








