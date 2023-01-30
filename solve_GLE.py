#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:24:19 2023

@author: Max Kerr Winter

A script to solve the overdamped GLE using the julia package of I. Pihlajamaa
and a user provided kernel (e.g. one extracted from data by a network).
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

start = time.time()

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from julia import Pkg
Pkg.add("ModeCouplingTheory")
from julia import ModeCouplingTheory as mct

base_path = './'

# Load original F curve
input_data_type = 'Simulation'
input_dir = base_path + 'data/simulation_data/'
sim_name = 'F.txt'
F_inpath = input_dir + sim_name
Omega_inpath = input_dir + 'Omega.txt'
t_inpath = input_dir + 't_array.txt'
noise_str = '-2'

sim_F = np.loadtxt(F_inpath)
S_kstar = sim_F[0]
sim_F /= sim_F[0]
Omega = float(np.loadtxt(Omega_inpath))
sim_t = np.loadtxt(t_inpath)


# Load kernel extracted by a particular model
model_name = 'F_to_K_test_model'
name_stub = 'model_{}_sim_{}_noise_{}.txt'.format(model_name, sim_name[:-4], 
                                                  noise_str)
input_dir = base_path + 'extracted_kernels/'
kernel_inpath = input_dir+'kernel_{}'.format(name_stub)
time_inpath = input_dir+'times_for_kernel_{}'.format(name_stub)

net_M = np.loadtxt(kernel_inpath)
net_t = np.loadtxt(time_inpath)


# Define system parameters and initial conditions 
m = 1.0
a = 0.0
b = 1.0
F0 = 1.0
dF0 = 0.0
c = Omega

# Solve GLE
kernel = mct.InterpolatingKernel(net_t, net_M)
equation = mct.LinearMCTEquation(a, b, c, F0, dF0, kernel)
solver = mct.FuchsSolver(N=128, Δt=10**-5, t_max=10.0**5, max_iterations=10**8, 
                         tolerance=10**-6, verbose=False)
t, F, K = mct.solve(equation, solver)

# Plot results
plot_stub = '{}plots/extracted_F/'.format(base_path)
parent_dir_exists = os.path.isdir(plot_stub)
if not parent_dir_exists:
    os.mkdir(plot_stub)
    
plot_output_dir = '{}simulation_data/'.format(plot_stub)    
dir_exists = os.path.isdir(plot_output_dir)
if not dir_exists:
    os.mkdir(plot_output_dir)

fig, ax = plt.subplots()
plt.plot(sim_t, sim_F, label='Original')
plt.plot(t, F, linestyle='dashed', label='Net GLE')
plt.legend()
plt.xscale('log')
plt.title('F vs time')
F_name = 'F_{}_sim_{}.pdf'.format(model_name, sim_name[:-4])
filename = '{}{}'.format(plot_output_dir, F_name)
plt.savefig(filename, bbox_inches='tight')

print('Running time = ', time.time()-start)


