#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 08:19:31 2023

@author: mwinter
"""

# A script to generate training data by solving the overdamped GLE for a set
# of memory kernels

import numpy as np
import time
import os

start = time.time()

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from julia import Pkg
Pkg.add("ModeCouplingTheory")
from julia import ModeCouplingTheory as mct

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
    
# Load from file: kernel, initial conditions, Omega parameter
input_dir = '{}/data/example_1D_kernels/'.format(base_path)
output_dir = '{}/data/example_GLE_data/'.format(base_path)

dir_exists = os.path.isdir(output_dir)

if not dir_exists:
    os.mkdir(output_dir)

numbers = []
files = os.listdir(input_dir)

for file in files:
    if file[:9] != 'K_example':
        continue
    
    str_number = file[10:-4]
    number = int(str_number)
    numbers.append(number)

numbers.sort()

for number in numbers:
    K_inpath = '{}K_example_{}.txt'.format(input_dir, number)
    F0_inpath = '{}S_example_{}.txt'.format(input_dir, number)
    dF0_inpath = '{}dF0_example_{}.txt'.format(input_dir, number)
    Omega_inpath = '{}Omega_example_{}.txt'.format(input_dir, number)
    time_inpath = '{}t_example_{}.txt'.format(input_dir, number)
    
    K = np.loadtxt(K_inpath)
    F0 = float(np.loadtxt(F0_inpath))
    dF0 = float(np.loadtxt(dF0_inpath))
    Omega = float(np.loadtxt(Omega_inpath))
    times = np.loadtxt(time_inpath)
    
    print('Solving GLE with kernel K_example_{}.txt'.format(number))
    
    # Define system parameters for equation ay''+by'+cy+\int(Ky')d\tau = 0
    a = 0.0
    b = 1.0
    c = Omega
        
    # Solve GLE
    kernel = mct.InterpolatingKernel(times, K)
    equation = mct.LinearMCTEquation(a, b, c, F0, dF0, kernel)
    solver = mct.FuchsSolver(N=128, Δt=10**-5, t_max=10.0**5, max_iterations=10**8, 
                              tolerance=10**-6, verbose=False)
    t, F, K = mct.solve(equation, solver)
    
    # Save output
    np.savetxt('{}F_GLE_solution_{}.txt'.format(output_dir, number), F)
    np.savetxt('{}K_GLE_solution_{}.txt'.format(output_dir, number), K)
    np.savetxt('{}t_GLE_solution_{}.txt'.format(output_dir, number), t)





