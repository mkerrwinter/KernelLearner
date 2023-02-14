#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 08:19:31 2023

@author: Max Kerr Winter

A script to generate training data by solving the overdamped GLE for a set
of user provided memory kernels. The overdamped GLE in some function f is
f' + \Omega f + \int_0^\infty dt'[K(t')f'(t-t')] = 0.
                                            
The GLE solver can be found at https://github.com/IlianPihlajamaa/ModeCouplingTheory.jl.

Before this code can be run, a working version of the programming language Julia should 
be installed, together with the PyCall package. To do so, install Julia from the website
https://julialang.org/downloads/ and, after having done so, run 
"import Pkg; Pkg.add("PyCall")" in a julia session. 

This code was run with ModeCouplingTheory.jl v0.6.2 and NumPy v1.20.3
"""



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

base_path = './'
input_dir = '{}/data/example_1D_kernels/'.format(base_path)
output_dir = '{}/data/example_GLE_data/'.format(base_path)

dir_exists = os.path.isdir(output_dir)
if not dir_exists:
    os.mkdir(output_dir)

# Order numbered input files
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
    # Load from file: kernels, initial conditions, frequency terms, time grids
    K_path = '{}K_example_{}.txt'.format(input_dir, number)
    F0_path = '{}S_example_{}.txt'.format(input_dir, number)
    dF0_path = '{}dF0_example_{}.txt'.format(input_dir, number)
    Omega_path = '{}Omega_example_{}.txt'.format(input_dir, number)
    time_path = '{}t_example_{}.txt'.format(input_dir, number)
    
    K = np.loadtxt(K_path)
    F0 = float(np.loadtxt(F0_path))
    dF0 = float(np.loadtxt(dF0_path))
    Omega = float(np.loadtxt(Omega_path))
    times = np.loadtxt(time_path)
    
    # Define system parameters for equation ay''+by'+cy+\int(Ky')d\tau = 0
    a = 0.0
    b = 1.0
    c = Omega
    d = 0.0    
    
    # Solve GLE
    print('Solving GLE with kernel K_example_{}.txt'.format(number))
    kernel = mct.InterpolatingKernel(times, K)
    equation = mct.MemoryEquation(a, b, c, d, F0, dF0, kernel)
    solver = mct.TimeDoublingSolver(N=128, Δt=10**-5, t_max=10.0**5, 
                             max_iterations=10**8, tolerance=10**-6, 
                             verbose=False)
    sol = mct.solve(equation, solver)
    
    # Save output
    Om = np.array([Omega])
    np.savetxt('{}F_GLE_solution_{}.txt'.format(output_dir, number), sol.F)
    np.savetxt('{}K_GLE_solution_{}.txt'.format(output_dir, number), sol.K)
    np.savetxt('{}Omega_GLE_solution_{}.txt'.format(output_dir, number), Om)
    np.savetxt('{}t_GLE_solution.txt'.format(output_dir), sol.t)





