#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:36:19 2022

@author: mwinter
"""


# A script to solve the GLE using the julia package of I. Pihlajamaa and a 
# given kernel

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.interpolate import interp1d

start = time.time()

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from julia import Pkg
Pkg.add("ModeCouplingTheory")
from julia import ModeCouplingTheory as mct


def Sk_PY_HS(k,phi,sigma,rho): #Single component PY for hard spheres
    # Sk = Sk_PY_HS(k,phi, sigma, rho) returns the structure factor S(k) for a 
    # hard-sphere system with sigma, rho, and phi denoting diameter, number 
    # density and volume fraction respectively. Results are calculated using 
    # the Percus-Yevick approximation and are based on (Hansen and McDonald 
    # Theory of Simple Liquids, and J.L. Lebowitz 1964). Function written by 
    # Vincent Debets

    r = np.linspace(0.000001,sigma,num=1000000)
    cr = np.zeros(len(r))
    ck = np.zeros(len(k)) 
    lambda1 = (1+2*phi)**2 / (1-phi)**4
    lambda2 = (1+phi/2)**2 / (1-phi)**4
    ir = np.flatnonzero(r<sigma) # c(r) = 0 for r>sigma
    cr[ir] = (-lambda1 + 6*phi*lambda2*(r[ir]/sigma) 
              - (phi/2)*lambda1*((r[ir]/sigma)**3))

    for i in range(len(k)):
        ki = k[i]
        ck[i] = 4*np.pi*np.trapz(r*np.sin(ki*r)*cr,x=r)/ki

    Sk = 1/(1-rho*ck)

    return Sk


# Load an example F(k, t) curve
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
        
input_dir = base_path + 'data/Brownian_data/'
kT_str = '1.0'
sim_name = 'F_kt_T_{}.txt'.format(kT_str)
Sk_name = 'Sk_T_{}.txt'.format(kT_str)
noise_str = 'NA'

F_inpath = input_dir + sim_name
Sk_inpath = input_dir + Sk_name
simulation_t_inpath = input_dir + 't_array.txt'
simulation_k_inpath = input_dir + 'k_array.txt'

sim_F = np.loadtxt(F_inpath)
sim_S_k_with_grid = np.loadtxt(Sk_inpath, delimiter=',')
sim_t = np.loadtxt(simulation_t_inpath)
sim_k = np.loadtxt(simulation_k_inpath)
k_grid_for_S = sim_S_k_with_grid[:, 0]
sim_S_k = sim_S_k_with_grid[:, 1]

# Interpolate S and F onto a good k grid
k_array_high_res = np.linspace(0.2, 39.8, 100)
sim_S_k_interp = interp1d(k_grid_for_S, sim_S_k, fill_value='extrapolate')
sim_S_interpolated = sim_S_k_interp(k_array_high_res)
sim_peak_k_idx = np.argmax(sim_S_interpolated) 
sim_peak_k = k_array_high_res[sim_peak_k_idx]

sim_F_interpolated = np.zeros((sim_F.shape[0], len(k_array_high_res)))
for row in range(sim_F.shape[0]):
    sim_F_interp = interp1d(sim_k, sim_F[row, :], fill_value='extrapolate')
    sim_F_interpolated[row, :] = sim_F_interp(k_array_high_res)

# Load the previously extracted kernel
# model_name = 'linear_weight_param_search_ic0_reg0_05_b5000_w8'
# model_name = 'balanced_data_param_search_ic0_reg0_001_b2500_w8' # best test loss
# model_name = 'balanced_data_param_search_ic1_reg0_1_b2500_w8' # Best non reg0_001 test loss 
model_name = 'F_final_data_param_search_ic4_reg0_001_b2500_w8'
name_stub = 'model_{}_sim_{}_noise_{}.txt'.format(model_name, sim_name[:-4], noise_str)
input_dir = base_path + 'extracted_kernels/'

kernel_inpath = input_dir+'kernel_{}'.format(name_stub)
time_inpath = input_dir+'times_for_kernel_{}'.format(name_stub)
k_inpath = input_dir+'wavenumber_for_kernel_{}'.format(name_stub)

net_M = np.loadtxt(kernel_inpath)
net_t = np.loadtxt(time_inpath)
wavenumber = np.loadtxt(k_inpath)   

k_idx = np.where(sim_k==wavenumber)[0][0]
F_star = sim_F_interpolated[:, sim_peak_k_idx]
F_star /= F_star[0]

# Define system parameters and initial conditions 
kT = float(kT_str) 

m = 1.0
a = 0.0
b = 1.0
F0 = 1.0
S_kstar = sim_S_interpolated[sim_peak_k_idx]
dF0 = 0.0
c = kT*sim_peak_k**2/(m*S_kstar) 

# Solve GLE
kernel = mct.InterpolatingKernel(net_t, net_M)
equation = mct.LinearMCTEquation(a, b, c, F0, dF0, kernel)
solver = mct.FuchsSolver(N=128, Δt=10**-5, t_max=10.0**5, max_iterations=10**8, 
                         tolerance=10**-6, verbose=False)
t, F, K = mct.solve(equation, solver)


# Solve WCA MCT equation
print('k* = ', wavenumber)
rho = 0.95
c = kT*k_array_high_res**2/(m*sim_S_interpolated) # THIS S SHOULD BE S(K) AT ALL K'S. Use S defined on good k grid that Ilian will send, and use the k's from this instead of a single wavenumber.

# interpolate S onto a grid that contains kstar and has k_array[0] = delta_k/2
F0_interpolated = sim_S_interpolated
dF0_interpolated = np.zeros_like(F0_interpolated)

MCT_WCA_kernel = mct.ModeCouplingKernel(rho, kT, m, k_array_high_res, sim_S_interpolated)
MCT_WCA_equation = mct.LinearMCTEquation(a, b, c, F0_interpolated, dF0_interpolated, MCT_WCA_kernel)
MCT_WCA_t, MCT_WCA_F, MCT_WCA_K = mct.solve(MCT_WCA_equation, solver)
MCT_WCA_F_star = MCT_WCA_F[sim_peak_k_idx, :]
MCT_WCA_F_star /= MCT_WCA_F_star.max()


# Solve PYHS MCT equation
sigma = 1.0 # Simulation particle diameter 
phi = rho*(4.0/3.0)*np.pi*(sigma/2.0)**3
print('phi = ', phi)
PYHS_Sk = Sk_PY_HS(k_array_high_res, phi, sigma, rho)
F0_interpolated = PYHS_Sk
dF0_interpolated = np.zeros_like(F0_interpolated)
c = kT*k_array_high_res**2/(m*PYHS_Sk)

MCT_HS_kernel = mct.ModeCouplingKernel(rho, kT, m, k_array_high_res, PYHS_Sk)
MCT_HS_equation = mct.LinearMCTEquation(a, b, c, F0_interpolated, dF0_interpolated, MCT_HS_kernel)
MCT_HS_t, MCT_HS_F, MCT_HS_K = mct.solve(MCT_HS_equation, solver)
MCT_HS_F_star = MCT_HS_F[sim_peak_k_idx, :]
MCT_HS_F_star /= MCT_HS_F_star.max()

plot_output_dir = '{}plots/extracted_Fkt/Brownian_data/'.format(base_path)    
    
dir_exists = os.path.isdir(plot_output_dir)
if not dir_exists:
    os.mkdir(plot_output_dir)

fig, ax = plt.subplots()
plt.plot(k_array_high_res, PYHS_Sk, label='HS')
plt.plot(k_array_high_res, sim_S_interpolated, label='WCA', linestyle='dashed')
plt.legend()
plt.title('S(k) comparison')
F_name = 'Sk_comparison_{}_sim_{}.pdf'.format(model_name, sim_name[:-4])
filename = '{}{}'.format(plot_output_dir, F_name)
plt.savefig(filename, bbox_inches='tight')

fig, ax = plt.subplots()
plt.plot(net_t, net_M, label='Network K')
plt.plot(MCT_WCA_t, MCT_WCA_K[sim_peak_k_idx, sim_peak_k_idx, :], label='WCA MCT K')
plt.plot(MCT_HS_t, MCT_HS_K[sim_peak_k_idx, sim_peak_k_idx, :], label='PYHS MCT K')
plt.legend()
plt.title('Kernel comparison')
plt.xscale('log')
k_path = '{}/plots/extracted_kernels/Brownian_data/'.format(base_path)
k_name = 'kernel_comparison_model_{}_sim_{}.pdf'.format(model_name, 
                                                        sim_name[:-4])
k_outpath = k_path + k_name
plt.savefig(k_outpath, bbox_inches='tight')


fig, ax = plt.subplots()
plt.plot(sim_t, F_star, label='Sim')
plt.plot(t, F, linestyle='dashed', label='Net GLE')
plt.plot(MCT_WCA_t, MCT_WCA_F_star, linestyle='dashed', label='WCA MCT')
plt.plot(MCT_HS_t, MCT_HS_F_star, linestyle='dotted', label='PYHS MCT')
plt.legend()
plt.xscale('log')
plt.title('F vs time')
F_name = 'Fkt_model_{}_sim_{}.pdf'.format(model_name, sim_name[:-4])
filename = '{}{}'.format(plot_output_dir, F_name)
plt.savefig(filename, bbox_inches='tight')

print('Running time = ', time.time()-start)


