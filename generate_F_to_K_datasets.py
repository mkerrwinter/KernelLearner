#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:33:34 2022

@author: mwinter
"""

# A script to generate a training and testing dataset for extracting K from F,
# from an existing set of MCT F curves

import os
import sys
import numpy as np
import pandas as pd
import random
from sklearn import decomposition
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset
import time
import pickle
from scipy.interpolate import interp1d
import matplotlib

start = time.time()
plt.close('all')
debug = True
random.seed(1234)
matplotlib.rcParams.update({'font.size': 20})

def get_PCA_components(F_df, order, outpath):
    
    pca = decomposition.PCA()
    pca.n_components = F_df.shape[1]
    pca_data = pca.fit_transform(F_df)

    exp_var = pca.explained_variance_ratio_
    exp_var_short = exp_var[:20]
    
    # Plot explained variance
    filename = '{}PCA_explained_variance.pdf'.format(outpath)
    fig, ax = plt.subplots()
    plt.bar(range(0, len(exp_var_short)), exp_var_short, align='center')
    plt.ylabel('Explained variance')
    plt.xlabel('PCA index')
    plt.yscale('log')
    plt.xticks([0, 3, 6, 9, 12, 15, 18])
    plt.yticks([1e0, 1e-4, 1e-7])
    plt.savefig(filename, bbox_inches='tight')
    
    return pca_data[:, :order], pca


if __name__=='__main__':    
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
    data_inpath = '{}data/MCT_data/'.format(base_path)
    
    dir_exists = os.path.isdir(data_inpath)
    if not dir_exists:
        raise FileNotFoundError('MCT data not found.')
    
    # Pick noise strength
    noise_over_signal = 1e-2
    noise_str = '-2'
    # noise_over_signal = 0.0
    # noise_str = '0'
    
    # Check if data exists
    data_outpath = '{}data/F_to_M_data/noise_{}/'.format(base_path, noise_str)    
    dir_exists = os.path.isdir(data_outpath)
    data_files_exist = False
    
    if not dir_exists:
        os.mkdir(data_outpath)
    else:
        data_files_exist = os.path.isfile(data_outpath+'F_noisy_train.csv')
    
    # get all files at different packing fractions
    numbers = []
    files = os.listdir(data_inpath)
    
    for file in files:
        if file[:6] != 'F_PYHS':
            continue
        
        str_number = file[10:-15]
        number = float(str_number)
        numbers.append(number)
    
    numbers.sort()
    
    # Load data or calculate new data
    if data_files_exist and not debug:
        print('Loading existing data files from ', data_outpath)
        F_train_df = pd.read_csv(data_outpath+'F_noisy_train.csv', header=None)
        F_test_df = pd.read_csv(data_outpath+'F_noisy_test.csv', header=None)
        M_train_df = pd.read_csv(data_outpath+'M_noisy_train.csv', header=None)
        M_test_df = pd.read_csv(data_outpath+'M_noisy_test.csv', header=None)
        Omega_train_series = pd.read_csv(data_outpath+'Omega_noisy_train.csv', header=None)
        Omega_test_series = pd.read_csv(data_outpath+'Omega_noisy_test.csv', header=None)
        F_final_train_series = pd.read_csv(data_outpath+'F_final_noisy_train.csv', header=None)
        F_final_test_series = pd.read_csv(data_outpath+'F_final_noisy_test.csv', header=None)
        
        Omega_train = np.array(Omega_train_series)
        Omega_test = np.array(Omega_test_series)
        F_final_train = np.array(F_final_train_series)
        F_final_test = np.array(F_final_test_series)
    
        str_number = '{:.3f}'.format(numbers[0])
        t_path = '{}time_PYHS_phi{}_Dt1.0_MF11.npy'.format(data_inpath, 
                                                           str_number)
        times = np.load(t_path)
    
    else:
        print('Calculating data')
    
        F_data = []
        M_data = []
        Omegas = []
        F_finals = []
        
        for number in numbers:
            D_str = '1.0'
            str_number = '{:.3f}'.format(number)
            F_path = '{}F_PYHS_phi{}_Dt{}_MF11.npy'.format(data_inpath, 
                                                            str_number, D_str)
            t_path = '{}time_PYHS_phi{}_Dt{}_MF11.npy'.format(data_inpath, 
                                                            str_number, D_str)
            k_path = '{}k_PYHS_phi{}_Dt{}_MF11.npy'.format(data_inpath, 
                                                            str_number, D_str)
            M_path = '{}M_PYHS_phi{}_Dt{}_MF11.npy'.format(data_inpath, 
                                                            str_number, D_str)
            S_path = '{}Sk_PYHS_phi{}_Dt{}_MF11.npy'.format(data_inpath, 
                                                            str_number, D_str)
        
            F_k_t = np.load(F_path)
            times = np.load(t_path)
            k_vals = np.load(k_path)
            M_k_t = np.load(M_path)
            S_k_t = np.load(S_path)
            
            # Check raw data is of length 4352
            if len(times) != 4352:
                raise AssertionError("Input data is not of length 4352.")
            
            # Get only peak k value
            F = F_k_t[:, np.argmax(F_k_t[0, :])]
            M = M_k_t[:, np.argmax(F_k_t[0, :])]
            S = S_k_t[np.argmax(F_k_t[0, :])]
            k_star = k_vals[np.argmax(F_k_t[0, :])]
            
            # Normalise F by t=0 value
            F /= S
            
            # Calculate Omega
            D = float(D_str)
            Omega = D*k_star**2/S
            
            # Get normalised final F value
            F_final = F[-1]/F[0]
            
            # Downsample kernel
            M_func = interp1d(times, M)
            log_min_t = np.log10(min(times))
            log_max_t = np.log10(max(times))
            new_times = np.logspace(log_min_t, log_max_t, 100)
            M_interp = M_func(new_times)
    
            # Generate multiple noise realisations
            if debug:
                N_noise = 100
            else:
                N_noise = 1000
            
            for N in range(N_noise):
                max_sig = F.max() - F.min()
                noise_strength = max_sig*noise_over_signal
                F_noisy = F + np.random.randn(len(F))*noise_strength
                
                F_data.append(F_noisy)
                M_data.append(M_interp)
                Omegas.append(Omega)
                F_finals.append(F_final)
                    
        # Save downsampled time grid
        time_outpath = data_outpath+'times_for_plotting.csv'
        if not debug:
            np.savetxt(time_outpath, new_times)
        
        # Shuffle data and kernels
        unshuffled_F = F_data.copy()
        unshuffled_M = M_data.copy()
        unshuffled_Omegas = Omegas.copy()
        unshuffled_F_finals = F_finals.copy()
        
        indices = list(np.arange(len(F_data)))
        temp = list(zip(F_data, M_data, Omegas, F_finals, indices))
        random.shuffle(temp)
        F_tuple, M_tuple, Omegas_tuple, F_finals_tuple, ind_tuple = zip(*temp)
        
        F_data = list(F_tuple)
        M_data = list(M_tuple)
        Omegas = np.array(Omegas_tuple)
        F_finals = np.array(F_finals_tuple)
        shuffled_indices = list(ind_tuple)
        
        # Check shuffle process has worked by looking at an example
        F_check = F_data[12]
        M_check = M_data[12]
        Omega_check = Omegas[12]
        F_final_check = F_finals[12]
        old_index = shuffled_indices[12]
        
        old_F = unshuffled_F[old_index]
        old_M = unshuffled_M[old_index]
        old_Omega = unshuffled_Omegas[old_index]
        old_F_final = unshuffled_F_finals[old_index]
        
        F_incorrect = not np.array_equal(F_check, old_F)
        M_incorrect = not np.array_equal(M_check, old_M)
        Omega_incorrect = not np.array_equal(Omega_check, old_Omega)
        F_final_incorrect = not np.array_equal(F_final_check, old_F_final)
        
        if F_incorrect or M_incorrect or Omega_incorrect or F_final_incorrect:
            raise AssertionError("Error in data shuffling.")

        # Normalise Omegas
        max_O = max(Omegas)
        min_O = min(Omegas)
        if max_O != min_O:
            Omegas = (Omegas - min_O)/(max_O - min_O)
            
        # Save Omega scaling
        minmax_outpath = data_outpath+'min_max_Omegas.csv'
        if not debug:
            np.savetxt(minmax_outpath, np.array([min_O, max_O]))
        
        # Convert data to dataframe
        F_df = pd.DataFrame(F_data)
        
        F_train_df = F_df.loc[:F_df.shape[0]//2-1, :]
        F_test_df = F_df.loc[F_df.shape[0]//2:, :]
        
        M_train = M_data[:F_df.shape[0]//2]
        M_test = M_data[F_df.shape[0]//2:]
        M_train_df = pd.DataFrame(M_train)
        M_test_df = pd.DataFrame(M_test)
        
        Omega_train = Omegas[:F_df.shape[0]//2]
        Omega_test = Omegas[F_df.shape[0]//2:]
        Omega_train_series = pd.Series(Omega_train)
        Omega_test_series = pd.Series(Omega_test)
        
        F_final_train = F_finals[:F_df.shape[0]//2]
        F_final_test = F_finals[F_df.shape[0]//2:]
        F_final_train_series = pd.Series(F_final_train)
        F_final_test_series = pd.Series(F_final_test)
        
        # Save dataframes as text files
        if not debug:
            F_train_df.to_csv(data_outpath+'F_noisy_train.csv', header=False, 
                              index=False)
            F_test_df.to_csv(data_outpath+'F_noisy_test.csv', header=False, 
                             index=False)
            M_train_df.to_csv(data_outpath+'M_noisy_train.csv', header=False, 
                              index=False)
            M_test_df.to_csv(data_outpath+'M_noisy_test.csv', header=False, 
                             index=False)
            Omega_train_series.to_csv(data_outpath+'Omega_noisy_train.csv', 
                                      header=False, index=False)
            Omega_test_series.to_csv(data_outpath+'Omega_noisy_test.csv', 
                                     header=False, index=False)
            
            F_final_train_series.to_csv(data_outpath+'F_final_noisy_train.csv', 
                                      header=False, index=False)
            F_final_test_series.to_csv(data_outpath+'F_final_noisy_test.csv', 
                                     header=False, index=False)
    
    # Do PCA on training data
    order = 15
    pca_F_train_noisy, pca_obj = get_PCA_components(F_train_df, order, 
                                                    data_outpath)
    
    # Pickle PCA object
    if not debug:
        pkl_path = data_outpath + 'pca_object.pkl'
        with open(pkl_path, 'wb') as outfile:
            pickle.dump(pca_obj, outfile)

    # Add Omega and F_final to feature list
    if not data_files_exist:
        Omega_train = np.expand_dims(Omega_train, 1)
        F_final_train = np.expand_dims(F_final_train, 1)

    pca_F_train_noisy = np.concatenate([pca_F_train_noisy, 
                                        Omega_train, 
                                        F_final_train],axis=1)

    # Convert F_data and M_data in a pytorch friendly format
    pt_train = []
    pt_train_l = []
    
    for row in range(pca_F_train_noisy.shape[0]):
        data_row = pca_F_train_noisy[row, :]
        data_row = np.expand_dims(data_row, axis=0)
        data_row = np.expand_dims(data_row, axis=0)
    
        label_row = M_train_df.loc[row, :]
        label_row = np.expand_dims(label_row, axis=0)
        
        pt_train.append(torch.Tensor(data_row))
        pt_train_l.append(torch.Tensor(label_row))
    
    pt_train_tensor = torch.cat(pt_train)
    pt_train_tensor_l = torch.cat(pt_train_l)
    
    pt_train_data = TensorDataset(pt_train_tensor, pt_train_tensor_l)
    
    # PCA transform testing data
    pca_F_test_noisy = pca_obj.transform(F_test_df)
    pca_F_test_noisy = pca_F_test_noisy[:, :order]
    
    # Add Omega to feature list
    if not data_files_exist:
        Omega_test = np.expand_dims(Omega_test, 1)
        F_final_test = np.expand_dims(F_final_test, 1)
        
    pca_F_test_noisy = np.concatenate([pca_F_test_noisy, 
                                       Omega_test, 
                                       F_final_test],axis=1)

    # Convert F_data and M_data in a pytorch friendly format
    pt_test = []
    pt_test_l = []
    
    for row in range(pca_F_test_noisy.shape[0]):
        data_row = pca_F_test_noisy[row, :]
        data_row = np.expand_dims(data_row, axis=0)
        data_row = np.expand_dims(data_row, axis=0)
    
        label_row = M_test_df.loc[row, :]
        label_row = np.expand_dims(label_row, axis=0) 
        
        pt_test.append(torch.Tensor(data_row))
        pt_test_l.append(torch.Tensor(label_row))
    
    pt_test_tensor = torch.cat(pt_test)
    pt_test_tensor_l = torch.cat(pt_test_l)
    
    pt_test_data = TensorDataset(pt_test_tensor, pt_test_tensor_l)
    
    if not debug:
        # Save pytorch data
        fname = data_outpath + 'MCT_train.pt'
        torch.save(pt_train_data, fname)
        
        fname = data_outpath + 'MCT_test.pt'
        torch.save(pt_test_data, fname)
    
    print('Running time = ', time.time()-start)