#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:33:34 2023

@author: Max Kerr Winter

A script to generate a training and testing dataset for extracting kernels, 
K, from solutions to the GLE, F. The script takes clean solutions to the GLE,
adds multiple realisations of noise, and constructs a training and testing 
set out of the resulint noisy curves in a pytorch friendly format.
"""

import os
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

start = time.time()
plt.close('all')
debug = False # No data is saved if debug = True
random.seed(12345)

def perform_PCA(data, order, outpath, debug):
    """
    A function to calculate the PCA components of a dataset.
    
    Inputs
    ------
    data    : DataFrame
              A dataframe containing the data. For N datapoints each of 
              dimension M, data has shape (N, M).
              
    order   : int
              The number of PCA components to be retained.
              
    outpath : str
              The path where the explained variance plot is saved.
              
    debug   : bool
              If True, no plots are saved.
    
    Outputs
    -------
    pca_data_reduced : numpy array
                       The data expressed in the PCA basis up to the 
                       specified order.
                       
    pca              : decomposition.PCA
                       The PCA object that is capable of transforming data 
                       onto the PCA basis of the input data.
    """

    pca = decomposition.PCA()
    pca.n_components = data.shape[0]
    pca_data = pca.fit_transform(data)

    exp_var = pca.explained_variance_ratio_
    exp_var_short = exp_var[:20]
    
    # Plot explained variance
    filename = '{}PCA_explained_variance.pdf'.format(outpath)
    fig, ax = plt.subplots()
    plt.bar(range(0, len(exp_var_short)), exp_var_short, align='center')
    plt.ylabel('Explained variance')
    plt.xlabel('PCA index')
    plt.yscale('log')
    if not debug:
        plt.savefig(filename, bbox_inches='tight')
    
    pca_data_reduced = pca_data[:, :order]
    
    return pca_data_reduced, pca


if __name__=='__main__':    
    base_path = './'
    data_inpath = '{}data/example_GLE_data/'.format(base_path)
    
    dir_exists = os.path.isdir(data_inpath)
    if not dir_exists:
        raise FileNotFoundError('Data not found. '+ 
                                'Run generate_data_from_Ks.py first.')
    
    # Pick noise strength
    noise_over_signal = 1e-2
    noise_str = '-2'
    
    # Check if noisy data already exists
    data_outpath = '{}data/F_to_K_data/noise_{}/'.format(base_path, noise_str)    
    dir_exists = os.path.isdir(data_outpath)
    data_files_exist = False
    
    if not dir_exists:
        data_parent_outpath = '{}data/F_to_K_data/'.format(base_path)
        data_parent_exists = os.path.isdir(data_parent_outpath)
        if not data_parent_exists and not debug:
            os.mkdir(data_parent_outpath)
        
        if not debug:
            os.mkdir(data_outpath)
    else:
        data_files_exist = os.path.isfile(data_outpath+'F_noisy_train.csv')
    
    # Organise numbered files
    numbers = []
    files = os.listdir(data_inpath)
    
    for file in files:
        if file[:14] != 'F_GLE_solution':
            continue
        
        str_number = file[15:-4]
        number = int(str_number)
        numbers.append(number)
    
    numbers.sort()
    
    # Load data or calculate new data
    # if data_files_exist and not debug:
    if data_files_exist:
        print('Loading existing data files from {}.'.format(data_outpath))
        F_train_df = pd.read_csv(data_outpath+'F_noisy_train.csv', header=None)
        F_test_df  = pd.read_csv(data_outpath+'F_noisy_test.csv', header=None)
        
        K_train_df = pd.read_csv(data_outpath+'K_train.csv', header=None)
        K_test_df  = pd.read_csv(data_outpath+'K_test.csv', header=None)
        
        Omega_train_series = pd.read_csv(data_outpath+'Omega_train.csv', 
                                         header=None)
        Omega_test_series  = pd.read_csv(data_outpath+'Omega_test.csv', 
                                         header=None)
        
        F_final_train_series = pd.read_csv(data_outpath+'F_final_train.csv', 
                                           header=None)
        F_final_test_series  = pd.read_csv(data_outpath+'F_final_test.csv', 
                                           header=None)
        
        Omega_train = np.array(Omega_train_series)
        Omega_test  = np.array(Omega_test_series)
        F_final_train = np.array(F_final_train_series)
        F_final_test  = np.array(F_final_test_series)
    
        times = np.loadtxt(data_inpath+'t_GLE_solution.txt')
    
    else:
        print('Calculating noisy data.')
    
        F_data = []
        K_data = []
        Omegas = []
        F_finals = []
        
        for number in numbers:
            F_path = '{}F_GLE_solution_{}.txt'.format(data_inpath, number)
            t_path = '{}t_GLE_solution.txt'.format(data_inpath)
            K_path = '{}K_GLE_solution_{}.txt'.format(data_inpath, number)
            Omega_path = '{}Omega_GLE_solution_{}.txt'.format(data_inpath, 
                                                              number)
        
            F = np.loadtxt(F_path)
            times = np.loadtxt(t_path)
            K = np.loadtxt(K_path)
            Omega = float(np.loadtxt(Omega_path))
            
            # Normalise F by t=0 value
            F /= F[0]

            # Get final F value
            F_final = F[-1]
            
            # Downsample kernel
            K_func = interp1d(times, K)
            times_without_zero = np.array(times)[times>0]
            log_min_t = np.log10(min(times_without_zero))
            log_max_t = np.log10(max(times_without_zero))
            
            if len(times) == len(times_without_zero):
                new_times = np.logspace(log_min_t, log_max_t, 100)
                
            elif len(times) == len(times_without_zero)+1:
                new_times_temp = np.logspace(log_min_t, log_max_t, 99)
                new_times = np.zeros(len(new_times_temp)+1)
                new_times[0] = 0.0
                new_times[1:] = new_times_temp
                
            else:
                raise ValueError('There are more than one zeros in times')
            
            new_times[-1] = times[-1]
            K_interp = K_func(new_times)
    
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
                K_data.append(K_interp)
                Omegas.append(Omega)
                F_finals.append(F_final)
                    
        # Save downsampled time grid
        time_outpath = data_outpath+'times_for_interpolated_K.txt'
        if not debug:
            np.savetxt(time_outpath, new_times)
        
        # Shuffle data and kernels
        indices = list(np.arange(len(F_data)))
        temp = list(zip(F_data, K_data, Omegas, F_finals, indices))
        random.shuffle(temp)
        F_tuple, K_tuple, Omegas_tuple, F_finals_tuple, ind_tuple = zip(*temp)
        
        F_data = list(F_tuple)
        K_data = list(K_tuple)
        Omegas = np.array(Omegas_tuple)
        F_finals = np.array(F_finals_tuple)
        shuffled_indices = list(ind_tuple)

        # Normalise Omegas
        max_O = max(Omegas)
        min_O = min(Omegas)
        if max_O != min_O:
            Omegas = (Omegas - min_O)/(max_O - min_O)
            
        # Save Omega scaling
        minmax_outpath = data_outpath+'min_max_Omegas.txt'
        if not debug:
            np.savetxt(minmax_outpath, np.array([min_O, max_O]))
        
        # Convert data to dataframe
        F_df = pd.DataFrame(F_data)
        
        halfway_point = F_df.shape[0]//2
        # Note: Pandas indexing is inclusive of end point
        F_train_df = F_df.loc[:halfway_point-1, :]
        F_test_df = F_df.loc[halfway_point:, :]
        
        K_train = K_data[:halfway_point]
        K_test = K_data[halfway_point:]
        K_train_df = pd.DataFrame(K_train)
        K_test_df = pd.DataFrame(K_test)
        
        Omega_train = Omegas[:halfway_point]
        Omega_test = Omegas[halfway_point:]
        Omega_train_series = pd.Series(Omega_train)
        Omega_test_series = pd.Series(Omega_test)
        
        F_final_train = F_finals[:halfway_point]
        F_final_test = F_finals[halfway_point:]
        F_final_train_series = pd.Series(F_final_train)
        F_final_test_series = pd.Series(F_final_test)
        
        # Save dataframes as text files
        if not debug:
            F_train_df.to_csv(data_outpath+'F_noisy_train.csv', header=False, 
                              index=False)
            F_test_df.to_csv(data_outpath+'F_noisy_test.csv', header=False, 
                             index=False)
            
            K_train_df.to_csv(data_outpath+'K_train.csv', header=False, 
                              index=False)
            K_test_df.to_csv(data_outpath+'K_test.csv', header=False, 
                             index=False)
            
            Omega_train_series.to_csv(data_outpath+'Omega_train.csv', 
                                      header=False, index=False)
            Omega_test_series.to_csv(data_outpath+'Omega_test.csv', 
                                     header=False, index=False)
            
            F_final_train_series.to_csv(data_outpath+'F_final_train.csv', 
                                      header=False, index=False)
            F_final_test_series.to_csv(data_outpath+'F_final_test.csv', 
                                     header=False, index=False)
    
    # Do PCA on training data
    # In general order is chosen subjectively by looking at explained variance.
    order = 15 
    pca_F_train_noisy, pca_obj = perform_PCA(F_train_df, order, data_outpath, 
                                             debug)
    
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

    # Convert training data into a pytorch friendly format
    pt_train = []
    pt_train_l = []
    
    for row in range(pca_F_train_noisy.shape[0]):
        data_row = pca_F_train_noisy[row, :]
        data_row = np.expand_dims(data_row, axis=0)
        data_row = np.expand_dims(data_row, axis=0)
    
        label_row = K_train_df.loc[row, :]
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

    # Convert test data in a pytorch friendly format
    pt_test = []
    pt_test_l = []
    
    for row in range(pca_F_test_noisy.shape[0]):
        data_row = pca_F_test_noisy[row, :]
        data_row = np.expand_dims(data_row, axis=0)
        data_row = np.expand_dims(data_row, axis=0)
    
        label_row = K_test_df.loc[row, :]
        label_row = np.expand_dims(label_row, axis=0) 
        
        pt_test.append(torch.Tensor(data_row))
        pt_test_l.append(torch.Tensor(label_row))
    
    pt_test_tensor = torch.cat(pt_test)
    pt_test_tensor_l = torch.cat(pt_test_l)
    
    pt_test_data = TensorDataset(pt_test_tensor, pt_test_tensor_l)
    
    if not debug:
        # Save pytorch data
        train_fname = data_outpath + 'GLE_train.pt'
        torch.save(pt_train_data, train_fname)
        
        test_fname = data_outpath + 'GLE_test.pt'
        torch.save(pt_test_data, test_fname)
    
    print('Running time = ', time.time()-start)