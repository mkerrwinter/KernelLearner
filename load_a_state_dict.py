#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:06:03 2023

@author: Max Kerr Winter

A script to load a network that has been saved by writing a JSON of the 
state_dict, without any serialization. The resulting model is then saved in
the standard pytorch way using serialization.
"""

import torch

import auxiliary_functions as aux
from NN import NeuralNetwork

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

base_path = './'

direc = '{}models/best_trained_model/'.format(base_path)
param_path = '{}best_model_params.json'.format(direc)
json_path = '{}best_model_epoch_441.json'.format(direc)

loaded_model = aux.load_from_json_state_dict(param_path, json_path, device)

outpath = json_path[:-5] + '.pth'
torch.save(loaded_model.state_dict(), outpath)

