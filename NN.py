#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:56:02 2023

@author: Max Kerr Winter

A class defining a feed forward neural network with dropout layers.
"""

import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, n_in=28*28, n_out=10, h_layer_widths=[512], bias=True,
                 prob=0.5):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        layer_widths = [n_in] + h_layer_widths

        layers = []
        for count in range(len(layer_widths)-1):
            layers.append(nn.Linear(int(layer_widths[count]), 
                                    int(layer_widths[count+1]), bias=bias))
            
            layers.append(nn.Dropout(p=prob)) # Drops nodes from previous layer

            layers.append(nn.ReLU())

        layers.append(nn.Linear(int(layer_widths[-1]), n_out, bias=bias))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits

if __name__=='__main__':
    # Make a test network
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    N_inputs = 20
    N_outputs = 10
    h_layer_widths=[30, 40, 50, 60, 70, 80]
    
    model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs,
                          h_layer_widths=h_layer_widths).to(device)
    
    print(model)

