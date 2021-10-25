# coding: utf-8

"""
Code: train_neuralnet.py
Author: Shota Shirai
Input: mnist handwritten digit data (in 'data' directory)
Output: 
    - model.cpkl (option): trained network objects
    - params.pkl (option): weights and biases at an epoch
Required class/functions are provided by 'NetworkElements'

This code generates a multilayer neural network and trains it. In this code, MNIST
handwritten digit data is used for training and testing the built network. 
The trained network and trained parameters (weights and biases) are saved in pickle 
(or cloudpickle) files.

Note: Tested Python version 3.7.5

"""

###################################################################################
# Import libraries 
###################################################################################
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.preprocess_mnist import load_mnist
from NetworkElements.multi_layer_net import MultiLayerNetwork
from NetworkElements.trainer import *

###################################################################################
# Import data 
###################################################################################
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

###################################################################################
# Generate network
###################################################################################
# hidden layers
hidden_size = [50, 50, 50]
# Generate network
network = MultiLayerNetwork(
    input_size=784
    , hidden_size=hidden_size
    , output_size=10
    )

###################################################################################
# Train network
###################################################################################
trainer = Trainer(network, x_train, t_train, x_test, t_test
                , epochs=20, verbose=2, learning_rate=0.1
                , save_network=True, save_params_epoch=5)
trainer.train()


#TODO Visualization of the learning process