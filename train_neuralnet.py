import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.preprocess_mnist import load_mnist
from NetworkElements.multi_layer_net import MultiLayerNetwork
from NetworkElements.trainer import *

# import data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# hidden layers
hidden_size = [50, 50, 50]
# Generate network
network = MultiLayerNetwork(
    input_size=784
    , hidden_size=hidden_size
    , output_size=10
    )

trainer = Trainer(network, x_train, t_train, x_test, t_test
                , epochs=20, verbose=2, learning_rate=0.1
                , save_network=True, save_params_epoch=5)

trainer.train()

#TODO Visualization of the leraning process