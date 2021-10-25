# coding: utf-8

# Import libraries
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from collections import OrderedDict


# Import Network Element modules 
from NetworkElements.gradient import numerical_gradient
from NetworkElements.layers import *

class MultiLayerNetwork:
    """ Fully-connected Multi layer neural network
    
    Parameters
    ----------
    input_size: input size
    hidden_size (list): a list of nubmer of neurons in the hidden layers (e.g. [100, 100, 100])
    output_size: output size
    activation: activation functions ('relu' or 'sigmoid')
    weight_init_std: standard deviation of weights (e.g., 0.01)
        'relu' or 'he': He initialization
        'sigmoid' or 'xavier': Xavier initialization
    """ 
    def __init__(self, input_size, hidden_size, output_size
                 ,activation='relu', weight_init_std='relu', weight_decay_lambda=0
                #  , warm_start=False
                #  , dir_name=None, file_name=None
                 ):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_layer_num = len(hidden_size) # the number of hidden layers
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # self.warm_start = warm_start
        # self.dir_name = dir_name
        # self.file_name = file_name

        # Initialize weights ------------------------------------------------------------------------------------
        self.__init_weight(weight_init_std)
        # if self.warm_start:
        #     self.load_params(dir_name, file_name)

        # Generate layers --------------------------------------------------------------------------------------
        self.layers = OrderedDict()

        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        
        # define layers (Affine layer + activation layer)
        for i in range(1, self.hidden_layer_num+1):
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
            self.layers['Activation_function' + str(i)] = activation_layer[activation]()
        # last layer
        i = self.hidden_layer_num + 1
        self.layers['Affine'+str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
        self.last_layer = SoftmaxWithLoss()
    
    def __init_weight(self, weight_init_std):
        """
        Weigth initialization

        Parameters
        ----------
        weight_init_std: standard deviation of weights (e.g., 0.01)
            'relu' or 'he': He initialization (ref: https://arxiv.org/pdf/1502.01852.pdf)
            'sigmoid' or 'xavier': Xavier initialization (ref: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
        """
        network_size = [self.input_size] + self.hidden_size + [self.output_size]

        for i in range(1, len(network_size)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / network_size[i - 1])  # recommended the value for ReLU (He initialization)
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / network_size[i - 1])  # recommended the value for sigmoid (Xavier initialization)
        
            self.params['W' + str(i)] = scale * np.random.randn(network_size[i-1], network_size[i])
            self.params['b' + str(i)] = np.zeros(network_size[i])

    def predict(self, x):
        # forward propagation
        for layer in self.layers.values():
            x = layer.forward(x) #forward() is defined in Affine class
        
        return x

    def loss(self, x, t):
        """Loss function

        Parameters
        ----------
        x: input data
        t: training data

        Returns
        -------
        the value of loss fucntion 
        """

        y = self.predict(x)
        weight_decay = 0
        for i in range(1, self.hidden_layer_num+2):
            W = self.params['W' + str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay # cross entropy loss

    def accuracy(self, x, t):
        """
        Accuracy score
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """ Numerical gradient

        Parameters
        ----------
        x: input data
        t: training label

        Returns
        -------
        dictionary variable with gradients for each layer
            grads['W1'], grads['W2'], ... weight for each layer
            grads['b1'], grads['b2'], ... bias for each layer
        """
        loss_W = lambda W: self.loss(x, t)
        # Setting
        grads = {}
        for i in range(1, self.hidden_layer_num+2):
            grads['W' + str(i)] = numerical_gradient(loss_W, self.params['W' + str(i)])
            grads['b' + str(i)] = numerical_gradient(loss_W, self.params['b' + str(i)])

        return grads

    def gradient(self, x, t):
        """ Numerical gradient

        Parameters
        ----------
        x: input data
        t: training label

        Returns
        -------
        dictionary variable with gradients for each layer
            grads['W1'], grads['W2'],... weight for each layer
            grads['b1'], grads['b2'].... bias for each layer
        """

        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Setting
        grads = {}
        for i in range(1, self.hidden_layer_num+2):
            grads['W' + str(i)] = self.layers['Affine' + str(i)].dW + self.weight_decay_lambda * self.layers['Affine' + str(i)].W
            grads['b' + str(i)] = self.layers['Affine' + str(i)].db

        return grads
    
    def save_params(self, dir_name, file_name):
        """ Save weights and biases in a pickle file
        """
        # params: weights and biases
        params = {}

        for key, val in self.params.items():
            params[key] = val
        
        with open(dir_name + '/' + file_name + '.pkl', 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, dir_name, file_name):
        """ Load weights and biases from a pickle file
        """
        with open(dir_name + '/' + file_name + '.pkl', 'rb') as f:
            params = pickle.load(f)
        
        for key, val in params.items():
            self.params[key] = val

if __name__ == '__main__':
    print('This is a module to define a class of multi layer neural network.')