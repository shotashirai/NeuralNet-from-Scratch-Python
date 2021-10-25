# coding: utf-8

import numpy as np
from functions import *

class Relu:
    """ Rectified Linear Unit (ReLU)

    ReLU is defined as f(x) = max(0, x) (i.e. 0 (x<=0),  x (x >0))

    Ref:
        http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf
        https://www.nature.com/articles/nature14539
    """
    def __init__(self):
        self.mask = None

    #Forward propagation
    def forward(self, x): 
        # make 'mask' for negative values
        self.mask = (x <= 0)
        out = x.copy()
        # negative values are replaced with 0
        out[self.mask] = 0
        return out

    # Back propagation
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    """ Sigmoid function
    Sigmoid fuction is defined as x / (1 + exp(x))

    Output from sigmoid function is given by 'sigmoid()' in functions.py module.
    """
    def __init__(self):
        self.out = None
    # Forward propagation
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    # Back propagation
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx



class Affine:
    """ Affine layers (Fully-connected layer/Dense layer)
    Propagate siginals for one layer to the next layer
    """
    
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # differentiation of weights and biases
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape) # back to the original input shape
        return dx


class SoftmaxWithLoss:
    """ Softmax with loss function (cross entropy error)
    Softmax function is often used for output layer. 
    This class provides softmax function with loss function (cross entropy error).
    """

    def __init__(self):
        self.loss = None
        self.y = None # output of softmax
        self.t = None # training label

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # if training data is one-hot-vector
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx



# TODO Add more functionalities to the network layer (e.g. convolutional layers, pooling layers...)
# Future development allocation 
class Convolution:
    """ Convolutional layer
    """
    pass

class Pooling:
    """ Pooling layer
    """
    pass

class Dropout:
    """ Dropout
    (ref:http://arxiv.org/abs/1207.0580)
    """
    pass

class BatchNorm:
    """ Batch Normalization
    (ref: http://arxiv.org/abs/1502.03167)
    """
    pass


if __name__ == '__main__':
    print('This is a module to define a class of activation functions and layers.')