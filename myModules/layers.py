#coding: utf-8

import numpy as np
from functions import *

class Relu:
    def __init__(self):
        self.mask = None

    #Forward propagation
    def forward(self, x): 
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    # Back propagation
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:

    def __init__(self):
        self.out = None
    # Forward propagation
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    # Back propagation
    def backward(self, x):
        dx = dout * (1.0 - self.out) * self.out
        return dx



class Affine:
    
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

if __name__ == '__main__':
    print('This is a module to define a class of activation functions and layers.')