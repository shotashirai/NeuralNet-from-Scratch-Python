#coding: utf-8

import numpy as np
import matplotlib.pyplot  as plt

# Identify function
def identify_function(x):
    return x

# Step function
def step_function(x):
    return np.array(x > 0, dtype=np.int)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

# ReLU (Rectified Linear Unit) function
def relu_function(x):
    return np.maximum(0,x)

# Softmax
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # Avoid overflow
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# Cross Entropy Error
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # When the training data is one-hot-vector, convert to the index of the training label
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


if __name__ == '__main__':
    # Demo
    input_values = np.arange(-5, 5, 0.2,  dtype=float)
    fig, ax = plt.subplots()

    # ax.plot(input_values, identify_function(input_values)
    # ,  label='Identify function')
    ax.plot(input_values, step_function(input_values)
    , label='Step function')
    ax.plot(input_values, sigmoid(input_values)
    , label='Sigmoid function')
    ax.plot(input_values, relu_function(input_values)
    , label='ReLU')
    ax.plot(input_values, softmax(input_values)
    , label='Softmax')

    ax.legend()
    ax.set_xlabel('Input values')
    ax.set_ylabel('Output values') 
    ax.set_title('Activation functions')
    ax.set_ylim([-0.2, 2.5])

    plt.show()

