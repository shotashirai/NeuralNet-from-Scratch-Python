# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def numerical_gradient(f, x, dx=1e-6):
    """ Numerical Gradient

    Parameters
    ----------
    f: function (e.g. softmax, sigmoid, etc...))
    x: variables
    dx (default: 1e-6): an infinitesimal change in x

    Returns
    -------
    grad: numerical gradient
    """
    
    grad = np.zeros_like(x) # Initialize gradient values as zeros
    
    # Get iterator object for numpy arrays
    it = np.nditer(x, flags=['multi_index'])

    while not it.finished: 
        # index for an iteration
        idx = it.multi_index

        # Calculate f(x+h)
        x_dx1 = x[idx]  + dx
        f_dx1 = f(x_dx1) # f(x+h)
        
        # Calculate f(x-h)
        x_dx2 = x[idx]  - dx
        f_dx2 = f(x_dx2) # f(x-h)

        # Get gradient value at x[idx]
        # (f(x+dx) - f(x-dx))/2*dx
        grad[idx] = (f_dx1 - f_dx2) / (2*dx)
        
        
        # Go to the next iteration
        it.iternext()   
        
    return grad


def test_func(x):
    if (x.ndim == 1) | (x.ndim == 0):
        return np.sum(x**2)
    else: return np.sum(x**2, axis=1)

def test_func_1D(x):
    return x**3 - 3*x

if __name__ == '__main__':
    print('This is a module to defnie a function for calcualtion of numerical gradient.')
    # ------ Gradient for 1d data ------------
    # x0 = np.arange(-3, 3.5, 0.25)
    # grad_1D = numerical_gradient(test_func_1D, x0)
    # TODO Visualize the 1-D gradient
    # ------ Gradient for 2-d data -----------
    x0 = np.arange(-3, 3.5, 0.25)
    x1 = np.arange(-3, 3.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()
    Z = np.array([X, Y]).T

    grad = numerical_gradient(test_func, Z).T

    fig, ax = plt.subplots()
    ax.quiver(X, Y, -grad[0], -grad[1]
            , angles='xy', color='#666666')

    ax.set_xlabel('x0')
    ax.set_ylabel('x1') 
    ax.set_title('2D gradient vector map')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])

    plt.grid()
    plt.draw()
    plt.show()