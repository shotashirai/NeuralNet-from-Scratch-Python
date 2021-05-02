# Neural Network from scratch in Python
Build a neural network from scratch in Python.

## Project Summary
This project aims to build and train a multi-layer neural network from scratch in Python without using existing machine learning libraries. Numpy is used for matrix operation and mathematical calculation. The built network is evaluated with a task of the **handwritten digit recognition** using the MNIST dataset.

## Network structure
The network has three types of layers: input layer, hidden layers and output layer. The network is defined by ```MultilayerNetwork``` class and the number of the hidden layers can be set to be an arbitrary number (In this project, three hidden layers are used for training network and classification task.).   
As activation functions, ReLU or Sigmoid are used and **cross-entropy loss** is used as loss function. To reduce the variance of the model, weight decay (L2 regularization) is applied during training the network. To reduce training time, **mini-batch learning** is also used.


## Code
- train_neuralnet.py: main code to implement training and classification task for MNIST handwritten recognition.  
- functions.py: define functions (e.g., ```cross_entropy_error()```)  
- layers.py: define classes of Affine layer, activation layer (ReLU, Sigmoid) and softmax function with a loss function.  
- gradient.py: define a function for the calculation of numerical gradient.  
- preprocess_minist.py: define a function to process the MNIST dataset.  

## Todo (in progress)
Add more functions and methods to the current version of the network e.g., Batch normalization, dropout, convolutional layer, optimization (Stochastic Gradient Descent(SDG), Momentum SDG, Nesterov's Accelerated Gradient, AdaGrad, RMSprop, Adma)
