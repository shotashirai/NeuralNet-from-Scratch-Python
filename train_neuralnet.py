# coding: utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.preprocess_mnist import load_mnist
from myModules.multi_layer_net import MultiLayerNetwork


# import data 
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# hidden layers
hidden_size = [50, 50, 50]
# Generate network
network = MultiLayerNetwork(input_size=784, hidden_size=hidden_size, output_size=10)

# parameter settings
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

current_epoch = 0

train_loss_list = [] # list of loss function 
train_acc_list = [] # list of accuracy for training
test_acc_list = [] # list of accuracy for test

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # randomly select images for taraining network
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # gradient
    grad = network.gradient(x_batch, t_batch)

    # update weights and biases
    for idx in range(1, len(hidden_size)+1):
        network.params['W' + str(idx)] -= learning_rate * grad['W' + str(idx)]
        network.params['b' + str(idx)] -= learning_rate * grad['b' + str(idx)]

    # loss function (Cross Entropy Loss)
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        current_epoch += 1
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # print(train_acc, test_acc)
        print(
            "=== epoch:" + str(current_epoch) 
            + ", train acc:" + str(train_acc) 
            + ", test acc:" + str(test_acc)
        )

    if i == iters_num-1:
        test_acc = network.accuracy(x_test, t_test)
        print("============= Final Test Accuracy ============= ")
        print("test acc: " + str(test_acc))