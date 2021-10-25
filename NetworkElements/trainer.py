# coding: utf-8

import sys, os
sys.path.append(os.pardir)
import numpy as np

from NetworkElements.load_save import *

class Trainer:

    def __init__(self, network, x_train, t_train, x_test, t_test
                , epochs, batch_size=100, verbose=0
                , learning_rate=0.1
                , save_network=True, save_params_epoch=None):
        
        # Initialize input
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        # Get parameters from data
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        
        # TODO Add optimizer (SGD, MomentumSGD, Adagrad...)
        # TODO Add convolutional/pooling layer options

        # Initialize epoch and iteration
        self.current_epoch = 0
        self.current_iter = 0

        # allocation for logs of loss fucntion, train accuracy, test accuracy 
        self.train_loss = []
        self.train_acc = []
        self.test_acc = []

        # Save network/parameters
        self.save_network = save_network
        self.save_params_epoch = save_params_epoch

    def train_iter(self):
        """ training at an iteration
        """
        batch_mask = np.random.choice(self.train_size, self.bathc_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # Calculate gradient for batch data
        grad = self.network.gradient(x_batch, t_batch)

        # Based on grad, update weights and biases
        for idx in range(1, len(self.hidden_size)+1):
            self.network.params['W' + str(idx)] -= self.learning_rate * grad['W' + str(idx)]
            self.network.params['b' + str(idx)] -= self.learning_rate * grad['b' + str(idx)]
        # TODO Replace with different optimizer defined by class or function and add options 

        loss_func = self.network.loss(x_batch, t_batch)
        # save the value from loss functio to the listn 
        self.train_loss.append(loss_func)

        if self.verbose==2:
            print(' === train loss (epoch:' + str(self.current_epoch) + '/' + str(self.epochs) + ') ===')
            print(str(loss_func))
        
        # Update Epoch
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            # Get accuracy of the prediction on the current learning
            acc_train = self.network.accuracy(self.x_train, self.t_train)
            acc_test = self.network.accuracy(self.x_test, self.t_test)

            # Append the accuracy scores to the lists
            self.train_acc.append(acc_train)
            self.test_acc.append(acc_test)

            if self.verbose == 1:
                print('=== Accuracy Score (epoch:' + str(self.current_epoch) + '/' + str(self.epochs) + ') ===')
                print('Train: ' + str(acc_train))
                print('Test: ' + str(acc_test))
        
        # Save parameters
        if not self.save_params_epoch is None:
            if (self.current_iter % self.iter_per_epoch == 0) & (self.save_params_epoch == self.current_epoch):
                file_name = 'params-epoch' + str(self.current_epoch)
                self.network.save_params(dir_name='trained_params', file_name=file_name)
                print('==== Parameters (weights & biases) at epoch ' 
                        + str(self.current_epoch)
                        +  ' are saved! ====')

        # Update the current iteration
        self.current_iter += 1

    def train(self):
        print('*** Network training started... ***')

        for i in range(self.max_iter):
            self.train_iter()
        
        final_acc_test = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose == 1:
            print('========= Final Accuracy Score =========')
            print('Test: ' + str(final_acc_test))
            print('========================================')

        print('*** Network training has been done! ***')

        if self.save_network:
            print('=================================================')
            print('==== Trained network was saved in .pkl file! ====')
            print('=================================================')

        save_model(dir_name='trained_model', file_name='model', model_obj=self.network)