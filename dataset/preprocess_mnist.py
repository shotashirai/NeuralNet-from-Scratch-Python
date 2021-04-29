# coding: utf-8

import os
import os.path
import numpy as np
import pickle
import cv2

url_mnist = 'http://yann.lecun.com/exdb/mnist/'
data_file = os.path.dirname(os.path.abspath(__file__)) + "/mnist.pkl"

n_train = 60000
n_test = 10000
dim_img = (1, 28, 28)
size_img = 784

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """ Load and preprocess MNIST dataset 
    Parameters:
        - normalize: normalize image pixel values between 0.0 and 1.0
        - one_hot_label: 
            True: one_hot_label is returned as one-hot array
            ** One_hot array: ex) [0,0,0,0,1,0,0,0,0]
        - flatten: convert images into one dimensional array

    Returns: (training image, trainign label), (test image, test label)
    """
    with open(data_file, 'rb') as f: # 'rb': open a file in binary format
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label']  = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

if __name__ == '__main__':
    print('This is a module for preprocessing MNIST dataset')