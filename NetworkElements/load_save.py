# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import cloudpickle as cp



def load_model(dir_name, file_name):
    """ Load built network

    Parameters
    ----------
    dir_name: Directory name for saving a model
    file_name (.cpkl): file name for saving 
    Returns
    -------
    network (object)
    """
    with open(dir_name + '/' + file_name + '.cpkl', 'rb') as f:
            network = cp.load(f)
    return network

    
def save_model(dir_name, file_name, model_obj):
    """ Save network structure (network objects)
    The model is saved as .cpkl file using cloudpickle
    Parameters
    ----------
    dir_name: Directory name for saving a model
    file_name (.cpkl): file name for saving 
    Returns
    -------
    None
    """
    with open(dir_name + '/' + file_name + '.cpkl', 'wb') as f:
            cp.dump(model_obj, f)


# def save_params(dir_name, file_name, params_model):
#     """ Save weights and biases in a pickle file
#     """
#     # params: weights and biases
#     params = {}

#     for key, val in params_model.items():
#         params[key] = val
    
#     with open(dir_name + '/' + file_name, 'wb') as f:
#         pickle.dump(params, f)
