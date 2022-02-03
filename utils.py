import numpy as np


def euclidean_distance(x1, x2):
    
    return np.sqrt(np.sum((x1-x2)**2))



def manhattan_distance(x1, x2):
    
    return np.sum(abs(x1-x2))



def identity(x, derivative = False):
   
    if not derivative:
        return x
    else:
        return 1
    


def sigmoid(x, derivative = False):
    
    if not derivative:
        return 1/(1 + np.exp(-x))
    else:
        return sigmoid(x)*(1 - sigmoid(x))


def tanh(x, derivative = False):
    
    if not derivative:
        return np.tanh(x)
    else:
         return 1 - (tanh(x)**2)

def relu(x, derivative = False):
    
    if not derivative:
        return np.where(x>0, x, 0.0)
    else:    
        return np.where(x>0, 1.0, 0.0)
    


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):

    return -np.mean(y * np.log(p+0.000001))
    

def one_hot_encoding(y):
    
    total_classes = max(y) + 1
    one_hot_matrix = np.zeros((len(y), total_classes))

    for count, _class in enumerate(y):
        one_hot_matrix[count][_class] = 1
    return one_hot_matrix

