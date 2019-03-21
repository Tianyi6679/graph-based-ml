import numpy as np

def softmax(X):
    ps = np.exp(X)
    sum = np.sum(ps, axis = 1)
    ps /= sum.reshape(-1,1)
    return sum, ps

def normalize(X):
    return X/X.sum(axis=1).reshape(len(X),1)