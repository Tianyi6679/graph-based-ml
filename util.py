import numpy as np

def softmax(X, theta):
    ps = np.exp(X-np.max(X))
    ps /= np.sum(ps)
    return ps

def normalize(X):
    return X/X.sum(axis=1).reshape(len(X),1)