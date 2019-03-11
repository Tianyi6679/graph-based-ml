import numpy as np

def softmax(X, theta):
    ps = np.exp(X-np.max(X))
    ps /= np.sum(ps)
    return ps

def normalize(X):
    for row in X:
        row /= row.sum()
    return X