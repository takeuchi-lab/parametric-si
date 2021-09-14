import numpy as np

from . import cv

def sfs(X,y,k):

    A = []
    A_c = list(range(X.shape[1]))
    s = []

    for t in range(0,k):
        X_A = X[:,A]
        correlation = (y - X_A @ np.linalg.inv(X_A.T @ X_A) @ X_A.T @ y).T @ X[:,A_c]
        index = np.argmax(np.abs(correlation))
        s.append(np.sign(correlation[index]))
        A.append(A_c[index])
        A_c.remove(A_c[index])
    
    return A,s

def sfs_CV(X,y,K,k_cv):
    k = K[np.argmin([cv.cv_error(k,X,y,k_cv,lambda X,y,k:sfs(X,y,k)[0]) for k in K])]
    A,s = sfs(X,y,k)
    return A,k