import numpy as np

from . import cv

def sfs(X,y,k):

    A = []
    A_c = list(range(X.shape[1]))
    s = []

    correlation = y.T @ X
    j_0 = A_c[np.argmax(np.abs(correlation))]
    s_0 = np.sign(correlation[j_0])

    A.append(j_0)
    s.append(s_0)

    for t in range(1,k):
        X_A = X[:,A]
        correlation = (y - X_A @ np.linalg.inv(X_A.T @ X_A) @ X_A.T @ y).T @ X[:,A_c]
        j_t = A_c[np.argmax(np.abs(correlation))]
        A.append(j_t)
        s.append(np.sign(correlation[j_t]))
    
    return A,s

def sfs_CV(X,y,K,k_cv):
    k = K[np.argmin([cv.cv_error(sfs(X,y,k)[0],X,y,k_cv) for k in K])]
    return sfs(X,y,k),k