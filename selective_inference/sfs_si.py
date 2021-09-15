import numpy as np
import portion as p

from . import sfs
from . import si

def parametric_sfs_si(X,y,k):

    A,s = sfs.sfs(X,y,k)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_p(X,y,A,k,Sigma,region)

def parametric_sfs_ci(X,y,k):

    A,s = sfs.sfs(X,y,k)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_ci(X,y,A,k,Sigma,region)

def parametric_sfs_cv_si(X,y,k_candidates,k_folds):

    A,k = sfs.sfs_CV(X,y,k_candidates,k_folds)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_cv_p(X,y,A,k,k_candidates,Sigma,region,k_folds)

def region(X,y,k,a,b):

    A,s = sfs.sfs(X,y,k)
    A_c = list(range(X.shape[1]))
    L,U = -np.inf,np.inf

    for i in range(k):
        X_Ai = X[:,A[0:i]]
        x_ji = X[:,A[i]]
        P = np.identity(X.shape[0]) - X_Ai @ np.linalg.inv(X_Ai.T @ X_Ai) @ X_Ai.T
        A_c.remove(A[i])
        for j in A_c:
            x_j = X[:,j]

            a_plus = (x_j - s[i] * x_ji).T @ P @ a
            b_plus = (x_j - s[i] * x_ji).T @ P @ b
            a_minus = (-x_j - s[i] * x_ji).T @ P @ a
            b_minus = (-x_j - s[i] * x_ji).T @ P @ b

            if b_plus < 0:
                L = max(L,-a_plus/b_plus)
            elif b_plus > 0:
                U = min(U,-a_plus/b_plus)
            
            if b_minus < 0:
                L = max(L,-a_minus/b_minus)
            elif b_minus > 0:
                U = min(U,-a_minus/b_minus)
    
    assert L < U

    return L,U,A