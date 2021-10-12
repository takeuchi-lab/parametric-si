import numpy as np
import portion as p

from . import sfs
from . import si
from . import si_cv

from typing import List

def parametric_sfs_si(X:np.ndarray,y:np.ndarray,k:int,sigma:int=1,alpha:float=0.05)-> si.SI_result:
    """Compute selective p-values and selective confidence intervals for the coefficients of the features selected by forward SFS at a fixed value of the hyperparameter k.

    This function computes selective p-values and selective confidence intervals for the coefficients of the features selected by forward SFS at a fixed value of the hyperparameter k. 

    Args:
        X (np.ndarray): feature matrix of shape (n_samples, p_features)
        y (np.ndarray): response vector of shape (n_samples, 1)
        k (int): number of features to be selected (hyperparameter)
        sigma (int, optional): variance for selective inference, default=1.0.
        alpha (float, optional): significance level for confidence intervals, default=0.05.

    Returns:
        si.SI_result: reffer to document of SI_result
    """

    A,s = sfs.sfs(X,y,k)

    return si.parametric_si(X,y,A,k,sigma,region,alpha)

def parametric_sfs_cv_si(X:np.ndarray,y:np.ndarray,k_candidates:List[float],k_folds:int,sigma:int=1,alpha:float=0.05)-> si.SI_result:
    """Compute selective p-values and selective confidence intervals for the coefficients of the features selected by forward SFS at the value of the hyperparameter k chosen by cross-validation.

    This function computes selective p-values and confidence intervals for the coefficient of the features selected by forward SFS at the value of the hyperparameter k chosen by cross-validation.

    Args:
        X (np.ndarray): feature matrix of shape (n_samples, p_features)
        y (np.ndarray): response vector of shape (n_samples, 1)
        k_candidates (List[float]): list of candidates for hyperparameter k
        k_folds (int): number of folds in cross validation
        sigma (int, optional): variance for selective inference, default=1.0.
        alpha (float, optional): significance level for confidence intervals, default=0.05.


    Returns:
        si.SI_result: please reffer to document of SI_result
    """

    A,k = sfs.sfs_CV(X,y,k_candidates,k_folds)

    return si_cv.parametric_si_cv(X,y,A,k,k_candidates,sigma,region,k_folds,alpha)

def region(X,y,z,k,a,b):

    y =  a + b * z

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