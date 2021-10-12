from turtle import shape
import numpy as np
import portion as p

from . import lars
from . import si
from . import si_cv

from typing import List

np.seterr(divide='ignore', invalid='ignore')

def parametric_lars_si(X:np.matrix,y:np.matrix,k:int,sigma=1,alpha=0.05):
    """Compute selective p-values and selective confidence intervals for the coefficients estimated by LARS algorithm at a fixed value of the hyperparameter k.

    This function computes selective p-values and selective confidence intervals for the coefficients of features selected by LARS at a fixed value of the hyperparameter k. Feature matrix must be centered and scaled, and the response vector must be centered.

    Args:
        X (np.ndarray): feature matrix of shape (n_samples, p_features)
        y (np.ndarray): response vector of shape (n_samples, 1)
        k (int): number of feature to be selected (hyperparameter)
        sigma (int, optional): variance for selective inference, default=1.0.
        alpha (float, optional): significance level for confidence intervals, default=0.05.
    Returns:
        si.SI_result: refer to document of SI_result
    """

    A = lars.lars(X,y,k)[0][-1]

    return si.parametric_si(X,y,A,k,sigma,region,alpha)

def parametric_lars_cv_si(X,y,k_candidates,k_folds,sigma=1,alpha=0.05):
    """Compute selective p-values and selective confidence intervals for the coefficients estimated by LARS algorithm at the value of the hyperparameter k chosen by cross-validation.

    This function computes selective selective p-values and selective confidence intervals for the coefficients of features selected by LARS at the value of the hyperparameter k chosen by cross-validation. Feature matrix must be centered and scaled, and the response vector must be centered.

    Args:
        X (np.ndarray): feature matrix of shape (n_samples, p_features)
        y (np.ndarray): response vector of shape (n_samples, 1)
        k_candidates (List[float]): list of candidates for k
        k_folds (int): number of folds in cross validation
        sigma (int, optional): variance for selective inference, default=1.0.
        alpha (float, optional): significance level for confidence intervals, default=0.05.

    Returns:
        si.SI_result: please refer to document of SI_result
    """

    A,k = lars.lars_CV(X,y,k_candidates,k_folds)

    return si_cv.parametric_si_cv(X,y,A,k,k_candidates,sigma,region,k_folds,alpha)
    
def region(X,y,z,step,a,b):

    y = a + b * z

    A,A_c,signs,S,Sb = lars.lars(X,y,step)
    A_mat = np.empty(0)

    # 1st step
    jk = A[0]
    xk = X[:,jk]
    sk = signs[0]

    # when s = -1
    A_tmp1 = np.empty([len(A_c[0]),X.shape[0]])
    for i,l in enumerate(A_c[0]):
        alpha = X[:,l] - (sk * xk).reshape(-1)
        A_tmp1[i,:] = alpha
    
    # when s = +1
    A_tmp2 = np.empty([len(A_c[0]),X.shape[0]])
    for i,l in enumerate(A_c[0]):
        alpha = -1 * X[:,l] - (sk * xk).reshape(-1)
        A_tmp2[i,:] = alpha
    
    A_mat = np.vstack([A_tmp1,A_tmp2])

    # after first step 
    for k in range(1,step):
        jk_1 = jk
        sk_1 = sk

        jk = A[k][-1]
        sk = signs[k][-1]

        s_1 = signs[k-1]
        s_2 = []
        A_1 = A[k-1]

        X_A1 = X[:,A[k-1]]
        X_A2 = []

        if k > 1:
            X_A2 = X[:,A[k-2]]
            s_2 = signs[k-2]
        
        pinvkS1 = np.linalg.solve(X_A1.T @ X_A1,s_1)
        pinvkS2 = []

        if k > 1:
            pinvkS2 = np.linalg.solve(X_A2.T @ X_A2,s_2)
        
        P_1 = (np.identity(X.shape[0]) - X_A1 @ np.linalg.solve(X_A1.T @ X_A1,X_A1.T))
        P_2 = []

        if k > 1:
            P_2 = (np.identity(X.shape[0]) - X_A2 @ np.linalg.solve(X_A2.T @ X_A2,X_A2.T))

        # c(jk,sk)>c(j,s) for all (j,s) \in S[k] \ {(jk,sk)}
        Sk = S[k]
        Sk.remove([jk,sk])
        A_tmp3 = np.empty([len(Sk),X.shape[0]])
        for i,(j,s) in enumerate(Sk):
            alpha = c(j,s,P_1,pinvkS1,X_A1,X) - c(jk,sk,P_1,pinvkS1,X_A1,X)
            A_tmp3[i,:] = alpha.reshape(-1)
        
        # -c() < 0 
        A_tmp4 = (-1 * c_(jk,sk,A_1,s_1,X)).reshape(1,-1)

        # c(j,s,Ak-1,sk-2) < c(jk-1,sk-1,Ak-2,sk-2) for (j,s) \in Sk
        A_tmp5 = np.empty([len(Sk),X.shape[0]])
        for i,(j,s) in enumerate(S[k]):
            alpha = np.empty(0)
            if k == 1:
                alpha = c(j,s,P_1,pinvkS1,X_A1,X) - (sk_1 * X[:,jk_1]).reshape(-1)
            else :
                alpha = c(j,s,P_1,pinvkS1,X_A1,X) - c(jk_1 , sk_1,P_2,pinvkS2,X_A2,X)
            A_tmp5[i,:] = alpha.reshape(-1)

        # c(jk-1,sk-1,Ak-2,sk-2)-c(j,s,Ak-1,sk-1) for all (j,s) \in A_C*{-1,1}\Sk
        A_tmp6 = np.empty([len(Sb[k]),X.shape[0]])
        for i,(j,s) in enumerate(Sb[k]):
            alpha = np.empty(0)
            if k == 1:
                alpha =  (sk * X[:,jk_1]).reshape(-1) - c(j,s,P_1,pinvkS1,X_A1,X)
            else : 
                alpha =  c(jk_1,sk_1,P_2,pinvkS2,X_A2,X) - c(j,s,P_1,pinvkS1,X_A1,X)

            A_tmp6[i,:] = alpha.reshape(-1)
        
        A_mat = np.vstack([A_mat,A_tmp3,A_tmp4,A_tmp5,A_tmp6])
    
    b_Aa = -1 * (A_mat @ a)
    temp1 = A_mat @ b
    temp2 = b_Aa.astype(np.float128) / temp1.astype(np.float128)

    temp2 = temp2.astype(np.float64)

    L = max(temp2[temp1 < 0],default=-np.inf)
    U = min(temp2[temp1 > 0],default=np.inf)

    assert L < U

    return L,U,A[-1]

def c_(j,s,A,S,X):
    n,d = X.shape
    X_A = X[:,A]
    X_j = X[:,j]
    P = (np.identity(n) - X_A @ np.linalg.pinv(X_A.T @ X_A) @ X_A.T)
    temp = (P @ X_j) /(s - X_j @ X_A @ np.linalg.pinv(X_A.T @ X_A) @ S)
    return temp

def c(j,s,P,X_pinvS,X_A,X):
    X_j = X[:,j]
    temp1 = (P @ X_j).astype(np.float128)
    temp2 = (s - X_j @ X_A @ X_pinvS).astype(np.float128)
    return  (temp1 / temp2).astype(np.float64)