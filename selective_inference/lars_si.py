import numpy as np
import portion as p

from . import lars
from . import si
from . import si_cv

from typing import List

np.seterr(divide='ignore', invalid='ignore')

def parametric_lars_si(X:np.matrix,y:np.matrix,k:int,sigma=1,alpha=0.05):
    """parametric selective inference for lars

    Args:
        X (np.ndarray): design matrix(n x p)
        y (np.ndarray): obejective variable(n x 1)
        k (int): number of feature to be selected(hyperparameter)
        sigma (int, optional): variance for selective inference. Defaults to 1.
        alpha (float, optional): significance level. Defaults to 0.05.

    Returns:
        si.SI_result: reffer to document of SI_result
    """

    A = lars.lars(X,y,k)[0][-1]

    return si.parametric_si(X,y,A,k,sigma,region,alpha)

def parametric_lars_cv_si(X,y,k_candidates,k_folds,sigma=1,alpha=0.05):
    """parametic selective inference for lars with cross validation

    Args:
        X (np.ndarray): design matrix(n x p)
        y (np.ndarray): obejective variable(n x 1)
        k_candidates (List[float]): list of k candidates
        k_folds (int): fold number in cross validation
        sigma (float, optional): variance for selective inference. Defaults to 1.
        alpha (float, optional): significance level. Defaults to 0.05.

    Returns:
        si.SI_result: please reffer to document of SI_result
    """

    A,k = lars.lars_CV(X,y,k_candidates,k_folds)

    return si_cv.parametric_si_cv(X,y,A,k,k_candidates,sigma,region,k_folds,alpha)
    
def region(X,z,step,a,b):

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
        
        pinvk1 = np.linalg.pinv(X_A1.T @ X_A1)
        pinvk2 = []

        if k > 1:
            pinvk2 = np.linalg.pinv(X_A2.T @ X_A2)
        
        P_1 = (np.identity(X.shape[0]) - X_A1 @ pinvk1 @ X_A1.T)
        P_2 = []

        if k > 1:
            P_2 = (np.identity(X.shape[0]) - X_A2 @ pinvk2 @ X_A2.T)

        # c(jk,sk)>c(j,s) for all (j,s) \in S[k] \ {(jk,sk)}
        Sk = S[k]
        Sk.remove([jk,sk])
        A_tmp3 = np.empty([len(Sk),X.shape[0]])
        for i,(j,s) in enumerate(Sk):
            alpha = c(j,s,P_1,pinvk1,X_A1,s_1,X) - c(jk,sk,P_1,pinvk1,X_A1,s_1,X)
            A_tmp3[i,:] = alpha.reshape(-1)
        
        # -c() < 0 
        A_tmp4 = (-1 * c_(jk,sk,A_1,s_1,X)).reshape(1,-1)

        # c(j,s,Ak-1,sk-2) < c(jk-1,sk-1,Ak-2,sk-2) for (j,s) \in Sk
        A_tmp5 = np.empty([len(Sk),X.shape[0]])
        for i,(j,s) in enumerate(S[k]):
            alpha = np.empty(0)
            if k == 1:
                alpha = c(j,s,P_1,pinvk1,X_A1,s_1,X) - (sk_1 * X[:,jk_1]).reshape(-1)
            else :
                alpha = c(j,s,P_1,pinvk1,X_A1,s_1,X) - c(jk_1 , sk_1,P_2,pinvk2,X_A2,s_2,X)
            A_tmp5[i,:] = alpha.reshape(-1)

        # c(jk-1,sk-1,Ak-2,sk-2)-c(j,s,Ak-1,sk-1) for all (j,s) \in A_C*{-1,1}\Sk
        A_tmp6 = np.empty([len(Sb[k]),X.shape[0]])
        for i,(j,s) in enumerate(Sb[k]):
            alpha = np.empty(0)
            if k == 1:
                alpha =  (sk * X[:,jk_1]).reshape(-1) - c(j,s,P_1,pinvk1,X_A1,s_1,X)
            else : 
                alpha =  c(jk_1,sk_1,P_2,pinvk2,X_A2,s_2,X) - c(j,s,P_1,pinvk1,X_A1,s_1,X)

            A_tmp6[i,:] = alpha.reshape(-1)
        
        A_mat = np.vstack([A_mat,A_tmp3,A_tmp4,A_tmp5,A_tmp6])
    
    b_Aa = -1 * (A_mat @ a)
    temp1 = A_mat @ b
    temp2 = b_Aa / temp1

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

def c(j,s,P,X_pinv,X_A,S,X):
    X_j = X[:,j]
    return (P @ X_j) / (s - X_j @ X_A @ X_pinv @ S)