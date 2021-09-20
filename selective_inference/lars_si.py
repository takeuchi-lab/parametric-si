import numpy as np
import portion as p

from . import lars
from . import si
from . import ci
from . import p_value


def parametric_lars_si(X:np.matrix,y:np.matrix,k:int):

    A = lars.lars(X,y,k)[0][-1]
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_p(X,y,A,k,Sigma,region)

def parametric_lars_ci(X,y,k):

    A = lars.lars(X,y,k)[0][-1]
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_ci(X,y,A,k,Sigma,region)

def parametric_lars_cv_si(X,y,k_candidates,k_folds):

    A,k = lars.lars_CV(X,y,k_candidates,k_folds)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_cv_p(X,y,A,k,k_candidates,Sigma,region,k_folds)
    

#TODO need refactaring because it is too complicated
def region(X,y,step,a,b):

    A,A_c,signs,S,Sb = lars.lars(X,y,step)

    L = -np.inf
    U = np.inf

    # 1st step
    jk = A[0]
    xk = X[:,jk]
    sk = signs[0]

    # when s = -1
    for l in A_c[0]:
        alpha = X[:,l] - (sk * xk).reshape(-1)
        beta = 0
        L, U = solve_inequality(alpha,beta,a,b,L,U)
    
    # when s = +1
    for l in A_c[0]:
        alpha = -1 * X[:,l] - (sk * xk).reshape(-1)
        beta = 0
        L, U = solve_inequality(alpha,beta,a,b,L,U)

    # after 2step 
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
        for (j,s) in Sk:
            alpha = c(j,s,P_1,pinvk1,X_A1,s_1,X) - c(jk,sk,P_1,pinvk1,X_A1,s_1,X)
            L,U = solve_inequality(alpha,0,a,b,L,U)
        
        # -c() < 0 
        alpha = -1 * c_(jk,sk,A_1,s_1,X)
        L,U = solve_inequality(alpha,0,a,b,L,U)

        # c(j,s,Ak-1,sk-2) < c(jk-1,sk-1,Ak-2,sk-2) for (j,s) \in Sk
        for (j,s) in S[k]:
            if k == 1:
                alpha = c(j,s,P_1,pinvk1,X_A1,s_1,X) - (sk_1 * X[:,jk_1]).reshape(-1)
                L,U = solve_inequality(alpha,0,a,b,L,U)
            else :
                alpha = c(j,s,P_1,pinvk1,X_A1,s_1,X) - c(jk_1 , sk_1,P_2,pinvk2,X_A2,s_2,X)
                L,U = solve_inequality(alpha,0,a,b,L,U)
        
        # c(jk-1,sk-1,Ak-2,sk-2)-c(j,s,Ak-1,sk-1) for all (j,s) \in A_C*{-1,1}\Sk
        for (j,s) in Sb[k]:
            if k == 1:
                alpha =  (sk * X[:,jk_1]).reshape(-1) - c(j,s,P_1,pinvk1,X_A1,s_1,X)
                L,U = solve_inequality(alpha,0,a,b,L,U)
            else : 
                alpha =  c(jk_1,sk_1,P_2,pinvk2,X_A2,s_2,X) - c(j,s,P_1,pinvk1,X_A1,s_1,X)
                L,U = solve_inequality(alpha,0,a,b,L,U)
        
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

def solve_inequality(alpha,beta,a,b,L,U):
    temp1 = alpha @ b
    temp2 = beta - (alpha @ a)

    l = L
    u = U

    if temp1 > 0:
        u = min(U,temp2 / temp1)
    elif temp1 < 0:
        l = max(L, temp2 / temp1)
    else:
        return L,U

    return l,u