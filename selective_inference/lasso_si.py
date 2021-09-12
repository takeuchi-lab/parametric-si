import numpy as np
import portion as p

from . import lasso
from . import si

def parametric_lasso_si(X,y,alpha):

    A,s = lasso.lasso(X,y,alpha)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_p(X,y,A,alpha,Sigma,region)

def parametric_lasso_ci(X,y,alpha):

    A,s = lasso.lasso(X,y,alpha)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_ci(X,y,A,alpha,Sigma,region)

def region(X,y,alpha,a,b):

    L,U = -np.inf,np.inf
    A,s = lasso.lasso(X,y,alpha)

    Ac = [i for i in range(X.shape[1]) if i not in A]
    print(A)
    print(Ac)

    X_A = X[:,A]
    X_Ac = X[:,Ac]

    P = X_A @ np.linalg.inv(X_A.T @ X_A) @ X_A.T

    X_A_inv = np.linalg.inv(X_A.T @ X_A)

    # please refer to lee et al paper
    A0_plus  = 1 / alpha * (X_Ac.T @ (np.identity(X.shape[0]) - P))
    A0_minus  = 1 / alpha * (-1 * X_Ac.T @ (np.identity(X.shape[0]) - P))
    b0_plus =  np.ones(X_Ac.shape[1]) - (X_Ac.T @ np.linalg.inv(X_A @ X_A.T) @ X_A @ s)
    b0_minus =  np.ones(X_Ac.shape[1]) + (X_Ac.T @ np.linalg.inv(X_A @ X_A.T) @ X_A @ s)
    A1 = -1 * np.diag(s) @ X_A_inv @ X_A.T
    b1 = -1 * alpha * np.diag(s) @ X_A_inv @ s

    # A_mat(a+bz) \leq b_vec
    A_mat = np.vstack([A0_plus,A0_minus,A1])
    b_vec = np.hstack([b0_plus,b0_minus,b1])

    Ab = A_mat @ b
    Aa = A_mat @ a

    temp = (b_vec - Aa) / Ab
    print(temp[Ab < 0])
    print(temp[Ab > 0])

    # caliculate interval
    L = np.max(temp[Ab < 0])
    U = np.min(temp[Ab > 0])

    temp = p.closed(L,U)
    print(L,U)
    print(temp)

    return p.closed(L,U),A