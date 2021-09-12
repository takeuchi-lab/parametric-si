import numpy as np
import portion as p

from . import lasso
from . import si

def parametric_lasso_si(X,y,alpha):

    A,s = lasso.lasso(X,y,alpha)
    print(A)
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

    X_A = X[:,A]
    X_Ac = X[:,Ac]

    P = X_A @ np.linalg.pinv(X_A.T @ X_A) @ X_A.T

    X_A_pinv = np.linalg.pinv(X_A.T @ X_A)

    # please refer to lee et al paper
    A0_plus  = 1 / (alpha*X.shape[0]) * (X_Ac.T @ (np.identity(X.shape[0]) - P))
    A0_minus  = 1 / (alpha*X.shape[0]) * (-1 * X_Ac.T @ (np.identity(X.shape[0]) - P))
    b0_plus =  np.ones(X_Ac.shape[1]) - (X_Ac.T @ np.linalg.pinv(X_A @ X_A.T) @ X_A @ s)
    b0_minus =  np.ones(X_Ac.shape[1]) + (X_Ac.T @ np.linalg.pinv(X_A @ X_A.T) @ X_A @ s)
    A1 = -1 * np.diag(s) @ X_A_pinv @ X_A.T
    b1 = -1 * (alpha*X.shape[0]) * np.diag(s) @ X_A_pinv @ s

    L,U = solve(A0_plus,b0_plus,a,b,L,U)
    L,U = solve(A0_minus,b0_minus,a,b,L,U)
    L,U = solve(A1,b1,a,b,L,U)

    return L,U,A

def solve(C,d,a,b,L,U):

    for i in range(C.shape[0]):
        numerator = d[i] - C[i,:] @ a
        denominator = C[i,:] @ b
        if denominator > 0:
            U = min(U,numerator/denominator)
        elif denominator < 0:
            L = max(L,numerator/denominator)

    return L,U
