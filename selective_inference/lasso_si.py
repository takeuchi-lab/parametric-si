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

def parametric_lasso_cv_si(X,y,k_candidates,k_folds):

    A,k = lasso.lasso_CV(X,y,k_candidates,k_folds)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_cv_p(X,y,A,k,k_candidates,Sigma,region,k_folds)


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

    A_mat = np.vstack((A0_plus,A0_minus,A1))
    b_vec = np.hstack((b0_plus,b0_minus,b1))

    b_Aa = b_vec - (A_mat @ a)
    temp1 = A_mat @ b
    temp2 = b_Aa / temp1

    L = max(temp2[temp1 < 0],default=-np.inf)
    U = min(temp2[temp1 > 0],default=np.inf)

    assert L < U

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
