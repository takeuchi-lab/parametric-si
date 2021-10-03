import numpy as np
import portion as p
from sklearn import linear_model

from . import lasso
from . import si

def compute_quotient(numerator,denominator):
    if denominator == 0:
        return np.Inf
    else:
        quotient = numerator / denominator

        if quotient <= 0:
            return np.Inf

        return quotient

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

def region(X,z,alpha,a,b):

    y = a * b + z

    n = X.shape[0]

    clf = linear_model.Lasso(alpha=alpha,fit_intercept=False,normalize=False,tol=1e-10)
    clf.fit(X,y)
    coef = clf.coef_

    A = np.where(coef!=0)[0].tolist()
    Ac = np.where(coef==0)[0].tolist()
    XA = X[:,A]
    XAc = X[:,Ac]
    beta_hat = coef[A]

    psiA = np.empty(0)

    if len(A) != 0:
        psiA = np.linalg.pinv(XA.T @ XA) @ XA.T @ b

    sign_hat = np.empty(0)
    gammaA = np.empty(0)

    if len(Ac) != 0:
        if len(A) == 0:
            e1 = y
            gammaA = (XAc.T @ b) / n
        else :
            e1 = y - XA @ beta_hat
            gammaA = ((XAc.T @ b) - (XAc.T @ XA @ psiA)) / n
        
        e2 = XAc.T @ e1
        sign_hat = e2 / (alpha * n)

    min1 = np.Inf
    min2 = np.Inf

    if len(A) != 0:
        min1 = np.min(np.vectorize(compute_quotient)(beta_hat,psiA))
    if len(Ac) != 0:
        min2 = np.min(np.vectorize(compute_quotient)((np.sign(gammaA)-sign_hat)*alpha,gammaA))

    L = z - si.EPS
    U = min(min1,min2)
    print(L,U)

    return L,U,A

def region(X,z,alpha,a,b):

    y = a + b * z

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