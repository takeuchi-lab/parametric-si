import numpy as np
import portion as p
from sklearn import linear_model

from . import lasso
from . import si
from . import si_cv

from typing import List

np.seterr(divide='ignore', invalid='ignore')

def parametric_lasso_si(X,y,k,sigma=1,alpha=0.05):
    """Compute selective p-values and selective confidence intervals for the coefficients of features selected by Lasso at a fixed value of the hyperparameter k. 

    This function computes selective p-values and selective confidence intervals for the coefficients of features selected by Lasso at a fixed value of the hyperparameter k.

    Args:
        X (np.ndarray): feature matrix of shape (n_samples, p_features)
        y (np.ndarray): response vector of shape (n_samples, 1)
        k (int): regularization parameter of lasso
        sigma (int, optional): variance for selective inference, default=1.0.
        alpha (float, optional): significance level for confidence intervals, default=0.05.

    Returns:
        si.SI_result: reffer to document of SI_result
    """

    A,s = lasso.lasso(X,y,k)

    return si.parametric_si(X,y,A,k,sigma,region,alpha)

def parametric_lasso_cv_si(X,y,k_candidates,k_folds,sigma=1,alpha=0.05):
    """Compute selective p-values and selective confidence intervals for the coefficients of features selected by Lasso at the value of the hyperparameter k chosen by cross-validation.
    
    This function computes selective p-values and selective confidence intervals for the coefficients of features selected by Lasso at the value of the regularization parameter k chosen by cross-validation.

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

    A,k = lasso.lasso_CV(X,y,k_candidates,k_folds)

    return si_cv.parametric_si_cv(X,y,A,k,k_candidates,sigma,region,k_folds,alpha)

def lee_et_all_region(X,z,alpha,a,b):
    """this function is made with reference to the lasso selection event of Lee et al. (2016)

    Args:
        X ([type]): [description]
        z ([type]): [description]
        alpha ([type]): [description]
        a ([type]): [description]
        b ([type]): [description]

    Returns:
        [type]: [description]
    """

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

def region(X,y,z,k,a,b):
    n,p = X.shape
    
    yz_flatten = a + b * z
    yz = yz_flatten.reshape(-1,1)

    clf = linear_model.Lasso(alpha=k, fit_intercept=False,tol=1e-10)
    clf.fit(X, yz)
    bhz = clf.coef_

    Az, XAz, Acz, XAcz, bhAz = construct_A_XA_Ac_XAc_bhA(X, bhz, n, p)

    if XAz is not None:
        inv = np.linalg.pinv(np.dot(XAz.T, XAz))
        invXAzT = np.dot(inv, XAz.T)
        etaAz = np.dot(invXAzT, b)

    shAz = np.array([])
    gammaAz = np.array([])

    if XAcz is not None:
        if XAz is None:
            e1 = yz
        else:
            e1 = yz - np.dot(XAz, bhAz)

        e2 = np.dot(XAcz.T, e1)
        shAz = e2/(k * n)

        if XAz is None:
            gammaAz = (np.dot(XAcz.T, b)) / n
        else:
            gammaAz = (np.dot(XAcz.T, b) - np.dot(np.dot(XAcz.T, XAz), etaAz)) / n
    
    bhAz = bhAz.flatten()
    etaAz = etaAz.flatten()
    shAz = shAz.flatten()
    gammaAz = gammaAz.flatten()

    min1 = np.Inf
    min2 = np.Inf

    for j in range(len(etaAz)):
        numerator = - bhAz[j]
        denominator = etaAz[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min1:
            min1 = quotient

    for j in range(len(gammaAz)):
        numerator = (np.sign(gammaAz[j]) - shAz[j])*k
        denominator = gammaAz[j]

        quotient = compute_quotient(numerator, denominator)

        if quotient < min2:
            min2 = quotient

    skz = min(min1, min2)
    return z-si.EPS,z+skz, np.where(bhz != 0)[0].tolist()

def construct_A_XA_Ac_XAc_bhA(X, bh, n, p):
    A = []
    Ac = []
    bhA = []

    for j in range(p):
        bhj = bh[j]
        if bhj != 0:
            A.append(j)
            bhA.append(bhj)
        else:
            Ac.append(j)

    XA = X[:, A]
    XAc = X[:, Ac]
    bhA = np.array(bhA).reshape((len(A), 1))

    return A, XA, Ac, XAc, bhA

def compute_quotient(numerator,denominator):
    if denominator == 0:
        return np.Inf
    else:
        quotient = numerator / denominator

        if quotient <= 0:
            return np.Inf

        return quotient
