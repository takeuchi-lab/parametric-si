from dataclasses import dataclass
import numpy as np
import portion as p
from sklearn import linear_model

from .p_value import p_value
from . import ci

from typing import List

EPS = 5e-3

@dataclass
class SI_result:
    """this class have result of selective inference. each selective inference function return this class.

    Attributes:
        A(List[int]):selected feature 
        k(float):hyperparameter of feature selection algorithm
        sigma(float):variance used for inference.
        p_values(List[float]):p-values of selected features
        CIs(List[portion.interval.Interval]):confidence intervals of selected features
    """

    A : list
    k : float
    sigma : float
    p_values: list
    CIs : list

def estimate_sigma(X:np.ndarray,y:np.ndarray)->float:
    """this function estimate variance of the mean squared residual with.
    
    Args:
        X (np.ndarray): design matrix(n x p)
        y (np.ndarray): obejective variable(n x 1)

    Returns:
        float: estimated variacnce 
    """

    n,p = X.shape

    e = y - X @ np.linalg.inv(X.T @ X) @ X.T @ y
    sigma = (e.T @ e) / (n - p - 1)

    return sigma

def estimate_sigma_lasso(X:np.ndarray,y:np.ndarray)->float:
    """this function estimate variance of the mean squared residual with LassoCV.

    Args:
        X (np.ndarray): design matrix(n x p)
        y (np.ndarray): obejective variable(n x 1)

    Returns:
        float: estimated variacnce
    """

    clf = linear_model.LassoCV()
    clf.fit(X,y)
    coef = clf.coef_
    e = y - X @ coef
    sigma = (e.T @ e) / (X.shape[0] - np.count_nonzero(coef!=0) - 1)

    return sigma

def construct_teststatistics(A,i,X,y,Sigma):
    """construct variables for selective inference

    Args:
        A (list): active features
        i (int): number of the active feature of interest
        X (numpy.ndarray): design matrix(n x p)
        y (numpy.ndarray): object variable(n x 1)
        Sigma (numpy.ndarray): covariance matrix of y(n x n)

    Returns:
        4-element tuple containing

        - numpy.ndarray: a(n x 1) are used to construct y on the direction of test statistic.
        - numpy.ndarray: b(n x 1) are used to construct y on the direction of test statistic.
        - float: variance for truncated normal distribution.,
        - float: the observed value of the test statistic.
    """

    X_A = X[:,A]
    e = np.zeros(len(A))
    e[i] = 1

    eta = X_A @ np.linalg.inv(X_A.T @ X_A) @ e
    var = eta.T @ Sigma @ eta

    z_obs = eta @ y
    b = Sigma @ eta / var
    a = (np.identity(X.shape[0]) - b.reshape(-1,1) @ eta.reshape(1,-1)) @ y

    return a,b,var,z_obs

def compute_solution_path(k,X,y,a,b,z_min,z_max,region):
    """compute list of interval and its model on the direction of test statistic

    Args:
        k (step number): step number of algorithm(in lasso this is regulization parameter,in sfs or lars this is number of features to choose)
        X (numpy.ndarray): design matrix(n x p)
        y (numpy.ndarray): object variable(n x 1)
        a (numpy.ndarray): direction of test statistic(n x 1)
        b (numpy.ndarray): direction of test statistic(n x 1)
        z_min (): minumum value of test statistic to search
        z_max (): maximum value of test statistic to search
        region (function): function to compute interval for each algorithm

    Returns:
        tuple : intervals is a list of closed interval that is [lower,upper]
                models is a list of active features that are in the interval
    """

    z = z_min

    intervals = []
    models = []

    while z < z_max:

        L,U,model_z = region(X,y,z,k,a,b)
        
        models.append(model_z)
        intervals.append(p.closed(max(L,z_min),min(U,z_max)))
        if type(intervals[-1].upper) == type(p.empty().upper):
            assert False
        z = U + EPS
    
    return intervals,models

def parametric_si(X,y,A,k,sigma,region,alpha):
    """calculate selective p-value for each active feature

    Args:
        X (numpy.ndarray): design matrix(n x p)
        y (numpy.ndarray): object variable(n x 1)
        A (list): active features
        k (int): hyperparameter
        Sigma (numpy.ndarray): covariance matrix of y(n x n)
        region (function): function to compute interval for each algorithm

    Returns:
        tuple : p_valus is a list of p-value for each active features
                A is list of active features
    """
    Sigma = np.identity(X.shape[0]) * sigma

    p_values = []
    CIs = []

    for i in range(len(A)):
        intervals = p.empty()

        a,b,var,z_obs = construct_teststatistics(A,i,X,y,Sigma)

        std = np.sqrt(var)
        z_min = -1 * std * 10
        z_max = std * 10

        regions,models = compute_solution_path(k,X,y,a,b,z_min,z_max,region)

        for r,model in zip(regions,models):
            if set(A) == set(model):
                intervals = intervals | r
        
        p_values.append(p_value(z_obs,intervals,std))
        CIs.append(ci.confidence_interval(intervals,z_obs,std,alpha))

    return SI_result(A,k,sigma,p_values,CIs)

def parametric_si_ci(X,y,A,k,Sigma,region,alpha=0.05):
    """calculate selective confidence intervals for each active feature

    Args:
        X (numpy.ndarray): design matrix(n x p)
        y (numpy.ndarray): object variable(n x 1)
        A (list): active features
        k (int): hyperparameter
        Sigma (numpy.ndarray): covariance matrix of y(n x n)
        region (function): function to compute interval for each algorithm

    Returns:
        tuple : cis is a list of selective confidence interval for each active features
                A is list of active feature
    """

    cis = []

    for i in range(len(A)):
        intervals = []
        a,b,var,z_obs = construct_teststatistics(A,i,X,y,Sigma)

        sigma = np.sqrt(var)
        z_min = -1 * sigma * 20
        z_max = sigma * 20

        regions,models = compute_solution_path(k,X,a,b,z_min,z_max,region)

        for r,model in zip(regions,models):
            if set(A) == set(model):
                intervals.append(r)
    
        cis.append(ci.confidence_interval(intervals,z_obs,sigma,alpha))
    
    return cis

