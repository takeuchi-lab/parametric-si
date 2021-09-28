import numpy as np
import portion as p

from . import ci
from . import cv
from . import si_cv
from .p_value import p_value

EPS = 5e-3

def construct_teststatistics(A,i,X,y,Sigma):

    X_A = X[:,A]
    e = np.zeros(len(A))
    e[i] = 1

    eta = X_A @ np.linalg.inv(X_A.T @ X_A) @ e
    var = eta.T @ Sigma @ eta

    z_obs = eta @ y
    b = Sigma @ eta / var
    a = (np.identity(X.shape[0]) - b.reshape(-1,1) @ eta.reshape(1,-1)) @ y

    return a,b,var,z_obs

def compute_solution_path(k,X,a,b,z_min,z_max,region):

    z = z_min

    intervals = []
    models = []

    while z < z_max:

        y = a + b * z 

        L,U,model_z = region(X,y,k,a,b)
        
        models.append(model_z)
        intervals.append(p.closed(max(L,z_min),min(U,z_max)))
        if type(intervals[-1].upper) == type(p.empty().upper):
            assert False
        z = intervals[-1].upper + EPS
    
    return intervals,models

def parametric_si_p(X,y,A,k,Sigma,region):

    p_values = []

    for i in range(len(A)):
        intervals = p.empty()

        a,b,var,z_obs = construct_teststatistics(A,i,X,y,Sigma)

        sigma = np.sqrt(var)
        z_min = -1 * sigma * 10
        z_max = sigma * 10

        regions,models = compute_solution_path(k,X,a,b,z_min,z_max,region)

        for r,model in zip(regions,models):
            if set(A) == set(model):
                intervals = intervals | r
        
        p_values.append(p_value(z_obs,intervals,sigma))

    return p_values, A

def parametric_si_ci(X,y,A,k,Sigma,region,alpha=0.05):

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

