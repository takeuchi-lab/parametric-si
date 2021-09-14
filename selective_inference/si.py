import numpy as np
import portion as p

from . import p_value 
from . import ci
from . import cv
from . import si_cv

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

    p_values = np.zeros(len(A))

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
        
        p_values[i] = p_value.compute_p_value(z_obs,intervals,sigma)

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

def parametric_si_cv_p(X,y,A,k_obs,k_candidates,Sigma,region,k_folds):

    p_values = []

    for i in range(len(A)):
        a,b,var,z_obs = construct_teststatistics(A,i,X,y,Sigma)
        sigma = np.sqrt(var)
        z_min = -20 * sigma
        z_max = 20 * sigma

        paths = [si_cv.compute_cv_path(k,X,a,b,z_min,z_max,k_folds,region) for k in k_candidates]

        Z = [paths[i][0] for i,j in enumerate(k_candidates)]
        E = [paths[i][1] for i,j in enumerate(k_candidates)]

        Z_CV = si_cv.compute_Z_CV(k_obs,k_candidates,Z,E)
        
        Z_alg = p.empty()
        intervals,models = compute_solution_path(k_obs,X,a,b,z_min,z_max,region)

        for interval,model in zip(intervals,models):
            if set(A) == set(model):
                Z_alg = Z_alg | interval
        
        Z = Z_alg & Z_CV

        p_values.append(p_value.compute_p_value(z_obs,Z,sigma))

    return p_values,A,k_obs

def parametric_si_cv_ci(X,y,A,k,k_candidates,Sigma,region,k_folds,alpha=0.05):

    cis = []

    for i in range(len(A[-1])):
        a,b,var,z_obs = construct_teststatistics(k,i,X,y,i,Sigma)
        sigma = np.sqrt(var)
        z_min = -20 * sigma
        z_max = 20 * sigma

        paths = [si_cv.compute_cv_path(k,X,a,b,z_min,z_max,k_folds,region) for k in k_candidates]
        Z = [paths[i][0] for i,j in enumerate(k_candidates)]
        E = [paths[i][1] for i,j in enumerate(k_candidates)]

        Z_CV = si_cv.compute_Z_CV(k,k_candidates,Z,E)
        Z_alg = p.empty()

        intervals,models = compute_solution_path(k,X,a,b,z_min,z_max,region)

        for interval,model in zip(intervals,models):
            if set(A) == set(model):
                Z_alg = Z_alg | p.closed(interval[0],interval[1])

        Z = Z_alg & Z_CV

        cis.append(ci.confidence_interval(Z,z_obs,sigma,alpha))

    return cis, A