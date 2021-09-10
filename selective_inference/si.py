import numpy as np
import portion as p

from . import p_value 
from . import ci

EPS = 1e-4

def construct_teststatistics(model,i,X,y,Sigma):

    X_A = X[:,model]
    e = np.zeros(model.shape[0])
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

        y = a + b*z 

        interval,model_z = region(k,X,y,a,b)
        models.append(model_z)
        intervals.append(interval)

        z = interval.upper + EPS
    
    return intervals,models

def parametric_si_p(X,y,A,k,Sigma,region):

    intervals = p.empty()
    p_values = np.zeros(k)

    for i in range(len(A)):
        a,b,var,z_obs = construct_teststatistics(A,i,X,y,Sigma)

        sigma = np.sqrt(var)
        z_min = -1 * sigma * 20
        z_max = sigma * 20

        regions,models = compute_solution_path(k,X,a,b,z_min,z_max,region)

        for r,model in zip(regions,models):
            if set(A) == set(model):
                intervals = intervals | r
    
        p_values[i] = p_value.compute_p_value(z_obs,regions,sigma)
    
    return p_values, A

def parametric_si_ci(X,y,A,k,Sigma,region,alpha=0.05):

    intervals = p.empty()
    cis = np.zeros(k)

    for i in range(len(A)):
        a,b,var,z_obs = construct_teststatistics(A,i,X,y,Sigma)

        sigma = np.sqrt(var)
        z_min = -1 * sigma * 20
        z_max = sigma * 20

        regions,models = compute_solution_path(k,X,a,b,z_min,z_max,region)

        for r,model in zip(regions,models):
            if set(A) == set(model):
                intervals = intervals | r
    
        cis.append(ci.confidence_interval(intervals,z_obs,sigma,alpha))
    
    return cis, A