import numpy as np

def construct_teststatistics(model,i,X,y,Sigma):

    X_A = X[:,model]
    e = np.zeros(model.shape[0])
    e[i] = 1

    eta = X_A @ np.linalg.inv(X_A.T @ X_A) @ e
    var = eta.T @ Sigma @ eta
    return var

    z_obs = eta @ y
    b = Sigma @ eta / var
    a = (np.identity(X.shape[0]) - b.reshape(-1,1) @ eta.reshape(1,-1)) @ y

    return a,b,z_obs,var

#TODO is this appropreate function name??
def compute_solution_path(k,X,a,b,z_min,z_max):

    intervals = []
    models = []

    return intervals,models
