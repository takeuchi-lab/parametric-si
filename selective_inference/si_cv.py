from operator import index
import numpy as np
import portion as p

from .quadratic import Quadratic
from .cv import train_val_index,train_val_split_mat,train_val_split_vec
from . import si
from . import ci
from .p_value import p_value

def parametric_si_cv(X,y,A,k_obs,k_candidates,sigma,region,k_folds,alpha):

    Sigma = np.identity(X.shape[0]) * sigma

    p_values = []
    CIs = []

    for i in range(len(A)):
        a,b,var,z_obs = si.construct_teststatistics(A,i,X,y,Sigma)
        std = np.sqrt(var)
        z_min = -20 * std
        z_max = 20 * std

        paths = [compute_cv_path(k,X,y,a,b,z_min,z_max,k_folds,region) for k in k_candidates]

        Z = [paths[i][0] for i,j in enumerate(k_candidates)]
        E = [paths[i][1] for i,j in enumerate(k_candidates)]

        Z_CV = compute_Z_CV(k_obs,k_candidates,Z,E)
        
        Z_alg = p.empty()
        intervals,models = si.compute_solution_path(k_obs,X,y,a,b,z_min,z_max,region)

        for interval,model in zip(intervals,models):
            if set(A) == set(model):
                Z_alg = Z_alg | interval
        
        Z = Z_alg & Z_CV

        p_values.append(p_value(z_obs,Z,std))
        CIs.append(ci.confidence_interval(Z,z_obs,std,alpha))

    return si.SI_result(A,k_obs,sigma,p_values,CIs)

def parametric_si_cv_ci(X,y,A,k_obs,k_candidates,Sigma,region,k_folds,alpha=0.05):

    cis = []

    for i in range(len(A[-1])):
        a,b,var,z_obs = si.construct_teststatistics(k_obs,i,X,y,i,Sigma)
        sigma = np.sqrt(var)
        z_min = -20 * sigma
        z_max = 20 * sigma

        paths = [compute_cv_path(k,X,y,a,b,z_min,z_max,k_folds,region) for k in k_candidates]
        Z = [paths[i][0] for i,j in enumerate(k_candidates)]
        E = [paths[i][1] for i,j in enumerate(k_candidates)]

        Z_CV = compute_Z_CV(k_obs,k_candidates,Z,E)
        Z_alg = p.empty()

        intervals,models = si.compute_solution_path(k_obs,X,a,b,z_min,z_max,region)

        for interval,model in zip(intervals,models):
            if set(A) == set(model):
                Z_alg = Z_alg | p.closed(interval[0],interval[1])

        Z = Z_alg & Z_CV

        cis.append(ci.confidence_interval(Z,z_obs,sigma,alpha))

    return cis, A, k_obs

def validation_error(X_train,X_val,a_train,a_val,b_train,b_val):
    """compute validation error for a given model expressed as Quadratic objects

    Args:
        X_train (numpy.ndarray): feature matrix for train set
        X_val ([type]): feature matrix for validation set
        a_train ([type]): vector of test statistics for train set
        a_val ([type]): vector of test statistics for validation set
        b_train ([type]): vector of test statistics for train set
        b_val ([type]): vector of test statistics for validation set

    Returns:
        Quadratic: validation error for a given model
    """

    X_inv = np.linalg.pinv(X_train.T @ X_train) @ X_train.T
    a = a_val - X_val @ X_inv @ a_train
    b = b_val - X_val @ X_inv  @ b_train

    return Quadratic(b.T@b, 2*a.T@b ,a.T @ a)

def compute_val_error_path(k,X_train,X_val,y_train,y_val,a_train,a_val,b_train,b_val,z_min,z_max,region):
    """ compute list of model and its interval and validation error on direction of test statistic

    Args:
        k ([type]): [description]
        X_train ([type]): [description]
        X_val ([type]): [description]
        y_train ([type]): [description]
        y_val ([type]): [description]
        a_train ([type]): [description]
        a_val ([type]): [description]
        b_train ([type]): [description]
        b_val ([type]): [description]
        z_min ([type]): [description]
        z_max ([type]): [description]
        region ([type]): [description]

    Returns:
        [type]: [description]
    """

    z_k,A_k = si.compute_solution_path(k,X_train,y_train,a_train,b_train,z_min,z_max,region)
    E_k = [validation_error(X_train[:,A],X_val[:,A],a_train,a_val,b_train,b_val) for A in A_k]

    return z_k,E_k

def compute_cv_path(k,X,y,a,b,z_min,z_max,k_cv,region):

    index = train_val_index(len(a),k_cv)
    paths = []

    for i in range(k_cv):
        X_train,X_val = train_val_split_mat(X,index[i],index[i+1])
        a_train,a_val = train_val_split_vec(a,index[i],index[i+1])
        b_train,b_val = train_val_split_vec(b,index[i],index[i+1])
        y_train,y_val = train_val_split_vec(y,index[i],index[i+1])
        z_k,E_k = compute_val_error_path(k,X_train,X_val,y_train,y_val,a_train,a_val,b_train,b_val,z_min,z_max,region)
        paths.append([z_k,E_k])

    Z_k = [paths[i][0] for i in range(k_cv)]
    E_k = [paths[i][1] for i in range(k_cv)]

    pointers = np.zeros(k_cv,dtype=int)

    Z = []
    E = []

    z_left,z_right = z_min,z_min

    while z_right < z_max-1e-4:
        temp = [Z_k[i][pointers[i]].upper for i in range(k_cv)]
        next_point = np.argmin(temp)
        z_right = Z_k[next_point][pointers[next_point]].upper
        Z.append(p.closed(z_left,z_right))
        E.append(Quadratic.mean([E_k[j][pointers[j]] for j in range(k_cv)]))

        z_left = z_right
        pointers[next_point] += 1

        if pointers[next_point] >= len(Z_k[next_point]):
            break

    return Z,E

def compute_Z_CV(k_obs,k_candidates,Z,E):
    index_k_obs = k_candidates.index(k_obs)
    z_min,z_max = Z[index_k_obs][0].lower,Z[index_k_obs][-1].upper
    Z_CV = p.empty()
    pointers = np.zeros(len(k_candidates),dtype=int)
    z_left,z_right = z_min,z_min

    while z_right < z_max:
        k_next = np.argmin([Z[j][pointers[j]].upper for j,l in enumerate(k_candidates)])
        z_right = Z[k_next][pointers[k_next]].upper

        if z_left == z_right:
            pointers[k_next]+=1
            continue

        I = p.closed(z_left,z_right)
        for i,k in enumerate(k_candidates):

            if k_obs == k:
                continue
            
            Z_I = E[index_k_obs][pointers[index_k_obs]].or_less(E[i][pointers[i]])
            I = I & Z_I

        Z_CV = Z_CV | I
        pointers[k_next] += 1
        z_left = z_right
        if pointers[k_next] >= len(Z[k_next]):
            break

    return Z_CV

# def print_Z_CV(k_obs,k_candidates,Z,E,Z_CV):
    # m = 0
    # plt.figure(figsize=(8,6))

    # for i,k in enumerate(k_candidates):
        # x = np.empty(0)
        # y = np.empty(0)

        # for z,e in zip(Z[i],E[i]):
            # x_temp = np.arange(z.lower,z.upper,0.01)
            # y_temp = e.f(x_temp)
            # x = np.hstack([x,x_temp])
            # y = np.hstack([y,y_temp])

        # if k == k_obs:
            # plt.plot(x,y,label=f'k_obs={k_obs}',linewidth=0.5)
            # m = np.min(y)
            
        # else:
            # plt.plot(x,y,label=f'k={k}',linewidth=0.5)
    
        # plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    # for z in Z_CV:
        # plt.plot([z.lower,z.upper],[m-1,m-1],color='r',linewidth=1.5)

    # plt.show()