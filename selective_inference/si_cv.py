from operator import index
import numpy as np
import portion as p
import matplotlib.pyplot as plt

from .quadratic import Quadratic
from .cv import train_val_index,train_val_split_mat,train_val_split_vec
from . import si

def validation_error(X_train,X_val,a_train,a_val,b_train,b_val):

    X_inv = np.linalg.pinv(X_train.T @ X_train) @ X_train.T
    a = a_val - X_val @ X_inv @ a_train
    b = b_val - X_val @ X_inv  @ b_train

    return Quadratic(b.T@b, 2*a.T@b ,a.T @ a)

def compute_val_error_path(k,X_train,X_val,a_train,a_val,b_train,b_val,z_min,z_max,region):

    z_k,A_k = si.compute_solution_path(k,X_train,a_train,b_train,z_min,z_max,region)
    E_k = [validation_error(X_train[:,A],X_val[:,A],a_train,a_val,b_train,b_val) for A in A_k]

    return z_k,E_k

def compute_cv_path(k,X,a,b,z_min,z_max,k_cv,region):

    index = train_val_index(len(a),k_cv)
    paths = []

    for i in range(k_cv):
        X_train,X_val = train_val_split_mat(X,index[i],index[i+1])
        a_train,a_val = train_val_split_vec(a,index[i],index[i+1])
        b_train,b_val = train_val_split_vec(b,index[i],index[i+1])
        z_k,E_k = compute_val_error_path(k,X_train,X_val,a_train,a_val,b_train,b_val,z_min,z_max,region)
        paths.append([z_k,E_k])

    Z_k = [paths[i][0] for i in range(k_cv)]
    E_k = [paths[i][1] for i in range(k_cv)]

    pointers = np.zeros(k_cv,dtype=int)

    Z = []
    E = []

    z_left,z_right = z_min,z_min
    while z_right < z_max:
        temp = [Z_k[i][pointers[i]].upper for i in range(k_cv)]
        next_point = np.argmin(temp)
        z_right = Z_k[next_point][pointers[next_point]].upper
        Z.append(p.closed(z_left,z_right))
        E.append(Quadratic.mean([E_k[j][pointers[j]] for j in range(k_cv)]))

        z_left = z_right
        pointers[next_point] += 1
    
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

    return Z_CV

def print_Z_CV(k_obs,k_candidates,Z,E,Z_CV):
    m = 0
    plt.figure(figsize=(8,6))

    for i,k in enumerate(k_candidates):
        x = np.empty(0)
        y = np.empty(0)

        for z,e in zip(Z[i],E[i]):
            x_temp = np.arange(z.lower,z.upper,0.01)
            y_temp = e.f(x_temp)
            x = np.hstack([x,x_temp])
            y = np.hstack([y,y_temp])

        if k == k_obs:
            plt.plot(x,y,label=f'k_obs={k_obs}',linewidth=0.5)
            m = np.min(y)
            
        else:
            plt.plot(x,y,label=f'k={k}',linewidth=0.5)
    
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    for z in Z_CV:
        plt.plot([z.lower,z.upper],[m-1,m-1],color='r',linewidth=1.5)

    plt.show()