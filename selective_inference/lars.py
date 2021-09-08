import numpy as np
import copy

from . import cv

def knot_value(j,s_j,A,s,X):
    X_A = X[:,A]
    x_j = X[:,j]
    P = (np.identity(X.shape[0])) - X_A @ np.linalg.inv(X_A.T @ X_A) @ X_A.T
    temp = (P @ x_j) / (s_j - x_j @ X_A @ np.linalg.inv(X_A.T @ X_A) @ s)

    return temp

#TODO: require refactraring
def lars(X,y,k):

    S = []
    A_c = []
    A = []
    Sb= []
    s= []

    step = k
    features = list(range(X.shape[1]))

    k = 0
    c = X.T @ y
    j = np.abs(c).argmax()
    s_j = np.sign(c[j])
    S_j = [[l,m] for l in features for m in [-1,1]]

    S.append(S_j)
    A.append([j])
    s.append([s_j])
    A_c_j = copy.copy(features)
    A_c_j.remove(j)
    A_c.append(A_c_j)
    Sb.append([])

    knot = s_j * X[:,j] @ y
    k += 1
    while k < d and k < step:

        S_k = []
        Sb_k = []
        
        Lambda = np.zeros((len(A_c[k-1]),2))
        for i,feature in enumerate(A_c[k-1]):
            for j,sign in enumerate([-1,1]):
                temp = knot_value(feature,sign,A[k-1],s[k-1],X) @ y 
                Lambda[i,j] = temp if temp <= knot else 0
                if Lambda[i,j] <= knot:
                    S_k.append([feature,sign])
                else :
                    Sb_k.append([feature,sign])
        
        knot = np.max(Lambda)
        j,s_j = np.unravel_index(np.argmax(Lambda),Lambda.shape)

        A.append(A[k-1] + [A_c[k-1][j]])
        s.append(s[k-1] + [[-1,1][s_j]])
        A_c_j = copy.copy(A_c[k-1])
        A_c_j.remove(A_c[k-1][j])
        A_c.append(A_c_j)
        S.append(S_k)
        Sb.append(Sb_k)

        k += 1

    return A[-1]

def lars_CV(X,y,k_candidate,k_cv):
    error = list(map(lambda x:cv.cv_error(x,X,y,k_cv),k_candidate))
    k = k_candidate[np.argmin(error)]
    return lars(X,y,k)