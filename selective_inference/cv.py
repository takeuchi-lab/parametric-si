import numpy as np

def train_val_index(N,k_cv):
    return [min(i * int(np.ceil(N/k_cv)),N) for i in range(k_cv+1)]

def train_val_split_mat(X,begin,end):

    X_val = X[begin:end,:]
    index_train = np.ones(X.shape[0],dtype=bool)
    index_train[begin:end] = False
    X_train = X[index_train,:]

    return X_train,X_val

def train_val_split_vec(y,begin,end):

    y_val = y[begin:end]
    index_train = np.ones(y.shape[0],dtype=bool)
    index_train[begin:end] = False
    y_train = y[index_train]

    return y_train,y_val

def validation_error(k,X_train,X_val,y_train,y_val,algorithm):

    A = algorithm(X_train,y_train,k)

    X_train_A = X_train[:,A]
    X_val_A = X_val[:,A]
    Beta_train = np.linalg.inv(X_train_A.T @ X_train_A) @  X_train_A.T @ y_train
    e = y_val - X_val_A @ Beta_train

    return e.T @ e

def cv_error(k,X,y,k_cv,algorithm):
    
    index = train_val_index(X.shape[0],k_cv)
    error_array = np.zeros(k_cv)

    for i in range(k_cv):
        X_train,X_val = train_val_split_mat(X,index[i],index[i+1])
        y_train,y_val = train_val_split_vec(y,index[i],index[i+1])
        error_array[i] = validation_error(k,X_train,X_val,y_train,y_val,algorithm)
    
    return np.average(error_array)