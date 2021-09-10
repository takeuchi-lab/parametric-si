import numpy as np
import portion as p

from . import lasso
from . import si

def parametric_lasso_si(X,y,alpha):

    A = lasso.lasso(X,y,alpha)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_p(X,y,A,len(A),Sigma,region)

def parametric_lasso_ci(X,y,alpha):

    A = lasso.lasso(X,y,alpha)
    Sigma = np.identity(X.shape[0])

    return si.parametric_si_ci(X,y,A,len(A),Sigma,region)

def region(X,y,alpha,a,b):

    L,U = -np.inf,np.inf
    A = lasso.lasso(X,y,alpha)

    X_A = X[:,A]

    inv = np.linalg.inv(X_A.T @ X_A)

    return p.closed(L,U),A