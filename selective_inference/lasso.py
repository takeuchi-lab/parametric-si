from sklearn import linear_model
import numpy as np

from . import cv

def lasso(X,y,alpha):

    clf = linear_model.Lasso(alpha=alpha,fit_intercept=False)
    clf.fit(X,y)
    coef = clf.coef_
    A = np.where(coef!=0)[0].tolist()
    s = np.sign(coef[A])

    return A,s

def lasso_CV(X,y,alpha_candidate,k_cv):

    error = [cv.cv_error(alpha,X,y,k_cv,lambda X,y,k:lasso(X,y,k)[0]) for alpha in alpha_candidate]
    alpha = alpha_candidate[np.argmin(error)]
    return lasso(X,y,alpha)[0],alpha