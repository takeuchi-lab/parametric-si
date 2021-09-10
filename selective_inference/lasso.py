from sklearn import linear_model
import numpy as np

from . import cv

def lasso(X,y,alpha):

    clf = linear_model.Lasso(alpha=alpha,fit_intercept=False,normalize=False)
    clf.fit(X,y)
    A = clf.coef_

    return np.where(A!=0)[0].tolist()

def lasso_CV(X,y,alpha_candidate,k_cv):

    alpha = alpha_candidate[np.argmin([cv.cv_error(lasso(X,y,alpha),X,y,k_cv) for alpha in alpha_candidate])]
    return lasso(X,y,alpha)