from sklearn import linear_model
import numpy as np

from . import cv

def lasso(X,y,alpha):

    clf = linear_model.Lasso(alpha=alpha,fit_intercept=False,normalize=False)
    clf.fit(X,y)
    coef = clf.coef_
    A = np.where(coef!=0)[0].tolist()
    s = np.sign(coef[A])

    return A,s

def lasso_CV(X,y,alpha_candidate,k_cv):

    alpha = alpha_candidate[np.argmin([cv.cv_error(lasso(X,y,alpha),X,y,k_cv) for alpha in alpha_candidate])]
    return lasso(X,y,alpha)