import numpy as np
import portion as p
from scipy import optimize
from . import probability as prob

def f_inverse(p,z,intervals,sigma):

    def f(mu):
        return prob.truncated_normal_cdf(z,intervals,mu,sigma)

    a = intervals[0].lower
    b = intervals[-1].upper

    fa = f(a)
    fb = f(b)

    while fb >= p:
        extent = b - a
        b_tmp = b + extent
        fb_tmp = f(b_tmp)

        while np.isnan(fb_tmp):
            extent /= 10
            b_tmp = b + extent
            fb_tmp = f(b_tmp)
        
        b = b_tmp
        fb = fb_tmp
    
    while p >= fa:
        extent = b - a 
        a_tmp = a - extent
        fa_tmp = f(a_tmp)

        while np.isnan(fa_tmp):
            extent /= 10
            a_tmp = a - extent
            fa_tmp = f(a_tmp)
        
        a = a_tmp
        fa = fa_tmp
    
    mu = optimize.bisect(lambda mu:f(mu)-p,a,b)

    return mu

def confidence_interval(intervals,z,sigma,alpha):

    l = f_inverse(1-alpha/2,z,intervals,sigma)
    u = f_inverse(alpha/2,z,intervals,sigma)

    return p.closed(l,u)