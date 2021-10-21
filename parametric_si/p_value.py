from . import probability as prob

def p_value(z_obs,intervals,sigma):

    p = prob.truncated_normal_cdf(z_obs,intervals,0,sigma)

    return 2 * min(p,1-p)