import portion as p
from scipy import stats

def prob(interval,mu,sigma):
    return stats.norm.cdf(interval.upper,loc=mu,scale=sigma)-stats.norm.cdf(interval.lower,loc=mu,scale=sigma)

def truncated_normal_cdf(z,intervals,mu,sigma):
    assert intervals is not None
    numerator = 0
    denominator = 0

    for interval in intervals:
        denominator += prob(interval,mu,sigma)
        if interval.lower <= z <= interval.upper:
            numerator += prob(p.closed(interval.lower,z),mu,sigma)
        elif interval.upper <= z:
            numerator += prob(interval,mu,sigma)

    return numerator / denominator

# def prob(interval,mu,sigma):
    # return stats.norm.cdf(interval[1],loc=mu,scale=sigma)-stats.norm.cdf(interval[0],loc=mu,scale=sigma)

# def truncated_normal_cdf(z,intervals,mu,sigma):
    # numerator = 0
    # denominator = 0

    # for interval in intervals:
        # denominator += prob(interval,mu,sigma)
        # if interval[0] <= z <= interval[1]:
            # numerator += prob([interval[0],z],mu,sigma)
        # elif interval[1] <= z:
            # numerator += prob(interval,mu,sigma)

    
    # return numerator / denominator