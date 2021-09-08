import portion as p
from scipy import stats

def prob(interval,mu,sigma):
    return stats.norm.cdf(interval.upper,mu,sigma)-stats.norm.cdf(interval.lower,mu,sigma)

def truncated_normal_cdf(z,intervals,mu,sigma):
    numerator = 0
    denominator = 0

    for interval in intervals:
        denominator += (prob(interval,mu,sigma))
        if interval.lower <= z <= interval.upper:
            numerator += prob(interval,mu,sigma)
        elif z <= interval.lower:
            numerator += prob(interval,mu,sigma)

    return numerator / denominator