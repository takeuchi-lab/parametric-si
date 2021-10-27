import portion as p
import numpy as np

class Quadratic:

    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c
    
    def or_less(self,g):
        a = self.a - g.a
        b = self.b - g.b
        c = self.c - g.c

        h = Quadratic(a,b,c)

        return h.or_less_zero()
    
    @staticmethod
    def mean(quadratic_array):
        a = 0
        b = 0
        c = 0

        n = len(quadratic_array)
        for q in quadratic_array:
            a += q.a
            b += q.b
            c += q.c

        return Quadratic(a/n,b/n,c/n)

    def or_less_zero(self):

        if self.a == 0:
            if self.b > 0:
                return p.closed(-np.inf,-self.c/self.b)
            elif self.b == 0:
                if self.c > 0:
                    return p.empty()
                else: return p.closed(-np.inf,np.inf)
            else:
                return p.closed(-self.c/self.b, np.inf)
        else :
            d = self.b**2 - 4*self.a*self.c
            if d < 0:
                if self.a > 0:
                    return p.empty()
                else :
                    return p.closed(-np.inf,np.inf)
            else :
                x_left,x_right = (-self.b - np.sqrt(d))/(2*self.a),(-self.b + np.sqrt(d))/(2*self.a)
                if self.a >0:
                    return p.closed(x_left,x_right)
                else :
                    return p.closed(-np.inf,x_right) | p.closed(x_left,np.inf)
    def  f(self,x):
        return self.a*x**2 + self.b*x + self.c