import selective_inference as si
import numpy as np
X = np.random.randn(100,10)
beta = np.zeros(10)
y = X @ beta + np.random.randn(100)
print(si.parametric_sfs_si(X,y,3))
