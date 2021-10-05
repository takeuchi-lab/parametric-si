import selective_inference as si
import numpy as np

p_result_lars = np.empty(0)
p_result_sfs = np.empty(0)
p_result_lasso = np.empty(0)

for i in range(1000):
    print(i)
    X = np.random.randn(50,5)
    beta = np.zeros(5)
    y = X @ beta + np.random.randn(50)

    try:
        p_result_lars = np.concatenate([p_result_lars,si.parametric_lars_si(X,y,3).p_values])
        p_result_lasso = np.concatenate([p_result_lasso,si.parametric_lasso_si(X,y,0.01).p_values])
        p_result_sfs = np.concatenate([p_result_sfs,si.parametric_sfs_si(X,y,3).p_values])
    except:
        print("error")

print(np.count_nonzero(p_result_lasso<0.05)/p_result_lasso.shape[0])
print(np.count_nonzero(p_result_lars<0.05)/p_result_lars.shape[0])
print(np.count_nonzero(p_result_sfs<0.05)/p_result_sfs.shape[0])