import pytest
import numpy as np 
from selective_inference.sfs_si import parametric_sfs_si,parametric_sfs_ci,parametric_sfs_cv_si

@pytest.mark.parametrize("p_test,A_test",[([0.02518984995324214, 0.43062551107376085, 0.46988125622144206, 0.6659805544944182], [2, 4, 1, 0])])
def test_sfs_si(data_generate,p_test,A_test):
    p_values,A = parametric_sfs_si(data_generate[0],data_generate[1],4)
    np.testing.assert_allclose(p_values,p_test)
    assert A == A_test

@pytest.mark.parametrize("p_test,A_test,k_test",[([0.07359934142517388], [2], 1)])
def test_sfs_cv_si(data_generate,p_test,A_test,k_test):
    p_values,A,k = parametric_sfs_cv_si(data_generate[0],data_generate[1],[1,2,3,3,4,5],5)
    np.testing.assert_allclose(p_values,p_test)
    assert A == A_test
    assert k == k_test