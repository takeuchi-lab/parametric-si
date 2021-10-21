import pytest
import numpy as np 
from parametric_si.lars_si import parametric_lars_si,parametric_lars_cv_si

@pytest.mark.parametrize("p_test,A_test",[([0.02516785736871907, 0.42930776257205366, 0.4697014120604279, 0.66639917133858], [2, 4, 1, 0])])
def test_lars_si(data_generate,p_test,A_test):
    result = parametric_lars_si(data_generate[0],data_generate[1],4)
    np.testing.assert_allclose(result.p_values,p_test,rtol=1e-5)
    assert result.A == A_test

@pytest.mark.parametrize("p_test,A_test,k_test",[([0.07359934142517388], [2], 1)])
def test_lars_cv_si(data_generate,p_test,A_test,k_test):
    result = parametric_lars_cv_si(data_generate[0],data_generate[1],[1,2,3,3,4,5],5)
    np.testing.assert_allclose(result.p_values,p_test,rtol=1e-05)
    assert result.A == A_test
    assert result.k == k_test