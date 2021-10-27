import pytest
import numpy as np 
from parametric_si.lasso_si import parametric_lasso_si,parametric_lasso_cv_si

@pytest.mark.parametrize("p_test,A_test",[([0.7088775232016662, 0.502334063812671, 0.028037418987861384, 0.4595646996593246], [0, 1, 2, 4])])
def test_lasso_si(data_generate,p_test,A_test):
    result = parametric_lasso_si(data_generate[0],data_generate[1],0.01)
    np.testing.assert_allclose(result.p_values,p_test)
    assert result.A == A_test

@pytest.mark.parametrize("p_test,A_test,k_test",[([0.9213079628616301], [2], 0.1)])
def test_lasso_cv_si(data_generate,p_test,A_test,k_test):
    result = parametric_lasso_cv_si(data_generate[0],data_generate[1],np.geomspace(0.001,10,5).tolist(),5)
    np.testing.assert_allclose(result.p_values,p_test)
    assert result.A == A_test
    assert result.k == k_test