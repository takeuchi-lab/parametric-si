import pytest
import numpy as np 
from selective_inference.lasso import lasso,lasso_CV

@pytest.mark.parametrize("A_test",[([0, 1, 2, 4])])
def test_lasso(data_generate,A_test):
    alpha = 0.01
    assert lasso(data_generate[0],data_generate[1],alpha)[0] == A_test

@pytest.mark.parametrize("A_test,k_test",[([2], 0.1)])
def test_lasso_CV(data_generate,A_test,k_test):
    assert lasso_CV(data_generate[0],data_generate[1],[0.001,0.01,0.1,1,10],5) == (A_test,k_test)