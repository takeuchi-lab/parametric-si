import pytest
import numpy as np 
from parametric_si.lars import lars,lars_CV

@pytest.mark.parametrize("A_test",[([2,4,1,0])])
def test_lars(data_generate,A_test):
    k = 4
    assert lars(data_generate[0],data_generate[1],k)[0][-1] == A_test

@pytest.mark.parametrize("A_test,k_test",[([2], 1)])
def test_lars_CV(data_generate,A_test,k_test):
    assert lars_CV(data_generate[0],data_generate[1],[1,2,3,3,4,5],5) == (A_test,k_test)