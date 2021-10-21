import pytest
import numpy as np 
from parametric_si.sfs import sfs,sfs_CV

@pytest.mark.parametrize("A_test",[([2, 4, 1, 0])])
def test_sfs(data_generate,A_test):
    k = 4
    assert sfs(data_generate[0],data_generate[1],k)[0] == A_test

@pytest.mark.parametrize("A_test,k_test",[([2], 1)])
def test_sfs_CV(data_generate,A_test,k_test):
    assert sfs_CV(data_generate[0],data_generate[1],[1,2,3,3,4,5],5) == (A_test,k_test)