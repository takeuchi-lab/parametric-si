import pytest
import numpy as np

@pytest.fixture
def data_generate():
    np.random.seed(0000)
    X = np.random.randn(100,5)
    beta = np.zeros(5)
    y = X @ beta + np.random.randn(100)
    return X,y