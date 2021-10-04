selective_inference for feature selection algorithm
===================================================

This package provides selective inference for feature selection algorithms.

See the paper https://arxiv.org/abs/2004.09794 for more details.

add desicription about SI

============
Requirements
============
selective_inference requires the following packages:

* numpy
* scipy
* skelearn
* portion

==============================
Installing selective_inference
==============================
Use pip to install 

.. code-block:: console
    
    $ pip install selective_inference

=======
Example
=======

.. code-block:: python

    import selective_inference as si
    import numpy as np
    X = np.random.randn(100,10)
    beta = np.zeros(10)
    y = X @ beta + np.random.randn(100)
    print(si.parametric_sfs_si(X,y,3))

    # SI_result(A=[6, 2, 9], k=3, p_values=[0.8757497138742228, 0.8871201546774179, 0.9058273799617409], CIs=[])

=============
API Reference
=============
API reference is available at https://hogehoge.hogehoge