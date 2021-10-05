selective_inference for feature selection algorithm
===================================================

This package provides selective inference for SFS,Lars,Lasso.

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
Use pip to install selective_inference package. Required packages are also installed automatically.

.. code-block:: console
    
    $ pip install selective_inference

=======
Example
=======

example for sfs 

.. code-block:: python

    import selective_inference as si
    import numpy as np
    X = np.random.randn(100,10)
    beta = np.zeros(10)
    y = X @ beta + np.random.randn(100)

    # number of features selected is 3
    print(si.parametric_sfs_si(X,y,3))

    # SI_result(A=[6, 2, 9], k=3, p_values=[0.8757497138742228, 0.8871201546774179, 0.9058273799617409], CIs=[])

example for lasso with cv

.. code-block:: python

    import selective_inference as si
    import numpy as np
    X = np.random.randn(100,10)
    beta = np.zeros(10)
    y = X @ beta + np.random.randn(100)

    # folds number in si is 5 and hyperparameter(number of features selected) candidates are 1,2,3,4 and 5.
    print(si.parametric_lars_cv_si(X,y,[1,2,3,4,5],5))

    # SI_result(A=[5], k=1, p_values=[0.4218494699946196], CIs=[])    

=============
API Reference
=============
API reference is available at https://hogehoge.hogehoge