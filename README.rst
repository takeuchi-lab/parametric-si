selective_inference for feature selection algorithm
===================================================

This package provides selective inference for SFS,Lars,Lasso.

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
Use pip to install selective_inference package. Required packages will be also installed automatically.

.. code-block:: console
    
    $ pip install selective_inference

=======
Example
=======

.. code-block:: python

    # import modules
    import selective_inference as si
    import numpy as np

    # generate data
    X = np.random.randn(50,10)
    beta = np.zeros(10)
    beta[0:2] = 1
    y = X @ beta + np.random.randn(50)

    # run selective inference
    print("---- k = 3 ----")
    print("lars si resutl")
    print(si.parametric_lars_si(X,y,3))
    print()
    print("sfs si result")
    print(si.parametric_sfs_si(X,y,3))
    print()
    print("lasso si result")
    print(si.parametric_lasso_si(X,y,0.1))
    print()

    print("---- with cv ----")
    print("lars si resutl")
    print(si.parametric_lars_cv_si(X,y,[1,2,3,4,5],5))
    print()
    print("sfs si result")
    print(si.parametric_sfs_cv_si(X,y,[1,2,3,4,5],5))
    print()
    print("lasso si result")
    print(si.parametric_lasso_cv_si(X,y,[0.01,0.1,1,10,100],5))
    print()

=============
API Reference
=============
API reference is available at https://daikimiwa.github.io/selective_inference_document/