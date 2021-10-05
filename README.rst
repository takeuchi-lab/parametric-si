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
    beta[0:3] = 0.5
    y = X @ beta + np.random.randn(100)

    # number of features selected is 3
    print(si.parametric_sfs_si(X,y,3))

    # SI_result(A=[0, 2, 1], k=3, p_values=[5.654553492107084e-09, 1.9619835046613687e-05, 0.10271336317135393], CIs=[[0.40216497119816363,0.7878657243366332], [0.2791430678431459,0.7046883232985647], [-0.024572104161974907,0.4558392581078039]])

example for lasso with cv

.. code-block:: python

    import selective_inference as si
    import numpy as np
    X = np.random.randn(100,10)
    beta = np.zeros(10)
    beta[0:3] = 0.5
    y = X @ beta + np.random.randn(100)

    # folds number in si is 5 and hyperparameter(number of features selected) candidates are 1,2,3,4 and 5.
    print(si.parametric_lars_cv_si(X,y,[1,2,3,4,5],5))

    # SI_result(A=[0, 1, 2, 4, 6, 9], k=0.1, p_values=[8.187213480570499e-05, 0.006541369052107893, 4.247307444882331e-05, 0.8455031989177867, 0.3413830199921571, 0.7169605201041223], CIs=[[0.2637282632180834,0.7963180250269161], [0.16402101262098456,0.6811812382735888], [0.2922869643467601,1.0836698526867241], [-0.24377227610907795,0.1185952203064659], [-0.3313975848916705,0.06665679761286254], [-0.10652630894251695,0.24857774369408472]])

=============
API Reference
=============
API reference is available at https://hogehoge.hogehoge