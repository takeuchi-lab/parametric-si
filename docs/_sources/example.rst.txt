Example
=======

.. toctree::
   :caption: Contents:
   :maxdepth: 2

Simple example for parametric_si

.. code-block:: python

   # import modules
   import parametric_si as psi
   import numpy as np

   # generate data
   X = np.random.randn(100,10)
   beta = np.array([0,0,0,0,0,0,0,0,0,0])
   y = X @ beta + np.random.randn(100)

   # execute selective inference
   result_lasso  = psi.parametric_lasso_si(X,y,α)
   result_stepwise = psi.parametric_sfs_si(X,y,k)
   result_lars = psi.parametric_lars_si(X,y,k)

   # execute selective inference with cv
   result_lasso_cv = psi.parametric_lasso_cv_si(X,y,[0.01,0.1,1,10,100],5)
   result_stepwise_cv = psi.parametric_sfs_cv_si(X,y,[1,2,3,4,5],5)
   result_lars_cv = psi.parametric_lars_cv_si(X,y,[1,2,3,4,5],5)