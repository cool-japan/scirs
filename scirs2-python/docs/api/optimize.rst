Optimization (``scirs2.optimize``)
====================================

Numerical optimization: unconstrained, constrained, global, and
curve-fitting routines.

.. automodule:: scirs2.optimize
   :members:
   :undoc-members:
   :show-inheritance:

Function Reference
------------------

Unconstrained Minimization (``minimize_py``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = scirs2.minimize_py(
       fun=lambda x: (x[0]-1)**2 + (x[1]-2)**2,
       x0=[0.0, 0.0],
       method="L-BFGS-B",
   )
   print(result["x"], result["fun"])

Supported methods: ``Nelder-Mead``, ``BFGS``, ``L-BFGS-B``, ``CG``,
``Newton-CG``, ``SLSQP``

Constrained Minimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   constraints = [{"type": "ineq", "fun": lambda x: x[0] - 0.5}]
   result = scirs2.minimize_py(fun, x0, method="SLSQP",
                               constraints=constraints)

Global Optimization
~~~~~~~~~~~~~~~~~~~

- ``differential_evolution_py(fun, bounds)``
- ``basin_hopping_py(fun, x0)``

Curve Fitting
~~~~~~~~~~~~~

.. code-block:: python

   def model(x, a, b):
       return a * np.exp(-b * x)

   popt, pcov = scirs2.curve_fit_py(model, x_data, y_data)

Extended
~~~~~~~~

- ``sqp_py`` — Sequential Quadratic Programming
- ``interior_point_lp_py`` — Linear programming (interior-point)
- ``interior_point_qp_py`` — Quadratic programming (interior-point)
