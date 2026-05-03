Integration (``scirs2.integrate``)
====================================

Numerical integration and ODE solvers.

.. automodule:: scirs2.integrate
   :members:
   :undoc-members:
   :show-inheritance:

Quadrature
----------

.. code-block:: python

   # 1-D adaptive quadrature
   result, err = scirs2.quad_py(lambda x: x**2, 0.0, 1.0)
   # result ≈ 0.3333

   # Array-based
   x = np.linspace(0, 1, 100)
   y = x**2
   result = scirs2.trapezoid(y, x)
   result = scirs2.simpson(y, x)
   result = scirs2.romberg(lambda x: x**2, 0.0, 1.0)

Available functions: ``quad_py``, ``dblquad_py``, ``trapezoid``,
``simpson``, ``cumulative_trapezoid``, ``romberg``

ODE Solvers (``solve_ivp_py``)
-------------------------------

.. code-block:: python

   def f(t, y):
       return [-y[0]]    # exponential decay

   sol = scirs2.solve_ivp_py(
       fun=f,
       t_span=(0.0, 5.0),
       y0=[1.0],
       method="RK45",
       t_eval=np.linspace(0, 5, 50),
   )
   # sol["t"], sol["y"] — solution arrays

Supported methods:

.. list-table::
   :header-rows: 1

   * - Method
     - Description
   * - ``RK45``
     - Explicit Runge-Kutta 4(5) — default
   * - ``RK23``
     - Explicit Runge-Kutta 2(3)
   * - ``DOP853``
     - Explicit Runge-Kutta 8(5,3)
   * - ``BDF``
     - Implicit multi-step (stiff problems)
   * - ``Radau``
     - Implicit Runge-Kutta (stiff)
   * - ``LSODA``
     - Adams/BDF with automatic stiffness detection
   * - ``RK4``
     - Fixed-step classic 4th-order
   * - ``Euler``
     - Fixed-step forward Euler
