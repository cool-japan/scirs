Statistics (``scirs2.stats``)
==============================

Statistical functions, distributions, hypothesis tests, MCMC samplers,
and Gaussian process regression.

.. automodule:: scirs2.stats
   :members:
   :undoc-members:
   :show-inheritance:

Descriptive Statistics
----------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``mean_py(data)``
     - Arithmetic mean
   * - ``std_py(data, ddof)``
     - Standard deviation
   * - ``var_py(data, ddof)``
     - Variance
   * - ``median_py(data)``
     - Median
   * - ``percentile_py(data, q)``
     - Percentile
   * - ``iqr_py(data)``
     - Interquartile range
   * - ``skew_py(data)``
     - Skewness (6-23x faster than SciPy on small data)
   * - ``kurtosis_py(data)``
     - Excess kurtosis (5-24x faster than SciPy)
   * - ``describe_py(data)``
     - Summary statistics dictionary

Correlation
-----------

- ``correlation_py(x, y)`` — Pearson correlation
- ``covariance_py(x, y, ddof)`` — Covariance
- ``pearsonr_py(x, y)`` — Pearson r with p-value
- ``spearmanr_py(x, y)`` — Spearman rank correlation

Distributions
-------------

All distributions expose ``pdf``, ``cdf``, ``ppf``, ``rvs`` methods:

.. code-block:: python

   dist = scirs2.norm()
   p = dist.pdf(0.0)     # 0.3989…
   q = dist.cdf(1.96)    # 0.9750…
   x = dist.ppf(0.975)   # 1.96…

Available distributions:

- **Continuous**: ``norm``, ``expon``, ``uniform``, ``beta``, ``gamma``,
  ``chi2``, ``t``, ``f``, ``cauchy``, ``lognorm``, ``weibull_min``,
  ``laplace``, ``logistic``, ``pareto``
- **Discrete**: ``binom``, ``poisson``, ``geom``, ``bernoulli``,
  ``nbinom``, ``hypergeom``

MCMC Samplers
-------------

.. code-block:: python

   import scirs2, numpy as np

   def log_prob(x):
       return -0.5 * float(np.dot(x, x))   # standard normal

   sampler = scirs2.MetropolisHastings(log_prob, step_size=0.5)
   samples = sampler.sample(np.zeros(2), n_samples=1000, burn_in=100)

Available samplers: ``MetropolisHastings``, ``HamiltonianMC``, ``NUTS``

Gaussian Process Regression
----------------------------

.. code-block:: python

   gp = scirs2.GaussianProcessRegressor(kernel="rbf", noise=1e-6)
   gp.fit(X_train, y_train)
   mean, std = gp.predict(X_test, return_std=True)

Survival Analysis
-----------------

- ``KaplanMeier`` — Non-parametric survival estimator
- ``NelsonAalen`` — Cumulative hazard estimator
- ``CoxPH`` — Proportional hazards model
