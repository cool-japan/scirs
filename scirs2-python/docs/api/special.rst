Special Functions (``scirs2.special``)
========================================

Mathematical special functions: Bessel, gamma family, hypergeometric,
Airy, and more.

.. automodule:: scirs2.special
   :members:
   :undoc-members:
   :show-inheritance:

Function Groups
---------------

Gamma Family
~~~~~~~~~~~~

``gamma``, ``lgamma``, ``digamma``, ``polygamma``, ``gammaln``,
``gammasgn``, ``gammainc``, ``gammaincc``, ``gammaincinv``

Bessel Functions
~~~~~~~~~~~~~~~~

Real-order Bessel functions of the first and second kind:
``j0``, ``j1``, ``jn``, ``y0``, ``y1``, ``yn``,
modified: ``i0``, ``i1``, ``iv``, ``k0``, ``k1``, ``kn``

Hypergeometric
~~~~~~~~~~~~~~

``hyp0f1``, ``hyp1f1``, ``hyp2f1``, ``hyperu``

Airy Functions
~~~~~~~~~~~~~~

``airy_ai``, ``airy_aip``, ``airy_bi``, ``airy_bip``

Error / Exponential Integrals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``erf``, ``erfc``, ``erfinv``, ``erfcinv``,
``sici_si``, ``sici_ci``, ``shichi_shi``, ``shichi_chi``

Beta / Incomplete Beta
~~~~~~~~~~~~~~~~~~~~~~

``beta``, ``betaln``, ``betainc``, ``betaincinv``

Zeta Functions
~~~~~~~~~~~~~~

``zeta``, ``hurwitz_zeta``, ``zetac``

Vectorized Wrappers
~~~~~~~~~~~~~~~~~~~

Functions operating on NumPy arrays:
``lgamma_array``, ``erfc_array``, ``digamma_array``, ``j1_array``
