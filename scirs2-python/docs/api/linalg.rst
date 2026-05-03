Linear Algebra (``scirs2.linalg``)
===================================

Provides linear algebra operations backed by OxiBLAS — a pure-Rust BLAS
implementation.  All functions accept and return NumPy arrays.

.. automodule:: scirs2.linalg
   :members:
   :undoc-members:
   :show-inheritance:

Function Reference
------------------

Basic Operations
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``det_py(A)``
     - Matrix determinant
   * - ``inv_py(A)``
     - Matrix inverse
   * - ``trace_py(A)``
     - Matrix trace
   * - ``matrix_norm_py(A, ord)``
     - Matrix norm (Frobenius, spectral, …)
   * - ``vector_norm_py(v, ord)``
     - Vector norm (L1, L2, ∞, …)
   * - ``matrix_rank_py(A)``
     - Numerical matrix rank
   * - ``cond_py(A)``
     - Condition number
   * - ``pinv_py(A)``
     - Moore-Penrose pseudoinverse

Decompositions
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function
     - Returns
   * - ``lu_py(A)``
     - (P, L, U) pivot matrices
   * - ``qr_py(A)``
     - (Q, R) orthogonal decomposition
   * - ``svd_py(A)``
     - (U, S, Vt) singular-value decomposition
   * - ``cholesky_py(A)``
     - Lower triangular Cholesky factor
   * - ``eig_py(A)``
     - (eigenvalues, eigenvectors) — complex
   * - ``eigh_py(A)``
     - (eigenvalues, eigenvectors) — real symmetric/Hermitian
   * - ``schur_py(A)``
     - Schur decomposition

Solvers
~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - ``solve_py(A, b)``
     - Exact linear solve
   * - ``lstsq_py(A, b)``
     - Least-squares solution

Batch / Vectorized APIs
~~~~~~~~~~~~~~~~~~~~~~~

All ``batch_*`` functions accept a stack of matrices and operate in
parallel via Rayon:

- ``batch_matmul_py(A_stack, B_stack)``
- ``batch_svd_py(A_stack)``
- ``batch_solve_py(A_stack, b_stack)``
- ``batch_matrix_norm_py(A_stack, ord)``

Examples
--------

.. code-block:: python

   import numpy as np
   import scirs2

   A = np.array([[1.0, 2.0], [3.0, 4.0]])

   det = scirs2.det_py(A)
   inv = scirs2.inv_py(A)
   U, S, Vt = scirs2.svd_py(A)

   b = np.array([1.0, 2.0])
   x = scirs2.solve_py(A, b)
