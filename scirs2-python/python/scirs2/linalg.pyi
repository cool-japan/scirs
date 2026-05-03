"""Type stubs for scirs2.linalg — Linear algebra module."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

def det_py(matrix: NDArray[np.float64]) -> float:
    """Compute the determinant of a square matrix."""
    ...

def inv_py(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the inverse of a square matrix."""
    ...

def trace_py(matrix: NDArray[np.float64]) -> float:
    """Sum of diagonal elements."""
    ...

def lu_py(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """LU decomposition: returns (P, L, U)."""
    ...

def qr_py(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """QR decomposition: returns (Q, R)."""
    ...

def svd_py(
    matrix: NDArray[np.float64],
    full_matrices: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Singular value decomposition: returns (U, S, Vt)."""
    ...

def cholesky_py(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cholesky factorisation (lower-triangular L where A = L @ L.T)."""
    ...

def eig_py(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """General eigenvalue decomposition: returns (eigenvalues, eigenvectors)."""
    ...

def eigh_py(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetric eigenvalue decomposition: returns (eigenvalues, eigenvectors)."""
    ...

def solve_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve the linear system A @ x = b."""
    ...

def lstsq_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], Optional[float], int, NDArray[np.float64]]:
    """Least-squares solution: returns (x, residuals, rank, sv)."""
    ...

def matrix_norm_py(matrix: NDArray[np.float64], ord: str = "fro") -> float:
    """Matrix norm (Frobenius, '1', '2', 'inf')."""
    ...

def vector_norm_py(vector: NDArray[np.float64], ord: Union[int, float] = 2) -> float:
    """Vector p-norm."""
    ...

def cond_py(matrix: NDArray[np.float64]) -> float:
    """Condition number of the matrix."""
    ...

def matrix_rank_py(matrix: NDArray[np.float64], tol: Optional[float] = None) -> int:
    """Numerical rank of the matrix."""
    ...

def expm_py(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrix exponential."""
    ...

def logm_py(matrix: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Matrix logarithm."""
    ...

def sqrtm_py(matrix: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Matrix square root."""
    ...

def schur_py(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Schur decomposition: returns (T, Z)."""
    ...

def kron_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Kronecker product of two matrices."""
    ...
