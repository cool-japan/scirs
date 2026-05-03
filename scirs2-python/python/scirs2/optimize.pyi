"""Type stubs for scirs2.optimize — Optimization module."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class OptimizeResult:
    """Optimization result container (mirrors scipy.optimize.OptimizeResult)."""
    x: NDArray[np.float64]
    success: bool
    fun: float
    nit: int
    message: str
    def __init__(self, **kwargs: Any) -> None: ...

# ---------------------------------------------------------------------------
# Minimization
# ---------------------------------------------------------------------------

def minimize_py(
    fun: Callable[[NDArray[np.float64]], float],
    x0: ArrayLike,
    method: str = "L-BFGS-B",
    jac: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    tol: Optional[float] = None,
    maxiter: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Minimise a scalar function of one or more variables.

    Mirrors the interface of ``scipy.optimize.minimize``.
    """
    ...

def minimize_scalar_py(
    fun: Callable[[float], float],
    bounds: Optional[Tuple[float, float]] = None,
    method: str = "brent",
) -> Dict[str, Any]:
    """Minimise a scalar function of a single variable."""
    ...

# ---------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------

def brentq_py(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 2e-12,
    maxiter: int = 100,
) -> float:
    """Find a root of f in the interval [a, b] using Brent's method."""
    ...

def fsolve_py(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: ArrayLike,
    tol: float = 1.49012e-8,
) -> NDArray[np.float64]:
    """Find roots of a multivariate function."""
    ...

# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def curve_fit_py(
    f: Callable[..., NDArray[np.float64]],
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: Optional[ArrayLike] = None,
    sigma: Optional[ArrayLike] = None,
    maxfev: int = 1000,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit a curve to data using non-linear least squares.

    Returns (popt, pcov) — optimal parameters and covariance matrix.
    """
    ...

# ---------------------------------------------------------------------------
# Linear programming
# ---------------------------------------------------------------------------

def linprog_py(
    c: ArrayLike,
    a_ub: Optional[NDArray[np.float64]] = None,
    b_ub: Optional[NDArray[np.float64]] = None,
    a_eq: Optional[NDArray[np.float64]] = None,
    b_eq: Optional[NDArray[np.float64]] = None,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
) -> Dict[str, Any]:
    """Minimise c^T x subject to linear constraints."""
    ...
