"""Type stubs for scirs2.integrate — Numerical integration module."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

def quad_py(
    fun: Callable[[float], float],
    a: float,
    b: float,
    limit: int = 50,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
) -> Tuple[float, float]:
    """Adaptive quadrature of a scalar function from a to b.

    Returns (integral, error_estimate).
    """
    ...

def trapezoid_array_py(
    y: ArrayLike,
    x: Optional[ArrayLike] = None,
    dx: float = 1.0,
) -> float:
    """Trapezoidal rule integration."""
    ...

def simpson_array_py(
    y: ArrayLike,
    x: Optional[ArrayLike] = None,
    dx: float = 1.0,
) -> float:
    """Simpson's rule integration (requires odd number of points)."""
    ...

def cumulative_trapezoid_py(
    y: ArrayLike,
    x: Optional[ArrayLike] = None,
    dx: float = 1.0,
    initial: Optional[float] = None,
) -> NDArray[np.float64]:
    """Cumulative trapezoidal integration."""
    ...

def romberg_array_py(
    y: ArrayLike,
    dx: float = 1.0,
) -> float:
    """Romberg integration on a uniformly-spaced sample."""
    ...

# ---------------------------------------------------------------------------
# ODE solvers
# ---------------------------------------------------------------------------

class OdeSolution:
    """Dense output from an ODE solver (callable interpolant)."""
    t: NDArray[np.float64]
    y: NDArray[np.float64]
    def __call__(self, t: Union[float, ArrayLike]) -> NDArray[np.float64]: ...

def solve_ivp_py(
    fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]],
    t_span: Tuple[float, float],
    y0: ArrayLike,
    method: str = "RK45",
    t_eval: Optional[ArrayLike] = None,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    max_step: float = float("inf"),
) -> Dict[str, Any]:
    """Solve an initial-value ODE problem.

    Returns a dict with keys ``t``, ``y``, ``success``, ``message``.
    Mirrors ``scipy.integrate.solve_ivp``.
    """
    ...
