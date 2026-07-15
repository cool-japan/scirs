"""
Type stubs for the scirs2 Python package.

This package-level stub re-exports the public API and adds Protocol-based
type definitions that allow duck-typed Rust/Python boundaries to be checked
by mypy.

PEP 561 compliance: py.typed marker is present next to this file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Re-export scalar type aliases used across sub-modules
Float64Array = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntArray = NDArray[np.int64]

# ---------------------------------------------------------------------------
# Protocols for duck-typed array-like inputs
# ---------------------------------------------------------------------------

@runtime_checkable
class ArrayProtocol(Protocol):
    """Protocol satisfied by any object that exposes ``__array__``.

    Accepted wherever scirs2 functions take array-like arguments.
    """
    def __array__(self, dtype: Optional[np.dtype[Any]] = None) -> np.ndarray[Any, Any]: ...

@runtime_checkable
class DLPackProtocol(Protocol):
    """Protocol for DLPack-compatible tensors (PyTorch, JAX, CuPy, etc.)."""
    def __dlpack__(self, *, stream: Optional[int] = None) -> Any: ...
    def __dlpack_device__(self) -> Tuple[int, int]: ...

@runtime_checkable
class DataFrameProtocol(Protocol):
    """Minimal Protocol for pandas-like DataFrames."""
    @property
    def values(self) -> np.ndarray[Any, Any]: ...
    @property
    def columns(self) -> Any: ...
    def to_numpy(self) -> np.ndarray[Any, Any]: ...

@runtime_checkable
class FittedModel(Protocol):
    """Protocol for fitted scikit-learn-style estimators."""
    def predict(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "FittedModel": ...

@runtime_checkable
class Transformer(Protocol):
    """Protocol for estimators that transform data."""
    def fit(self, X: ArrayLike) -> "Transformer": ...
    def transform(self, X: ArrayLike) -> NDArray[np.float64]: ...
    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Module-level convenience type aliases
# ---------------------------------------------------------------------------

# 1-D, 2-D and dynamic array type shortcuts
Vec = NDArray[np.float64]
Mat = NDArray[np.float64]

# Sparse matrix (COO/CSR/CSC) placeholder — actual type provided by sub-module
SparseMatrix = Any

# ---------------------------------------------------------------------------
# Version / metadata
# ---------------------------------------------------------------------------

__version__: str
__author__: str

# ---------------------------------------------------------------------------
# Core clustering API (re-exported from scirs2.cluster)
# ---------------------------------------------------------------------------

class KMeans:
    """K-Means clustering algorithm."""
    labels: NDArray[np.int32]
    inertia_: float
    def __init__(self, n_clusters: int) -> None: ...
    def fit(self, data: NDArray[np.float64]) -> None: ...
    def predict(self, data: NDArray[np.float64]) -> NDArray[np.int32]: ...

# ---------------------------------------------------------------------------
# Linear algebra (re-exported from scirs2.linalg)
# ---------------------------------------------------------------------------

def det_py(matrix: NDArray[np.float64]) -> float: ...
def inv_py(matrix: NDArray[np.float64]) -> NDArray[np.float64]: ...
def trace_py(matrix: NDArray[np.float64]) -> float: ...
def solve_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def eig_py(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128]]: ...
def svd_py(
    matrix: NDArray[np.float64],
    full_matrices: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...

# ---------------------------------------------------------------------------
# FFT (re-exported from scirs2.fft)
# ---------------------------------------------------------------------------

def fft_py(x: ArrayLike, n: Optional[int] = None) -> NDArray[np.complex128]: ...
def ifft_py(x: ArrayLike, n: Optional[int] = None) -> NDArray[np.complex128]: ...
def rfft_py(x: ArrayLike, n: Optional[int] = None) -> NDArray[np.complex128]: ...
def irfft_py(x: ArrayLike, n: Optional[int] = None) -> NDArray[np.float64]: ...
def fftfreq_py(n: int, d: float = 1.0) -> NDArray[np.float64]: ...
def rfftfreq_py(n: int, d: float = 1.0) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# Statistics (re-exported from scirs2.stats)
# ---------------------------------------------------------------------------

def mean_py(data: ArrayLike, axis: Optional[int] = None) -> Union[float, NDArray[np.float64]]: ...
def std_py(data: ArrayLike, axis: Optional[int] = None) -> Union[float, NDArray[np.float64]]: ...
def var_py(data: ArrayLike, axis: Optional[int] = None) -> Union[float, NDArray[np.float64]]: ...
def median_py(data: ArrayLike) -> float: ...
def describe_py(data: NDArray[np.float64]) -> Dict[str, float]: ...

# ---------------------------------------------------------------------------
# Optimization (re-exported from scirs2.optimize)
# ---------------------------------------------------------------------------

def minimize_py(
    fun: Any,
    x0: ArrayLike,
    method: str = "L-BFGS-B",
    tol: Optional[float] = None,
    maxiter: Optional[int] = None,
) -> Dict[str, Any]: ...

# ---------------------------------------------------------------------------
# Integration (re-exported from scirs2.integrate)
# ---------------------------------------------------------------------------

def quad_py(
    fun: Any,
    a: float,
    b: float,
) -> Tuple[float, float]: ...

def solve_ivp_py(
    fun: Any,
    t_span: Tuple[float, float],
    y0: ArrayLike,
    method: str = "RK45",
    t_eval: Optional[ArrayLike] = None,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> Dict[str, Any]: ...

# ---------------------------------------------------------------------------
# Interpolation (re-exported from scirs2.interpolate)
# ---------------------------------------------------------------------------

def interp1d_py(
    x: ArrayLike,
    y: ArrayLike,
    kind: str = "linear",
) -> Any: ...

# ---------------------------------------------------------------------------
# Signal processing (re-exported from scirs2.signal)
# ---------------------------------------------------------------------------

def butter_py(
    N: int,
    Wn: Union[float, Sequence[float]],
    btype: str = "low",
    fs: Optional[float] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]: ...

def sosfilt_py(
    sos: NDArray[np.float64],
    x: NDArray[np.float64],
) -> NDArray[np.float64]: ...

# ---------------------------------------------------------------------------
# DLPack tensor interop (re-exported from scirs2 top-level)
# ---------------------------------------------------------------------------

def from_dlpack(capsule: Any) -> NDArray[np.float64]:
    """Convert a DLPack capsule (PyTorch, JAX, CuPy, …) to a scirs2 array.

    Parameters
    ----------
    capsule : PyCapsule
        A ``PyCapsule`` named ``"dltensor"`` — typically obtained by calling
        ``tensor.__dlpack__()`` on a PyTorch, JAX or CuPy tensor.

    Returns
    -------
    numpy.ndarray
        A NumPy array, matching the source tensor's shape, whose contents
        are *copied* from the DLPack tensor (this is a copying bridge, not a
        zero-copy view). CPU tensors of dtype int8/16/32/64, uint8/16/32/64,
        float32, or float64 are supported.

        Non-contiguous source tensors (a transposed PyTorch tensor from
        ``t.t().__dlpack__()``, a reversed/negative-stride view, a sliced
        view, …) are read according to their *actual* reported strides, so
        the copied data is always logically correct — never silently
        misread as if the buffer were C-contiguous.

    Raises
    ------
    TypeError
        If ``capsule`` is not a ``PyCapsule``, if the tensor resides on a
        non-CPU device, or if its dtype is not one of the supported types
        listed above.
    ValueError
        If the capsule is not named ``"dltensor"`` (e.g. it was already
        consumed), if the tensor has a null data pointer, or if the
        tensor's shape/strides metadata is malformed (e.g. an internal
        offset computation would overflow).
    RuntimeError
        If the capsule pointer cannot be read, or if importing/calling
        into NumPy fails.

    Examples
    --------
    >>> import torch, scirs2
    >>> t = torch.ones(3, 4)
    >>> arr = scirs2.from_dlpack(t.__dlpack__())
    >>> # Non-contiguous (transposed) source tensors are handled correctly:
    >>> arr_t = scirs2.from_dlpack(t.t().__dlpack__())
    """
    ...

def to_dlpack(array: NDArray[np.float64]) -> Any:
    """Export a scirs2 array as a DLPack ``PyCapsule``.

    Parameters
    ----------
    array : numpy.ndarray
        A NumPy-compatible array whose data should be shared. It is first
        coerced to a contiguous float64 array (via ``numpy.ascontiguousarray``)
        before being copied into the capsule.

    Returns
    -------
    PyCapsule
        A capsule named ``"dltensor"``, backed by an owned copy of the
        array's data (not a zero-copy view), consumable by
        ``torch.from_dlpack``, ``jax.dlpack.from_dlpack``, etc. The
        capsule's ``DLManagedTensor.deleter`` frees that copy once the
        consumer releases the tensor.

    Raises
    ------
    TypeError
        If ``array`` cannot be converted to a float64 NumPy array, or if
        its shape cannot be extracted.
    RuntimeError
        If importing NumPy fails or the underlying capsule allocation
        cannot be constructed.

    Examples
    --------
    >>> import numpy as np, torch, scirs2
    >>> arr = np.ones((3, 4))
    >>> cap = scirs2.to_dlpack(arr)
    >>> t = torch.from_dlpack(cap)
    """
    ...
