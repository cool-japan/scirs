"""Type stubs for scirs2.fft — Fast Fourier Transform module."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

def fft_py(x: ArrayLike, n: Optional[int] = None) -> NDArray[np.complex128]:
    """1-D discrete Fourier transform."""
    ...

def ifft_py(x: ArrayLike, n: Optional[int] = None) -> NDArray[np.complex128]:
    """Inverse 1-D DFT."""
    ...

def rfft_py(x: ArrayLike, n: Optional[int] = None) -> NDArray[np.complex128]:
    """DFT of a real-valued sequence (positive frequencies only)."""
    ...

def irfft_py(x: ArrayLike, n: Optional[int] = None) -> NDArray[np.float64]:
    """Inverse DFT of a Hermitian-symmetric spectrum."""
    ...

def fft2_py(
    x: ArrayLike,
    s: Optional[tuple[int, int]] = None,
) -> NDArray[np.complex128]:
    """2-D DFT."""
    ...

def dct_py(x: ArrayLike, type: int = 2) -> NDArray[np.float64]:
    """Discrete Cosine Transform (types 1–4)."""
    ...

def idct_py(x: ArrayLike, type: int = 2) -> NDArray[np.float64]:
    """Inverse Discrete Cosine Transform."""
    ...

def fftfreq_py(n: int, d: float = 1.0) -> NDArray[np.float64]:
    """DFT sample frequencies."""
    ...

def rfftfreq_py(n: int, d: float = 1.0) -> NDArray[np.float64]:
    """DFT frequencies for real-input FFT (positive half only)."""
    ...

def fftshift_py(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Shift zero-frequency component to the centre of the spectrum."""
    ...

def ifftshift_py(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Inverse of fftshift."""
    ...

def next_fast_len_py(n: int) -> int:
    """Smallest FFT-efficient length >= n."""
    ...
