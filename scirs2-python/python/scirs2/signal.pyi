"""Type stubs for scirs2.signal — Signal processing module."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Filter design
# ---------------------------------------------------------------------------

def butter_py(
    N: int,
    Wn: Union[float, Sequence[float]],
    btype: str = "low",
    analog: bool = False,
    output: str = "ba",
    fs: Optional[float] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Design a Butterworth filter.

    Returns (b, a) coefficients or (sos,) depending on *output*.
    Mirrors ``scipy.signal.butter``.
    """
    ...

def sosfilt_py(
    sos: NDArray[np.float64],
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a second-order section (SOS) IIR filter to a signal."""
    ...

def filtfilt_py(
    b: NDArray[np.float64],
    a: NDArray[np.float64],
    x: NDArray[np.float64],
    padlen: Optional[int] = None,
) -> NDArray[np.float64]:
    """Zero-phase forward-backward filter."""
    ...

# ---------------------------------------------------------------------------
# Convolution / correlation
# ---------------------------------------------------------------------------

def convolve_py(
    in1: NDArray[np.float64],
    in2: NDArray[np.float64],
    mode: str = "full",
) -> NDArray[np.float64]:
    """1-D convolution of two signals."""
    ...

def correlate_py(
    in1: NDArray[np.float64],
    in2: NDArray[np.float64],
    mode: str = "full",
) -> NDArray[np.float64]:
    """Cross-correlation of two 1-D signals."""
    ...

# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def periodogram_py(
    x: NDArray[np.float64],
    fs: float = 1.0,
    window: str = "boxcar",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Power spectral density estimate via periodogram.

    Returns (frequencies, power_spectral_density).
    """
    ...

def welch_py(
    x: NDArray[np.float64],
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Welch's method for power spectral density estimation.

    Returns (frequencies, power_spectral_density).
    """
    ...

# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def hann_window_py(n: int) -> NDArray[np.float64]:
    """Hann window of length n."""
    ...

def hamming_window_py(n: int) -> NDArray[np.float64]:
    """Hamming window of length n."""
    ...

def blackman_window_py(n: int) -> NDArray[np.float64]:
    """Blackman window of length n."""
    ...
