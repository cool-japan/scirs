"""Type stubs for scirs2.stats — Statistical functions module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def mean_py(data: ArrayLike, axis: Optional[int] = None) -> Union[float, NDArray[np.float64]]:
    """Arithmetic mean."""
    ...

def std_py(
    data: ArrayLike,
    axis: Optional[int] = None,
    ddof: int = 0,
) -> Union[float, NDArray[np.float64]]:
    """Standard deviation."""
    ...

def var_py(
    data: ArrayLike,
    axis: Optional[int] = None,
    ddof: int = 0,
) -> Union[float, NDArray[np.float64]]:
    """Variance."""
    ...

def median_py(data: ArrayLike) -> float:
    """Median value."""
    ...

def percentile_py(data: ArrayLike, q: float) -> float:
    """q-th percentile (0 <= q <= 100)."""
    ...

def iqr_py(data: ArrayLike) -> float:
    """Interquartile range (75th - 25th percentile)."""
    ...

def skew_py(data: ArrayLike) -> float:
    """Fisher-Pearson skewness."""
    ...

def kurtosis_py(data: ArrayLike) -> float:
    """Excess kurtosis (Fisher definition)."""
    ...

def correlation_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Pearson correlation coefficient."""
    ...

def covariance_py(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> float:
    """Sample covariance."""
    ...

def describe_py(data: NDArray[np.float64]) -> Dict[str, float]:
    """Descriptive statistics: mean, std, min, max, median, etc."""
    ...

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def ttest_1samp_py(data: NDArray[np.float64], popmean: float) -> Tuple[float, float]:
    """One-sample t-test: returns (statistic, p-value)."""
    ...

def ttest_ind_py(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    equal_var: bool = True,
) -> Tuple[float, float]:
    """Independent-samples t-test: returns (statistic, p-value)."""
    ...

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

class NormDist:
    """Normal distribution."""
    def __init__(self, loc: float = 0.0, scale: float = 1.0) -> None: ...
    def pdf(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def cdf(self, x: ArrayLike) -> NDArray[np.float64]: ...
    def ppf(self, q: ArrayLike) -> NDArray[np.float64]: ...
    def rvs(self, size: Optional[int] = None) -> NDArray[np.float64]: ...

class BinomDist:
    """Binomial distribution."""
    def __init__(self, n: int, p: float) -> None: ...
    def pmf(self, k: ArrayLike) -> NDArray[np.float64]: ...
    def cdf(self, k: ArrayLike) -> NDArray[np.float64]: ...
    def rvs(self, size: Optional[int] = None) -> NDArray[np.int64]: ...
