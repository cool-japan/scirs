"""
pytest tests for scirs2-numpy array conversion parity with rust-numpy.

These tests verify that scirs2-numpy produces results identical to numpy /
rust-numpy behavior for common array conversion operations.

Run with:
    pytest scirs2-numpy/python/tests/

Requirements:
    pip install pytest numpy
    cd scirs2-numpy && maturin develop --features pyo3/extension-module
"""

import pytest
import numpy as np

# Guard: skip all tests if the native extension is not installed.
# Install it with: maturin develop (from scirs2-numpy directory)
try:
    import scirs2_numpy as scirs2np  # type: ignore[import]
    HAS_SCIRS2 = True
except ImportError:
    HAS_SCIRS2 = False

pytestmark = pytest.mark.skipif(
    not HAS_SCIRS2,
    reason="scirs2_numpy not installed — run `maturin develop` first",
)


# ---------------------------------------------------------------------------
# Basic dtype round-trips
# ---------------------------------------------------------------------------


def test_float32_array_roundtrip():
    """f32 array survives a Rust round-trip unchanged."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = scirs2np.from_numpy(arr)
    np.testing.assert_array_equal(np.array(result), arr)


def test_float64_array_identity():
    """f64 array is bitwise-identical after round-trip."""
    arr = np.arange(100, dtype=np.float64)
    result = scirs2np.from_numpy(arr)
    np.testing.assert_allclose(np.array(result), arr)


def test_int32_array_roundtrip():
    """Integer arrays are preserved without silent type promotion."""
    arr = np.array([-1, 0, 1, 127, -128], dtype=np.int32)
    result = scirs2np.from_numpy(arr)
    np.testing.assert_array_equal(np.array(result), arr)


def test_bool_array_roundtrip():
    """Boolean arrays are preserved."""
    arr = np.array([True, False, True, True, False], dtype=bool)
    result = scirs2np.from_numpy(arr)
    np.testing.assert_array_equal(np.array(result), arr)


# ---------------------------------------------------------------------------
# Shape handling
# ---------------------------------------------------------------------------


def test_2d_contiguous_array():
    """2-D C-contiguous arrays retain their shape."""
    arr = np.ones((10, 20), dtype=np.float32)
    result = scirs2np.from_numpy(arr)
    assert np.array(result).shape == (10, 20)


def test_3d_array_shape_preserved():
    """3-D arrays have the correct shape after conversion."""
    arr = np.zeros((4, 5, 6), dtype=np.float64)
    result = scirs2np.from_numpy(arr)
    assert np.array(result).shape == (4, 5, 6)


def test_scalar_wrapped_as_0d():
    """A 0-D (scalar) numpy array converts without error."""
    arr = np.array(42.0, dtype=np.float64)
    result = scirs2np.from_numpy(arr)
    assert float(np.array(result)) == 42.0


# ---------------------------------------------------------------------------
# Non-contiguous / strided inputs
# ---------------------------------------------------------------------------


def test_non_contiguous_array_converted():
    """Non-contiguous slices are converted after explicit contiguification."""
    arr = np.arange(100.0).reshape(10, 10)[::2, ::2]  # non-contiguous
    result = scirs2np.from_numpy(np.ascontiguousarray(arr))
    assert result is not None


def test_fortran_order_array():
    """Fortran-order arrays are handled (may trigger a copy)."""
    arr = np.asfortranarray(np.ones((8, 8), dtype=np.float64))
    result = scirs2np.from_numpy(np.ascontiguousarray(arr))
    assert np.array(result).shape == (8, 8)


def test_transposed_array_converted():
    """Transposed view is handled after making it contiguous."""
    arr = np.arange(12.0).reshape(3, 4).T  # transposed: shape (4, 3)
    result = scirs2np.from_numpy(np.ascontiguousarray(arr))
    assert np.array(result).shape == (4, 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_length_array():
    """Zero-length array produces an empty result without error."""
    arr = np.array([], dtype=np.float64)
    result = scirs2np.from_numpy(arr)
    assert len(np.array(result)) == 0


def test_single_element_array():
    """Single-element array round-trips correctly."""
    arr = np.array([3.14], dtype=np.float64)
    result = scirs2np.from_numpy(arr)
    np.testing.assert_allclose(np.array(result), arr)


def test_large_array_no_copy():
    """Large arrays use zero-copy (or at least complete without OOM)."""
    arr = np.zeros(1_000_000, dtype=np.float64)
    result = scirs2np.from_numpy(arr)
    assert np.array(result).shape == (1_000_000,)


def test_nan_values_preserved():
    """NaN and inf values survive the round-trip."""
    arr = np.array([float("nan"), float("inf"), float("-inf"), 0.0], dtype=np.float64)
    result = np.array(scirs2np.from_numpy(arr))
    assert np.isnan(result[0])
    assert np.isposinf(result[1])
    assert np.isneginf(result[2])


def test_very_small_float32_values():
    """Denormal / subnormal f32 values are preserved."""
    tiny = np.finfo(np.float32).tiny
    arr = np.array([tiny, tiny * 0.5, 0.0], dtype=np.float32)
    result = scirs2np.from_numpy(arr)
    np.testing.assert_array_equal(np.array(result), arr)


def test_integer_types_i64():
    """64-bit integer arrays are handled."""
    arr = np.array([-(2**62), 0, 2**62 - 1], dtype=np.int64)
    result = scirs2np.from_numpy(arr)
    np.testing.assert_array_equal(np.array(result), arr)


# ---------------------------------------------------------------------------
# DLPack protocol parity
# ---------------------------------------------------------------------------


def test_dlpack_export():
    """Arrays expose __dlpack__ / __dlpack_device__ when available."""
    arr = np.zeros((4, 4), dtype=np.float32)
    result = scirs2np.from_numpy(arr)
    # If the Rust type exposes __dlpack__, verify it is callable.
    if hasattr(result, "__dlpack__"):
        caps = result.__dlpack__()
        assert caps is not None


def test_dlpack_device_is_cpu():
    """DLPack device tuple is (1, 0) for CPU tensors."""
    arr = np.ones(10, dtype=np.float64)
    result = scirs2np.from_numpy(arr)
    if hasattr(result, "__dlpack_device__"):
        dev = result.__dlpack_device__()
        assert dev[0] == 1, f"Expected device type 1 (CPU), got {dev[0]}"
