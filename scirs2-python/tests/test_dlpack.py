"""Tests for scirs2 DLPack tensor interop (to_dlpack / from_dlpack).

Regression coverage for a memory-safety bug found during 0.6.1 Python-surface
validation: `to_dlpack`'s internal `BackingStore` wrapper was missing
`#[repr(C)]`, so the pointer stashed in the capsule did not actually address
a `DLManagedTensor` once Rust's (unspecified) default layout reordered the
struct's fields. Consuming or simply garbage-collecting a `to_dlpack`
capsule crashed the interpreter with SIGBUS. A second, related bug in the
capsule destructor did not account for capsules already consumed by
`from_dlpack` (renamed to `"used_dltensor"`), leaking a `ValueError` out of
`gc.collect()`. Both are fixed in `scirs2-python/src/dlpack.rs`; the tests
below exercise exactly the scenarios that used to crash or raise.
"""

import gc

import numpy as np
import pytest

import scirs2


class TestToDlpackFromDlpackRoundTrip:
    """Round-trip scirs2.to_dlpack() -> scirs2.from_dlpack()."""

    def test_1d_float64_round_trip(self):
        """Basic 1D float64 self round-trip preserves values."""
        a = np.array([1.0, 2.0, 3.0, 4.5, -7.25], dtype=np.float64)
        capsule = scirs2.to_dlpack(a)
        back = scirs2.from_dlpack(capsule)
        assert np.array_equal(a, back)

    def test_2d_contiguous_round_trip(self):
        """2D contiguous array preserves shape and values."""
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        capsule = scirs2.to_dlpack(a)
        back = scirs2.from_dlpack(capsule)
        assert back.shape == (3, 4)
        assert np.array_equal(a, back)


class TestFromDlpackStrideHandling:
    """from_dlpack must honor non-contiguous producer strides, not assume
    C-contiguity (exercises numpy itself as the DLPack *producer*)."""

    def test_transposed_2d_view(self):
        """A transposed (non-contiguous) view must be read in logical order."""
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        a_t = a.T
        assert not a_t.flags["C_CONTIGUOUS"]
        back = scirs2.from_dlpack(a_t.__dlpack__())
        assert np.array_equal(a_t, back)

    def test_negative_stride_reversed_slice(self):
        """A reversed (negative-stride) slice must be read correctly."""
        a = np.arange(10, dtype=np.float64)[::-1]
        back = scirs2.from_dlpack(a.__dlpack__())
        assert np.array_equal(a, back)

    def test_row_strided_2d_slice(self):
        """A non-contiguous row-strided slice must be read correctly."""
        a = np.arange(20, dtype=np.float64).reshape(5, 4)[::2, :]
        back = scirs2.from_dlpack(a.__dlpack__())
        assert np.array_equal(a, back)

    @pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.int64, np.float32])
    def test_supported_dtypes(self, dtype):
        """int/uint 8-64 and float32/64 producers must all be consumable."""
        a = np.array([0, 1, 2, 3, 4], dtype=dtype)
        back = scirs2.from_dlpack(a.__dlpack__())
        assert np.array_equal(a, back.astype(dtype))


class TestCapsuleLifecycleMemorySafety:
    """Regression tests for the SIGBUS / double-free bugs.

    These must merely *complete without crashing the interpreter or raising*
    -- that is the entire point of the regression.
    """

    def test_to_dlpack_alone_garbage_collected_without_consumption(self):
        """A to_dlpack capsule that is never passed to from_dlpack must still
        be freed safely by its own destructor when garbage-collected.

        Before the `#[repr(C)]` fix, this alone crashed the process with a
        SIGBUS (the destructor reinterpreted unrelated bytes as a
        DLManagedTensor and jumped to a garbage `deleter` function pointer).
        """
        for _ in range(20):
            arr = np.random.rand(4, 4)
            capsule = scirs2.to_dlpack(arr)
            del capsule
        gc.collect()

    def test_consumed_capsule_garbage_collected_cleanly(self):
        """A capsule already consumed by from_dlpack (renamed to
        "used_dltensor") must not raise when its destructor later runs.

        Before the destructor fix, the destructor unconditionally looked up
        the capsule pointer under the original "dltensor" name, which
        CPython rejects with ValueError once the name has changed --
        surfacing as a SystemError out of gc.collect().
        """
        for _ in range(20):
            arr = np.random.rand(7, 5)
            capsule = scirs2.to_dlpack(arr)
            back = scirs2.from_dlpack(capsule)
            assert np.allclose(arr, back)
            del capsule, back
        gc.collect()

    def test_mixed_consumed_and_unconsumed_capsules(self):
        """Interleaving consumed and never-consumed capsules must not
        interfere with each other's cleanup."""
        kept_alive = []
        for i in range(20):
            arr = np.random.rand(3, 3)
            capsule = scirs2.to_dlpack(arr)
            if i % 2 == 0:
                back = scirs2.from_dlpack(capsule)
                assert np.allclose(arr, back)
            else:
                kept_alive.append(capsule)
        del kept_alive
        gc.collect()
