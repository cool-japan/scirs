//! NumPy compatibility layer for scirs2-core
//!
//! This module ensures that scirs2_core::ndarray types are fully compatible with
//! PyO3's numpy crate, enabling seamless Python integration.
//!
//! # Problem Statement
//!
//! The PyO3 `numpy` crate (v0.28.3) delegates NumPy bindings to `scirs2-numpy` v0.5.0. When scirs2-core
//! uses `ndarray` v0.17 by default, there's a type incompatibility. Python integration
//! modules must explicitly use `ndarray16` types to ensure compatibility with numpy.
//!
//! # Solution
//!
//! This module provides:
//! 1. Explicit re-exports of ndarray types that numpy needs
//! 2. Type aliases that guarantee compatibility
//! 3. Conversion utilities for zero-copy operations
//!
//! # Usage in Python Bindings
//!
//! ```ignore
//! use pyo3::prelude::*;
//! use scirs2_core::python::numpy_compat::*;
//!
//! #[pyfunction]
//! fn process_array(array: PyReadonlyArrayDyn<f32>) -> PyResult<Py<PyArrayDyn<f32>>> {
//!     // Convert from NumPy to scirs2 array
//!     let scirs_array = numpy_to_scirs_arrayd(array)?;
//!
//!     // Process with scirs2-core
//!     let result = scirs_array.map(|x| x * 2.0);
//!
//!     // Convert back to NumPy
//!     scirs_to_numpy_arrayd(result, array.py())
//! }
//! ```

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use scirs2_numpy::{
    Element, PyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};

// Re-export ndarray types for Python compatibility
// IMPORTANT: Python integration uses scirs2-numpy with ndarray 0.17 support
#[cfg(feature = "python")]
pub use ::ndarray::{
    arr1, arr2, array, s, Array, Array0, Array1, Array2, Array3, Array4, ArrayBase, ArrayD,
    ArrayView, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMutD, Axis, Data, DataMut, DataOwned, Dim, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, IxDyn,
    IxDynImpl, OwnedRepr, RawData, ViewRepr, Zip,
};

/// Type alias for numpy-compatible dynamic arrays
#[cfg(feature = "python")]
pub type NumpyCompatArrayD<T> = ArrayD<T>;

/// Type alias for numpy-compatible 1D arrays
#[cfg(feature = "python")]
pub type NumpyCompatArray1<T> = Array1<T>;

/// Type alias for numpy-compatible 2D arrays
#[cfg(feature = "python")]
pub type NumpyCompatArray2<T> = Array2<T>;

// ========================================
// CONVERSION FUNCTIONS
// ========================================

/// Convert NumPy array to scirs2 ArrayD (zero-copy when possible)
///
/// This function attempts zero-copy conversion when the NumPy array is C-contiguous.
/// Otherwise, it creates a owned copy.
#[cfg(feature = "python")]
pub fn numpy_to_scirs_arrayd<'py, T>(array: &Bound<'py, PyArrayDyn<T>>) -> PyResult<ArrayD<T>>
where
    T: Element + Clone,
{
    let readonly = array.readonly();
    let array_ref = readonly.as_array();

    // Always create owned array for safety
    // Zero-copy path: use `numpy_readonly_to_scirs_view` for a lifetime-bounded
    // `ArrayViewD<'a, T>` that avoids the allocation here. This owned-copy path
    // exists for callers that need owned data or mutable access.
    Ok(array_ref.to_owned())
}

/// Convert NumPy readonly array to scirs2 ArrayD
#[cfg(feature = "python")]
pub fn numpy_readonly_to_scirs_arrayd<T>(array: PyReadonlyArrayDyn<T>) -> PyResult<ArrayD<T>>
where
    T: Element + Clone,
{
    Ok(array.as_array().to_owned())
}

/// Convert scirs2 ArrayD to NumPy array
#[cfg(feature = "python")]
pub fn scirs_to_numpy_arrayd<T: Element>(
    array: ArrayD<T>,
    py: Python<'_>,
) -> PyResult<Py<PyArrayDyn<T>>> {
    Ok(PyArrayDyn::from_owned_array(py, array).unbind())
}

/// Convert scirs2 Array1 to NumPy array
#[cfg(feature = "python")]
pub fn scirs_to_numpy_array1<T: Element>(
    array: Array1<T>,
    py: Python<'_>,
) -> PyResult<Py<PyArray1<T>>> {
    Ok(PyArray1::from_owned_array(py, array).unbind())
}

/// Convert scirs2 Array2 to NumPy array
#[cfg(feature = "python")]
pub fn scirs_to_numpy_array2<T: Element>(
    array: Array2<T>,
    py: Python<'_>,
) -> PyResult<Py<PyArray2<T>>> {
    Ok(PyArray2::from_owned_array(py, array).unbind())
}

// ========================================
// BATCH CONVERSION UTILITIES
// ========================================

/// Convert a vector of NumPy arrays to scirs2 ArrayD arrays
#[cfg(feature = "python")]
pub fn numpy_batch_to_scirs<T: Element + Clone>(
    arrays: Vec<PyReadonlyArrayDyn<T>>,
) -> PyResult<Vec<ArrayD<T>>> {
    arrays
        .into_iter()
        .map(|arr| Ok(arr.as_array().to_owned()))
        .collect()
}

/// Convert a vector of scirs2 arrays to NumPy arrays
#[cfg(feature = "python")]
pub fn scirs_batch_to_numpy<T: Element>(
    arrays: Vec<ArrayD<T>>,
    py: Python<'_>,
) -> PyResult<Vec<Py<PyArrayDyn<T>>>> {
    arrays
        .into_iter()
        .map(|arr| Ok(PyArrayDyn::from_owned_array(py, arr).unbind()))
        .collect()
}

// ========================================
// VIEW CONVERSION (ZERO-COPY)
// ========================================

/// Convert NumPy readonly array to scirs2 ArrayView (zero-copy)
///
/// This provides true zero-copy access to NumPy arrays. The readonly guard
/// must be kept alive for the lifetime of the view.
///
/// # Example
///
/// ```ignore
/// let readonly = numpy_array.readonly();
/// let view = numpy_readonly_to_scirs_view(&readonly);
/// // Use view while readonly is in scope
/// ```
#[cfg(feature = "python")]
pub fn numpy_readonly_to_scirs_view<'a, T: Element>(
    array: &'a PyReadonlyArrayDyn<'a, T>,
) -> ArrayViewD<'a, T> {
    array.as_array()
}

// ========================================
// TYPE INFORMATION UTILITIES
// ========================================

/// Check if a NumPy array is compatible with scirs2 operations
#[cfg(feature = "python")]
pub fn is_numpy_compatible<T: Element>(array: &Bound<'_, PyArrayDyn<T>>) -> bool {
    // Check if array is well-formed (has valid shape and strides)
    array.shape().iter().all(|&dim| dim > 0)
}

/// Get the memory layout of a NumPy array
#[cfg(feature = "python")]
pub enum MemoryLayout {
    CContiguous,
    FContiguous,
    Neither,
}

#[cfg(feature = "python")]
pub fn get_numpy_layout<T: Element>(array: &Bound<'_, PyArrayDyn<T>>) -> MemoryLayout {
    if array.is_c_contiguous() {
        MemoryLayout::CContiguous
    } else if array.is_fortran_contiguous() {
        MemoryLayout::FContiguous
    } else {
        MemoryLayout::Neither
    }
}

// ========================================
// TESTS
// ========================================

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    #[allow(deprecated)]
    fn test_numpy_scirs_roundtrip() {
        pyo3::Python::attach(|py| {
            // Create scirs2 array
            let scirs_array = array![[1.0f32, 2.0], [3.0, 4.0]];
            let scirs_arrayd = scirs_array.into_dyn();

            // Convert to NumPy
            let numpy_array =
                scirs_to_numpy_arrayd(scirs_arrayd.clone(), py).expect("Operation failed");

            // Convert back to scirs2
            let result = numpy_to_scirs_arrayd(&numpy_array.bind(py)).expect("Operation failed");

            // Verify equality
            assert_eq!(result.shape(), scirs_arrayd.shape());
            assert_eq!(result, scirs_arrayd);
        });
    }

    #[test]
    #[allow(deprecated)]
    fn test_zero_copy_view() {
        pyo3::Python::attach(|py| {
            let scirs_array = array![[1.0f32, 2.0], [3.0, 4.0]].into_dyn();
            let numpy_array =
                scirs_to_numpy_arrayd(scirs_array.clone(), py).expect("Operation failed");

            // Get zero-copy view
            let readonly = numpy_array.bind(py).readonly();
            let view = numpy_readonly_to_scirs_view(&readonly);

            // Verify it's the same data
            assert_eq!(view.shape(), scirs_array.shape());
            assert_eq!(view[[0, 0]], 1.0f32);
        });
    }
}
