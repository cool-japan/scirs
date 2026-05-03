//! Extended integer type support: `i128` and `u128`.
//!
//! NumPy does not expose native `int128` / `uint128` dtypes, but PyO3 can
//! round-trip Rust `i128` / `u128` values through Python's arbitrary-precision
//! `int` type.  This module provides:
//!
//! - Free functions [`to_i128`], [`from_i128`], [`to_u128`], [`from_u128`] for
//!   single-value conversion.
//! - [`I128Array`] and [`U128Array`] — Python-visible fixed-size containers
//!   backed by `Vec<i128>` / `Vec<u128>` respectively.

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

// ──────────────────────────────────────────────────────────────────────────────
// Free-standing conversion helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Convert a Python `int` to a Rust `i128`.
///
/// PyO3 supports extracting `i128` directly from Python's arbitrary-precision
/// integer type; values outside `[i128::MIN, i128::MAX]` raise `OverflowError`.
///
/// # Errors
/// Returns a [`PyErr`] if the Python object is not an integer or if it
/// overflows `i128`.
#[pyfunction]
pub fn to_i128(val: &Bound<'_, PyAny>) -> PyResult<i128> {
    val.extract::<i128>()
}

/// Convert a Rust `i128` to a Python `int`.
///
/// The resulting Python object is an ordinary `int` and can be used with any
/// Python code that accepts integers.
///
/// # Errors
/// Returns a [`PyErr`] on internal conversion failure (should not happen in
/// practice).
#[pyfunction]
pub fn from_i128(py: Python<'_>, val: i128) -> PyResult<Py<PyAny>> {
    Ok(val.into_pyobject(py)?.into_any().unbind())
}

/// Convert a Python `int` to a Rust `u128`.
///
/// Values outside `[0, u128::MAX]` raise `OverflowError`.
///
/// # Errors
/// Returns a [`PyErr`] if the value is not a non-negative integer or if it
/// overflows `u128`.
#[pyfunction]
pub fn to_u128(val: &Bound<'_, PyAny>) -> PyResult<u128> {
    val.extract::<u128>()
}

/// Convert a Rust `u128` to a Python `int`.
///
/// # Errors
/// Returns a [`PyErr`] on internal conversion failure (should not happen in
/// practice).
#[pyfunction]
pub fn from_u128(py: Python<'_>, val: u128) -> PyResult<Py<PyAny>> {
    Ok(val.into_pyobject(py)?.into_any().unbind())
}

// ──────────────────────────────────────────────────────────────────────────────
// I128Array
// ──────────────────────────────────────────────────────────────────────────────

/// A Python-visible fixed-size array of `i128` values.
///
/// NumPy does not expose a native `int128` dtype, so this type provides a
/// pure-Rust container that can be passed to and from Python via PyO3's
/// arbitrary-precision integer support.
#[pyclass(name = "I128Array")]
pub struct I128Array {
    /// Backing storage.
    data: Vec<i128>,
}

#[pymethods]
impl I128Array {
    /// Construct a zero-initialised array of `size` elements.
    #[new]
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0i128; size],
        }
    }

    /// Return the element at `idx`.
    ///
    /// # Errors
    /// Returns `IndexError` if `idx >= self.len()`.
    pub fn get(&self, idx: usize) -> PyResult<i128> {
        self.data.get(idx).copied().ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for I128Array of length {}",
                idx,
                self.data.len()
            ))
        })
    }

    /// Set the element at `idx` to `val`.
    ///
    /// # Errors
    /// Returns `IndexError` if `idx >= self.len()`.
    pub fn set(&mut self, idx: usize, val: i128) -> PyResult<()> {
        let len = self.data.len();
        self.data.get_mut(idx).map(|e| *e = val).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for I128Array of length {}",
                idx, len
            ))
        })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compute the sum of all elements.
    pub fn sum(&self) -> i128 {
        self.data.iter().sum()
    }

    /// Return all elements as a Python list of `int` values.
    pub fn to_list(&self) -> Vec<i128> {
        self.data.clone()
    }

    /// Fill the array from a Python list of integers.
    ///
    /// # Errors
    /// Returns a `ValueError` if `values` has a different length than `self`,
    /// or a `TypeError`/`OverflowError` if an element cannot be converted to
    /// `i128`.
    pub fn from_list(&mut self, values: Vec<i128>) -> PyResult<()> {
        if values.len() != self.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} elements, got {}",
                self.data.len(),
                values.len()
            )));
        }
        self.data.copy_from_slice(&values);
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// U128Array
// ──────────────────────────────────────────────────────────────────────────────

/// A Python-visible fixed-size array of `u128` values.
///
/// Mirrors [`I128Array`] but for unsigned 128-bit integers.
#[pyclass(name = "U128Array")]
pub struct U128Array {
    /// Backing storage.
    data: Vec<u128>,
}

#[pymethods]
impl U128Array {
    /// Construct a zero-initialised array of `size` elements.
    #[new]
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u128; size],
        }
    }

    /// Return the element at `idx`.
    ///
    /// # Errors
    /// Returns `IndexError` if `idx >= self.len()`.
    pub fn get(&self, idx: usize) -> PyResult<u128> {
        self.data.get(idx).copied().ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for U128Array of length {}",
                idx,
                self.data.len()
            ))
        })
    }

    /// Set the element at `idx` to `val`.
    ///
    /// # Errors
    /// Returns `IndexError` if `idx >= self.len()`.
    pub fn set(&mut self, idx: usize, val: u128) -> PyResult<()> {
        let len = self.data.len();
        self.data.get_mut(idx).map(|e| *e = val).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for U128Array of length {}",
                idx, len
            ))
        })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compute the sum of all elements.
    pub fn sum(&self) -> u128 {
        self.data.iter().sum()
    }

    /// Return all elements as a Python list of `int` values.
    pub fn to_list(&self) -> Vec<u128> {
        self.data.clone()
    }

    /// Fill the array from a Python list of integers.
    ///
    /// # Errors
    /// Returns a `ValueError` if `values` has a different length than `self`,
    /// or a `TypeError`/`OverflowError` if an element cannot be converted to
    /// `u128`.
    pub fn from_list(&mut self, values: Vec<u128>) -> PyResult<()> {
        if values.len() != self.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} elements, got {}",
                self.data.len(),
                values.len()
            )));
        }
        self.data.copy_from_slice(&values);
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Module registration
// ──────────────────────────────────────────────────────────────────────────────

/// Register extended-integer types and conversion functions into a PyO3 module.
///
/// Exposes [`I128Array`], [`U128Array`], [`to_i128`], [`from_i128`],
/// [`to_u128`], and [`from_u128`].
pub fn register_extended_int_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<I128Array>()?;
    m.add_class::<U128Array>()?;
    m.add_function(wrap_pyfunction!(to_i128, m)?)?;
    m.add_function(wrap_pyfunction!(from_i128, m)?)?;
    m.add_function(wrap_pyfunction!(to_u128, m)?)?;
    m.add_function(wrap_pyfunction!(from_u128, m)?)?;
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i128_array_basic_operations() {
        let mut arr = I128Array::new(3);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.sum(), 0);

        arr.set(0, 100).expect("set index 0");
        arr.set(1, 200).expect("set index 1");
        arr.set(2, 300).expect("set index 2");

        assert_eq!(arr.get(1).expect("get index 1"), 200);
        assert_eq!(arr.sum(), 600);
        assert_eq!(arr.to_list(), vec![100i128, 200, 300]);
    }

    #[test]
    fn i128_array_out_of_bounds() {
        let arr = I128Array::new(2);
        assert!(arr.get(99).is_err());
    }

    #[test]
    fn i128_to_from_roundtrip() {
        Python::attach(|py| {
            // i128::MAX round-trips through Python int.
            let py_int = from_i128(py, i128::MAX).expect("from_i128 failed");
            let val = to_i128(py_int.bind(py)).expect("to_i128 failed");
            assert_eq!(val, i128::MAX);
        });
    }

    #[test]
    fn u128_roundtrip() {
        Python::attach(|py| {
            let py_int = from_u128(py, u128::MAX).expect("from_u128 failed");
            let val = to_u128(py_int.bind(py)).expect("to_u128 failed");
            assert_eq!(val, u128::MAX);
        });
    }

    #[test]
    fn u128_array_basic_operations() {
        let mut arr = U128Array::new(2);
        arr.set(0, u128::MAX).expect("set 0");
        assert_eq!(arr.get(0).expect("get 0"), u128::MAX);
        assert!(arr.get(5).is_err());
    }
}
