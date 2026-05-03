//! Array subclass support for duck-typed Python objects.
//!
//! Provides utilities to accept any Python object that looks like an array:
//! - NumPy ndarrays (via `__array__` protocol)
//! - pandas Series (has a `.values` attribute returning an ndarray)
//! - Any list-like object supporting `__len__` and `__getitem__`
//!
//! Also exposes [`SubclassArrayWrapper`], a `#[pyclass]` that wraps a flat
//! `f64` buffer with shape metadata and looks enough like a NumPy array for
//! downstream code that uses duck-typing.

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

// ──────────────────────────────────────────────────────────────────────────────
// Free-standing extraction helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Extract `f32` values from any Python array-like object.
///
/// Attempts the following strategies in order:
/// 1. `.values` attribute (pandas Series / masked array).
/// 2. `.__array__()` method (NumPy array protocol).
/// 3. Direct iteration via `__len__` + `__getitem__`.
///
/// # Errors
/// Returns a [`PyErr`] if none of the strategies succeeds or if an element
/// cannot be converted to `f32`.
#[pyfunction]
pub fn from_array_like_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Strategy 1: .values attribute (pandas Series / masked array).
    // Guard: only recurse if the attribute is NOT already a plain ndarray
    // (to avoid infinite recursion on ndarray objects that happen to lack .values).
    if let Ok(values) = obj.getattr("values") {
        // If `values` itself has a .values attribute we stop (avoid deep recursion).
        if values.getattr("values").is_err() {
            return from_array_like_f32(&values);
        }
    }

    // Strategy 2: __array__ protocol.
    if let Ok(arr) = obj.call_method0("__array__") {
        // Recursing here is safe: the resulting ndarray has no .values attribute.
        return from_array_like_f32(&arr);
    }

    // Strategy 3: direct iteration.
    let len = obj.len()?;
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let item = obj.get_item(i)?;
        let val: f32 = item.extract()?;
        result.push(val);
    }
    Ok(result)
}

/// Extract `f64` values from any Python array-like object.
///
/// Attempts the following strategies in order:
/// 1. `.values` attribute (pandas Series / masked array).
/// 2. `.__array__()` method (NumPy array protocol).
/// 3. Direct iteration via `__len__` + `__getitem__`.
///
/// # Errors
/// Returns a [`PyErr`] if none of the strategies succeeds or if an element
/// cannot be converted to `f64`.
#[pyfunction]
pub fn from_array_like_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    // Strategy 1: .values attribute — with depth guard.
    if let Ok(values) = obj.getattr("values") {
        if values.getattr("values").is_err() {
            return from_array_like_f64(&values);
        }
    }

    // Strategy 2: __array__ protocol.
    if let Ok(arr) = obj.call_method0("__array__") {
        return from_array_like_f64(&arr);
    }

    // Strategy 3: direct iteration.
    let len = obj.len()?;
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let item = obj.get_item(i)?;
        let val: f64 = item.extract()?;
        result.push(val);
    }
    Ok(result)
}

// ──────────────────────────────────────────────────────────────────────────────
// SubclassArrayWrapper
// ──────────────────────────────────────────────────────────────────────────────

/// A Python-visible wrapper around a flat `f64` data buffer with shape metadata.
///
/// Implements enough of the NumPy duck-typing surface to be accepted by code
/// that inspects `.shape`, `.dtype`, `.__len__`, and `.__getitem__`.
#[pyclass(name = "SubclassArrayWrapper")]
pub struct SubclassArrayWrapper {
    /// Flat data buffer in C (row-major) order.
    data: Vec<f64>,
    /// Logical shape of the array.
    shape: Vec<usize>,
    /// NumPy-compatible dtype string (e.g. `"float64"`).
    dtype: String,
}

#[pymethods]
impl SubclassArrayWrapper {
    /// Construct a new wrapper.
    ///
    /// # Arguments
    /// * `data`  – flat element buffer; length must equal the product of `shape`.
    /// * `shape` – logical shape; `[]` is interpreted as a 0-d scalar.
    /// * `dtype` – NumPy-compatible dtype string such as `"float64"`.
    #[new]
    #[pyo3(signature = (data, shape, dtype = "float64".to_string()))]
    pub fn new(data: Vec<f64>, shape: Vec<usize>, dtype: String) -> PyResult<Self> {
        let n: usize = shape.iter().product::<usize>().max(1);
        if shape.is_empty() {
            // 0-d: exactly one element.
            if data.len() != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "0-d SubclassArrayWrapper requires exactly one element",
                ));
            }
        } else if data.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "data length {} does not match shape product {}",
                data.len(),
                n
            )));
        }
        Ok(Self { data, shape, dtype })
    }

    /// Number of elements (flat).
    pub fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Return element at flat index `idx`.
    ///
    /// # Errors
    /// Returns `IndexError` if `idx` is out of bounds.
    pub fn __getitem__(&self, idx: usize) -> PyResult<f64> {
        self.data.get(idx).copied().ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of bounds for array of length {}",
                idx,
                self.data.len()
            ))
        })
    }

    /// Return `self` — mirrors pandas `.values` which returns the underlying array.
    ///
    /// This makes `SubclassArrayWrapper` itself accepted by [`from_array_like_f64`].
    pub fn values(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Logical shape tuple.
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// NumPy-compatible dtype string.
    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    /// Flat copy of the data as a Python list.
    pub fn to_list(&self) -> Vec<f64> {
        self.data.clone()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Module registration
// ──────────────────────────────────────────────────────────────────────────────

/// Register array-subclass helpers and [`SubclassArrayWrapper`] into a PyO3 module.
pub fn register_array_subclass_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_array_like_f32, m)?)?;
    m.add_function(wrap_pyfunction!(from_array_like_f64, m)?)?;
    m.add_class::<SubclassArrayWrapper>()?;
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_like_extracts_from_list() {
        Python::attach(|py| {
            let list = py
                .eval(pyo3::ffi::c_str!("[1.0, 2.0, 3.0]"), None, None)
                .expect("eval failed");
            let result = from_array_like_f64(&list).expect("extraction failed");
            assert_eq!(result, vec![1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn array_like_wrapper_len_correct() {
        let wrapper =
            SubclassArrayWrapper::new(vec![1.0, 2.0, 3.0], vec![3], "float64".to_string())
                .expect("construction failed");
        assert_eq!(wrapper.__len__(), 3);
    }

    #[test]
    fn subclass_wrapper_getitem_correct() {
        let wrapper =
            SubclassArrayWrapper::new(vec![10.0, 20.0, 30.0], vec![3], "float64".to_string())
                .expect("construction failed");
        assert!((wrapper.__getitem__(1).expect("index valid") - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn subclass_wrapper_getitem_oob() {
        let wrapper = SubclassArrayWrapper::new(vec![1.0], vec![1], "float64".to_string())
            .expect("construction failed");
        assert!(wrapper.__getitem__(99).is_err());
    }

    #[test]
    fn subclass_wrapper_shape_and_dtype() {
        let wrapper = SubclassArrayWrapper::new(vec![1.0, 2.0], vec![2], "float64".to_string())
            .expect("construction failed");
        assert_eq!(wrapper.shape(), vec![2usize]);
        assert_eq!(wrapper.dtype(), "float64");
    }
}
