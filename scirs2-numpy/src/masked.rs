//! Masked array support: a NumPy `numpy.ma`-compatible interface.
//!
//! A masked array pairs a data buffer with a boolean mask indicating which
//! elements are valid.  Elements where the mask is `true` are considered
//! invalid (masked out) and are excluded from aggregate operations such as
//! [`MaskedArray::mean`] and [`MaskedArray::sum`].

use pyo3::prelude::*;

/// A masked array of `f64` values.
///
/// Each element has an associated boolean mask value:
/// - `false` → element is valid and participates in computations.
/// - `true`  → element is masked (invalid) and is replaced by `fill_value`
///   when the filled view is requested.
#[pyclass(name = "MaskedArray")]
pub struct MaskedArray {
    /// Flat data buffer.
    data: Vec<f64>,
    /// Flat mask buffer; parallel to `data`.
    mask: Vec<bool>,
    /// Logical shape of the array.
    shape: Vec<usize>,
    /// Value used to fill masked positions in [`Self::filled`].
    fill_value: f64,
}

#[pymethods]
impl MaskedArray {
    /// Construct a new masked array.
    ///
    /// # Arguments
    /// * `data`       – flat element buffer; length must equal the product of `shape`.
    /// * `mask`       – optional flat mask; defaults to all-`false` (nothing masked).
    /// * `shape`      – logical shape; must have product equal to `data.len()`.
    /// * `fill_value` – value substituted for masked elements (default: `f64::NAN`).
    #[new]
    pub fn new(
        data: Vec<f64>,
        mask: Option<Vec<bool>>,
        shape: Vec<usize>,
        fill_value: Option<f64>,
    ) -> PyResult<Self> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "data length {} does not match shape product {}",
                data.len(),
                n
            )));
        }
        let mask = mask.unwrap_or_else(|| vec![false; n]);
        if mask.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mask length does not match shape product",
            ));
        }
        Ok(Self {
            data,
            mask,
            shape,
            fill_value: fill_value.unwrap_or(f64::NAN),
        })
    }

    /// Return the data with masked positions replaced by `fill_value`.
    pub fn filled(&self) -> Vec<f64> {
        self.data
            .iter()
            .zip(self.mask.iter())
            .map(|(&d, &m)| if m { self.fill_value } else { d })
            .collect()
    }

    /// Return the number of unmasked (valid) elements.
    pub fn count(&self) -> usize {
        self.mask.iter().filter(|&&m| !m).count()
    }

    /// Return the mean of unmasked elements, or `None` if all elements are masked.
    pub fn mean(&self) -> Option<f64> {
        let valid: Vec<f64> = self
            .data
            .iter()
            .zip(self.mask.iter())
            .filter(|(_, &m)| !m)
            .map(|(&d, _)| d)
            .collect();
        if valid.is_empty() {
            None
        } else {
            Some(valid.iter().sum::<f64>() / valid.len() as f64)
        }
    }

    /// Return the sum of unmasked elements.
    pub fn sum(&self) -> f64 {
        self.data
            .iter()
            .zip(self.mask.iter())
            .filter(|(_, &m)| !m)
            .map(|(&d, _)| d)
            .sum()
    }

    /// Return the logical shape.
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Return the raw data buffer.
    pub fn data(&self) -> Vec<f64> {
        self.data.clone()
    }

    /// Return the mask buffer.
    pub fn mask(&self) -> Vec<bool> {
        self.mask.clone()
    }

    /// Return the fill value used for masked positions.
    pub fn fill_value(&self) -> f64 {
        self.fill_value
    }

    /// Set the mask flag for a single flat element index.
    pub fn mask_element(&mut self, idx: usize, masked: bool) -> PyResult<()> {
        if idx >= self.mask.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of bounds",
            ));
        }
        self.mask[idx] = masked;
        Ok(())
    }

    /// Apply an element-wise unary operation to unmasked values.
    ///
    /// Masked positions receive `fill_value`.  Supported operations:
    /// - `"abs"` – absolute value
    /// - `"sqrt"` – square root
    /// - `"log"` – natural logarithm
    pub fn apply_unmasked(&self, op: &str) -> PyResult<Vec<f64>> {
        let fill = self.fill_value;
        match op {
            "abs" => Ok(self
                .data
                .iter()
                .zip(self.mask.iter())
                .map(|(&d, &m)| if m { fill } else { d.abs() })
                .collect()),
            "sqrt" => Ok(self
                .data
                .iter()
                .zip(self.mask.iter())
                .map(|(&d, &m)| if m { fill } else { d.sqrt() })
                .collect()),
            "log" => Ok(self
                .data
                .iter()
                .zip(self.mask.iter())
                .map(|(&d, &m)| if m { fill } else { d.ln() })
                .collect()),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown operation '{op}'; supported: abs, sqrt, log"
            ))),
        }
    }
}

/// Create a 1-D masked array from parallel data and mask vectors.
///
/// This mirrors `numpy.ma.array(data, mask=mask)`.
///
/// # Examples
///
/// ```
/// use scirs2_numpy::masked::masked_array;
///
/// // Third element (index 2) is masked.
/// let ma = masked_array(vec![1.0, 2.0, 3.0], vec![false, false, true]).unwrap();
/// assert_eq!(ma.count(), 2);
/// assert!((ma.sum() - 3.0).abs() < 1e-12); // 1.0 + 2.0
/// ```
#[pyfunction]
pub fn masked_array(data: Vec<f64>, mask: Vec<bool>) -> PyResult<MaskedArray> {
    let n = data.len();
    MaskedArray::new(data, Some(mask), vec![n], None)
}

/// Create a 1-D masked array with all elements below `threshold` masked.
///
/// Mirrors `numpy.ma.masked_less(data, threshold)`.
///
/// # Examples
///
/// ```
/// use scirs2_numpy::masked::masked_less;
///
/// // Values < 3.0 are masked: [1.0, 2.0] masked, [3.0, 4.0, 5.0] valid.
/// let ma = masked_less(vec![1.0, 2.0, 3.0, 4.0, 5.0], 3.0);
/// assert_eq!(ma.count(), 3);
/// assert!((ma.sum() - 12.0).abs() < 1e-12); // 3.0 + 4.0 + 5.0
/// ```
#[pyfunction]
pub fn masked_less(data: Vec<f64>, threshold: f64) -> MaskedArray {
    let n = data.len();
    let mask: Vec<bool> = data.iter().map(|&d| d < threshold).collect();
    MaskedArray {
        data,
        mask,
        shape: vec![n],
        fill_value: f64::NAN,
    }
}

/// Register masked-array classes and functions into a PyO3 module.
///
/// Call this from your `#[pymodule]` init function to expose `MaskedArray`,
/// `masked_array`, and `masked_less`.
pub fn register_masked_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MaskedArray>()?;
    m.add_function(wrap_pyfunction!(masked_array, m)?)?;
    m.add_function(wrap_pyfunction!(masked_less, m)?)?;
    Ok(())
}
