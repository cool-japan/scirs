//! Implementation of masked arrays for handling missing or invalid data
//!
//! Masked arrays are useful in scientific computing when working with:
//! - Data that contains missing values
//! - Operations that produce invalid results (NaN, Inf)
//! - Operations that should be applied only to a subset of data
//! - Statistical computations that should ignore certain values
//!
//! The implementation is inspired by ``NumPy``'s `MaskedArray` and provides similar functionality
//! in a Rust-native way.

use ndarray::{Array, ArrayBase, Data, Dimension, Ix1};
use num_traits::{Float, Zero};
use std::cmp::PartialEq;
use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// Error type for array operations
#[derive(Debug, Clone)]
pub enum ArrayError {
    /// Shape mismatch error
    ShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
        msg: String,
    },
    /// Value error
    ValueError(String),
}

impl std::fmt::Display for ArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch {
                expected,
                found,
                msg,
            } => {
                write!(
                    f,
                    "Shape mismatch: expected {expected:?}, found {found:?}: {msg}"
                )
            }
            Self::ValueError(msg) => write!(f, "Value error: {msg}"),
        }
    }
}

impl std::error::Error for ArrayError {}

/// Represents an array with a mask to identify invalid or missing values
#[derive(Clone)]
pub struct MaskedArray<A, S, D>
where
    A: Clone,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    /// The underlying data array
    pub data: ArrayBase<S, D>,

    /// The mask array: true = masked (invalid), false = valid
    pub mask: Array<bool, D>,

    /// The fill value used when creating a filled array
    pub fill_value: A,
}

/// Represents a "no mask" indicator, which is equivalent to an array of all false
pub struct NoMask;

/// The global "no mask" constant
pub const NOMASK: NoMask = NoMask;

impl<A, S, D> MaskedArray<A, S, D>
where
    A: Clone + PartialEq,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    /// Create a new `MaskedArray` from data and mask
    ///
    /// # Errors
    /// Returns `ArrayError::ShapeMismatch` if the mask shape doesn't match the data shape.
    pub fn new(
        data: ArrayBase<S, D>,
        mask: Option<Array<bool, D>>,
        fill_value: Option<A>,
    ) -> Result<Self, ArrayError> {
        let mask = match mask {
            Some(m) => {
                // Validate mask shape matches data shape
                if m.shape() != data.shape() {
                    return Err(ArrayError::ShapeMismatch {
                        expected: data.shape().to_vec(),
                        found: m.shape().to_vec(),
                        msg: "Mask shape must match data shape".to_string(),
                    });
                }
                m
            }
            None => Array::<bool, D>::from_elem(data.raw_dim(), false),
        };

        // Use provided fill value or create a default
        let fill_value = fill_value.map_or_else(|| default_fill_value(&data), |v| v);

        Ok(Self {
            data,
            mask,
            fill_value,
        })
    }

    /// Get a view of the data with masked values replaced by `fill_value`
    pub fn filled(&self, fill_value: Option<A>) -> Array<A, D>
    where
        <D as Dimension>::Pattern: ndarray::NdIndex<D>,
    {
        let fill = fill_value.map_or_else(|| self.fill_value.clone(), |v| v);

        // Create new array with same shape as data
        let mut result = Array::from_elem(self.data.raw_dim(), fill);

        // Copy unmasked values from original data
        for (i, val) in self.data.iter().enumerate() {
            if !*self.mask.iter().nth(i).unwrap_or(&true) {
                // Only copy if not masked
                if let Some(v) = result.iter_mut().nth(i) {
                    *v = val.clone();
                }
            }
        }

        result
    }

    /// Returns true if the array has at least one masked element
    pub fn has_masked(&self) -> bool {
        self.mask.iter().any(|&x| x)
    }

    /// Returns the count of non-masked elements
    pub fn count(&self) -> usize {
        self.mask.iter().filter(|&&x| !x).count()
    }

    /// Get a copy of the current mask
    pub fn get_mask(&self) -> Array<bool, D> {
        self.mask.clone()
    }

    /// Set a new mask for the array
    ///
    /// # Errors
    /// Returns `ArrayError::ShapeMismatch` if the mask shape doesn't match the data shape.
    pub fn set_mask(&mut self, mask: Array<bool, D>) -> Result<(), ArrayError> {
        // Validate mask shape
        if mask.shape() != self.data.shape() {
            return Err(ArrayError::ShapeMismatch {
                expected: self.data.shape().to_vec(),
                found: mask.shape().to_vec(),
                msg: "Mask shape must match data shape".to_string(),
            });
        }

        self.mask = mask;
        Ok(())
    }

    /// Set the fill value for the array
    pub fn set_fill_value(&mut self, fill_value: A) {
        self.fill_value = fill_value;
    }

    /// Returns a new array containing only unmasked values
    pub fn compressed(&self) -> Array<A, Ix1> {
        // Count non-masked elements
        let count = self.count();

        // Create output array
        let mut result = Vec::with_capacity(count);

        // Fill output array with non-masked elements
        for (i, val) in self.data.iter().enumerate() {
            if !self.mask.iter().nth(i).unwrap_or(&true) {
                result.push(val.clone());
            }
        }

        // Convert to ndarray
        Array::from_vec(result)
    }

    /// Returns the shape of the array
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns the number of dimensions of the array
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Returns the number of elements in the array
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns a tuple of (data, mask)
    pub const fn data_and_mask(&self) -> (&ArrayBase<S, D>, &Array<bool, D>) {
        (&self.data, &self.mask)
    }

    /// Creates a new masked array with the given mask operation applied
    #[must_use]
    pub fn mask_op<F>(&self, op: F) -> Self
    where
        F: Fn(&Array<bool, D>) -> Array<bool, D>,
    {
        let new_mask = op(&self.mask);

        Self {
            data: self.data.clone(),
            mask: new_mask,
            fill_value: self.fill_value.clone(),
        }
    }

    /// Creates a new masked array with a hardened mask (copy)
    #[must_use]
    pub fn harden_mask(&self) -> Self {
        // Create a copy with the same mask
        Self {
            data: self.data.clone(),
            mask: self.mask.clone(),
            fill_value: self.fill_value.clone(),
        }
    }

    /// Create a new masked array with a softened mask (copy)
    #[must_use]
    pub fn soften_mask(&self) -> Self {
        // Create a copy with the same mask
        Self {
            data: self.data.clone(),
            mask: self.mask.clone(),
            fill_value: self.fill_value.clone(),
        }
    }

    /// Create a new masked array where the result of applying the function to each element is masked
    #[must_use]
    pub fn mask_where<F>(&self, condition: F) -> Self
    where
        F: Fn(&A) -> bool,
    {
        // Apply condition to each element of data
        let new_mask = self.data.mapv(|ref x| condition(x));

        // Combine with existing mask
        let combined_mask = &self.mask | &new_mask;

        Self {
            data: self.data.clone(),
            mask: combined_mask,
            fill_value: self.fill_value.clone(),
        }
    }

    /// Create a logical OR of the mask with another mask
    ///
    /// # Errors
    /// Returns `ArrayError::ShapeMismatch` if the mask shapes don't match.
    pub fn mask_or(&self, other_mask: &Array<bool, D>) -> Result<Self, ArrayError> {
        // Check that shapes match
        if self.mask.shape() != other_mask.shape() {
            return Err(ArrayError::ShapeMismatch {
                expected: self.mask.shape().to_vec(),
                found: other_mask.shape().to_vec(),
                msg: "Mask shapes must match for mask_or operation".to_string(),
            });
        }

        // Combine masks
        let combined_mask = &self.mask | other_mask;

        Ok(Self {
            data: self.data.clone(),
            mask: combined_mask,
            fill_value: self.fill_value.clone(),
        })
    }

    /// Create a logical AND of the mask with another mask
    ///
    /// # Errors
    /// Returns `ArrayError::ShapeMismatch` if the mask shapes don't match.
    pub fn mask_and(&self, other_mask: &Array<bool, D>) -> Result<Self, ArrayError> {
        // Check that shapes match
        if self.mask.shape() != other_mask.shape() {
            return Err(ArrayError::ShapeMismatch {
                expected: self.mask.shape().to_vec(),
                found: other_mask.shape().to_vec(),
                msg: "Mask shapes must match for mask_and operation".to_string(),
            });
        }

        // Combine masks
        let combined_mask = &self.mask & other_mask;

        Ok(Self {
            data: self.data.clone(),
            mask: combined_mask,
            fill_value: self.fill_value.clone(),
        })
    }

    /// Reshape the masked array
    ///
    /// # Errors
    /// Returns `ArrayError::ValueError` if the reshape operation fails.
    pub fn reshape<E>(
        &self,
        shape: E,
    ) -> Result<MaskedArray<A, ndarray::OwnedRepr<A>, E>, ArrayError>
    where
        E: Dimension,
        D: Dimension,
    {
        // Try to reshape the data and mask
        let reshaped_data = match self.data.clone().into_shape_with_order(shape.clone()) {
            Ok(d) => d,
            Err(e) => {
                return Err(ArrayError::ValueError(format!(
                    "Failed to reshape data: {e}"
                )))
            }
        };

        let reshaped_mask = match self.mask.clone().into_shape_with_order(shape) {
            Ok(m) => m,
            Err(e) => {
                return Err(ArrayError::ValueError(format!(
                    "Failed to reshape mask: {e}"
                )))
            }
        };

        Ok(MaskedArray {
            data: reshaped_data.into_owned(),
            mask: reshaped_mask,
            fill_value: self.fill_value.clone(),
        })
    }

    /// Convert to a different type
    ///
    /// # Errors
    /// Currently this method doesn't return errors, but the signature is kept for future compatibility.
    pub fn astype<B>(&self) -> Result<MaskedArray<B, ndarray::OwnedRepr<B>, D>, ArrayError>
    where
        A: Into<B> + Clone,
        B: Clone + PartialEq + 'static,
    {
        // Convert each element
        let converted_data = self.data.mapv(std::convert::Into::into);

        Ok(MaskedArray {
            data: converted_data,
            mask: self.mask.clone(),
            fill_value: self.fill_value.clone().into(),
        })
    }
}

/// Implementation of statistical methods
impl<A, S, D> MaskedArray<A, S, D>
where
    A: Clone + PartialEq + num_traits::NumAssign + num_traits::Zero + num_traits::One + PartialOrd,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    /// Compute the sum of all unmasked elements
    pub fn sum(&self) -> A {
        let mut sum = A::zero();

        for (i, val) in self.data.iter().enumerate() {
            if !*self.mask.iter().nth(i).unwrap_or(&true) {
                sum += val.clone();
            }
        }

        sum
    }

    /// Compute the product of all unmasked elements
    pub fn product(&self) -> A {
        let mut product = A::one();

        for (i, val) in self.data.iter().enumerate() {
            if !*self.mask.iter().nth(i).unwrap_or(&true) {
                product *= val.clone();
            }
        }

        product
    }

    /// Find the minimum value among unmasked elements
    pub fn min(&self) -> Option<A> {
        let mut min_val = None;

        for (i, val) in self.data.iter().enumerate() {
            if !*self.mask.iter().nth(i).unwrap_or(&true) {
                if let Some(ref current_min) = min_val {
                    if val < current_min {
                        min_val = Some(val.clone());
                    }
                } else {
                    min_val = Some(val.clone());
                }
            }
        }

        min_val
    }

    /// Find the maximum value among unmasked elements
    pub fn max(&self) -> Option<A> {
        let mut max_val = None;

        for (i, val) in self.data.iter().enumerate() {
            if !*self.mask.iter().nth(i).unwrap_or(&true) {
                if let Some(ref current_max) = max_val {
                    if val > current_max {
                        max_val = Some(val.clone());
                    }
                } else {
                    max_val = Some(val.clone());
                }
            }
        }

        max_val
    }

    /// Find the index of the minimum value among unmasked elements
    pub fn argmin(&self) -> Option<usize> {
        let mut min_idx = None;
        let mut min_val = None;

        for (i, val) in self.data.iter().enumerate() {
            if !*self.mask.iter().nth(i).unwrap_or(&true) {
                if let Some(ref current_min) = min_val {
                    if val < current_min {
                        min_val = Some(val.clone());
                        min_idx = Some(i);
                    }
                } else {
                    min_val = Some(val.clone());
                    min_idx = Some(i);
                }
            }
        }

        min_idx
    }

    /// Find the index of the maximum value among unmasked elements
    pub fn argmax(&self) -> Option<usize> {
        let mut max_idx = None;
        let mut max_val = None;

        for (i, val) in self.data.iter().enumerate() {
            if !*self.mask.iter().nth(i).unwrap_or(&true) {
                if let Some(ref current_max) = max_val {
                    if val > current_max {
                        max_val = Some(val.clone());
                        max_idx = Some(i);
                    }
                } else {
                    max_val = Some(val.clone());
                    max_idx = Some(i);
                }
            }
        }

        max_idx
    }
}

/// Implementation of statistical methods for floating point types
impl<A, S, D> MaskedArray<A, S, D>
where
    A: Clone + PartialEq + num_traits::Float + std::iter::Sum<A>,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    /// Compute the mean of all unmasked elements
    ///
    /// # Panics
    /// Panics if the count cannot be converted to type A.
    pub fn mean(&self) -> Option<A> {
        let count = self.count();

        if count == 0 {
            return None;
        }

        let sum: A = self
            .data
            .iter()
            .enumerate()
            .filter(|(i, _)| !*self.mask.iter().nth(*i).unwrap_or(&true))
            .map(|(_, val)| *val)
            .sum();

        Some(sum / A::from(count).unwrap())
    }

    /// Compute the variance of all unmasked elements
    ///
    /// # Panics
    /// Panics if the count cannot be converted to type A.
    pub fn var(&self, ddof: usize) -> Option<A> {
        let count = self.count();

        if count <= ddof {
            return None;
        }

        // Calculate mean
        let mean = self.mean()?;

        // Calculate sum of squared differences
        let sum_sq_diff: A = self
            .data
            .iter()
            .enumerate()
            .filter(|(i, _)| !*self.mask.iter().nth(*i).unwrap_or(&true))
            .map(|(_, val)| (*val - mean) * (*val - mean))
            .sum();

        // Apply degrees of freedom correction
        Some(sum_sq_diff / A::from(count - ddof).unwrap())
    }

    /// Compute the standard deviation of all unmasked elements
    pub fn std(&self, ddof: usize) -> Option<A> {
        self.var(ddof).map(num_traits::Float::sqrt)
    }

    /// Check if all unmasked elements are finite
    pub fn all_finite(&self) -> bool {
        self.data
            .iter()
            .enumerate()
            .filter(|(i, _)| !*self.mask.iter().nth(*i).unwrap_or(&true))
            .all(|(_, val)| val.is_finite())
    }
}

/// Helper function to create a default fill value for a given type
fn default_fill_value<A, S, D>(data: &ArrayBase<S, D>) -> A
where
    A: Clone,
    S: Data<Elem = A>,
    D: Dimension,
{
    // In a real implementation, this would use type traits to determine
    // appropriate default values based on the type (like `NumPy` does)
    data.iter().next().map_or_else(
        || {
            // This is a placeholder - in reality you'd need to handle this by type
            panic!("Cannot determine default fill value for empty array");
        },
        std::clone::Clone::clone,
    )
}

/// Function to check if a value is masked
pub const fn is_masked<A>(_value: &A) -> bool
where
    A: PartialEq,
{
    // In `NumPy` this would check against the masked singleton
    // Here we just return false as a placeholder
    false
}

/// Create a masked array with elements equal to a given value masked
pub fn masked_equal<A, S, D>(data: ArrayBase<S, D>, value: A) -> MaskedArray<A, S, D>
where
    A: Clone + PartialEq,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    // Create a mask indicating where elements equal the value
    let mask = data.mapv(|x| x == value);

    MaskedArray {
        data,
        mask,
        fill_value: value,
    }
}

/// Create a masked array with NaN and infinite values masked
pub fn masked_invalid<A, S, D>(data: ArrayBase<S, D>) -> MaskedArray<A, S, D>
where
    A: Clone + PartialEq + Float,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    // Create a mask indicating where elements are NaN or infinite
    let mask = data.mapv(|val| val.is_nan() || val.is_infinite());

    let fill_value = A::nan();

    MaskedArray {
        data,
        mask,
        fill_value,
    }
}

/// Create a masked array
///
/// # Errors
/// Returns `ArrayError::ShapeMismatch` if the mask shape doesn't match the data shape.
pub fn mask_array<A, S, D>(
    data: ArrayBase<S, D>,
    mask: Option<Array<bool, D>>,
    fill_value: Option<A>,
) -> Result<MaskedArray<A, S, D>, ArrayError>
where
    A: Clone + PartialEq,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    MaskedArray::new(data, mask, fill_value)
}

/// Create a masked array with values outside a range masked
pub fn masked_outside<A, S, D>(
    data: ArrayBase<S, D>,
    min_val: &A,
    max_val: &A,
) -> MaskedArray<A, S, D>
where
    A: Clone + PartialEq + PartialOrd,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    // Create a mask indicating where elements are outside the range
    let mask = data.mapv(|x| x < *min_val || x > *max_val);

    // Choose a fill value (using min_val as default)
    let fill_value = min_val.clone();

    MaskedArray {
        data,
        mask,
        fill_value,
    }
}

/// Create a masked array with values inside a range masked
pub fn masked_inside<A, S, D>(
    data: ArrayBase<S, D>,
    min_val: &A,
    max_val: &A,
) -> MaskedArray<A, S, D>
where
    A: Clone + PartialEq + PartialOrd,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    // Create a mask indicating where elements are inside the range
    let mask = data.mapv(|x| x >= *min_val && x <= *max_val);

    // Choose a fill value (using min_val as default)
    let fill_value = min_val.clone();

    MaskedArray {
        data,
        mask,
        fill_value,
    }
}

/// Create a masked array with values greater than a given value masked
pub fn masked_greater<A, S, D>(data: ArrayBase<S, D>, value: &A) -> MaskedArray<A, S, D>
where
    A: Clone + PartialEq + PartialOrd,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    // Create a mask indicating where elements are greater than the value
    let mask = data.mapv(|x| x > *value);

    // Use the specified value as fill_value
    let fill_value = value.clone();

    MaskedArray {
        data,
        mask,
        fill_value,
    }
}

/// Create a masked array with values less than a given value masked
pub fn masked_less<A, S, D>(data: ArrayBase<S, D>, value: &A) -> MaskedArray<A, S, D>
where
    A: Clone + PartialEq + PartialOrd,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    // Create a mask indicating where elements are less than the value
    let mask = data.mapv(|x| x < *value);

    // Use the specified value as fill_value
    let fill_value = value.clone();

    MaskedArray {
        data,
        mask,
        fill_value,
    }
}

/// Create a masked array with values where a condition is true
pub fn masked_where<A, S, D, F>(
    condition: F,
    data: ArrayBase<S, D>,
    fill_value: Option<A>,
) -> MaskedArray<A, S, D>
where
    A: Clone + PartialEq,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
    F: Fn(&A) -> bool,
{
    // Create a mask by applying the condition function to each element
    let mask = data.mapv(|ref x| condition(x));

    // Use provided fill value or create a default
    let fill_value = fill_value.map_or_else(|| default_fill_value(&data), |v| v);

    MaskedArray {
        data,
        mask,
        fill_value,
    }
}

/// Implementation of Display for `MaskedArray`
impl<A, S, D> fmt::Display for MaskedArray<A, S, D>
where
    A: Clone + PartialEq + fmt::Display,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MaskedArray(")?;

        writeln!(f, "  data=[")?;
        for (i, elem) in self.data.iter().enumerate() {
            if i > 0 && i % 10 == 0 {
                writeln!(f)?;
            }
            if *self.mask.iter().nth(i).unwrap_or(&false) {
                write!(f, " --,")?;
            } else {
                write!(f, " {elem},")?;
            }
        }
        writeln!(f, "\n  ],")?;

        writeln!(f, "  mask=[")?;
        for (i, &elem) in self.mask.iter().enumerate() {
            if i > 0 && i % 10 == 0 {
                writeln!(f)?;
            }
            write!(f, " {elem},")?;
        }
        writeln!(f, "\n  ],")?;

        writeln!(f, "  fill_value={}", self.fill_value)?;
        write!(f, ")")
    }
}

/// Implementation of Debug for `MaskedArray`
impl<A, S, D> fmt::Debug for MaskedArray<A, S, D>
where
    A: Clone + PartialEq + fmt::Debug,
    S: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaskedArray")
            .field("data", &self.data)
            .field("mask", &self.mask)
            .field("fill_value", &self.fill_value)
            .finish()
    }
}

// Arithmetic operations for MaskedArray
// These could be expanded to handle more operations and types

impl<A, S1, S2, D> Add<&MaskedArray<A, S2, D>> for &MaskedArray<A, S1, D>
where
    A: Clone + Add<Output = A> + PartialEq,
    S1: Data<Elem = A> + Clone + ndarray::RawDataClone,
    S2: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    type Output = MaskedArray<A, ndarray::OwnedRepr<A>, D>;

    fn add(self, rhs: &MaskedArray<A, S2, D>) -> Self::Output {
        // Create combined mask: true if either input is masked
        let combined_mask = &self.mask | &rhs.mask;

        // Create output data by adding input data
        let data = &self.data + &rhs.data;

        MaskedArray {
            data,
            mask: combined_mask,
            fill_value: self.fill_value.clone(),
        }
    }
}

impl<A, S1, S2, D> Sub<&MaskedArray<A, S2, D>> for &MaskedArray<A, S1, D>
where
    A: Clone + Sub<Output = A> + PartialEq,
    S1: Data<Elem = A> + Clone + ndarray::RawDataClone,
    S2: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    type Output = MaskedArray<A, ndarray::OwnedRepr<A>, D>;

    fn sub(self, rhs: &MaskedArray<A, S2, D>) -> Self::Output {
        // Create combined mask: true if either input is masked
        let combined_mask = &self.mask | &rhs.mask;

        // Create output data by subtracting input data
        let data = &self.data - &rhs.data;

        MaskedArray {
            data,
            mask: combined_mask,
            fill_value: self.fill_value.clone(),
        }
    }
}

impl<A, S1, S2, D> Mul<&MaskedArray<A, S2, D>> for &MaskedArray<A, S1, D>
where
    A: Clone + Mul<Output = A> + PartialEq,
    S1: Data<Elem = A> + Clone + ndarray::RawDataClone,
    S2: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    type Output = MaskedArray<A, ndarray::OwnedRepr<A>, D>;

    fn mul(self, rhs: &MaskedArray<A, S2, D>) -> Self::Output {
        // Create combined mask: true if either input is masked
        let combined_mask = &self.mask | &rhs.mask;

        // Create output data by multiplying input data
        let data = &self.data * &rhs.data;

        MaskedArray {
            data,
            mask: combined_mask,
            fill_value: self.fill_value.clone(),
        }
    }
}

impl<A, S1, S2, D> Div<&MaskedArray<A, S2, D>> for &MaskedArray<A, S1, D>
where
    A: Clone + Div<Output = A> + PartialEq + Zero,
    S1: Data<Elem = A> + Clone + ndarray::RawDataClone,
    S2: Data<Elem = A> + Clone + ndarray::RawDataClone,
    D: Dimension,
{
    type Output = MaskedArray<A, ndarray::OwnedRepr<A>, D>;

    fn div(self, rhs: &MaskedArray<A, S2, D>) -> Self::Output {
        // Create combined mask: true if either input is masked or rhs is zero
        let mut combined_mask = &self.mask | &rhs.mask;

        // Also mask division by zero
        let zero = A::zero();
        let additional_mask = rhs.data.mapv(|x| x == zero);

        // Update combined mask to also mask division by zero
        combined_mask = combined_mask | additional_mask;

        // Create output data by dividing input data
        let data = &self.data / &rhs.data;

        MaskedArray {
            data,
            mask: combined_mask,
            fill_value: self.fill_value.clone(),
        }
    }
}

// Add more arithmetic operations as needed

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_masked_array_creation() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = array![false, true, false, true, false];

        let ma = MaskedArray::new(data.clone(), Some(mask.clone()), None).unwrap();

        assert_eq!(ma.data, data);
        assert_eq!(ma.mask, mask);
        assert_eq!(ma.count(), 3);
        assert!(ma.has_masked());
    }

    #[test]
    fn test_masked_array_filled() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mask = array![false, true, false, true, false];

        let ma = MaskedArray::new(data, Some(mask), Some(999.0)).unwrap();

        let filled = ma.filled(None);
        assert_eq!(filled, array![1.0, 999.0, 3.0, 999.0, 5.0]);

        let filled_custom = ma.filled(Some(-1.0));
        assert_eq!(filled_custom, array![1.0, -1.0, 3.0, -1.0, 5.0]);
    }

    #[test]
    fn test_masked_equal() {
        let data = array![1.0, 2.0, 3.0, 2.0, 5.0];

        let ma = masked_equal(data, 2.0);

        assert_eq!(ma.mask, array![false, true, false, true, false]);
        assert_eq!(ma.count(), 3);
    }

    #[test]
    fn test_masked_invalid() {
        let data = array![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];

        let ma = masked_invalid(data);

        // Cannot directly compare masks with NaN values using assert_eq
        // So we check each element individually
        assert!(!ma.mask[0]); // 1.0 is valid
        assert!(ma.mask[1]); // NaN is invalid
        assert!(!ma.mask[2]); // 3.0 is valid
        assert!(ma.mask[3]); // INFINITY is invalid
        assert!(!ma.mask[4]); // 5.0 is valid

        assert_eq!(ma.count(), 3);
    }

    #[test]
    fn test_masked_array_arithmetic() {
        let a = MaskedArray::new(
            array![1.0, 2.0, 3.0, 4.0, 5.0],
            Some(array![false, true, false, false, false]),
            Some(0.0),
        )
        .unwrap();

        let b = MaskedArray::new(
            array![5.0, 4.0, 3.0, 2.0, 1.0],
            Some(array![false, false, false, true, false]),
            Some(0.0),
        )
        .unwrap();

        // Addition
        let c = &a + &b;
        assert_eq!(c.data, array![6.0, 6.0, 6.0, 6.0, 6.0]);
        assert_eq!(c.mask, array![false, true, false, true, false]);

        // Subtraction
        let d = &a - &b;
        assert_eq!(d.data, array![-4.0, -2.0, 0.0, 2.0, 4.0]);
        assert_eq!(d.mask, array![false, true, false, true, false]);

        // Multiplication
        let e = &a * &b;
        assert_eq!(e.data, array![5.0, 8.0, 9.0, 8.0, 5.0]);
        assert_eq!(e.mask, array![false, true, false, true, false]);

        // Division
        let f = &a / &b;
        assert_eq!(f.data, array![0.2, 0.5, 1.0, 2.0, 5.0]);
        assert_eq!(f.mask, array![false, true, false, true, false]);

        // Division by zero
        let g = MaskedArray::new(
            array![1.0, 2.0, 3.0],
            Some(array![false, false, false]),
            Some(0.0),
        )
        .unwrap();

        let h = MaskedArray::new(
            array![1.0, 0.0, 3.0],
            Some(array![false, false, false]),
            Some(0.0),
        )
        .unwrap();

        let i = &g / &h;
        assert_eq!(i.mask, array![false, true, false]);
    }

    #[test]
    fn test_compressed() {
        let ma = MaskedArray::new(
            array![1.0, 2.0, 3.0, 4.0, 5.0],
            Some(array![false, true, false, true, false]),
            Some(0.0),
        )
        .unwrap();

        let compressed = ma.compressed();
        assert_eq!(compressed, array![1.0, 3.0, 5.0]);
    }
}
