//! Moment calculation functions for arrays

use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::error::{NdimageError, Result};

/// Find the center of mass of an array
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * `Result<Vec<T>>` - Center of mass (centroid)
pub fn center_of_mass<T, D>(input: &Array<T, D>) -> Result<Vec<T>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if input.is_empty() {
        return Err(NdimageError::InvalidInput("Input array is empty".into()));
    }

    let ndim = input.ndim();
    let shape = input.shape();

    // Calculate total mass (sum of all values)
    let total_mass = input.sum();

    if total_mass == T::zero() {
        // If total mass is zero, return center of array
        let center: Vec<T> = shape
            .iter()
            .map(|&dim| T::from_usize(dim).unwrap() / (T::one() + T::one()))
            .collect();
        return Ok(center);
    }

    // Calculate center of mass for each dimension
    let mut center_of_mass = vec![T::zero(); ndim];

    // Convert to dynamic array for easier indexing
    let input_dyn = input.clone().into_dyn();

    // Iterate through all elements in the array
    for (idx, &value) in input_dyn.indexed_iter() {
        if value != T::zero() {
            // Add weighted coordinates
            for (dim, &coord) in idx.as_array_view().iter().enumerate() {
                center_of_mass[dim] += T::from_usize(coord).unwrap() * value;
            }
        }
    }

    // Normalize by total mass
    for coord in center_of_mass.iter_mut() {
        *coord /= total_mass;
    }

    Ok(center_of_mass)
}

/// Find the moment of inertia tensor of an array
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// * `Result<Array<T, ndarray::Ix2>>` - Moment of inertia tensor
pub fn moments_inertia_tensor<T, D>(input: &Array<T, D>) -> Result<Array<T, ndarray::Ix2>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Placeholder implementation
    let dim = input.ndim();
    Ok(Array::<T, _>::zeros((dim, dim)))
}

/// Calculate image moments
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
///
/// # Returns
///
/// * `Result<Array<T, ndarray::Ix1>>` - Array of moments
pub fn moments<T, D>(input: &Array<T, D>, order: usize) -> Result<Array<T, ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Placeholder implementation
    let num_moments = (order + 1).pow(input.ndim() as u32);
    Ok(Array::<T, _>::zeros(num_moments))
}

/// Calculate central moments (moments around centroid)
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
/// * `center` - Center coordinates (if None, uses center of mass)
///
/// # Returns
///
/// * `Result<Array<T, ndarray::Ix1>>` - Array of central moments
pub fn central_moments<T, D>(
    input: &Array<T, D>,
    order: usize,
    center: Option<&[T]>,
) -> Result<Array<T, ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if let Some(c) = center {
        if c.len() != input.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Center must have same length as input dimensions (got {} expected {})",
                c.len(),
                input.ndim()
            )));
        }
    }

    // Placeholder implementation
    let num_moments = (order + 1).pow(input.ndim() as u32);
    Ok(Array::<T, _>::zeros(num_moments))
}

/// Calculate normalized moments
///
/// # Arguments
///
/// * `input` - Input array
/// * `order` - Maximum order of moments to calculate
///
/// # Returns
///
/// * `Result<Array<T, ndarray::Ix1>>` - Array of normalized moments
pub fn normalized_moments<T, D>(input: &Array<T, D>, order: usize) -> Result<Array<T, ndarray::Ix1>>
where
    T: Float + FromPrimitive + Debug + NumAssign,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    // Placeholder implementation
    let num_moments = (order + 1).pow(input.ndim() as u32);
    Ok(Array::<T, _>::zeros(num_moments))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_center_of_mass() {
        let input: Array2<f64> = Array2::eye(3);
        let com = center_of_mass(&input).unwrap();
        assert_eq!(com.len(), input.ndim());
    }

    #[test]
    fn test_moments_inertia_tensor() {
        let input: Array2<f64> = Array2::eye(3);
        let tensor = moments_inertia_tensor(&input).unwrap();
        assert_eq!(tensor.shape(), &[input.ndim(), input.ndim()]);
    }

    #[test]
    fn test_moments() {
        let input: Array2<f64> = Array2::eye(3);
        let order = 2;
        let mom = moments(&input, order).unwrap();
        assert!(!mom.is_empty());
    }
}
