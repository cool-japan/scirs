//! Utility functions for morphological operations

use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

use super::MorphBorderMode;
use crate::error::{NdimageError, Result};

/// Apply padding to an array based on the specified border mode for morphological operations
///
/// # Arguments
///
/// * `input` - Input array to pad
/// * `pad_width` - Width of padding in each dimension (before, after)
/// * `mode` - Border handling mode
/// * `constant_value` - Value to use for constant mode
///
/// # Returns
///
/// * `Result<Array<T, D>>` - Padded array
pub fn pad_array<T, D>(
    input: &Array<T, D>,
    pad_width: &[(usize, usize)],
    _mode: &MorphBorderMode,
    _constant_value: T,
) -> Result<Array<T, D>>
where
    T: Float + FromPrimitive + Debug + Clone,
    D: Dimension,
{
    // Validate inputs
    if input.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Input array cannot be 0-dimensional".into(),
        ));
    }

    if pad_width.len() != input.ndim() {
        return Err(NdimageError::DimensionError(format!(
            "Pad width must have same length as input dimensions (got {} expected {})",
            pad_width.len(),
            input.ndim()
        )));
    }

    // No padding needed - return copy of input
    if pad_width.iter().all(|&(a, b)| a == 0 && b == 0) {
        return Ok(input.to_owned());
    }

    // Placeholder implementation returning a copy of the input
    // This will be properly implemented with padding logic
    Ok(input.to_owned())
}

/// Check if an array is a valid structuring element
///
/// # Arguments
///
/// * `structure` - Structuring element to check
///
/// # Returns
///
/// * `Result<()>` - Ok if valid, Error otherwise
pub fn validate_structure<D>(structure: &Array<bool, D>) -> Result<()>
where
    D: Dimension,
{
    // Validate inputs
    if structure.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Structure cannot be 0-dimensional".into(),
        ));
    }

    // Require at least one True value
    if !structure.iter().any(|&x| x) {
        return Err(NdimageError::InvalidInput(
            "Structure must have at least one True value".into(),
        ));
    }

    // For proper validation we would also check:
    // - All dimensions are odd (so there's a clear center)
    // - The structure has a center element that is True

    Ok(())
}

/// Get center indices of a structuring element
///
/// # Arguments
///
/// * `structure` - Structuring element
/// * `origin` - Origin of the structuring element (if None, uses the center)
///
/// # Returns
///
/// * `Result<Vec<isize>>` - Center indices
pub fn get_structure_center<D>(
    structure: &Array<bool, D>,
    origin: Option<&[isize]>,
) -> Result<Vec<isize>>
where
    D: Dimension,
{
    // Validate inputs
    if structure.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Structure cannot be 0-dimensional".into(),
        ));
    }

    // If origin is specified, validate and use it
    if let Some(orig) = origin {
        if orig.len() != structure.ndim() {
            return Err(NdimageError::DimensionError(format!(
                "Origin must have same length as structure dimensions (got {} expected {})",
                orig.len(),
                structure.ndim()
            )));
        }

        // Check that origin is within bounds
        for (i, &o) in orig.iter().enumerate() {
            let dim = structure.shape()[i] as isize;
            if o < -(dim / 2) || o > dim / 2 {
                return Err(NdimageError::InvalidInput(format!(
                    "Origin {} is out of bounds for dimension {} of size {}",
                    o, i, dim
                )));
            }
        }

        return Ok(orig.to_vec());
    }

    // Otherwise, calculate center
    let mut center = Vec::with_capacity(structure.ndim());
    for &dim in structure.shape() {
        center.push((dim as isize) / 2);
    }

    Ok(center)
}

/// Get the center of a structuring element (dynamic version)
///
/// # Arguments
///
/// * `structure` - Structuring element
/// * `origin` - Optional origin offset
///
/// # Returns
///
/// * `Result<Vec<isize>>` - Center indices
pub(crate) fn get_structure_center_dyn(
    structure: &Array<bool, ndarray::IxDyn>,
    origin: Option<&[isize]>,
) -> Result<Vec<isize>> {
    get_structure_center(structure, origin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_validate_structure() {
        let structure = Array2::from_elem((3, 3), true);
        let result = validate_structure(&structure);
        assert!(result.is_ok());

        let empty_structure = Array2::from_elem((3, 3), false);
        let result = validate_structure(&empty_structure);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_structure_center() {
        let structure = Array2::from_elem((3, 3), true);
        let center = get_structure_center(&structure, None).unwrap();
        assert_eq!(center, vec![1, 1]);

        let structure = Array2::from_elem((5, 5), true);
        let center = get_structure_center(&structure, None).unwrap();
        assert_eq!(center, vec![2, 2]);
    }
}
