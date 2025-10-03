//! Safe numeric conversion utilities for spatial algorithms

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::numeric::Float;

/// Safely convert a numeric literal to type T
///
/// This function replaces the pattern `T::from(value).unwrap()` with proper error handling
#[allow(dead_code)]
pub fn safe_from<T: Float>(value: f64, context: &str) -> SpatialResult<T> {
    T::from(value).ok_or_else(|| {
        SpatialError::ComputationError(format!(
            "Failed to convert {value} to numeric type in {context}"
        ))
    })
}

/// Safely convert usize to type T
#[allow(dead_code)]
pub fn safe_from_usize<T: Float>(value: usize, context: &str) -> SpatialResult<T> {
    T::from(value).ok_or_else(|| {
        SpatialError::ComputationError(format!(
            "Failed to convert usize {value} to numeric type in {context}"
        ))
    })
}

/// Safely perform partial comparison with proper error handling
#[allow(dead_code)]
pub fn safe_partial_cmp<T: PartialOrd>(
    a: &T,
    b: &T,
    context: &str,
) -> SpatialResult<std::cmp::Ordering> {
    a.partial_cmp(b).ok_or_else(|| {
        SpatialError::ComputationError(format!(
            "Failed to compare values in {context} (possibly NaN or incomparable values)"
        ))
    })
}

/// Safely get the minimum of an iterator
#[allow(dead_code)]
pub fn safe_min<I, T>(iter: I, context: &str) -> SpatialResult<T>
where
    I: Iterator<Item = T>,
    T: PartialOrd,
{
    iter.min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| {
            SpatialError::ComputationError(format!(
                "Failed to find minimum value in {context} (empty iterator or all NaN)"
            ))
        })
}

/// Safely get the maximum of an iterator
#[allow(dead_code)]
pub fn safe_max<I, T>(iter: I, context: &str) -> SpatialResult<T>
where
    I: Iterator<Item = T>,
    T: PartialOrd,
{
    iter.max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| {
            SpatialError::ComputationError(format!(
                "Failed to find maximum value in {context} (empty iterator or all NaN)"
            ))
        })
}

/// Safely convert array view to slice
#[allow(dead_code)]
pub fn safe_as_slice<'a, T>(
    array: &'a scirs2_core::ndarray::ArrayView1<'a, T>,
    context: &str,
) -> SpatialResult<&'a [T]> {
    array.as_slice().ok_or_else(|| {
        SpatialError::ComputationError(format!(
            "Failed to convert array to slice in {context} (non-contiguous memory layout)"
        ))
    })
}

/// Safely get first element of a slice
#[allow(dead_code)]
pub fn safe_first<'a, T>(slice: &'a [T], context: &str) -> SpatialResult<&'a T> {
    slice.first().ok_or_else(|| {
        SpatialError::ComputationError(format!(
            "Failed to get first element in {context} (empty slice)"
        ))
    })
}

/// Safely get last element of a slice
#[allow(dead_code)]
pub fn safe_last<'a, T>(slice: &'a [T], context: &str) -> SpatialResult<&'a T> {
    slice.last().ok_or_else(|| {
        SpatialError::ComputationError(format!(
            "Failed to get last element in {context} (empty slice)"
        ))
    })
}

/// Safely index into an array with bounds checking
#[allow(dead_code)]
pub fn safe_index<T: Clone>(array: &[T], index: usize, context: &str) -> SpatialResult<T> {
    array.get(index).cloned().ok_or_else(|| {
        SpatialError::ComputationError(format!(
            "Index {index} out of bounds for array of length {} in {context}",
            array.len()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_from() {
        let result: SpatialResult<f32> = safe_from(2.0, "test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2.0f32);
    }

    #[test]
    fn test_safe_partial_cmp() {
        let result = safe_partial_cmp(&2.0, &3.0, "test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_safe_min() {
        let values = vec![3.0, 1.0, 4.0, 2.0];
        let result = safe_min(values.into_iter(), "test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1.0);
    }

    #[test]
    fn test_safe_index() {
        let array = vec![1.0, 2.0, 3.0];

        let result = safe_index(&array, 1, "test");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2.0);

        let result = safe_index(&array, 5, "test");
        assert!(result.is_err());
    }
}
