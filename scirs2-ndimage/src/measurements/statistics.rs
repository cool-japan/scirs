//! Statistical measurement functions for labeled arrays

use ndarray::{Array, Array1, Dimension};
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::error::{NdimageError, Result};

/// Sum the values of an array for each labeled region
///
/// # Arguments
///
/// * `input` - Input array
/// * `labels` - Label array
/// * `index` - Labels to include (if None, includes all labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Sum of values for each label
pub fn sum_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> Result<Array1<T>>
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

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Find unique labels
    let unique_labels: std::collections::HashSet<usize> = if let Some(idx) = index {
        idx.iter().cloned().collect()
    } else {
        labels.iter().cloned().collect()
    };

    let mut sorted_labels: Vec<usize> = unique_labels.into_iter().collect();
    sorted_labels.sort();

    // Remove label 0 if it exists (typically background)
    if sorted_labels.first() == Some(&0) {
        sorted_labels.remove(0);
    }

    if sorted_labels.is_empty() {
        return Ok(Array1::<T>::zeros(0));
    }

    // Create a mapping from label to index
    let label_to_idx: std::collections::HashMap<usize, usize> = sorted_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // Initialize sums array
    let mut sums = vec![T::zero(); sorted_labels.len()];

    // Sum values for each label
    for (input_val, label_val) in input.iter().zip(labels.iter()) {
        if let Some(&idx) = label_to_idx.get(label_val) {
            sums[idx] += *input_val;
        }
    }

    Ok(Array1::from_vec(sums))
}

/// Calculate the mean of an array for each labeled region
///
/// # Arguments
///
/// * `input` - Input array
/// * `labels` - Label array
/// * `index` - Labels to include (if None, includes all labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Mean of values for each label
pub fn mean_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    index: Option<&[usize]>,
) -> Result<Array1<T>>
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

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Get sums and counts for each label
    let sums = sum_labels(input, labels, index)?;
    let counts = count_labels(labels, index)?;

    if sums.len() != counts.len() {
        return Err(NdimageError::InvalidInput(
            "Mismatch between sums and counts arrays".into(),
        ));
    }

    // Calculate means (sum / count for each label)
    let means: Vec<T> = sums
        .iter()
        .zip(counts.iter())
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / T::from_usize(count).unwrap()
            } else {
                T::zero()
            }
        })
        .collect();

    Ok(Array1::from_vec(means))
}

/// Calculate the variance of an array for each labeled region
///
/// # Arguments
///
/// * `input` - Input array
/// * `labels` - Label array
/// * `index` - Labels to include (if None, includes all labels)
///
/// # Returns
///
/// * `Result<Array1<T>>` - Variance of values for each label
pub fn variance_labels<T, D>(
    input: &Array<T, D>,
    labels: &Array<usize, D>,
    _index: Option<&[usize]>,
) -> Result<Array1<T>>
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

    if labels.shape() != input.shape() {
        return Err(NdimageError::DimensionError(
            "Labels array must have same shape as input array".to_string(),
        ));
    }

    // Placeholder implementation
    Ok(Array1::<T>::zeros(1))
}

/// Count the number of elements in each labeled region
///
/// # Arguments
///
/// * `labels` - Label array
/// * `index` - Labels to include (if None, includes all labels)
///
/// # Returns
///
/// * `Result<Array1<usize>>` - Count of elements for each label
pub fn count_labels<D>(labels: &Array<usize, D>, index: Option<&[usize]>) -> Result<Array1<usize>>
where
    D: Dimension,
{
    // Validate inputs
    if labels.ndim() == 0 {
        return Err(NdimageError::InvalidInput(
            "Labels array cannot be 0-dimensional".into(),
        ));
    }

    // Find unique labels
    let unique_labels: std::collections::HashSet<usize> = if let Some(idx) = index {
        idx.iter().cloned().collect()
    } else {
        labels.iter().cloned().collect()
    };

    let mut sorted_labels: Vec<usize> = unique_labels.into_iter().collect();
    sorted_labels.sort();

    // Remove label 0 if it exists (typically background)
    if sorted_labels.first() == Some(&0) {
        sorted_labels.remove(0);
    }

    if sorted_labels.is_empty() {
        return Ok(Array1::<usize>::zeros(0));
    }

    // Create a mapping from label to index
    let label_to_idx: std::collections::HashMap<usize, usize> = sorted_labels
        .iter()
        .enumerate()
        .map(|(i, &label)| (label, i))
        .collect();

    // Initialize counts array
    let mut counts = vec![0usize; sorted_labels.len()];

    // Count occurrences of each label
    for &label_val in labels.iter() {
        if let Some(&idx) = label_to_idx.get(&label_val) {
            counts[idx] += 1;
        }
    }

    Ok(Array1::from_vec(counts))
}

/// Calculate histogram of labeled array
///
/// # Arguments
///
/// * `input` - Input array
/// * `min` - Minimum value of range
/// * `max` - Maximum value of range
/// * `bins` - Number of bins
/// * `labels` - Label array (if None, uses whole array)
/// * `index` - Labels to include (if None, includes all labels)
///
/// # Returns
///
/// * `Result<(Array1<usize>, Array1<T>)>` - Histogram counts and bin edges
pub fn histogram<T, D>(
    input: &Array<T, D>,
    min: T,
    max: T,
    bins: usize,
    labels: Option<&Array<usize, D>>,
    _index: Option<&[usize]>,
) -> Result<(Array1<usize>, Array1<T>)>
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

    if min >= max {
        return Err(NdimageError::InvalidInput(format!(
            "min must be less than max (got min={:?}, max={:?})",
            min, max
        )));
    }

    if bins == 0 {
        return Err(NdimageError::InvalidInput(
            "bins must be greater than 0".into(),
        ));
    }

    if let Some(lab) = labels {
        if lab.shape() != input.shape() {
            return Err(NdimageError::DimensionError(
                "Labels array must have same shape as input array".to_string(),
            ));
        }
    }

    // Placeholder implementation
    let hist = Array1::<usize>::zeros(bins);
    let edges = Array1::<T>::linspace(min, max, bins + 1);

    Ok((hist, edges))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sum_labels() {
        let input: Array2<f64> = Array2::eye(3);
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result = sum_labels(&input, &labels, None).unwrap();

        assert!(!result.is_empty());
    }

    #[test]
    fn test_mean_labels() {
        let input: Array2<f64> = Array2::eye(3);
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result = mean_labels(&input, &labels, None).unwrap();

        assert!(!result.is_empty());
    }

    #[test]
    fn test_count_labels() {
        let labels: Array2<usize> = Array2::from_elem((3, 3), 1);
        let result = count_labels(&labels, None).unwrap();

        assert!(!result.is_empty());
    }
}
