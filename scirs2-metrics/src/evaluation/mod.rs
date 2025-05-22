//! Model evaluation utilities
//!
//! This module provides utilities for evaluating machine learning models,
//! such as cross-validation, train-test split, and learning curves.

use ndarray::{Array1, ArrayBase, Data, Dimension};
use num_traits::NumCast;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::error::{MetricsError, Result};

// Re-export evaluation submodules
pub mod cross_validation;
pub mod statistical;
pub mod workflow;

// Re-export common functions for backward compatibility
pub use cross_validation::{
    grouped_k_fold, k_fold_cross_validation, leave_one_out_cv, nested_cross_validation,
    stratified_k_fold, time_series_split,
};
pub use statistical::{
    bootstrap_confidence_interval, cochrans_q_test, friedman_test, mcnemars_test,
    wilcoxon_signed_rank_test,
};

/// Type alias for the train-test split result
pub type TrainTestSplitResult<T> = (Vec<Array1<T>>, Vec<Array1<T>>);

/// Splits arrays or matrices into random train and test subsets
///
/// # Arguments
///
/// * `arrays` - Sequence of arrays to be split (not modified)
/// * `test_size` - Proportion of the dataset to include in the test split (float between 0.0 and 1.0)
/// * `random_seed` - Seed for the random number generator
///
/// # Returns
///
/// * A tuple of arrays `(train_arrays, test_arrays)` where each item contains the training
///   and testing portion of the original arrays
///
/// # Examples
///
/// ```
/// use ndarray::{Array, Ix1};
/// use scirs2_metrics::evaluation::train_test_split;
///
/// let x = Array::<f64, _>::linspace(0., 9., 10).into_shape(Ix1(10)).unwrap();
/// let y = &x * 2.;
///
/// let (train_arrays, test_arrays) = train_test_split(&[&x, &y], 0.3, Some(42)).unwrap();
///
/// // Unpack the results
/// let x_train = &train_arrays[0];
/// let y_train = &train_arrays[1];
/// let x_test = &test_arrays[0];
/// let y_test = &test_arrays[1];
///
/// assert_eq!(x_train.len(), 7);  // 70% of the data
/// assert_eq!(x_test.len(), 3);   // 30% of the data
/// ```
pub fn train_test_split<T>(
    arrays: &[&ArrayBase<impl Data<Elem = T>, impl Dimension>],
    test_size: f64,
    random_seed: Option<u64>,
) -> Result<TrainTestSplitResult<T>>
where
    T: Clone + NumCast,
{
    if arrays.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No arrays provided for splitting".to_string(),
        ));
    }

    // Check all arrays have the same first dimension
    let n_samples = arrays[0].shape()[0];
    for (i, arr) in arrays.iter().enumerate().skip(1) {
        if arr.shape()[0] != n_samples {
            return Err(MetricsError::InvalidInput(format!(
                "Arrays have different lengths: arrays[0]: {}, arrays[{}]: {}",
                n_samples,
                i,
                arr.shape()[0]
            )));
        }
    }

    // Validate test_size
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(MetricsError::InvalidInput(format!(
            "test_size must be between 0 and 1, got {}",
            test_size
        )));
    }

    // Calculate the size of the test set
    let n_test = (n_samples as f64 * test_size).round() as usize;
    if n_test == 0 || n_test >= n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "test_size={} resulted in an invalid test set size: {}",
            test_size, n_test
        )));
    }

    // Generate shuffled indices
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Initialize random number generator with provided seed
    // In rand 0.9.0 we need to use rng() instead of thread_rng()
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            // In rand 0.9.0, from_rng returns the RNG directly, not a Result
            let mut r = rand::rng();
            StdRng::from_rng(&mut r)
        }
    };

    indices.shuffle(&mut rng);

    // Divide indices into train and test sets
    let test_indices = &indices[0..n_test];
    let train_indices = &indices[n_test..];

    // Split each array according to train and test indices
    let mut train_arrays = Vec::with_capacity(arrays.len());
    let mut test_arrays = Vec::with_capacity(arrays.len());

    for &arr in arrays {
        // For train set
        let mut train_arr = Vec::with_capacity(train_indices.len());
        for &idx in train_indices {
            // Safe to use index access here
            let value = match arr.iter().nth(idx) {
                Some(v) => v.clone(),
                None => {
                    return Err(MetricsError::InvalidInput(format!(
                        "Index out of bounds: {} for array of shape {:?}",
                        idx,
                        arr.shape()
                    )))
                }
            };
            train_arr.push(value);
        }
        train_arrays.push(Array1::from(train_arr));

        // For test set
        let mut test_arr = Vec::with_capacity(test_indices.len());
        for &idx in test_indices {
            // Safe to use index access here
            let value = match arr.iter().nth(idx) {
                Some(v) => v.clone(),
                None => {
                    return Err(MetricsError::InvalidInput(format!(
                        "Index out of bounds: {} for array of shape {:?}",
                        idx,
                        arr.shape()
                    )))
                }
            };
            test_arr.push(value);
        }
        test_arrays.push(Array1::from(test_arr));
    }

    Ok((train_arrays, test_arrays))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split() {
        let x = ndarray::Array::linspace(0.0, 9.0, 10);
        let y = &x * 2.0;

        let (train_arrays, test_arrays) = train_test_split(&[&x, &y], 0.3, Some(42)).unwrap();

        // Check arrays dimension
        assert_eq!(train_arrays.len(), 2);
        assert_eq!(test_arrays.len(), 2);

        // Check split ratio
        assert_eq!(train_arrays[0].len(), 7); // 70% of the data
        assert_eq!(test_arrays[0].len(), 3); // 30% of the data

        // Check that y_train = 2 * x_train
        for (x_val, y_val) in train_arrays[0].iter().zip(train_arrays[1].iter()) {
            assert_eq!(*y_val, *x_val * 2.0);
        }

        // Check that y_test = 2 * x_test
        for (x_val, y_val) in test_arrays[0].iter().zip(test_arrays[1].iter()) {
            assert_eq!(*y_val, *x_val * 2.0);
        }
    }
}
