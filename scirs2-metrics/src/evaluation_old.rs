//! Model evaluation utilities
//!
//! This module provides utilities for evaluating machine learning models,
//! such as cross-validation, train-test split, and learning curves.

use ndarray::{Array1, ArrayBase, Data, Dimension};
use num_traits::NumCast;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::HashMap;

use crate::error::{MetricsError, Result};

/// Type alias for the train-test split result
pub type TrainTestSplitResult<T> = (Vec<Array1<T>>, Vec<Array1<T>>);

/// Splits arrays or matrices into random train and test subsets
///
/// # Arguments
///
/// * `arrays` - Sequence of arrays to be split
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

/// K-fold cross-validator
///
/// Provides train/test indices to split data in train/test sets. Split dataset into k
/// consecutive folds (without shuffling by default).
///
/// # Arguments
///
/// * `n` - Total number of samples
/// * `n_folds` - Number of folds
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `random_seed` - Seed for the random number generator (if shuffle is true)
///
/// # Returns
///
/// * A vector of tuples, each containing a pair of train and test indices
///
/// # Examples
///
/// ```ignore
/// use scirs2_metrics::evaluation::k_fold_cross_validation;
///
/// let splits = k_fold_cross_validation(10, 3, false, None).unwrap();
/// assert_eq!(splits.len(), 3); // 3 folds
///
/// // Check first fold
/// let (train_indices, test_indices) = &splits[0];
/// assert_eq!(train_indices.len(), 7); // 7 samples in training
/// assert_eq!(test_indices.len(), 3);  // 3 samples in testing
/// ```
pub fn k_fold_cross_validation(
    n: usize,
    n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n <= 1 {
        return Err(MetricsError::InvalidInput(
            "Number of samples must be greater than 1".to_string(),
        ));
    }

    if n_folds < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    if n_folds > n {
        return Err(MetricsError::InvalidInput(format!(
            "Number of folds ({}) cannot be greater than number of samples ({})",
            n_folds, n
        )));
    }

    // Generate indices
    let mut indices: Vec<usize> = (0..n).collect();

    // Shuffle if requested
    if shuffle {
        let mut rng = match random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut r = rand::rng();
                StdRng::from_rng(&mut r)
            }
        };

        indices.shuffle(&mut rng);
    }

    // Calculate fold sizes
    let fold_sizes = (0..n_folds)
        .map(|i| (n - i) / n_folds + ((n - i) % n_folds > 0) as usize)
        .collect::<Vec<_>>();

    let mut current = 0;
    let mut folds = Vec::with_capacity(n_folds);

    // Create folds
    for fold_size in fold_sizes {
        // Extract test indices for this fold
        let test_indices = indices[current..(current + fold_size)].to_vec();

        // Extract train indices by excluding test indices
        let mut train_indices = Vec::with_capacity(n - fold_size);
        train_indices.extend_from_slice(&indices[0..current]);
        train_indices.extend_from_slice(&indices[(current + fold_size)..]);

        folds.push((train_indices, test_indices));
        current += fold_size;
    }

    Ok(folds)
}

/// Leave-one-out cross-validation (LOOCV)
///
/// Returns indices for leave-one-out cross-validation: each sample is used once as a test set
/// while the remaining samples form the training set.
///
/// # Arguments
///
/// * `n` - Total number of samples
///
/// # Returns
///
/// * A vector of tuples, each containing a pair of train and test indices
///
/// # Examples
///
/// ```
/// use scirs2_metrics::evaluation::leave_one_out_cv;
///
/// let splits = leave_one_out_cv(5).unwrap();
/// assert_eq!(splits.len(), 5); // 5 splits for 5 samples
///
/// // Check first split
/// let (train_indices, test_indices) = &splits[0];
/// assert_eq!(train_indices.len(), 4); // 4 samples in training
/// assert_eq!(test_indices.len(), 1);  // 1 sample in testing
/// assert_eq!(test_indices[0], 0);     // First sample in test set
/// ```
pub fn leave_one_out_cv(n: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n <= 1 {
        return Err(MetricsError::InvalidInput(
            "Number of samples must be greater than 1".to_string(),
        ));
    }

    let mut splits = Vec::with_capacity(n);

    for i in 0..n {
        let test_indices = vec![i];

        let mut train_indices = Vec::with_capacity(n - 1);
        for j in 0..n {
            if j != i {
                train_indices.push(j);
            }
        }

        splits.push((train_indices, test_indices));
    }

    Ok(splits)
}

/// Stratified k-fold cross-validator
///
/// Provides train/test indices to split data in train/test sets.
/// This cross-validation object is a variation of KFold that returns stratified folds.
/// The folds are made by preserving the percentage of samples for each class.
///
/// # Arguments
///
/// * `y` - Array of target values
/// * `n_folds` - Number of folds
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `random_seed` - Seed for the random number generator (if shuffle is true)
///
/// # Returns
///
/// * A vector of tuples, each containing a pair of train and test indices
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::evaluation::stratified_k_fold;
///
/// let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
/// let splits = stratified_k_fold(&y, 3, true, Some(42)).unwrap();
/// assert_eq!(splits.len(), 3); // 3 folds
/// ```
pub fn stratified_k_fold<T>(
    y: &ArrayBase<impl Data<Elem = T>, impl Dimension>,
    n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>>
where
    T: Clone + std::hash::Hash + Eq + std::fmt::Debug,
{
    let n_samples = y.len();

    if n_samples <= 1 {
        return Err(MetricsError::InvalidInput(
            "Number of samples must be greater than 1".to_string(),
        ));
    }

    if n_folds < 2 {
        return Err(MetricsError::InvalidInput(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    if n_folds > n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "Number of folds ({}) cannot be greater than number of samples ({})",
            n_folds, n_samples
        )));
    }

    // Count class occurrences
    let mut class_counts = HashMap::new();
    for (i, val) in y.iter().enumerate() {
        class_counts
            .entry(val.clone())
            .or_insert_with(Vec::new)
            .push(i);
    }

    // Check that each class has enough instances
    for (class, indices) in &class_counts {
        if indices.len() < n_folds {
            return Err(MetricsError::InvalidInput(format!(
                "Class {:?} has only {} samples, which is less than n_folds={}",
                class,
                indices.len(),
                n_folds
            )));
        }
    }

    // Initialize random number generator if needed
    let mut rng = match random_seed {
        Some(seed) => Some(StdRng::seed_from_u64(seed)),
        None if shuffle => {
            let mut r = rand::rng();
            Some(StdRng::from_rng(&mut r))
        }
        None => None,
    };

    // Shuffle class indices if needed
    if shuffle {
        let rng = rng.as_mut().unwrap();

        for indices in class_counts.values_mut() {
            indices.shuffle(rng);
        }
    }

    // Allocate samples to folds, respecting the class distribution
    let mut folds = vec![Vec::new(); n_folds];

    for indices in class_counts.values() {
        for (i, &idx) in indices.iter().enumerate() {
            folds[i % n_folds].push(idx);
        }
    }

    // Generate train/test splits
    let mut splits = Vec::with_capacity(n_folds);

    for i in 0..n_folds {
        let test_indices = folds[i].clone();

        let mut train_indices = Vec::with_capacity(n_samples - test_indices.len());
        for (j, fold) in folds.iter().enumerate() {
            if j != i {
                train_indices.extend_from_slice(fold);
            }
        }

        // Sort indices for deterministic behavior
        train_indices.sort_unstable();

        splits.push((train_indices, test_indices));
    }

    Ok(splits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

    #[test]
    fn test_k_fold_cross_validation() {
        // Test 10 samples with 3 folds
        let splits = k_fold_cross_validation(10, 3, false, None).unwrap();

        // Check number of folds
        assert_eq!(splits.len(), 3);

        // Check fold sizes (roughly equal)
        for (train_indices, test_indices) in &splits {
            assert_eq!(train_indices.len() + test_indices.len(), 10);
            assert!(test_indices.len() >= 3); // Each fold should have at least floor(10/3) samples
            assert!(test_indices.len() <= 4); // Each fold should have at most ceil(10/3) samples
        }

        // Check that all indices are used exactly once in test sets
        let mut all_test_indices = Vec::new();
        for (_, test_indices) in &splits {
            all_test_indices.extend_from_slice(test_indices);
        }
        all_test_indices.sort_unstable();

        assert_eq!(all_test_indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_leave_one_out_cv() {
        let splits = leave_one_out_cv(5).unwrap();

        // Check number of splits
        assert_eq!(splits.len(), 5);

        // Check that each split has exactly one test sample
        for (train_indices, test_indices) in &splits {
            assert_eq!(train_indices.len(), 4);
            assert_eq!(test_indices.len(), 1);
        }

        // Check that each index is used exactly once as a test index
        let test_indices: Vec<usize> = splits.iter().map(|(_, test)| test[0]).collect();

        assert_eq!(test_indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_stratified_k_fold() {
        // Create a dataset with imbalanced classes: 4 of class 0, 3 of class 1, 6 of class 2
        // We need at least 3 samples for each class when n_folds=3
        let y = array![0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2];

        let splits = stratified_k_fold(&y, 3, false, None).unwrap();

        // Check number of folds
        assert_eq!(splits.len(), 3);

        // Check that each fold's test set contains a stratified sample
        for (_, test_indices) in &splits {
            // Count classes in this test fold
            let mut class_counts = HashMap::new();
            for &idx in test_indices {
                let class = y[idx];
                *class_counts.entry(class).or_insert(0) += 1;
            }

            // Check class distribution is approximately preserved
            // Each fold should get ~1/3 of each class, but exact values may vary
            // Class 0: 4 samples => 1-2 per fold
            assert!(class_counts.get(&0).map_or(0, |&c| c) >= 1);
            assert!(class_counts.get(&0).map_or(0, |&c| c) <= 2);

            // Class 1: 3 samples => 1 per fold
            assert!(class_counts.get(&1).map_or(0, |&c| c) >= 1);
            assert!(class_counts.get(&1).map_or(0, |&c| c) <= 1);

            // Class 2: 6 samples => 2 per fold
            assert!(class_counts.get(&2).map_or(0, |&c| c) >= 2);
            assert!(class_counts.get(&2).map_or(0, |&c| c) <= 2);
        }

        // Check that all indices are used exactly once in test sets
        let mut all_test_indices = Vec::new();
        for (_, test_indices) in &splits {
            all_test_indices.extend_from_slice(test_indices);
        }
        all_test_indices.sort_unstable();

        assert_eq!(all_test_indices, (0..13).collect::<Vec<_>>());
    }
}
