//! Data splitting utilities for machine learning workflows
//!
//! This module provides various functions for splitting datasets into training,
//! validation, and test sets. It includes support for simple train-test splits,
//! cross-validation (both standard and stratified), and time series splitting.

use crate::error::{DatasetsError, Result};
use crate::utils::Dataset;
use ndarray::Array1;
use rand::prelude::*;
use rand::rng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// Cross-validation fold indices
///
/// Each element is a tuple of (train_indices, validation_indices)
/// where indices refer to samples in the original dataset.
pub type CrossValidationFolds = Vec<(Vec<usize>, Vec<usize>)>;

/// Split a dataset into training and test sets
///
/// This function creates a random split of the dataset while preserving
/// the metadata and feature information in both resulting datasets.
///
/// # Arguments
///
/// * `dataset` - The dataset to split
/// * `test_size` - Fraction of samples to include in test set (0.0 to 1.0)
/// * `random_seed` - Optional random seed for reproducible splits
///
/// # Returns
///
/// A tuple of (train_dataset, test_dataset)
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_datasets::utils::{Dataset, train_test_split};
///
/// let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
/// let dataset = Dataset::new(data, None);
///
/// let (train, test) = train_test_split(&dataset, 0.3, Some(42)).unwrap();
/// assert_eq!(train.n_samples() + test.n_samples(), 10);
/// ```
pub fn train_test_split(
    dataset: &Dataset,
    test_size: f64,
    random_seed: Option<u64>,
) -> Result<(Dataset, Dataset)> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(DatasetsError::InvalidFormat(
            "test_size must be between 0 and 1".to_string(),
        ));
    }

    let n_samples = dataset.n_samples();
    let n_test = (n_samples as f64 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    if n_train == 0 || n_test == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Both train and test sets must have at least one sample".to_string(),
        ));
    }

    // Create shuffled indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };
    indices.shuffle(&mut rng);

    let train_indices = &indices[0..n_train];
    let test_indices = &indices[n_train..];

    // Create training dataset
    let train_data = dataset.data.select(ndarray::Axis(0), train_indices);
    let train_target = dataset
        .target
        .as_ref()
        .map(|t| t.select(ndarray::Axis(0), train_indices));

    let mut train_dataset = Dataset::new(train_data, train_target);
    if let Some(feature_names) = &dataset.feature_names {
        train_dataset = train_dataset.with_feature_names(feature_names.clone());
    }
    if let Some(description) = &dataset.description {
        train_dataset = train_dataset.with_description(description.clone());
    }

    // Create test dataset
    let test_data = dataset.data.select(ndarray::Axis(0), test_indices);
    let test_target = dataset
        .target
        .as_ref()
        .map(|t| t.select(ndarray::Axis(0), test_indices));

    let mut test_dataset = Dataset::new(test_data, test_target);
    if let Some(feature_names) = &dataset.feature_names {
        test_dataset = test_dataset.with_feature_names(feature_names.clone());
    }
    if let Some(description) = &dataset.description {
        test_dataset = test_dataset.with_description(description.clone());
    }

    Ok((train_dataset, test_dataset))
}

/// Performs K-fold cross-validation splitting
///
/// Splits the dataset into k consecutive folds. Each fold is used once as a validation
/// set while the remaining k-1 folds form the training set.
///
/// # Arguments
///
/// * `n_samples` - Number of samples in the dataset
/// * `n_folds` - Number of folds (must be >= 2 and <= n_samples)
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `random_seed` - Optional random seed for reproducible shuffling
///
/// # Returns
///
/// A vector of (train_indices, validation_indices) tuples for each fold
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::utils::k_fold_split;
///
/// let folds = k_fold_split(10, 3, true, Some(42)).unwrap();
/// assert_eq!(folds.len(), 3);
///
/// // Each fold should have roughly equal size
/// for (train_idx, val_idx) in &folds {
///     assert!(val_idx.len() >= 3 && val_idx.len() <= 4);
///     assert_eq!(train_idx.len() + val_idx.len(), 10);
/// }
/// ```
pub fn k_fold_split(
    n_samples: usize,
    n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<CrossValidationFolds> {
    if n_folds < 2 {
        return Err(DatasetsError::InvalidFormat(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    if n_folds > n_samples {
        return Err(DatasetsError::InvalidFormat(
            "Number of folds cannot exceed number of samples".to_string(),
        ));
    }

    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        let mut rng = match random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut r = rng();
                StdRng::seed_from_u64(r.next_u64())
            }
        };
        indices.shuffle(&mut rng);
    }

    let mut folds = Vec::new();
    let fold_size = n_samples / n_folds;
    let remainder = n_samples % n_folds;

    for i in 0..n_folds {
        let start = i * fold_size + i.min(remainder);
        let end = start + fold_size + if i < remainder { 1 } else { 0 };

        let validation_indices = indices[start..end].to_vec();
        let mut train_indices = Vec::new();
        train_indices.extend(&indices[0..start]);
        train_indices.extend(&indices[end..]);

        folds.push((train_indices, validation_indices));
    }

    Ok(folds)
}

/// Performs stratified K-fold cross-validation splitting
///
/// Splits the dataset into k folds while preserving the percentage of samples
/// for each target class in each fold. This is useful for classification tasks
/// with imbalanced datasets.
///
/// # Arguments
///
/// * `targets` - Target values for stratification
/// * `n_folds` - Number of folds (must be >= 2)
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `random_seed` - Optional random seed for reproducible shuffling
///
/// # Returns
///
/// A vector of (train_indices, validation_indices) tuples for each fold
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_datasets::utils::stratified_k_fold_split;
///
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
/// let folds = stratified_k_fold_split(&targets, 2, true, Some(42)).unwrap();
/// assert_eq!(folds.len(), 2);
///
/// // Each fold should maintain class proportions
/// for (train_idx, val_idx) in &folds {
///     assert_eq!(train_idx.len() + val_idx.len(), 6);
/// }
/// ```
pub fn stratified_k_fold_split(
    targets: &Array1<f64>,
    n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<CrossValidationFolds> {
    if n_folds < 2 {
        return Err(DatasetsError::InvalidFormat(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    let n_samples = targets.len();
    if n_folds > n_samples {
        return Err(DatasetsError::InvalidFormat(
            "Number of folds cannot exceed number of samples".to_string(),
        ));
    }

    // Group indices by target class
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();

    for (i, &target) in targets.iter().enumerate() {
        let class = target.round() as i64;
        class_indices.entry(class).or_default().push(i);
    }

    // Shuffle indices within each class if requested
    if shuffle {
        let mut rng = match random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut r = rng();
                StdRng::seed_from_u64(r.next_u64())
            }
        };

        for indices in class_indices.values_mut() {
            indices.shuffle(&mut rng);
        }
    }

    // Create folds while maintaining class proportions
    let mut folds = vec![Vec::new(); n_folds];

    for (_, indices) in class_indices {
        let class_size = indices.len();
        let fold_size = class_size / n_folds;
        let remainder = class_size % n_folds;

        for (i, fold) in folds.iter_mut().enumerate() {
            let start = i * fold_size + i.min(remainder);
            let end = start + fold_size + if i < remainder { 1 } else { 0 };
            fold.extend(&indices[start..end]);
        }
    }

    // Convert to (train, validation) pairs
    let cv_folds = (0..n_folds)
        .map(|i| {
            let validation_indices = folds[i].clone();
            let mut train_indices = Vec::new();
            for (j, fold) in folds.iter().enumerate() {
                if i != j {
                    train_indices.extend(fold);
                }
            }
            (train_indices, validation_indices)
        })
        .collect();

    Ok(cv_folds)
}

/// Performs time series cross-validation splitting
///
/// Creates splits suitable for time series data where future observations
/// should not be used to predict past observations. Each training set contains
/// all observations up to a certain point, and the validation set contains
/// the next `n_test_samples` observations.
///
/// # Arguments
///
/// * `n_samples` - Number of samples in the dataset
/// * `n_splits` - Number of splits to create
/// * `n_test_samples` - Number of samples in each test set
/// * `gap` - Number of samples to skip between train and test sets (default: 0)
///
/// # Returns
///
/// A vector of (train_indices, validation_indices) tuples for each split
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::utils::time_series_split;
///
/// let folds = time_series_split(100, 5, 10, 0).unwrap();
/// assert_eq!(folds.len(), 5);
///
/// // Training sets should be increasing in size
/// for i in 1..folds.len() {
///     assert!(folds[i].0.len() > folds[i-1].0.len());
/// }
/// ```
pub fn time_series_split(
    n_samples: usize,
    n_splits: usize,
    n_test_samples: usize,
    gap: usize,
) -> Result<CrossValidationFolds> {
    if n_splits < 1 {
        return Err(DatasetsError::InvalidFormat(
            "Number of splits must be at least 1".to_string(),
        ));
    }

    if n_test_samples < 1 {
        return Err(DatasetsError::InvalidFormat(
            "Number of test samples must be at least 1".to_string(),
        ));
    }

    // Calculate minimum samples needed
    let min_samples_needed = n_test_samples + gap + n_splits;
    if n_samples < min_samples_needed {
        return Err(DatasetsError::InvalidFormat(format!(
            "Not enough samples for time series split. Need at least {}, got {}",
            min_samples_needed, n_samples
        )));
    }

    let mut folds = Vec::new();
    let test_starts = (0..n_splits)
        .map(|i| {
            let split_size = (n_samples - n_test_samples - gap) / n_splits;
            split_size * (i + 1) + gap
        })
        .collect::<Vec<_>>();

    for &test_start in &test_starts {
        let train_end = test_start - gap;
        let test_end = test_start + n_test_samples;

        if test_end > n_samples {
            break;
        }

        let train_indices = (0..train_end).collect();
        let test_indices = (test_start..test_end).collect();

        folds.push((train_indices, test_indices));
    }

    if folds.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Could not create any valid time series splits".to_string(),
        ));
    }

    Ok(folds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_train_test_split() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let target = Some(array![0.0, 1.0, 0.0, 1.0, 0.0]);
        let dataset = Dataset::new(data, target);

        let (train, test) = train_test_split(&dataset, 0.4, Some(42)).unwrap();

        assert_eq!(train.n_samples() + test.n_samples(), 5);
        assert_eq!(test.n_samples(), 2); // 40% of 5 samples
        assert_eq!(train.n_samples(), 3); // Remaining samples
    }

    #[test]
    fn test_train_test_split_invalid_size() {
        let data = array![[1.0, 2.0]];
        let dataset = Dataset::new(data, None);

        // Test invalid test sizes
        assert!(train_test_split(&dataset, 0.0, None).is_err());
        assert!(train_test_split(&dataset, 1.0, None).is_err());
        assert!(train_test_split(&dataset, 1.5, None).is_err());
    }

    #[test]
    fn test_k_fold_split() {
        let folds = k_fold_split(10, 3, false, Some(42)).unwrap();

        assert_eq!(folds.len(), 3);

        // Check that all samples are covered exactly once in validation
        let mut all_validation_indices: Vec<usize> = Vec::new();
        for (_, val_indices) in &folds {
            all_validation_indices.extend(val_indices);
        }
        all_validation_indices.sort();

        let expected: Vec<usize> = (0..10).collect();
        assert_eq!(all_validation_indices, expected);
    }

    #[test]
    fn test_k_fold_split_invalid_params() {
        // Too few folds
        assert!(k_fold_split(10, 1, false, None).is_err());

        // Too many folds
        assert!(k_fold_split(5, 6, false, None).is_err());
    }

    #[test]
    fn test_stratified_k_fold_split() {
        let targets = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]; // 3 class 0, 3 class 1
        let folds = stratified_k_fold_split(&targets, 2, false, Some(42)).unwrap();

        assert_eq!(folds.len(), 2);

        // Check that all samples are covered
        let mut all_validation_indices: Vec<usize> = Vec::new();
        for (_, val_indices) in &folds {
            all_validation_indices.extend(val_indices);
        }
        all_validation_indices.sort();

        let expected: Vec<usize> = (0..6).collect();
        assert_eq!(all_validation_indices, expected);
    }

    #[test]
    fn test_time_series_split() {
        let folds = time_series_split(20, 3, 5, 1).unwrap();

        assert_eq!(folds.len(), 3);

        // Check that training sets are increasing in size
        for i in 1..folds.len() {
            assert!(folds[i].0.len() > folds[i - 1].0.len());
        }

        // Check that validation sets have correct size
        for (_, val_indices) in &folds {
            assert_eq!(val_indices.len(), 5);
        }
    }

    #[test]
    fn test_time_series_split_insufficient_data() {
        // Not enough samples
        assert!(time_series_split(5, 3, 5, 1).is_err());

        // Invalid parameters
        assert!(time_series_split(100, 0, 10, 0).is_err());
        assert!(time_series_split(100, 5, 0, 0).is_err());
    }
}
