//! Cross-validation utilities
//!
//! This module provides functions for cross-validation, a model validation technique
//! to evaluate the generalization performance of a model to an independent dataset.

use ndarray::ArrayBase;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::{HashMap, HashSet};

use crate::error::{MetricsError, Result};

/// Type alias for nested cross-validation result
/// Represents outer train indices, outer test indices, and inner fold splits
pub type NestedCVResult = Vec<(Vec<usize>, Vec<usize>, Vec<(Vec<usize>, Vec<usize>)>)>;

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
/// ```
/// use scirs2_metrics::evaluation::cross_validation::k_fold_cross_validation;
///
/// let splits = k_fold_cross_validation(10, 3, false, None).unwrap();
/// assert_eq!(splits.len(), 3); // 3 folds
///
/// // Check first fold
/// let (train_indices, test_indices) = &splits[0];
/// assert_eq!(train_indices.len(), 6); // 6 or 7 samples in training (depending on split)
/// assert_eq!(test_indices.len(), 4);  // 3 or 4 samples in testing (depending on split)
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
                // Create a StdRng from the global RNG (this suppresses the deprecation warning)
                #[allow(deprecated)]
                let mut r = rand::thread_rng();
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
/// use scirs2_metrics::evaluation::cross_validation::leave_one_out_cv;
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
/// use scirs2_metrics::evaluation::cross_validation::stratified_k_fold;
///
/// let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
/// let splits = stratified_k_fold(&y, 3, true, Some(42)).unwrap();
/// assert_eq!(splits.len(), 3); // 3 folds
/// ```
pub fn stratified_k_fold<T>(
    y: &ArrayBase<impl ndarray::Data<Elem = T>, impl ndarray::Dimension>,
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
            // Create a StdRng from the global RNG (this suppresses the deprecation warning)
            #[allow(deprecated)]
            let mut r = rand::thread_rng();
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

/// Time series cross-validator
///
/// Provides train/test indices to split time series data in train/test sets.
/// This cross-validation object is a variation of KFold which maintains
/// the time ordering of the samples - each training set consists of
/// samples that occur prior in time to the samples in the test set.
///
/// # Arguments
///
/// * `n` - Total number of samples
/// * `n_splits` - Number of splits
/// * `test_size` - Number of samples in each test set
/// * `gap` - Number of samples to exclude after the train set, before the test set
/// * `max_train_size` - Maximum size for the training set (None means use all available samples)
///
/// # Returns
///
/// * A vector of tuples, each containing a pair of train and test indices
///
/// # Examples
///
/// ```
/// use scirs2_metrics::evaluation::cross_validation::time_series_split;
///
/// let splits = time_series_split(10, 3, 2, 0, None).unwrap();
/// assert_eq!(splits.len(), 3); // 3 splits
///
/// // Check first split
/// let (train_indices, test_indices) = &splits[0];
/// assert_eq!(train_indices, &[0, 1, 2, 3]);
/// assert_eq!(test_indices, &[4, 5]);
///
/// // Check second split
/// let (train_indices, test_indices) = &splits[1];
/// assert_eq!(train_indices, &[0, 1, 2, 3, 4, 5]);
/// assert_eq!(test_indices, &[6, 7]);
/// ```
pub fn time_series_split(
    n: usize,
    n_splits: usize,
    test_size: usize,
    gap: usize,
    max_train_size: Option<usize>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n <= test_size {
        return Err(MetricsError::InvalidInput(format!(
            "Number of samples ({}) must be greater than test_size ({})",
            n, test_size
        )));
    }

    if test_size == 0 {
        return Err(MetricsError::InvalidInput(
            "test_size must be greater than 0".to_string(),
        ));
    }

    if n_splits < 1 {
        return Err(MetricsError::InvalidInput(
            "n_splits must be at least 1".to_string(),
        ));
    }

    let mut splits = Vec::with_capacity(n_splits);

    // Calculate the size needed for all splits
    let size_needed = (n_splits - 1) * (test_size + gap) + test_size;
    if size_needed > n {
        return Err(MetricsError::InvalidInput(format!(
            "Cannot perform {} splits with test_size={} and gap={} on {} samples",
            n_splits, test_size, gap, n
        )));
    }

    // Determine the end of the first test set
    let mut test_end = n - (n_splits - 1) * (test_size + gap);

    // Create splits
    for _ in 0..n_splits {
        let train_end = test_end - gap - test_size;
        let test_start = train_end + gap;

        // Get train indices, respecting max_train_size if specified
        let train_start = if let Some(max_size) = max_train_size {
            train_end.saturating_sub(max_size)
        } else {
            0
        };

        let train_indices: Vec<usize> = (train_start..train_end).collect();
        let test_indices: Vec<usize> = (test_start..test_start + test_size).collect();

        splits.push((train_indices, test_indices));

        // Update for next split
        test_end += test_size + gap;
    }

    Ok(splits)
}

/// Grouped K-fold cross-validator
///
/// Provides train/test indices to split data according to groups.
/// This cross-validation ensures that the same group is not present in
/// both training and testing sets. For example, when groups represent
/// patients, this ensures that the same patient won't be in both sets.
///
/// # Arguments
///
/// * `groups` - Array of group labels for the samples
/// * `n_folds` - Number of folds
///
/// # Returns
///
/// * A vector of tuples, each containing a pair of train and test indices
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_metrics::evaluation::grouped_k_fold;
///
/// // Each sample belongs to one of three groups: A, B, or C
/// let groups = array!["A", "A", "A", "B", "B", "C", "C", "C"];
/// let splits = grouped_k_fold(&groups, 3).unwrap();
/// assert_eq!(splits.len(), 3); // 3 folds
///
/// // Check if groups are properly separated
/// for (train_indices, test_indices) in &splits {
///     let mut train_groups = Vec::new();
///     let mut test_groups = Vec::new();
///     
///     for &idx in train_indices {
///         train_groups.push(groups[idx]);
///     }
///     
///     for &idx in test_indices {
///         test_groups.push(groups[idx]);
///     }
///     
///     // Verify no group appears in both train and test sets
///     let mut has_overlap = false;
///     for &test_group in &test_groups {
///         if train_groups.contains(&test_group) {
///             has_overlap = true;
///             break;
///         }
///     }
///     
///     assert!(!has_overlap);
/// }
/// ```
pub fn grouped_k_fold<T>(
    groups: &ArrayBase<impl ndarray::Data<Elem = T>, impl ndarray::Dimension>,
    n_folds: usize,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>>
where
    T: Clone + std::hash::Hash + Eq + std::fmt::Debug,
{
    let n_samples = groups.len();

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

    // Find all unique groups
    let mut unique_groups = HashSet::new();
    for group in groups.iter() {
        unique_groups.insert(group.clone());
    }

    let n_groups = unique_groups.len();

    if n_folds > n_groups {
        return Err(MetricsError::InvalidInput(format!(
            "Number of folds ({}) cannot be greater than number of groups ({})",
            n_folds, n_groups
        )));
    }

    // Create a map from group to sample indices
    let mut group_indices: HashMap<T, Vec<usize>> = HashMap::new();
    for (i, group) in groups.iter().enumerate() {
        group_indices.entry(group.clone()).or_default().push(i);
    }

    // Convert map values to a vector of sample index vectors
    let groups_list: Vec<Vec<usize>> = group_indices.values().cloned().collect();

    // Assign groups to folds using a greedy approach to balance fold sizes
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); n_folds];
    let mut fold_sizes = vec![0; n_folds];

    // Sort groups by size (largest first) for better balancing
    let mut groups_list_with_size: Vec<(usize, Vec<usize>)> = groups_list
        .into_iter()
        .map(|indices| (indices.len(), indices))
        .collect();
    groups_list_with_size.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    // Assign each group to the fold with the fewest samples
    for (_, indices) in groups_list_with_size {
        let fold_idx = fold_sizes
            .iter()
            .enumerate()
            .min_by_key(|&(_, &size)| size)
            .map(|(idx, _)| idx)
            .unwrap();

        folds[fold_idx].extend_from_slice(&indices);
        fold_sizes[fold_idx] += indices.len();
    }

    // Create train/test splits
    let mut splits = Vec::with_capacity(n_folds);

    for i in 0..n_folds {
        let test_indices = folds[i].clone();

        // Combine all other folds for training
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

/// Nested cross-validation
///
/// Performs nested cross-validation, which consists of an inner loop
/// for hyperparameter tuning and an outer loop for model evaluation.
/// This approach provides a less biased estimate of the model's performance.
///
/// # Arguments
///
/// * `n` - Total number of samples
/// * `outer_n_folds` - Number of folds for the outer cross-validation
/// * `inner_n_folds` - Number of folds for the inner cross-validation
/// * `shuffle` - Whether to shuffle the data before splitting
/// * `random_seed` - Seed for the random number generator (if shuffle is true)
///
/// # Returns
///
/// * A vector of tuples, each containing (outer_train_indices, outer_test_indices, inner_splits)
///   where inner_splits is a vector of tuples containing (inner_train_indices, inner_val_indices)
///
/// # Examples
///
/// ```
/// use scirs2_metrics::evaluation::nested_cross_validation;
///
/// let nested_cv = nested_cross_validation(20, 5, 3, true, Some(42)).unwrap();
/// assert_eq!(nested_cv.len(), 5); // 5 outer folds
///
/// // Check first outer fold
/// let (outer_train, outer_test, inner_splits) = &nested_cv[0];
/// assert_eq!(outer_train.len() + outer_test.len(), 20); // All samples are used
/// assert_eq!(inner_splits.len(), 3); // 3 inner folds
/// ```
pub fn nested_cross_validation(
    n: usize,
    outer_n_folds: usize,
    inner_n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<NestedCVResult> {
    if n <= outer_n_folds {
        return Err(MetricsError::InvalidInput(format!(
            "Number of samples ({}) must be greater than outer_n_folds ({})",
            n, outer_n_folds
        )));
    }

    if outer_n_folds < 2 {
        return Err(MetricsError::InvalidInput(
            "outer_n_folds must be at least 2".to_string(),
        ));
    }

    if inner_n_folds < 2 {
        return Err(MetricsError::InvalidInput(
            "inner_n_folds must be at least 2".to_string(),
        ));
    }

    // Get outer fold splits
    let outer_splits = k_fold_cross_validation(n, outer_n_folds, shuffle, random_seed)?;

    // For each outer fold, create inner folds
    let mut nested_splits = Vec::with_capacity(outer_n_folds);

    // If random seed is provided, we need different seeds for each inner CV
    // (This is used in the inner_seed calculation below)

    for (outer_fold_idx, (outer_train, outer_test)) in outer_splits.into_iter().enumerate() {
        // Generate a new seed for inner fold based on the outer fold index
        let inner_seed = random_seed.map(|seed| seed.wrapping_add(outer_fold_idx as u64));

        // Create inner folds using only the outer training data
        let n_inner = outer_train.len();
        let inner_raw_splits =
            k_fold_cross_validation(n_inner, inner_n_folds, shuffle, inner_seed)?;

        // Map inner indices back to original data indices
        let inner_splits = inner_raw_splits
            .into_iter()
            .map(|(inner_train_idx, inner_val_idx)| {
                let inner_train = inner_train_idx
                    .into_iter()
                    .map(|idx| outer_train[idx])
                    .collect();
                let inner_val = inner_val_idx
                    .into_iter()
                    .map(|idx| outer_train[idx])
                    .collect();
                (inner_train, inner_val)
            })
            .collect();

        nested_splits.push((outer_train, outer_test, inner_splits));
    }

    Ok(nested_splits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

    #[test]
    fn test_time_series_split() {
        // Test with 10 samples, 3 splits, test_size=2, no gap
        let splits = time_series_split(10, 3, 2, 0, None).unwrap();

        // Check number of splits
        assert_eq!(splits.len(), 3);

        // Check first split
        let (train_indices, test_indices) = &splits[0];
        assert_eq!(train_indices, &[0, 1, 2, 3]);
        assert_eq!(test_indices, &[4, 5]);

        // Check second split
        let (train_indices, test_indices) = &splits[1];
        assert_eq!(train_indices, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(test_indices, &[6, 7]);

        // Check third split
        let (train_indices, test_indices) = &splits[2];
        assert_eq!(train_indices, &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(test_indices, &[8, 9]);

        // Test with gap
        let splits = time_series_split(12, 3, 2, 1, None).unwrap();

        // Check first split
        let (train_indices, test_indices) = &splits[0];
        assert_eq!(train_indices, &[0, 1, 2]);
        assert_eq!(test_indices, &[4, 5]); // Gap of 1 between train and test

        // Test with max_train_size
        let splits = time_series_split(10, 3, 2, 0, Some(3)).unwrap();

        // Check splits with limited training size
        let (train_indices, test_indices) = &splits[0];
        assert_eq!(train_indices, &[1, 2, 3]); // Only last 3 samples
        assert_eq!(test_indices, &[4, 5]);

        let (train_indices, test_indices) = &splits[1];
        assert_eq!(train_indices, &[3, 4, 5]); // Only last 3 samples
        assert_eq!(test_indices, &[6, 7]);
    }

    #[test]
    fn test_grouped_k_fold() {
        // Create a dataset with 3 groups
        let groups = array!["A", "A", "A", "B", "B", "C", "C", "C"];

        let splits = grouped_k_fold(&groups, 3).unwrap();

        // Check number of folds
        assert_eq!(splits.len(), 3);

        // Check that each fold contains unique groups
        for (train_indices, test_indices) in &splits {
            let mut train_groups = HashSet::new();
            let mut test_groups = HashSet::new();

            for &idx in train_indices {
                train_groups.insert(groups[idx]);
            }

            for &idx in test_indices {
                test_groups.insert(groups[idx]);
            }

            // Verify no common groups between train and test
            for group in &test_groups {
                assert!(!train_groups.contains(group));
            }
        }

        // Check that all indices are used
        let mut all_test_indices = Vec::new();
        for (_, test_indices) in &splits {
            all_test_indices.extend_from_slice(test_indices);
        }
        all_test_indices.sort_unstable();

        assert_eq!(all_test_indices, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_nested_cross_validation() {
        // Test with 20 samples, 5 outer folds, 3 inner folds
        let nested_cv = nested_cross_validation(20, 5, 3, true, Some(42)).unwrap();

        // Check number of outer folds
        assert_eq!(nested_cv.len(), 5);

        for (outer_train, outer_test, inner_splits) in &nested_cv {
            // Check that outer train and test indices are disjoint
            for &test_idx in outer_test {
                assert!(!outer_train.contains(&test_idx));
            }

            // Check number of inner folds
            assert_eq!(inner_splits.len(), 3);

            // Check that inner splits only use indices from outer train
            for (inner_train, inner_val) in inner_splits {
                for &train_idx in inner_train {
                    assert!(outer_train.contains(&train_idx));
                }

                for &val_idx in inner_val {
                    assert!(outer_train.contains(&val_idx));
                }

                // Check that inner train and validation indices are disjoint
                for &val_idx in inner_val {
                    assert!(!inner_train.contains(&val_idx));
                }

                // Check that all samples in outer_train are used in inner CV
                assert_eq!(inner_train.len() + inner_val.len(), outer_train.len());
            }
        }

        // Check that all samples are used in the outer folds
        let mut all_test_indices = Vec::new();
        for (_, outer_test, _) in &nested_cv {
            all_test_indices.extend_from_slice(outer_test);
        }
        all_test_indices.sort_unstable();

        assert_eq!(all_test_indices, (0..20).collect::<Vec<_>>());
    }
}
