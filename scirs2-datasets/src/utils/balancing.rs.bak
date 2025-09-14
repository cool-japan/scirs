//! Data balancing utilities for handling imbalanced datasets
//!
//! This module provides various strategies for balancing datasets to handle
//! class imbalance problems in machine learning. It includes random oversampling,
//! random undersampling, and SMOTE-like synthetic sample generation.

use crate::error::{DatasetsError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand_distr::Uniform;
use scirs2_core::rng;
use std::collections::HashMap;

/// Balancing strategies for handling imbalanced datasets
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum BalancingStrategy {
    /// Random oversampling - duplicates minority class samples
    RandomOversample,
    /// Random undersampling - removes majority class samples
    RandomUndersample,
    /// SMOTE (Synthetic Minority Oversampling Technique) with specified k_neighbors
    SMOTE {
        /// Number of nearest neighbors to consider for synthetic sample generation
        k_neighbors: usize,
    },
}

/// Performs random oversampling to balance class distribution
///
/// Duplicates samples from minority classes to match the majority class size.
/// This is useful for handling imbalanced datasets in classification problems.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples, n_features)
/// * `targets` - Target values for each sample
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A tuple containing the resampled (data, targets) arrays
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use scirs2__datasets::utils::random_oversample;
///
/// let data = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Imbalanced: 2 vs 4
/// let (balanced_data, balanced_targets) = random_oversample(&data, &targets, Some(42)).unwrap();
/// // Now both classes have 4 samples each
/// ```
#[allow(dead_code)]
pub fn random_oversample(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    random_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if data.nrows() != targets.len() {
        return Err(DatasetsError::InvalidFormat(
            "Data rows and targets length must match".to_string(),
        ));
    }

    if data.is_empty() || targets.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Data and targets cannot be empty".to_string(),
        ));
    }

    // Group indices by class
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &target) in targets.iter().enumerate() {
        let class = target.round() as i64;
        class_indices.entry(class).or_default().push(i);
    }

    // Find the majority class size
    let max_class_size = class_indices.values().map(|v| v.len()).max().unwrap();

    let mut rng = match random_seed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Collect all resampled indices
    let mut resampled_indices = Vec::new();

    for (_, indices) in class_indices {
        let class_size = indices.len();

        // Add all original samples
        resampled_indices.extend(&indices);

        // Oversample if this class is smaller than the majority class
        if class_size < max_class_size {
            let samples_needed = max_class_size - class_size;
            for _ in 0..samples_needed {
                let random_idx = rng.sample(Uniform::new(0, class_size).unwrap());
                resampled_indices.push(indices[random_idx]);
            }
        }
    }

    // Create resampled data and targets
    let resampled_data = data.select(ndarray::Axis(0), &resampled_indices);
    let resampled_targets = targets.select(ndarray::Axis(0), &resampled_indices);

    Ok((resampled_data, resampled_targets))
}

/// Performs random undersampling to balance class distribution
///
/// Randomly removes samples from majority classes to match the minority class size.
/// This reduces the overall dataset size but maintains balance.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples, n_features)
/// * `targets` - Target values for each sample
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A tuple containing the undersampled (data, targets) arrays
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use scirs2__datasets::utils::random_undersample;
///
/// let data = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Imbalanced: 2 vs 4
/// let (balanced_data, balanced_targets) = random_undersample(&data, &targets, Some(42)).unwrap();
/// // Now both classes have 2 samples each
/// ```
#[allow(dead_code)]
pub fn random_undersample(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    random_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if data.nrows() != targets.len() {
        return Err(DatasetsError::InvalidFormat(
            "Data rows and targets length must match".to_string(),
        ));
    }

    if data.is_empty() || targets.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Data and targets cannot be empty".to_string(),
        ));
    }

    // Group indices by class
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &target) in targets.iter().enumerate() {
        let class = target.round() as i64;
        class_indices.entry(class).or_default().push(i);
    }

    // Find the minority class size
    let min_class_size = class_indices.values().map(|v| v.len()).min().unwrap();

    let mut rng = match random_seed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    // Collect undersampled indices
    let mut undersampled_indices = Vec::new();

    for (_, mut indices) in class_indices {
        if indices.len() > min_class_size {
            // Randomly sample down to minority class size
            indices.shuffle(&mut rng);
            undersampled_indices.extend(&indices[0..min_class_size]);
        } else {
            // Use all samples if already at or below minority class size
            undersampled_indices.extend(&indices);
        }
    }

    // Create undersampled data and targets
    let undersampled_data = data.select(ndarray::Axis(0), &undersampled_indices);
    let undersampled_targets = targets.select(ndarray::Axis(0), &undersampled_indices);

    Ok((undersampled_data, undersampled_targets))
}

/// Generates synthetic samples using SMOTE-like interpolation
///
/// Creates synthetic samples by interpolating between existing samples within each class.
/// This is useful for oversampling minority classes without simple duplication.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples, n_features)
/// * `targets` - Target values for each sample
/// * `target_class` - The class to generate synthetic samples for
/// * `n_synthetic` - Number of synthetic samples to generate
/// * `k_neighbors` - Number of nearest neighbors to consider for interpolation
/// * `random_seed` - Optional random seed for reproducible generation
///
/// # Returns
///
/// A tuple containing the synthetic (data, targets) arrays
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use scirs2__datasets::utils::generate_synthetic_samples;
///
/// let data = Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5]).unwrap();
/// let targets = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);
/// let (synthetic_data, synthetic_targets) = generate_synthetic_samples(&data, &targets, 0.0, 2, 2, Some(42)).unwrap();
/// assert_eq!(synthetic_data.nrows(), 2);
/// assert_eq!(synthetic_targets.len(), 2);
/// ```
#[allow(dead_code)]
pub fn generate_synthetic_samples(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    target_class: f64,
    n_synthetic: usize,
    k_neighbors: usize,
    random_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if data.nrows() != targets.len() {
        return Err(DatasetsError::InvalidFormat(
            "Data rows and targets length must match".to_string(),
        ));
    }

    if n_synthetic == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Number of _synthetic samples must be > 0".to_string(),
        ));
    }

    if k_neighbors == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Number of _neighbors must be > 0".to_string(),
        ));
    }

    // Find samples belonging to the target _class
    let class_indices: Vec<usize> = targets
        .iter()
        .enumerate()
        .filter(|(_, &target)| (target - target_class).abs() < 1e-10)
        .map(|(i, _)| i)
        .collect();

    if class_indices.len() < 2 {
        return Err(DatasetsError::InvalidFormat(
            "Need at least 2 samples of the target _class for _synthetic generation".to_string(),
        ));
    }

    if k_neighbors >= class_indices.len() {
        return Err(DatasetsError::InvalidFormat(
            "k_neighbors must be less than the number of samples in the target _class".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(_seed) => StdRng::seed_from_u64(_seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let n_features = data.ncols();
    let mut synthetic_data = Array2::zeros((n_synthetic, n_features));
    let synthetic_targets = Array1::from_elem(n_synthetic, target_class);

    for i in 0..n_synthetic {
        // Randomly select a sample from the target _class
        let base_idx = class_indices[rng.sample(Uniform::new(0, class_indices.len()).unwrap())];
        let base_sample = data.row(base_idx);

        // Find k nearest _neighbors within the same _class
        let mut distances: Vec<(usize, f64)> = class_indices
            .iter()
            .filter(|&&idx| idx != base_idx)
            .map(|&idx| {
                let neighbor = data.row(idx);
                let distance: f64 = base_sample
                    .iter()
                    .zip(neighbor.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (idx, distance)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let k_nearest = &distances[0..k_neighbors.min(distances.len())];

        // Select a random neighbor from the k nearest
        let neighbor_idx = k_nearest[rng.sample(Uniform::new(0, k_nearest.len()).unwrap())].0;
        let neighbor_sample = data.row(neighbor_idx);

        // Generate _synthetic sample by interpolation
        let alpha = rng.gen_range(0.0..1.0);
        for (j, synthetic_feature) in synthetic_data.row_mut(i).iter_mut().enumerate() {
            *synthetic_feature = base_sample[j] + alpha * (neighbor_sample[j] - base_sample[j]);
        }
    }

    Ok((synthetic_data, synthetic_targets))
}

/// Creates a balanced dataset using the specified balancing strategy
///
/// Automatically balances the dataset by applying oversampling, undersampling,
/// or synthetic sample generation based on the specified strategy.
///
/// # Arguments
///
/// * `data` - Feature matrix (n_samples, n_features)
/// * `targets` - Target values for each sample
/// * `strategy` - Balancing strategy to use
/// * `random_seed` - Optional random seed for reproducible balancing
///
/// # Returns
///
/// A tuple containing the balanced (data, targets) arrays
///
/// # Examples
///
/// ```rust
/// use ndarray::{Array1, Array2};
/// use scirs2__datasets::utils::{create_balanced_dataset, BalancingStrategy};
///
/// let data = Array2::from_shape_vec((6, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
/// let (balanced_data, balanced_targets) = create_balanced_dataset(&data, &targets, BalancingStrategy::RandomOversample, Some(42)).unwrap();
/// ```
#[allow(dead_code)]
pub fn create_balanced_dataset(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    strategy: BalancingStrategy,
    random_seed: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    match strategy {
        BalancingStrategy::RandomOversample => random_oversample(data, targets, random_seed),
        BalancingStrategy::RandomUndersample => random_undersample(data, targets, random_seed),
        BalancingStrategy::SMOTE { k_neighbors } => {
            // Apply SMOTE to minority classes
            let mut class_counts: HashMap<i64, usize> = HashMap::new();
            for &target in targets.iter() {
                let class = target.round() as i64;
                *class_counts.entry(class).or_default() += 1;
            }

            let max_count = *class_counts.values().max().unwrap();
            let mut combined_data = data.clone();
            let mut combined_targets = targets.clone();

            for (&class, &count) in &class_counts {
                if count < max_count {
                    let samples_needed = max_count - count;
                    let (synthetic_data, synthetic_targets) = generate_synthetic_samples(
                        data,
                        targets,
                        class as f64,
                        samples_needed,
                        k_neighbors,
                        random_seed,
                    )?;

                    // Concatenate with existing data
                    combined_data = ndarray::concatenate(
                        ndarray::Axis(0),
                        &[combined_data.view(), synthetic_data.view()],
                    )
                    .map_err(|_| {
                        DatasetsError::InvalidFormat("Failed to concatenate data".to_string())
                    })?;

                    combined_targets = ndarray::concatenate(
                        ndarray::Axis(0),
                        &[combined_targets.view(), synthetic_targets.view()],
                    )
                    .map_err(|_| {
                        DatasetsError::InvalidFormat("Failed to concatenate targets".to_string())
                    })?;
                }
            }

            Ok((combined_data, combined_targets))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Uniform;

    #[test]
    fn test_random_oversample() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Imbalanced: 2 vs 4

        let (balanced_data, balanced_targets) =
            random_oversample(&data, &targets, Some(42)).unwrap();

        // Check that we now have equal number of each class
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, 4); // Should be oversampled to match majority class
        assert_eq!(class_1_count, 4);

        // Check that total samples increased
        assert_eq!(balanced_data.nrows(), 8);
        assert_eq!(balanced_targets.len(), 8);

        // Check that data dimensions are preserved
        assert_eq!(balanced_data.ncols(), 2);
    }

    #[test]
    fn test_random_oversample_invalid_params() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array1::from(vec![0.0, 1.0]);

        // Mismatched data and targets
        assert!(random_oversample(&data, &targets, None).is_err());

        // Empty data
        let empty_data = Array2::zeros((0, 2));
        let empty_targets = Array1::from(vec![]);
        assert!(random_oversample(&empty_data, &empty_targets, None).is_err());
    }

    #[test]
    fn test_random_undersample() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Imbalanced: 2 vs 4

        let (balanced_data, balanced_targets) =
            random_undersample(&data, &targets, Some(42)).unwrap();

        // Check that we now have equal number of each class (minimum)
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, 2); // Should match minority class
        assert_eq!(class_1_count, 2); // Should be undersampled to match minority class

        // Check that total samples decreased
        assert_eq!(balanced_data.nrows(), 4);
        assert_eq!(balanced_targets.len(), 4);

        // Check that data dimensions are preserved
        assert_eq!(balanced_data.ncols(), 2);
    }

    #[test]
    fn test_random_undersample_invalid_params() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array1::from(vec![0.0, 1.0]);

        // Mismatched data and targets
        assert!(random_undersample(&data, &targets, None).is_err());

        // Empty data
        let empty_data = Array2::zeros((0, 2));
        let empty_targets = Array1::from(vec![]);
        assert!(random_undersample(&empty_data, &empty_targets, None).is_err());
    }

    #[test]
    fn test_generate_synthetic_samples() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5]).unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);

        let (synthetic_data, synthetic_targets) =
            generate_synthetic_samples(&data, &targets, 0.0, 2, 2, Some(42)).unwrap();

        // Check that we generated the correct number of synthetic samples
        assert_eq!(synthetic_data.nrows(), 2);
        assert_eq!(synthetic_targets.len(), 2);

        // Check that all synthetic targets are the correct class
        for &target in synthetic_targets.iter() {
            assert_eq!(target, 0.0);
        }

        // Check that data dimensions are preserved
        assert_eq!(synthetic_data.ncols(), 2);

        // Check that synthetic samples are interpolations (should be within reasonable bounds)
        for i in 0..synthetic_data.nrows() {
            for j in 0..synthetic_data.ncols() {
                let value = synthetic_data[[i, j]];
                assert!((0.5..=2.5).contains(&value)); // Should be within range of class 0 samples
            }
        }
    }

    #[test]
    fn test_generate_synthetic_samples_invalid_params() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 2.5, 2.5]).unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 0.0, 1.0]);

        // Mismatched data and targets
        let bad_targets = Array1::from(vec![0.0, 1.0]);
        assert!(generate_synthetic_samples(&data, &bad_targets, 0.0, 2, 2, None).is_err());

        // Zero synthetic samples
        assert!(generate_synthetic_samples(&data, &targets, 0.0, 0, 2, None).is_err());

        // Zero neighbors
        assert!(generate_synthetic_samples(&data, &targets, 0.0, 2, 0, None).is_err());

        // Too few samples of target class (only 1 sample of class 1.0)
        assert!(generate_synthetic_samples(&data, &targets, 1.0, 2, 2, None).is_err());

        // k_neighbors >= number of samples in class
        assert!(generate_synthetic_samples(&data, &targets, 0.0, 2, 3, None).is_err());
    }

    #[test]
    fn test_create_balanced_dataset_random_oversample() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let (balanced_data, balanced_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomOversample,
            Some(42),
        )
        .unwrap();

        // Check that classes are balanced
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, class_1_count);
        assert_eq!(balanced_data.nrows(), balanced_targets.len());
    }

    #[test]
    fn test_create_balanced_dataset_random_undersample() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let (balanced_data, balanced_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomUndersample,
            Some(42),
        )
        .unwrap();

        // Check that classes are balanced
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, class_1_count);
        assert_eq!(balanced_data.nrows(), balanced_targets.len());
    }

    #[test]
    fn test_create_balanced_dataset_smote() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0,
            ],
        )
        .unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]); // Already balanced for easier testing

        let (balanced_data, balanced_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::SMOTE { k_neighbors: 2 },
            Some(42),
        )
        .unwrap();

        // Check that classes remain balanced
        let class_0_count = balanced_targets.iter().filter(|&&x| x == 0.0).count();
        let class_1_count = balanced_targets.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(class_0_count, class_1_count);
        assert_eq!(balanced_data.nrows(), balanced_targets.len());
    }

    #[test]
    fn test_balancing_strategy_with_multiple_classes() {
        // Test with 3 classes of different sizes
        let data = Array2::from_shape_vec((9, 2), (0..18).map(|x| x as f64).collect()).unwrap();
        let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        // Class distribution: 0 (2 samples), 1 (4 samples), 2 (3 samples)

        // Test oversampling
        let (_over_data, over_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomOversample,
            Some(42),
        )
        .unwrap();

        let over_class_0_count = over_targets.iter().filter(|&&x| x == 0.0).count();
        let over_class_1_count = over_targets.iter().filter(|&&x| x == 1.0).count();
        let over_class_2_count = over_targets.iter().filter(|&&x| x == 2.0).count();

        // All classes should have 4 samples (majority class size)
        assert_eq!(over_class_0_count, 4);
        assert_eq!(over_class_1_count, 4);
        assert_eq!(over_class_2_count, 4);

        // Test undersampling
        let (_under_data, under_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomUndersample,
            Some(42),
        )
        .unwrap();

        let under_class_0_count = under_targets.iter().filter(|&&x| x == 0.0).count();
        let under_class_1_count = under_targets.iter().filter(|&&x| x == 1.0).count();
        let under_class_2_count = under_targets.iter().filter(|&&x| x == 2.0).count();

        // All classes should have 2 samples (minority class size)
        assert_eq!(under_class_0_count, 2);
        assert_eq!(under_class_1_count, 2);
        assert_eq!(under_class_2_count, 2);
    }
}
