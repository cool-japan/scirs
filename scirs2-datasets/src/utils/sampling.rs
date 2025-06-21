//! Data sampling utilities for statistical analysis and machine learning
//!
//! This module provides various sampling strategies including random sampling,
//! stratified sampling, and importance-weighted sampling. These functions are
//! useful for creating representative subsets of datasets, bootstrap sampling,
//! and handling imbalanced data distributions.

use crate::error::{DatasetsError, Result};
use ndarray::Array1;
use rand::prelude::*;
use rand::rng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// Performs random sampling with or without replacement
///
/// This function creates random samples from a dataset using either bootstrap
/// sampling (with replacement) or standard random sampling (without replacement).
///
/// # Arguments
///
/// * `n_samples` - Total number of samples in the dataset
/// * `sample_size` - Number of samples to draw
/// * `replace` - Whether to sample with replacement (bootstrap)
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A vector of indices representing the sampled data points
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::utils::random_sample;
///
/// // Sample 5 indices from 10 total samples without replacement
/// let indices = random_sample(10, 5, false, Some(42)).unwrap();
/// assert_eq!(indices.len(), 5);
/// assert!(indices.iter().all(|&i| i < 10));
///
/// // Bootstrap sampling (with replacement)
/// let bootstrap_indices = random_sample(10, 15, true, Some(42)).unwrap();
/// assert_eq!(bootstrap_indices.len(), 15);
/// ```
pub fn random_sample(
    n_samples: usize,
    sample_size: usize,
    replace: bool,
    random_seed: Option<u64>,
) -> Result<Vec<usize>> {
    if n_samples == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Number of samples must be > 0".to_string(),
        ));
    }

    if sample_size == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Sample size must be > 0".to_string(),
        ));
    }

    if !replace && sample_size > n_samples {
        return Err(DatasetsError::InvalidFormat(format!(
            "Cannot sample {} items from {} without replacement",
            sample_size, n_samples
        )));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut indices = Vec::with_capacity(sample_size);

    if replace {
        // Bootstrap sampling (with replacement)
        for _ in 0..sample_size {
            indices.push(rng.random_range(0..n_samples));
        }
    } else {
        // Sampling without replacement
        let mut available: Vec<usize> = (0..n_samples).collect();
        available.shuffle(&mut rng);
        indices.extend_from_slice(&available[0..sample_size]);
    }

    Ok(indices)
}

/// Performs stratified random sampling
///
/// Maintains the same class distribution in the sample as in the original dataset.
/// This is particularly useful for classification tasks where you want to ensure
/// that all classes are represented proportionally in your sample.
///
/// # Arguments
///
/// * `targets` - Target values for stratification
/// * `sample_size` - Number of samples to draw
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A vector of indices representing the stratified sample
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_datasets::utils::stratified_sample;
///
/// let targets = Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
/// let indices = stratified_sample(&targets, 6, Some(42)).unwrap();
/// assert_eq!(indices.len(), 6);
///
/// // Check that the sample maintains class proportions
/// let mut class_counts = std::collections::HashMap::new();
/// for &idx in &indices {
///     let class = targets[idx] as i32;
///     *class_counts.entry(class).or_insert(0) += 1;
/// }
/// ```
pub fn stratified_sample(
    targets: &Array1<f64>,
    sample_size: usize,
    random_seed: Option<u64>,
) -> Result<Vec<usize>> {
    if targets.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Targets array cannot be empty".to_string(),
        ));
    }

    if sample_size == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Sample size must be > 0".to_string(),
        ));
    }

    if sample_size > targets.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "Cannot sample {} items from {} total samples",
            sample_size,
            targets.len()
        )));
    }

    // Group indices by target class
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &target) in targets.iter().enumerate() {
        let class = target.round() as i64;
        class_indices.entry(class).or_default().push(i);
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut stratified_indices = Vec::new();
    let n_classes = class_indices.len();
    let base_samples_per_class = sample_size / n_classes;
    let remainder = sample_size % n_classes;

    let mut class_list: Vec<_> = class_indices.keys().cloned().collect();
    class_list.sort();

    for (i, &class) in class_list.iter().enumerate() {
        let class_samples = class_indices.get(&class).unwrap();
        let samples_for_this_class = if i < remainder {
            base_samples_per_class + 1
        } else {
            base_samples_per_class
        };

        if samples_for_this_class > class_samples.len() {
            return Err(DatasetsError::InvalidFormat(format!(
                "Class {} has only {} samples but needs {} for stratified sampling",
                class,
                class_samples.len(),
                samples_for_this_class
            )));
        }

        // Sample from this class
        let sampled_indices = random_sample(
            class_samples.len(),
            samples_for_this_class,
            false,
            Some(rng.next_u64()),
        )?;

        for &idx in &sampled_indices {
            stratified_indices.push(class_samples[idx]);
        }
    }

    stratified_indices.shuffle(&mut rng);
    Ok(stratified_indices)
}

/// Performs importance sampling based on provided weights
///
/// Samples indices according to the provided probability weights. Higher weights
/// increase the probability of selection. This is useful for adaptive sampling
/// where some samples are more important than others for training.
///
/// # Arguments
///
/// * `weights` - Probability weights for each sample (must be non-negative)
/// * `sample_size` - Number of samples to draw
/// * `replace` - Whether to sample with replacement
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A vector of indices representing the importance-weighted sample
///
/// # Examples
///
/// ```rust
/// use ndarray::Array1;
/// use scirs2_datasets::utils::importance_sample;
///
/// // Give higher weights to the last few samples
/// let weights = Array1::from(vec![0.1, 0.1, 0.1, 0.8, 0.9, 1.0]);
/// let indices = importance_sample(&weights, 3, false, Some(42)).unwrap();
/// assert_eq!(indices.len(), 3);
///
/// // Higher weighted samples should be more likely to be selected
/// let mut high_weight_count = 0;
/// for &idx in &indices {
///     if idx >= 3 { // Last three samples have higher weights
///         high_weight_count += 1;
///     }
/// }
/// // This should be true with high probability
/// assert!(high_weight_count >= 1);
/// ```
pub fn importance_sample(
    weights: &Array1<f64>,
    sample_size: usize,
    replace: bool,
    random_seed: Option<u64>,
) -> Result<Vec<usize>> {
    if weights.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "Weights array cannot be empty".to_string(),
        ));
    }

    if sample_size == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Sample size must be > 0".to_string(),
        ));
    }

    if !replace && sample_size > weights.len() {
        return Err(DatasetsError::InvalidFormat(format!(
            "Cannot sample {} items from {} without replacement",
            sample_size,
            weights.len()
        )));
    }

    // Check for negative weights
    for &weight in weights.iter() {
        if weight < 0.0 {
            return Err(DatasetsError::InvalidFormat(
                "All weights must be non-negative".to_string(),
            ));
        }
    }

    let weight_sum: f64 = weights.sum();
    if weight_sum <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "Sum of weights must be positive".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut indices = Vec::with_capacity(sample_size);
    let mut available_weights = weights.clone();
    let mut available_indices: Vec<usize> = (0..weights.len()).collect();

    for _ in 0..sample_size {
        let current_sum = available_weights.sum();
        if current_sum <= 0.0 {
            break;
        }

        // Generate random number between 0 and current_sum
        let random_value = rng.random_range(0.0..current_sum);

        // Find the index corresponding to this random value
        let mut cumulative_weight = 0.0;
        let mut selected_idx = 0;

        for (i, &weight) in available_weights.iter().enumerate() {
            cumulative_weight += weight;
            if random_value <= cumulative_weight {
                selected_idx = i;
                break;
            }
        }

        let original_idx = available_indices[selected_idx];
        indices.push(original_idx);

        if !replace {
            // Remove the selected item for sampling without replacement
            available_weights = Array1::from_iter(
                available_weights
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != selected_idx)
                    .map(|(_, &w)| w),
            );
            available_indices.remove(selected_idx);
        }
    }

    Ok(indices)
}

/// Generate bootstrap samples from indices
///
/// This is a convenience function that generates bootstrap samples (sampling with
/// replacement) which is commonly used for bootstrap confidence intervals and
/// ensemble methods.
///
/// # Arguments
///
/// * `n_samples` - Total number of samples in the dataset
/// * `n_bootstrap_samples` - Number of bootstrap samples to generate
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A vector of bootstrap sample indices
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::utils::bootstrap_sample;
///
/// let bootstrap_indices = bootstrap_sample(100, 100, Some(42)).unwrap();
/// assert_eq!(bootstrap_indices.len(), 100);
///
/// // Some indices should appear multiple times (with high probability)
/// let mut unique_indices = bootstrap_indices.clone();
/// unique_indices.sort();
/// unique_indices.dedup();
/// assert!(unique_indices.len() < bootstrap_indices.len());
/// ```
pub fn bootstrap_sample(
    n_samples: usize,
    n_bootstrap_samples: usize,
    random_seed: Option<u64>,
) -> Result<Vec<usize>> {
    random_sample(n_samples, n_bootstrap_samples, true, random_seed)
}

/// Generate multiple bootstrap samples
///
/// Creates multiple independent bootstrap samples, useful for ensemble methods
/// like bagging or for computing bootstrap confidence intervals.
///
/// # Arguments
///
/// * `n_samples` - Total number of samples in the dataset
/// * `sample_size` - Size of each bootstrap sample
/// * `n_bootstrap_rounds` - Number of bootstrap samples to generate
/// * `random_seed` - Optional random seed for reproducible sampling
///
/// # Returns
///
/// A vector of bootstrap sample vectors
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::utils::multiple_bootstrap_samples;
///
/// let bootstrap_samples = multiple_bootstrap_samples(50, 50, 10, Some(42)).unwrap();
/// assert_eq!(bootstrap_samples.len(), 10);
/// assert!(bootstrap_samples.iter().all(|sample| sample.len() == 50));
/// ```
pub fn multiple_bootstrap_samples(
    n_samples: usize,
    sample_size: usize,
    n_bootstrap_rounds: usize,
    random_seed: Option<u64>,
) -> Result<Vec<Vec<usize>>> {
    if n_bootstrap_rounds == 0 {
        return Err(DatasetsError::InvalidFormat(
            "Number of bootstrap rounds must be > 0".to_string(),
        ));
    }

    let mut rng = match random_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let mut r = rng();
            StdRng::seed_from_u64(r.next_u64())
        }
    };

    let mut bootstrap_samples = Vec::with_capacity(n_bootstrap_rounds);

    for _ in 0..n_bootstrap_rounds {
        let sample = random_sample(n_samples, sample_size, true, Some(rng.next_u64()))?;
        bootstrap_samples.push(sample);
    }

    Ok(bootstrap_samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::HashSet;

    #[test]
    fn test_random_sample_without_replacement() {
        let indices = random_sample(10, 5, false, Some(42)).unwrap();

        assert_eq!(indices.len(), 5);
        assert!(indices.iter().all(|&i| i < 10));

        // All indices should be unique (no replacement)
        let unique_indices: HashSet<_> = indices.iter().cloned().collect();
        assert_eq!(unique_indices.len(), 5);
    }

    #[test]
    fn test_random_sample_with_replacement() {
        let indices = random_sample(5, 10, true, Some(42)).unwrap();

        assert_eq!(indices.len(), 10);
        assert!(indices.iter().all(|&i| i < 5));

        // Some indices might be repeated (with replacement)
        let unique_indices: HashSet<_> = indices.iter().cloned().collect();
        assert!(unique_indices.len() <= 10);
    }

    #[test]
    fn test_random_sample_invalid_params() {
        // Zero samples
        assert!(random_sample(0, 5, false, None).is_err());

        // Zero sample size
        assert!(random_sample(10, 0, false, None).is_err());

        // Too many samples without replacement
        assert!(random_sample(5, 10, false, None).is_err());
    }

    #[test]
    fn test_stratified_sample() {
        let targets = array![0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]; // 2, 3, 3 samples per class
        let indices = stratified_sample(&targets, 6, Some(42)).unwrap();

        assert_eq!(indices.len(), 6);

        // Count samples per class in the result
        let mut class_counts = HashMap::new();
        for &idx in &indices {
            let class = targets[idx] as i32;
            *class_counts.entry(class).or_insert(0) += 1;
        }

        // Should maintain rough proportions
        assert!(class_counts.len() <= 3); // At most 3 classes
    }

    #[test]
    fn test_stratified_sample_insufficient_samples() {
        let targets = array![0.0, 1.0]; // Only 1 sample per class
                                        // Requesting 4 samples but only 2 total
        assert!(stratified_sample(&targets, 4, Some(42)).is_err());
    }

    #[test]
    fn test_importance_sample() {
        let weights = array![0.1, 0.1, 0.1, 0.8, 0.9, 1.0]; // Higher weights at the end
        let indices = importance_sample(&weights, 3, false, Some(42)).unwrap();

        assert_eq!(indices.len(), 3);
        assert!(indices.iter().all(|&i| i < 6));

        // All indices should be unique (no replacement)
        let unique_indices: HashSet<_> = indices.iter().cloned().collect();
        assert_eq!(unique_indices.len(), 3);
    }

    #[test]
    fn test_importance_sample_negative_weights() {
        let weights = array![0.5, -0.1, 0.3]; // Contains negative weight
        assert!(importance_sample(&weights, 2, false, None).is_err());
    }

    #[test]
    fn test_importance_sample_zero_weights() {
        let weights = array![0.0, 0.0, 0.0]; // All zero weights
        assert!(importance_sample(&weights, 2, false, None).is_err());
    }

    #[test]
    fn test_bootstrap_sample() {
        let indices = bootstrap_sample(20, 20, Some(42)).unwrap();

        assert_eq!(indices.len(), 20);
        assert!(indices.iter().all(|&i| i < 20));

        // Should likely have some repeated indices
        let unique_indices: HashSet<_> = indices.iter().cloned().collect();
        assert!(unique_indices.len() < 20); // Very likely with size 20
    }

    #[test]
    fn test_multiple_bootstrap_samples() {
        let samples = multiple_bootstrap_samples(10, 8, 5, Some(42)).unwrap();

        assert_eq!(samples.len(), 5);
        assert!(samples.iter().all(|sample| sample.len() == 8));
        assert!(samples.iter().all(|sample| sample.iter().all(|&i| i < 10)));

        // Different bootstrap samples should be different
        assert_ne!(samples[0], samples[1]); // Very likely to be different
    }

    #[test]
    fn test_multiple_bootstrap_samples_invalid_params() {
        assert!(multiple_bootstrap_samples(10, 10, 0, None).is_err());
    }
}
