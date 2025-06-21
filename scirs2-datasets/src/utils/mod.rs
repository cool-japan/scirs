//! Utility functions and data structures for datasets
//!
//! This module provides a comprehensive collection of utilities for dataset manipulation,
//! including data serialization, dataset structures, splitting, sampling, balancing,
//! scaling, feature engineering, and trait extensions.

// Import all submodules
pub mod balancing;
pub mod dataset;
pub mod extensions;
pub mod feature_engineering;
pub mod sampling;
pub mod scaling;
pub mod serialization;
pub mod splitting;

// Re-export main types and functions for backward compatibility

// Dataset and serialization
pub use dataset::Dataset;
pub use serialization::*;

// Data splitting
pub use splitting::{
    k_fold_split, stratified_k_fold_split, time_series_split, train_test_split,
    CrossValidationFolds,
};

// Data sampling
pub use sampling::{
    bootstrap_sample, importance_sample, multiple_bootstrap_samples, random_sample,
    stratified_sample,
};

// Data balancing
pub use balancing::{
    create_balanced_dataset, generate_synthetic_samples, random_oversample, random_undersample,
    BalancingStrategy,
};

// Data scaling and normalization
pub use scaling::{min_max_scale, normalize, robust_scale, StatsExt};

// Feature engineering
pub use feature_engineering::{
    create_binned_features, polynomial_features, statistical_features, BinningStrategy,
};

// Trait extensions
// pub use extensions::*;

// Type aliases for convenience

/// Convenience alias for ndarray 1D array
pub type Array1<T> = ndarray::Array1<T>;
/// Convenience alias for ndarray 2D array
pub type Array2<T> = ndarray::Array2<T>;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_module_integration() {
        // Test that all major functionality is accessible through the module
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let target = ndarray::Array1::from(vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        // Test dataset creation
        let dataset = Dataset::new(data.clone(), Some(target.clone()));
        assert_eq!(dataset.n_samples(), 6);
        assert_eq!(dataset.n_features(), 2);

        // Test data splitting
        let (train, test) = train_test_split(&dataset, 0.3, Some(42)).unwrap();
        assert_eq!(train.n_samples() + test.n_samples(), 6);

        // Test sampling
        let indices = random_sample(6, 3, false, Some(42)).unwrap();
        assert_eq!(indices.len(), 3);

        // Test balancing
        let (balanced_data, _balanced_targets) =
            random_oversample(&data, &target, Some(42)).unwrap();
        assert!(balanced_data.nrows() > data.nrows()); // Should have more samples after oversampling

        // Test scaling
        let mut scaled_data = data.clone();
        min_max_scale(&mut scaled_data, (0.0, 1.0));
        assert!(scaled_data.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // Test feature engineering
        let poly_features = polynomial_features(&data, 2, true).unwrap();
        assert!(poly_features.ncols() > data.ncols()); // Should have more features
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that the old API still works after refactoring
        use crate::utils::*;

        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let targets = ndarray::Array1::from(vec![0.0, 0.0, 1.0, 1.0]);

        // These should all work exactly as they did before refactoring
        let dataset = Dataset::new(data.clone(), Some(targets.clone()));
        let folds = k_fold_split(4, 2, false, Some(42)).unwrap();
        let sample_indices = stratified_sample(&targets, 2, Some(42)).unwrap();
        let (bal_data, _bal_targets) = create_balanced_dataset(
            &data,
            &targets,
            BalancingStrategy::RandomOversample,
            Some(42),
        )
        .unwrap();

        assert_eq!(dataset.n_samples(), 4);
        assert_eq!(folds.len(), 2);
        assert_eq!(sample_indices.len(), 2);
        assert!(bal_data.nrows() >= data.nrows());
    }

    #[test]
    fn test_cross_validation_compatibility() {
        // Test cross-validation functionality that spans multiple modules
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let targets = ndarray::Array1::from((0..10).map(|x| (x % 3) as f64).collect::<Vec<_>>());

        let dataset = Dataset::new(data, Some(targets.clone()));

        // Test k-fold splitting
        let folds = k_fold_split(dataset.n_samples(), 5, true, Some(42)).unwrap();
        assert_eq!(folds.len(), 5);

        // Test stratified splitting
        let stratified_folds = stratified_k_fold_split(&targets, 3, true, Some(42)).unwrap();
        assert_eq!(stratified_folds.len(), 3);

        // Test time series splitting
        let ts_folds = time_series_split(dataset.n_samples(), 3, 2, 1).unwrap();
        assert_eq!(ts_folds.len(), 3);
    }
}
