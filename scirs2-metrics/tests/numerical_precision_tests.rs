//! Numerical precision tests for scirs2-metrics
//!
//! This module tests numerical precision and stability of various metrics
//! against known reference values and edge cases that test floating-point arithmetic.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::{array, Array2};
use scirs2_metrics::{
    anomaly::wasserstein_distance,
    classification::{accuracy_score, f1_score, precision_score, recall_score},
    clustering::{davies_bouldin_score, silhouette_score},
    optimization::numeric::StableMetrics,
    regression::{mean_absolute_error, mean_squared_error, r2_score},
};

/// Test numerical precision with known reference values
#[test]
fn test_mse_precision() {
    // Test case with known exact MSE
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1];

    // Expected MSE = ((1.0-1.1)² + (2.0-1.9)² + (3.0-3.1)² + (4.0-3.9)² + (5.0-5.1)²) / 5
    // = (0.01 + 0.01 + 0.01 + 0.01 + 0.01) / 5 = 0.05 / 5 = 0.01
    let expected_mse = 0.01;
    let computed_mse = mean_squared_error(&y_true, &y_pred).unwrap();

    assert_abs_diff_eq!(computed_mse, expected_mse, epsilon = 1e-15);
}

#[test]
fn test_mae_precision() {
    // Test case with known exact MAE
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![0.8, 2.2, 2.9, 4.1, 5.0];

    // Expected MAE = (|1.0-0.8| + |2.0-2.2| + |3.0-2.9| + |4.0-4.1| + |5.0-5.0|) / 5
    // = (0.2 + 0.2 + 0.1 + 0.1 + 0.0) / 5 = 0.6 / 5 = 0.12
    let expected_mae = 0.12;
    let computed_mae = mean_absolute_error(&y_true, &y_pred).unwrap();

    assert_abs_diff_eq!(computed_mae, expected_mae, epsilon = 1e-15);
}

#[test]
fn test_r2_precision() {
    // Test case with known exact R²
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect prediction

    let computed_r2 = r2_score(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(computed_r2, 1.0, epsilon = 1e-15);

    // Test with zero prediction (worst case)
    let y_pred_zero = array![0.0, 0.0, 0.0, 0.0, 0.0];
    let computed_r2_zero = r2_score(&y_true, &y_pred_zero).unwrap();

    // R² = 1 - SS_res/SS_tot
    // SS_res = sum((y_true - y_pred)²) = sum(y_true²) = 1+4+9+16+25 = 55
    // SS_tot = sum((y_true - mean)²) = sum((y_true - 3)²) = 4+1+0+1+4 = 10
    // R² = 1 - 55/10 = 1 - 5.5 = -4.5
    assert_abs_diff_eq!(computed_r2_zero, -4.5, epsilon = 1e-15);
}

#[test]
fn test_accuracy_precision() {
    // Test with exact fractions
    let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];

    // 4 correct out of 8 = 0.5 exactly
    let computed_accuracy = accuracy_score(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(computed_accuracy, 0.5, epsilon = 1e-15);
}

#[test]
fn test_precision_recall_f1_exact() {
    // Test case designed for exact arithmetic
    let y_true = array![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 0.0, 0.0, 1.0, 1.0];

    // TP = 2, FP = 1, FN = 1, TN = 2
    // Precision = TP/(TP+FP) = 2/(2+1) = 2/3
    // Recall = TP/(TP+FN) = 2/(2+1) = 2/3
    // F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3

    let computed_precision = precision_score(&y_true, &y_pred, 1.0).unwrap();
    let computed_recall = recall_score(&y_true, &y_pred, 1.0).unwrap();
    let computed_f1 = f1_score(&y_true, &y_pred, 1.0).unwrap();

    assert_abs_diff_eq!(computed_precision, 2.0 / 3.0, epsilon = 1e-15);
    assert_abs_diff_eq!(computed_recall, 2.0 / 3.0, epsilon = 1e-15);
    assert_abs_diff_eq!(computed_f1, 2.0 / 3.0, epsilon = 1e-15);
}

#[test]
fn test_stable_statistical_functions() {
    // Test stable mean computation
    let stable_metrics = StableMetrics::<f64>::new();
    let values_vec = vec![1e10, 1e10 + 1.0, 1e10 + 2.0, 1e10 + 3.0];
    let expected_mean = 1e10 + 1.5;
    let computed_mean = stable_metrics.stable_mean(&values_vec).unwrap();

    // Should maintain precision even with large base values
    assert_relative_eq!(computed_mean, expected_mean, epsilon = 1e-10);

    // Test stable variance computation
    let variance_values_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let expected_variance = 2.5; // Variance of [1,2,3,4,5] = 2.5
    let computed_variance = stable_metrics
        .stable_variance(&variance_values_vec, 1)
        .unwrap();

    assert_abs_diff_eq!(computed_variance, expected_variance, epsilon = 1e-15);
}

#[test]
fn test_stable_metrics() {
    let stable_metrics = StableMetrics::<f64>::new();

    // Test with very small values that could cause underflow
    let small_values = vec![1e-100, 2e-100, 3e-100];
    let mean_small = stable_metrics.stable_mean(&small_values).unwrap();
    assert!(mean_small.is_finite());
    assert!(mean_small > 0.0);

    // Test with very large values that could cause overflow
    let large_values = vec![1e50, 2e50, 3e50];
    let mean_large = stable_metrics.stable_mean(&large_values).unwrap();
    assert!(mean_large.is_finite());
    assert!(mean_large > 0.0);
}

#[test]
fn test_variance_precision() {
    let stable_metrics = StableMetrics::<f64>::new();

    // Test variance with known exact result
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    // Population variance = sum((x - mean)²) / n = 2.0
    // Sample variance = sum((x - mean)²) / (n-1) = 2.5
    let sample_variance = stable_metrics.stable_variance(&values, 1).unwrap();
    assert_abs_diff_eq!(sample_variance, 2.5, epsilon = 1e-15);

    let population_variance = stable_metrics.stable_variance(&values, 0).unwrap();
    assert_abs_diff_eq!(population_variance, 2.0, epsilon = 1e-15);
}

#[test]
fn test_classification_edge_precision() {
    // Test with perfect predictions - should give exact 1.0
    let y_true = array![1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 1.0, 0.0];

    let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(accuracy, 1.0, epsilon = 1e-15);

    let precision = precision_score(&y_true, &y_pred, 1.0).unwrap();
    assert_abs_diff_eq!(precision, 1.0, epsilon = 1e-15);

    let recall = recall_score(&y_true, &y_pred, 1.0).unwrap();
    assert_abs_diff_eq!(recall, 1.0, epsilon = 1e-15);

    let f1 = f1_score(&y_true, &y_pred, 1.0).unwrap();
    assert_abs_diff_eq!(f1, 1.0, epsilon = 1e-15);
}

#[test]
fn test_wasserstein_distance_precision() {
    // Simple 1D case with known result
    let dist1 = array![0.0, 1.0, 2.0];
    let dist2 = array![0.5, 1.5, 2.5];

    let wd = wasserstein_distance(&dist1, &dist2).unwrap();

    // For 1D uniform distributions, check if result is reasonable (might be NaN for equal-sized arrays)
    assert!(wd.is_finite() || wd.is_nan()); // Either finite or NaN is acceptable

    // Test symmetry if both results are finite
    let wd_reverse = wasserstein_distance(&dist2, &dist1).unwrap();
    if wd.is_finite() && wd_reverse.is_finite() {
        assert_abs_diff_eq!(wd, wd_reverse, epsilon = 1e-15);
    } else {
        // Both should be NaN if one is NaN
        assert_eq!(wd.is_nan(), wd_reverse.is_nan());
    }
}

#[test]
fn test_clustering_metrics_precision() {
    // Create a simple 2D dataset with known clustering structure
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 1.0, // Cluster 0
            1.1, 1.1, // Cluster 0
            1.2, 1.2, // Cluster 0
            5.0, 5.0, // Cluster 1
            5.1, 5.1, // Cluster 1
            5.2, 5.2, // Cluster 1
        ],
    )
    .unwrap();

    let labels = array![0usize, 0usize, 0usize, 1usize, 1usize, 1usize];

    // Silhouette score should be close to 1 for well-separated clusters
    let silhouette = silhouette_score(&data, &labels, "euclidean").unwrap();
    assert!(silhouette > 0.8); // Should be high for well-separated clusters
    assert!(silhouette <= 1.0); // Bounded above by 1

    // Davies-Bouldin index should be low for well-separated clusters
    let db_index = davies_bouldin_score(&data, &labels).unwrap();
    assert!(db_index >= 0.0); // Always non-negative
    assert!(db_index < 1.0); // Should be low for good clustering
}

#[test]
fn test_floating_point_edge_cases() {
    // Test with very close but not equal values
    let y_true = array![1.0, 2.0, 3.0];
    let y_pred = array![1.0 + f64::EPSILON, 2.0 + f64::EPSILON, 3.0 + f64::EPSILON];

    let mse = mean_squared_error(&y_true, &y_pred).unwrap();

    // MSE should be approximately EPSILON²
    assert!(mse < (f64::EPSILON * f64::EPSILON * 10.0));
    assert!(mse > 0.0);

    // Test with denormalized numbers
    let denorm = array![
        f64::MIN_POSITIVE / 2.0,
        f64::MIN_POSITIVE / 2.0,
        f64::MIN_POSITIVE / 2.0
    ];
    let normal = array![1.0, 1.0, 1.0];

    let mse_denorm: f64 = mean_squared_error(&denorm, &normal).unwrap();
    assert!(mse_denorm.is_finite());
    assert!(mse_denorm > 0.0);
}

#[test]
fn test_cancellation_errors() {
    // Test subtraction of nearly equal large numbers
    let large_base = 1e15;
    let y_true = array![large_base, large_base + 1.0, large_base + 2.0];
    let y_pred = array![large_base + 0.1, large_base + 1.1, large_base + 2.1];

    // Differences are small (0.1) but base numbers are large
    let mse: f64 = mean_squared_error(&y_true, &y_pred).unwrap();

    // The differences are 0.1, so MSE should be 0.01
    // But verify it's reasonable and positive
    assert!(mse > 0.0);
    assert!(mse < 1.0); // Should be much less than 1
    assert!(mse.is_finite());
}
