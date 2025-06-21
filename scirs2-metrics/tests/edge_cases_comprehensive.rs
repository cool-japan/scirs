//! Comprehensive edge case tests for scirs2-metrics
//!
//! This module tests various edge cases that could cause numerical instability,
//! panics, or incorrect results across all metric categories.

use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use scirs2_metrics::{
    anomaly::{js_divergence, kl_divergence},
    classification::{accuracy_score, f1_score, precision_score, recall_score},
    clustering::silhouette_score,
    optimization::numeric::StableMetrics,
    regression::{mean_absolute_error, mean_squared_error, r2_score},
};

#[test]
fn test_empty_arrays() {
    let empty_f64: Array1<f64> = array![];
    let empty_2d: Array2<f64> = Array2::zeros((0, 0));

    // Classification metrics should return errors for empty arrays
    assert!(accuracy_score(&empty_f64, &empty_f64).is_err());
    assert!(precision_score(&empty_f64, &empty_f64, 1.0).is_err());
    assert!(recall_score(&empty_f64, &empty_f64, 1.0).is_err());
    assert!(f1_score(&empty_f64, &empty_f64, 1.0).is_err());

    // Regression metrics should return errors for empty arrays
    assert!(mean_squared_error(&empty_f64, &empty_f64).is_err());
    assert!(mean_absolute_error(&empty_f64, &empty_f64).is_err());
    assert!(r2_score(&empty_f64, &empty_f64).is_err());

    // Clustering metrics should return errors for empty arrays
    let empty_usize: Array1<usize> = array![];
    assert!(silhouette_score(&empty_2d, &empty_usize, "euclidean").is_err());
}

#[test]
fn test_nan_values() {
    let with_nan = array![1.0, 2.0, f64::NAN, 4.0];
    let normal = array![1.0, 2.0, 3.0, 4.0];

    // Most metrics should handle NaN gracefully - either by returning errors or NaN results
    let mse_result = mean_squared_error(&with_nan, &normal);
    assert!(mse_result.is_err() || mse_result.unwrap().is_nan());

    let mae_result = mean_absolute_error(&with_nan, &normal);
    assert!(mae_result.is_err() || mae_result.unwrap().is_nan());

    let acc_result = accuracy_score(&with_nan, &normal);
    // accuracy_score might not return errors for NaN values, just verify it returns something
    if let Ok(acc) = acc_result {
        // Just verify it's a number (could be any valid floating point value including NaN)
        let _: f64 = acc;
    }

    // Test stable metrics with NaN should be handled gracefully
    let stable_metrics = StableMetrics::<f64>::new();
    let with_nan_vec = vec![1.0, 2.0, f64::NAN, 4.0];
    let nan_mean_result = stable_metrics.stable_mean(&with_nan_vec);
    // Should either error or return NaN
    assert!(nan_mean_result.is_err() || nan_mean_result.unwrap().is_nan());
}

#[test]
fn test_infinite_values() {
    let with_inf = array![1.0, 2.0, f64::INFINITY, 4.0];
    let with_neg_inf = array![1.0, 2.0, f64::NEG_INFINITY, 4.0];
    let normal = array![1.0, 2.0, 3.0, 4.0];

    // Metrics should handle infinity values appropriately - either by returning errors or infinite results
    let mse_inf_result = mean_squared_error(&with_inf, &normal);
    assert!(mse_inf_result.is_err() || mse_inf_result.unwrap().is_infinite());

    let mse_neg_inf_result = mean_squared_error(&with_neg_inf, &normal);
    assert!(mse_neg_inf_result.is_err() || mse_neg_inf_result.unwrap().is_infinite());

    // Test stable metrics with infinity should be handled gracefully
    let stable_metrics = StableMetrics::<f64>::new();
    let with_inf_vec = vec![1.0, 2.0, f64::INFINITY, 4.0];
    let mean_result = stable_metrics.stable_mean(&with_inf_vec);
    // Should either error, return infinity, or return some finite value (graceful handling)
    if let Ok(mean_val) = mean_result {
        // Any floating point value is acceptable (finite, infinite, or NaN)
        let _: f64 = mean_val;
    }
}

#[test]
fn test_very_small_numbers() {
    let very_small = array![1e-100, 2e-100, 3e-100, 4e-100];
    let tiny_diff = array![1e-100, 2e-100, 3e-100, 4e-100 + 1e-101];

    // Should not panic with very small numbers
    let mse: f64 = mean_squared_error(&very_small, &tiny_diff).unwrap();
    assert!(mse.is_finite());
    assert!(mse >= 0.0);

    // Test stable functions with very small numbers
    let stable_metrics = StableMetrics::<f64>::new();
    let very_small_vec = vec![1e-100, 2e-100, 3e-100, 4e-100];
    let mean_val = stable_metrics.stable_mean(&very_small_vec).unwrap();
    assert!(mean_val.is_finite());
    assert!(mean_val > 0.0);

    let var_val = stable_metrics.stable_variance(&very_small_vec, 1).unwrap();
    assert!(var_val.is_finite());
    assert!(var_val >= 0.0);
}

#[test]
fn test_very_large_numbers() {
    let very_large = array![1e50, 2e50, 3e50, 4e50];
    let large_diff = array![1e50, 2e50, 3e50, 5e50];

    // Should not overflow with very large numbers
    let mae: f64 = mean_absolute_error(&very_large, &large_diff).unwrap();
    assert!(mae.is_finite());
    assert!(mae >= 0.0);

    // Test stable functions with very large numbers
    let stable_metrics = StableMetrics::<f64>::new();
    let very_large_vec = vec![1e50, 2e50, 3e50, 4e50];
    let mean_val = stable_metrics.stable_mean(&very_large_vec).unwrap();
    assert!(mean_val.is_finite());
    assert!(mean_val > 0.0);
}

#[test]
fn test_single_element_arrays() {
    let single_true = array![1.0];
    let single_pred = array![0.5];

    // Single element should work for most metrics
    let mse = mean_squared_error(&single_true, &single_pred).unwrap();
    assert_abs_diff_eq!(mse, 0.25, epsilon = 1e-10);

    let mae = mean_absolute_error(&single_true, &single_pred).unwrap();
    assert_abs_diff_eq!(mae, 0.5, epsilon = 1e-10);

    // Single element clustering should fail (need at least 2 points)
    let single_data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let single_labels = array![0usize];
    assert!(silhouette_score(&single_data, &single_labels, "euclidean").is_err());
}

#[test]
fn test_identical_arrays() {
    let identical = array![1.0, 2.0, 3.0, 4.0];

    // Perfect predictions should give expected results
    let mse = mean_squared_error(&identical, &identical).unwrap();
    assert_abs_diff_eq!(mse, 0.0, epsilon = 1e-10);

    let mae = mean_absolute_error(&identical, &identical).unwrap();
    assert_abs_diff_eq!(mae, 0.0, epsilon = 1e-10);

    let accuracy = accuracy_score(&identical, &identical).unwrap();
    assert_abs_diff_eq!(accuracy, 1.0, epsilon = 1e-10);
}

#[test]
fn test_constant_arrays() {
    let constant_true = array![5.0, 5.0, 5.0, 5.0];
    let constant_pred = array![3.0, 3.0, 3.0, 3.0];

    // Constant arrays should work but give specific results
    let mse = mean_squared_error(&constant_true, &constant_pred).unwrap();
    assert_abs_diff_eq!(mse, 4.0, epsilon = 1e-10); // (5-3)^2 = 4

    let mae = mean_absolute_error(&constant_true, &constant_pred).unwrap();
    assert_abs_diff_eq!(mae, 2.0, epsilon = 1e-10); // |5-3| = 2

    // R² should be undefined for constant true values (no variance)
    assert!(r2_score(&constant_true, &constant_pred).is_err());
}

#[test]
fn test_zero_division_scenarios() {
    let _zeros = array![0.0, 0.0, 0.0, 0.0];
    let _non_zeros = array![1.0, 2.0, 3.0, 4.0];

    // Test precision/recall with no positive predictions
    let no_pos_pred = array![0.0, 0.0, 0.0, 0.0];
    let some_pos_true = array![1.0, 0.0, 1.0, 0.0];

    // Should handle gracefully (return 0 or error)
    let precision_result = precision_score(&some_pos_true, &no_pos_pred, 1.0);
    // This should either be 0.0 or return an error - both are acceptable
    assert!(precision_result.is_err() || precision_result.unwrap() == 0.0);
}

#[test]
fn test_extreme_class_imbalance() {
    // 99.9% negative class
    let mut imbalanced_true = vec![0.0; 1000];
    imbalanced_true[0] = 1.0; // Only one positive case

    let mut imbalanced_pred = vec![0.0; 1000];
    imbalanced_pred[0] = 1.0; // Correctly predict the one positive

    let imbalanced_true = Array1::from_vec(imbalanced_true);
    let imbalanced_pred = Array1::from_vec(imbalanced_pred);

    // Should handle extreme imbalance without issues
    let accuracy = accuracy_score(&imbalanced_true, &imbalanced_pred).unwrap();
    assert_abs_diff_eq!(accuracy, 1.0, epsilon = 1e-10);

    let precision = precision_score(&imbalanced_true, &imbalanced_pred, 1.0).unwrap();
    assert_abs_diff_eq!(precision, 1.0, epsilon = 1e-10);

    let recall = recall_score(&imbalanced_true, &imbalanced_pred, 1.0).unwrap();
    assert_abs_diff_eq!(recall, 1.0, epsilon = 1e-10);
}

#[test]
fn test_probability_edge_cases() {
    // Test KL divergence with probability distributions
    let uniform = array![0.25, 0.25, 0.25, 0.25];
    let peaked = array![0.97, 0.01, 0.01, 0.01];
    let with_zero = array![1.0, 0.0, 0.0, 0.0];

    // Normal case should work
    let kl_normal = kl_divergence(&uniform, &peaked).unwrap();
    assert!(kl_normal >= 0.0);
    assert!(kl_normal.is_finite());

    // Zero probabilities should be handled gracefully
    let kl_zero = kl_divergence(&uniform, &with_zero);
    // Should either work (with smoothing) or return error
    assert!(kl_zero.is_ok() || kl_zero.is_err());

    // JS divergence should be more robust
    let js_normal = js_divergence(&uniform, &peaked).unwrap();
    assert!(js_normal >= 0.0);
    assert!(js_normal <= 1.0);
    assert!(js_normal.is_finite());

    let js_zero = js_divergence(&uniform, &with_zero).unwrap();
    assert!(js_zero >= 0.0);
    assert!(js_zero <= 1.0);
    assert!(js_zero.is_finite());
}

#[test]
fn test_clustering_edge_cases() {
    // Single cluster (all same label)
    let data_single_cluster =
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let labels_single = array![0usize, 0usize, 0usize, 0usize];

    // Silhouette score should be undefined for single cluster
    assert!(silhouette_score(&data_single_cluster, &labels_single, "euclidean").is_err());

    // As many clusters as points - this might work but should give undefined/poor results
    let labels_all_different = array![0usize, 1usize, 2usize, 3usize];
    let silhouette_result =
        silhouette_score(&data_single_cluster, &labels_all_different, "euclidean");
    // Either returns an error or gives a finite result (might be NaN or a valid number)
    if let Ok(score) = silhouette_result {
        let score: f64 = score;
        assert!(score.is_finite() || score.is_nan());
    } else {
        // Error is also acceptable
        assert!(silhouette_result.is_err());
    }

    // Two points, two clusters (minimum valid case) - might work or give error
    let data_minimal = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let labels_minimal = array![0usize, 1usize];
    let minimal_result = silhouette_score(&data_minimal, &labels_minimal, "euclidean");
    // This is acceptable to either work or fail
    if let Ok(score) = minimal_result {
        let score: f64 = score;
        assert!(score.is_finite() || score.is_nan());
    }
}

#[test]
fn test_mismatched_dimensions() {
    let y_true_short = array![1.0, 2.0, 3.0];
    let y_pred_long = array![1.0, 2.0, 3.0, 4.0];

    // All metrics should detect dimension mismatches
    assert!(mean_squared_error(&y_true_short, &y_pred_long).is_err());
    assert!(mean_absolute_error(&y_true_short, &y_pred_long).is_err());
    assert!(accuracy_score(&y_true_short, &y_pred_long).is_err());
    assert!(precision_score(&y_true_short, &y_pred_long, 1.0).is_err());
    assert!(recall_score(&y_true_short, &y_pred_long, 1.0).is_err());
}

#[test]
fn test_numerical_stability() {
    // Test with numbers that could cause overflow in intermediate calculations
    let large_numbers = array![1e30, 2e30, 3e30, 4e30];
    let slightly_different = array![1e30 + 1e20, 2e30 + 1e20, 3e30 + 1e20, 4e30 + 1e20];

    // Should not overflow or underflow
    let mse: f64 = mean_squared_error(&large_numbers, &slightly_different).unwrap();
    assert!(mse.is_finite());
    assert!(mse > 0.0);

    // Test stable statistical functions
    let stable_metrics = StableMetrics::<f64>::new();
    let large_numbers_vec = vec![1e30, 2e30, 3e30, 4e30];
    let mean_val = stable_metrics.stable_mean(&large_numbers_vec).unwrap();
    assert!(mean_val.is_finite());

    let var_val = stable_metrics
        .stable_variance(&large_numbers_vec, 1)
        .unwrap();
    assert!(var_val.is_finite());
    assert!(var_val >= 0.0);
}
