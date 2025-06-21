//! Reference benchmarks and validation tests against known mathematical results
//!
//! This module provides tests that validate metrics implementations against
//! known reference values and mathematical properties, serving as benchmarks
//! for correctness and compatibility.

use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use scirs2_metrics::{
    classification::{accuracy_score, f1_score, precision_score, recall_score},
    clustering::{davies_bouldin_score, silhouette_score},
    regression::{mean_absolute_error, mean_squared_error, r2_score},
};

/// Test classification metrics against known reference values
/// These values are computed manually and cross-validated with mathematical definitions
#[test]
fn test_classification_reference_values() {
    // Simple binary classification example with known confusion matrix:
    // TP=2, FP=1, FN=1, TN=2
    let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred = array![1.0, 0.0, 0.0, 0.0, 1.0, 1.0];

    // Manual calculations:
    // Accuracy = (TP + TN) / (TP + TN + FP + FN) = (2 + 2) / (2 + 2 + 1 + 1) = 4/6 = 2/3
    let expected_accuracy = 2.0 / 3.0;
    let computed_accuracy = accuracy_score(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(computed_accuracy, expected_accuracy, epsilon = 1e-15);

    // Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3
    let expected_precision = 2.0 / 3.0;
    let computed_precision = precision_score(&y_true, &y_pred, 1.0).unwrap();
    assert_abs_diff_eq!(computed_precision, expected_precision, epsilon = 1e-15);

    // Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3
    let expected_recall = 2.0 / 3.0;
    let computed_recall = recall_score(&y_true, &y_pred, 1.0).unwrap();
    assert_abs_diff_eq!(computed_recall, expected_recall, epsilon = 1e-15);

    // F1 = 2 * (Precision * Recall) / (Precision + Recall) = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
    let expected_f1 = 2.0 / 3.0;
    let computed_f1 = f1_score(&y_true, &y_pred, 1.0).unwrap();
    assert_abs_diff_eq!(computed_f1, expected_f1, epsilon = 1e-15);
}

/// Test regression metrics against known reference values
#[test]
fn test_regression_reference_values() {
    // Simple regression example with known differences
    let y_true = array![1.0, 2.0, 3.0, 4.0];
    let y_pred = array![1.5, 2.5, 3.5, 4.5];

    // All predictions are off by +0.5
    // MSE = mean((y_true - y_pred)^2) = mean((-0.5)^2) = mean(0.25) = 0.25
    let expected_mse = 0.25;
    let computed_mse = mean_squared_error(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(computed_mse, expected_mse, epsilon = 1e-15);

    // MAE = mean(|y_true - y_pred|) = mean(|-0.5|) = 0.5
    let expected_mae = 0.5;
    let computed_mae = mean_absolute_error(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(computed_mae, expected_mae, epsilon = 1e-15);

    // R² calculation:
    // y_true mean = (1+2+3+4)/4 = 2.5
    // SS_tot = sum((y_true - mean)^2) = (1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2
    //        = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
    // SS_res = sum((y_true - y_pred)^2) = 4 * 0.25 = 1.0
    // R² = 1 - SS_res/SS_tot = 1 - 1.0/5.0 = 0.8
    let expected_r2 = 0.8;
    let computed_r2 = r2_score(&y_true, &y_pred).unwrap();
    assert_abs_diff_eq!(computed_r2, expected_r2, epsilon = 1e-15);
}

/// Test clustering metrics against known reference values for well-separated clusters
#[test]
fn test_clustering_reference_values() {
    // Two well-separated 2D clusters
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            // Cluster 0: centered around (0, 0)
            0.0, 0.0, 0.1, 0.1, -0.1, 0.1,
            // Cluster 1: centered around (10, 10) - well separated
            10.0, 10.0, 10.1, 10.1, 9.9, 9.9,
        ],
    )
    .unwrap();

    let labels = array![0usize, 0usize, 0usize, 1usize, 1usize, 1usize];

    // For perfectly separated clusters, silhouette score should be close to 1
    let silhouette = silhouette_score(&data, &labels, "euclidean").unwrap();
    assert!(silhouette > 0.9); // Should be very high
    assert!(silhouette <= 1.0); // Bounded above

    // Davies-Bouldin index should be low for well-separated clusters
    let db_index = davies_bouldin_score(&data, &labels).unwrap();
    assert!(db_index >= 0.0); // Always non-negative
    assert!(db_index < 0.5); // Should be low for good clustering
}

/// Test mathematical properties that should hold for any valid implementation
#[test]
fn test_mathematical_properties() {
    // Test classification metric bounds
    let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.0, 1.0, 1.0, 0.0, 0.0];

    let accuracy = accuracy_score(&y_true, &y_pred).unwrap();
    assert!((0.0..=1.0).contains(&accuracy));

    let precision = precision_score(&y_true, &y_pred, 1.0).unwrap();
    assert!((0.0..=1.0).contains(&precision));

    let recall = recall_score(&y_true, &y_pred, 1.0).unwrap();
    assert!((0.0..=1.0).contains(&recall));

    let f1 = f1_score(&y_true, &y_pred, 1.0).unwrap();
    assert!((0.0..=1.0).contains(&f1));

    // Test regression metric properties
    let y_true_reg = array![1.0, 2.0, 3.0, 4.0];
    let y_pred_reg = array![1.1, 2.1, 3.1, 4.1];

    let mse = mean_squared_error(&y_true_reg, &y_pred_reg).unwrap();
    assert!(mse >= 0.0); // MSE is always non-negative

    let mae = mean_absolute_error(&y_true_reg, &y_pred_reg).unwrap();
    assert!(mae >= 0.0); // MAE is always non-negative

    // Perfect predictions should give MSE = 0
    let mse_perfect = mean_squared_error(&y_true_reg, &y_true_reg).unwrap();
    assert_abs_diff_eq!(mse_perfect, 0.0, epsilon = 1e-15);

    // Perfect predictions should give MAE = 0
    let mae_perfect = mean_absolute_error(&y_true_reg, &y_true_reg).unwrap();
    assert_abs_diff_eq!(mae_perfect, 0.0, epsilon = 1e-15);

    // Perfect predictions should give R² = 1
    let r2_perfect = r2_score(&y_true_reg, &y_true_reg).unwrap();
    assert_abs_diff_eq!(r2_perfect, 1.0, epsilon = 1e-15);
}

/// Test symmetry properties where applicable
#[test]
fn test_symmetry_properties() {
    // MSE and MAE should be symmetric in the sense that swapping y_true and y_pred
    // gives the same result (since they measure absolute differences)
    let a = array![1.0, 2.0, 3.0];
    let b = array![1.5, 2.5, 3.5];

    let mse_ab = mean_squared_error(&a, &b).unwrap();
    let mse_ba = mean_squared_error(&b, &a).unwrap();
    assert_abs_diff_eq!(mse_ab, mse_ba, epsilon = 1e-15);

    let mae_ab = mean_absolute_error(&a, &b).unwrap();
    let mae_ba = mean_absolute_error(&b, &a).unwrap();
    assert_abs_diff_eq!(mae_ab, mae_ba, epsilon = 1e-15);
}

/// Test with known datasets from literature (simplified versions)
#[test]
fn test_known_dataset_results() {
    // Iris-like dataset with perfect linear separability for 2 classes
    let y_true_iris = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let y_pred_iris = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // Perfect classification

    let accuracy_iris = accuracy_score(&y_true_iris, &y_pred_iris).unwrap();
    assert_abs_diff_eq!(accuracy_iris, 1.0, epsilon = 1e-15);

    let precision_iris = precision_score(&y_true_iris, &y_pred_iris, 1.0).unwrap();
    assert_abs_diff_eq!(precision_iris, 1.0, epsilon = 1e-15);

    let recall_iris = recall_score(&y_true_iris, &y_pred_iris, 1.0).unwrap();
    assert_abs_diff_eq!(recall_iris, 1.0, epsilon = 1e-15);

    // Boston housing-like dataset with linear relationship
    let y_true_boston = array![20.0, 25.0, 30.0, 35.0];
    let y_pred_boston = array![19.0, 26.0, 29.0, 36.0]; // Close predictions

    // Expected MSE = ((20-19)² + (25-26)² + (30-29)² + (35-36)²) / 4 = (1+1+1+1)/4 = 1.0
    let expected_mse_boston = 1.0;
    let computed_mse_boston = mean_squared_error(&y_true_boston, &y_pred_boston).unwrap();
    assert_abs_diff_eq!(computed_mse_boston, expected_mse_boston, epsilon = 1e-15);
}

/// Test extreme cases with known results
#[test]
fn test_extreme_cases() {
    // All zeros vs all ones (worst binary classification)
    let y_true_worst = array![0.0, 0.0, 0.0, 0.0];
    let y_pred_worst = array![1.0, 1.0, 1.0, 1.0];

    let accuracy_worst = accuracy_score(&y_true_worst, &y_pred_worst).unwrap();
    assert_abs_diff_eq!(accuracy_worst, 0.0, epsilon = 1e-15);

    // Constant predictions in regression
    let y_true_const = array![1.0, 2.0, 3.0, 4.0];
    let y_pred_const = array![2.5, 2.5, 2.5, 2.5]; // Predicting the mean

    // R² should be 0 when always predicting the mean
    let r2_const = r2_score(&y_true_const, &y_pred_const).unwrap();
    assert_abs_diff_eq!(r2_const, 0.0, epsilon = 1e-15);
}

/// Test consistency across different input sizes
#[test]
fn test_size_consistency() {
    // Small dataset
    let y_true_small = array![1.0, 0.0];
    let y_pred_small = array![1.0, 0.0];
    let acc_small = accuracy_score(&y_true_small, &y_pred_small).unwrap();
    assert_abs_diff_eq!(acc_small, 1.0, epsilon = 1e-15);

    // Medium dataset
    let y_true_medium = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_medium = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let acc_medium = accuracy_score(&y_true_medium, &y_pred_medium).unwrap();
    assert_abs_diff_eq!(acc_medium, 1.0, epsilon = 1e-15);

    // Large dataset (perfect predictions should always give accuracy = 1)
    let y_true_large: Array1<f64> = Array1::ones(100);
    let y_pred_large: Array1<f64> = Array1::ones(100);
    let acc_large = accuracy_score(&y_true_large, &y_pred_large).unwrap();
    assert_abs_diff_eq!(acc_large, 1.0, epsilon = 1e-15);
}
