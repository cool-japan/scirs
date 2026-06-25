//! Tests for the ML property-based validation utilities.
//!
//! Each test verifies that validators correctly accept valid inputs
//! and reject invalid inputs.

use scirs2_stability_tests::data_generators;
use scirs2_stability_tests::ml_properties::{
    classification, clustering, dim_reduction, metrics, regression, trees,
};

// ───────────────────── Clustering Properties ─────────────────────

#[test]
fn test_partition_valid() {
    assert!(clustering::validate_partition(&[0, 1, 2, 0, 1], 3));
    assert!(clustering::validate_partition(&[0, 0, 0], 1));
    assert!(clustering::validate_partition(&[], 0));
}

#[test]
fn test_partition_invalid() {
    // Label 3 out of range for 3 clusters (valid labels: 0,1,2)
    assert!(!clustering::validate_partition(&[0, 1, 3], 3));
    // Non-empty labels with 0 clusters
    assert!(!clustering::validate_partition(&[0], 0));
}

#[test]
fn test_non_empty_clusters_valid() {
    assert!(clustering::validate_non_empty_clusters(&[0, 1, 2], 3));
    assert!(clustering::validate_non_empty_clusters(&[0, 1, 0, 1], 2));
}

#[test]
fn test_non_empty_clusters_invalid() {
    // Cluster 2 is empty
    assert!(!clustering::validate_non_empty_clusters(&[0, 0, 1, 1], 3));
}

#[test]
fn test_elbow_property() {
    // Inertia decreasing: valid
    let inertias = vec![(1, 100.0), (2, 60.0), (3, 40.0), (4, 35.0)];
    assert!(clustering::validate_elbow_property(&inertias, 1e-10));

    // Inertia increasing: invalid
    let bad_inertias = vec![(1, 40.0), (2, 60.0)];
    assert!(!clustering::validate_elbow_property(&bad_inertias, 1e-10));
}

#[test]
fn test_determinism() {
    assert!(clustering::validate_determinism(&[0, 1, 2], &[0, 1, 2]));
    assert!(!clustering::validate_determinism(&[0, 1, 2], &[0, 1, 0]));
    assert!(!clustering::validate_determinism(&[0, 1], &[0, 1, 2]));
}

#[test]
fn test_contiguous_labels() {
    assert!(clustering::validate_contiguous_labels(&[0, 1, 2, 0]));
    assert!(clustering::validate_contiguous_labels(&[]));
    // Missing label 1
    assert!(!clustering::validate_contiguous_labels(&[0, 2, 2]));
}

#[test]
fn test_cluster_separation_ratio() {
    let centroids = &[[0.0, 0.0], [10.0, 10.0]];
    let (points, labels) = data_generators::clustered_data(centroids, 20);
    let ratio = clustering::compute_cluster_separation_ratio(&points, &labels);
    assert!(ratio.is_some());
    // Well-separated clusters should have ratio < 1
    assert!(ratio.expect("ratio should be Some") < 1.0);
}

// ───────────────────── Regression Properties ─────────────────────

#[test]
fn test_residual_sum_zero() {
    let residuals = vec![0.1, -0.05, -0.05, 0.02, -0.02];
    assert!(regression::validate_residual_sum_zero(&residuals, 1e-10));

    let bad_residuals = vec![1.0, 1.0, 1.0];
    assert!(!regression::validate_residual_sum_zero(&bad_residuals, 0.1));
}

#[test]
fn test_r_squared_range_valid() {
    assert!(regression::validate_r_squared_range(0.0));
    assert!(regression::validate_r_squared_range(0.5));
    assert!(regression::validate_r_squared_range(1.0));
}

#[test]
fn test_r_squared_range_invalid() {
    assert!(!regression::validate_r_squared_range(1.5));
    assert!(!regression::validate_r_squared_range(-0.5));
}

#[test]
fn test_interpolation() {
    let y_true = vec![1.0, 2.0, 3.0];
    let y_pred = vec![1.0, 2.0, 3.0];
    assert!(regression::validate_interpolation(&y_true, &y_pred, 1e-12));

    let y_pred_bad = vec![1.0, 2.5, 3.0];
    assert!(!regression::validate_interpolation(
        &y_true,
        &y_pred_bad,
        1e-2
    ));
}

#[test]
fn test_monotone_r2() {
    let r2s = vec![(1, 0.3), (2, 0.5), (3, 0.7)];
    assert!(regression::validate_monotone_r2(&r2s, 1e-10));

    let bad_r2s = vec![(1, 0.7), (2, 0.3)];
    assert!(!regression::validate_monotone_r2(&bad_r2s, 1e-10));
}

#[test]
fn test_compute_r_squared() {
    // Perfect fit
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let r2 = regression::compute_r_squared(&y, &y);
    assert!(r2.is_some());
    assert!((r2.expect("r2 should be Some") - 1.0).abs() < 1e-12);

    // Constant prediction = mean
    let y_true = vec![1.0, 2.0, 3.0, 4.0];
    let mean = 2.5;
    let y_pred = vec![mean; 4];
    let r2 = regression::compute_r_squared(&y_true, &y_pred);
    assert!(r2.is_some());
    assert!(r2.expect("r2 should be Some").abs() < 1e-12); // R^2 = 0
}

#[test]
fn test_linear_scaling() {
    let orig = vec![1.0, 2.0, 3.0];
    let scaled = vec![2.0, 4.0, 6.0];
    assert!(regression::validate_linear_scaling(
        &orig, &scaled, 2.0, 1e-12
    ));
    assert!(!regression::validate_linear_scaling(
        &orig, &scaled, 3.0, 1e-2
    ));
}

// ───────────────────── Classification Properties ─────────────────────

#[test]
fn test_label_set_valid() {
    assert!(classification::validate_label_set(
        &[0, 1, 2],
        &[0, 1, 2, 3]
    ));
}

#[test]
fn test_label_set_invalid() {
    assert!(!classification::validate_label_set(&[0, 1, 5], &[0, 1, 2]));
}

#[test]
fn test_accuracy_range() {
    assert!(classification::validate_accuracy_range(0.0));
    assert!(classification::validate_accuracy_range(1.0));
    assert!(classification::validate_accuracy_range(0.75));
    assert!(!classification::validate_accuracy_range(1.5));
    assert!(!classification::validate_accuracy_range(-0.1));
}

#[test]
fn test_compute_accuracy() {
    let acc = classification::compute_accuracy(&[0, 1, 2, 1], &[0, 1, 2, 1]);
    assert!(acc.is_some());
    assert!((acc.expect("acc should be Some") - 1.0).abs() < 1e-12);

    let acc2 = classification::compute_accuracy(&[0, 1, 2, 1], &[0, 0, 0, 0]);
    assert!(acc2.is_some());
    assert!((acc2.expect("acc2 should be Some") - 0.25).abs() < 1e-12);
}

#[test]
fn test_confusion_matrix_row_sums() {
    // 2x2 confusion matrix
    //   pred 0  pred 1
    // true 0:  3    1    => 4 samples of class 0
    // true 1:  2    4    => 6 samples of class 1
    let cm = vec![vec![3, 1], vec![2, 4]];
    let true_labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
    assert!(classification::validate_confusion_matrix(&cm, &true_labels));

    // Bad row sum
    let cm_bad = vec![vec![3, 2], vec![2, 4]];
    assert!(!classification::validate_confusion_matrix(
        &cm_bad,
        &true_labels
    ));
}

#[test]
fn test_probability_simplex_valid() {
    let probs = vec![vec![0.3, 0.7], vec![0.5, 0.5], vec![0.0, 1.0]];
    assert!(classification::validate_probability_simplex(&probs, 1e-10));
}

#[test]
fn test_probability_simplex_invalid() {
    // Sums to 0.8 not 1.0
    let probs = vec![vec![0.3, 0.5]];
    assert!(!classification::validate_probability_simplex(&probs, 1e-10));

    // Negative probability
    let probs_neg = vec![vec![-0.1, 1.1]];
    assert!(!classification::validate_probability_simplex(
        &probs_neg, 0.01
    ));
}

#[test]
fn test_precision_recall_f1() {
    let y_true = vec![true, true, false, false, true];
    let y_pred = vec![true, true, false, true, false];

    let result = classification::compute_precision_recall_f1(&y_true, &y_pred);
    assert!(result.is_some());
    let (p, r, f1) = result.expect("result should be Some");

    // TP=2, FP=1, FN=1 => P=2/3, R=2/3, F1=2/3
    assert!((p - 2.0 / 3.0).abs() < 1e-12);
    assert!((r - 2.0 / 3.0).abs() < 1e-12);
    assert!((f1 - 2.0 / 3.0).abs() < 1e-12);
}

// ───────────────────── Dimensionality Reduction ─────────────────────

#[test]
fn test_output_shape() {
    let output = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    assert!(dim_reduction::validate_output_shape(&output, 3, 2));
    assert!(!dim_reduction::validate_output_shape(&output, 3, 3));
    assert!(!dim_reduction::validate_output_shape(&output, 2, 2));
}

#[test]
fn test_orthogonal_components_identity() {
    // Standard basis vectors are orthogonal
    let components = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    assert!(dim_reduction::validate_orthogonal_components(
        &components,
        1e-12
    ));

    // Non-orthogonal
    let bad = vec![vec![1.0, 1.0], vec![1.0, 0.0]];
    assert!(!dim_reduction::validate_orthogonal_components(&bad, 1e-12));
}

#[test]
fn test_unit_components() {
    let components = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    assert!(dim_reduction::validate_unit_components(&components, 1e-12));

    let bad = vec![vec![2.0, 0.0]];
    assert!(!dim_reduction::validate_unit_components(&bad, 1e-12));
}

#[test]
fn test_explained_variance_valid() {
    assert!(dim_reduction::validate_explained_variance(
        &[0.6, 0.3, 0.1],
        1e-10
    ));
    assert!(dim_reduction::validate_explained_variance(
        &[0.5, 0.3],
        1e-10
    ));
}

#[test]
fn test_explained_variance_invalid() {
    // Sum > 1
    assert!(!dim_reduction::validate_explained_variance(
        &[0.6, 0.5],
        1e-10
    ));
    // Negative ratio
    assert!(!dim_reduction::validate_explained_variance(
        &[-0.1, 0.5],
        1e-10
    ));
}

#[test]
fn test_distance_correlation() {
    // Perfectly correlated distances
    let d_high = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let d_low = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // 2x scaling
    let corr = dim_reduction::validate_distance_correlation(&d_high, &d_low);
    assert!(corr.is_some());
    assert!((corr.expect("corr should be Some") - 1.0).abs() < 1e-12);
}

#[test]
fn test_component_count() {
    assert!(dim_reduction::validate_component_count(100, 50, 50));
    assert!(dim_reduction::validate_component_count(100, 50, 10));
    assert!(!dim_reduction::validate_component_count(100, 50, 51));
    assert!(!dim_reduction::validate_component_count(10, 50, 11));
}

// ───────────────────── Tree Properties ─────────────────────

#[test]
fn test_max_depth_constraint() {
    assert!(trees::validate_max_depth_constraint(2, 1));
    assert!(trees::validate_max_depth_constraint(4, 2));
    assert!(!trees::validate_max_depth_constraint(5, 2)); // 2^2=4, 5>4
    assert!(trees::validate_max_depth_constraint(1, 0)); // 2^0=1
}

#[test]
fn test_feature_importance_sum() {
    assert!(trees::validate_feature_importance_sum(
        &[0.4, 0.3, 0.2, 0.1],
        1e-10
    ));
    assert!(!trees::validate_feature_importance_sum(&[0.5, 0.6], 1e-2));
}

#[test]
fn test_ensemble_monotone() {
    let accs = vec![(10, 0.85), (20, 0.90), (50, 0.92)];
    assert!(trees::validate_ensemble_monotone(&accs, 1e-10));

    let bad_accs = vec![(10, 0.90), (20, 0.80)];
    assert!(!trees::validate_ensemble_monotone(&bad_accs, 1e-10));
}

// ───────────────────── Metric Properties ─────────────────────

#[test]
fn test_metric_range_bounds() {
    assert!(metrics::validate_metric_range(0.0));
    assert!(metrics::validate_metric_range(0.5));
    assert!(metrics::validate_metric_range(1.0));
    assert!(!metrics::validate_metric_range(1.1));
    assert!(!metrics::validate_metric_range(-0.1));
}

#[test]
fn test_f1_formula_consistent() {
    let p = 0.8;
    let r = 0.6;
    let expected_f1 = 2.0 * p * r / (p + r);
    assert!(metrics::validate_f1_formula(p, r, expected_f1, 1e-12));
    assert!(!metrics::validate_f1_formula(p, r, 0.5, 1e-2));
}

#[test]
fn test_auc_range() {
    assert!(metrics::validate_auc_range(0.0));
    assert!(metrics::validate_auc_range(0.5));
    assert!(metrics::validate_auc_range(1.0));
    assert!(!metrics::validate_auc_range(1.1));
}

#[test]
fn test_silhouette_range() {
    assert!(metrics::validate_silhouette_range(-1.0));
    assert!(metrics::validate_silhouette_range(0.0));
    assert!(metrics::validate_silhouette_range(1.0));
    assert!(!metrics::validate_silhouette_range(-1.1));
    assert!(!metrics::validate_silhouette_range(1.1));
}

#[test]
fn test_mcc_range() {
    assert!(metrics::validate_mcc_range(-1.0));
    assert!(metrics::validate_mcc_range(0.0));
    assert!(metrics::validate_mcc_range(1.0));
    assert!(!metrics::validate_mcc_range(-1.2));
}

#[test]
fn test_compute_mae() {
    let mae = metrics::compute_mae(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!(mae.is_some());
    assert!(mae.expect("mae should be Some").abs() < 1e-12);

    let mae2 = metrics::compute_mae(&[1.0, 2.0], &[2.0, 3.0]);
    assert!(mae2.is_some());
    assert!((mae2.expect("mae2 should be Some") - 1.0).abs() < 1e-12);
}

#[test]
fn test_compute_mse() {
    let mse = metrics::compute_mse(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
    assert!(mse.is_some());
    assert!(mse.expect("mse should be Some").abs() < 1e-12);

    let mse2 = metrics::compute_mse(&[0.0], &[3.0]);
    assert!(mse2.is_some());
    assert!((mse2.expect("mse2 should be Some") - 9.0).abs() < 1e-12);
}

#[test]
fn test_metric_symmetry() {
    assert!(metrics::validate_metric_symmetry(0.5, 0.5, 1e-12));
    assert!(!metrics::validate_metric_symmetry(0.5, 0.6, 1e-2));
}

// ───────────────────── Data Generator Tests ─────────────────────

#[test]
fn test_data_generator_linear_separable() {
    let (points, labels) = data_generators::linear_separable_2d(25);
    assert_eq!(points.len(), 50);
    assert_eq!(labels.len(), 50);

    // Class 0 should have negative x, class 1 positive x
    for (p, &l) in points.iter().zip(labels.iter()) {
        if l == 0 {
            assert!(p[0] < 0.0, "Class 0 point at x={} should be negative", p[0]);
        } else {
            assert!(p[0] > 0.0, "Class 1 point at x={} should be positive", p[0]);
        }
    }
}

#[test]
fn test_data_generator_concentric_rings() {
    let (points, labels) = data_generators::concentric_rings(30, 3);
    assert_eq!(points.len(), 90);
    assert_eq!(labels.len(), 90);

    // Each ring should have its points at approximately the right radius
    for (p, &l) in points.iter().zip(labels.iter()) {
        let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
        let expected_radius = (l + 1) as f64;
        assert!(
            (r - expected_radius).abs() < 0.01,
            "Ring {l} point at radius {r}, expected {expected_radius}"
        );
    }
}

#[test]
fn test_data_generator_regression_exact() {
    let (x, y) = data_generators::exact_linear_data(100, 2.0, 1.0);
    assert_eq!(x.len(), 100);
    assert_eq!(y.len(), 100);

    for (xi, yi) in x.iter().zip(y.iter()) {
        let expected = 2.0 * xi + 1.0;
        assert!(
            (yi - expected).abs() < 1e-12,
            "y={yi}, expected={expected} for x={xi}"
        );
    }
}

#[test]
fn test_data_generator_clusters_correct_count() {
    let centroids = &[[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]];
    let (points, labels) = data_generators::clustered_data(centroids, 15);
    assert_eq!(points.len(), 45);
    assert_eq!(labels.len(), 45);
    assert!(clustering::validate_partition(&labels, 3));
    assert!(clustering::validate_non_empty_clusters(&labels, 3));
}

#[test]
fn test_data_generator_high_dim() {
    let (points, labels) = data_generators::high_dim_clustered_data(5, 3, 10, 10.0);
    assert_eq!(points.len(), 30);
    assert_eq!(labels.len(), 30);
    assert!(points.iter().all(|p| p.len() == 5));
    assert!(clustering::validate_partition(&labels, 3));
}

#[test]
fn test_data_generator_xor() {
    let (points, labels) = data_generators::xor_pattern(10);
    assert_eq!(points.len(), 40);
    assert_eq!(labels.len(), 40);
    // Should have two classes
    assert!(clustering::validate_partition(&labels, 2));
}

#[test]
fn test_data_generator_polynomial() {
    // y = 1 + 2x + 3x^2
    let (x, y) = data_generators::polynomial_data(50, &[1.0, 2.0, 3.0]);
    assert_eq!(x.len(), 50);
    for (xi, yi) in x.iter().zip(y.iter()) {
        let expected = 1.0 + 2.0 * xi + 3.0 * xi * xi;
        assert!(
            (yi - expected).abs() < 1e-10,
            "y={yi}, expected={expected} for x={xi}"
        );
    }
}

#[test]
fn test_structured_correlation() {
    let matrix = data_generators::structured_correlation(&[2, 3], 0.8);
    assert_eq!(matrix.len(), 5);
    assert!(matrix.iter().all(|row| row.len() == 5));

    // Diagonal should be 1
    for (i, row) in matrix.iter().enumerate() {
        assert!((row[i] - 1.0).abs() < 1e-12);
    }

    // Within block 0 (indices 0,1)
    assert!((matrix[0][1] - 0.8).abs() < 1e-12);

    // Across blocks (index 0 with index 2)
    assert!((matrix[0][2]).abs() < 1e-12);
}
