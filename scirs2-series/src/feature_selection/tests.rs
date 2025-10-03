//! Tests for feature selection methods

use super::*;
use scirs2_core::ndarray::{array, Array2};

#[allow(dead_code)]
fn create_test_data() -> (Array2<f64>, Array1<f64>) {
    // Create synthetic data with known relationships
    let n_samples = 100;
    let n_features = 10;

    let mut features = Array2::zeros((n_samples, n_features));
    let mut target = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let t = i as f64;

        // Feature 0: strongly correlated with target
        features[[i, 0]] = t + (i as f64 * 0.1).sin();

        // Feature 1: weakly correlated with target
        features[[i, 1]] = t * 0.3 + (i as f64 * 0.2).cos();

        // Feature 2: noise
        features[[i, 2]] = (i as f64 * 0.05).sin() * 10.0;

        // Feature 3: constant (low variance)
        features[[i, 3]] = 5.0;

        // Features 4-9: random noise
        for j in 4..n_features {
            features[[i, j]] = (i * j) as f64 * 0.01;
        }

        // Target is mainly based on features 0 and 1
        target[i] = features[[i, 0]] * 2.0 + features[[i, 1]] * 0.5 + (i as f64 * 0.03).sin();
    }

    (features, target)
}

#[test]
#[allow(dead_code)]
fn test_variance_threshold() {
    let (features, _) = create_test_data();

    let result = FilterMethods::variance_threshold(&features, 0.1).unwrap();

    // Feature 3 (constant) should be filtered out
    assert!(!result.selected_features.contains(&3));
    assert!(result.selected_features.len() < features.ncols());
}

#[test]
#[allow(dead_code)]
fn test_correlation_selection() {
    let (features, target) = create_test_data();

    let result = FilterMethods::correlation_selection(&features, &target, 0.3).unwrap();

    // Should select features 0 and 1 which are correlated with target
    assert!(result.selected_features.contains(&0));
    assert!(result.selected_features.contains(&1));
    assert!(result.feature_scores[0] > result.feature_scores[2]);
}

#[test]
#[allow(dead_code)]
fn test_mutual_information_selection() {
    let (features, target) = create_test_data();

    let result =
        FilterMethods::mutual_information_selection(&features, &target, 5, Some(5)).unwrap();

    assert_eq!(result.selected_features.len(), 5);
    assert!(!result.feature_scores.iter().all(|&x| x == 0.0));
}

#[test]
#[allow(dead_code)]
fn test_f_test_selection() {
    let (features, target) = create_test_data();

    let result = FilterMethods::f_test_selection(&features, &target, 0.05).unwrap();

    // Should select some features based on F-test
    assert!(!result.selected_features.is_empty());
    assert!(result.feature_scores[0] > 0.0);
}

#[test]
#[allow(dead_code)]
fn test_autocorrelation_filter() {
    let (features, _) = create_test_data();

    let result = FilterMethods::autocorrelation_filter(&features, 5, 0.1).unwrap();

    assert!(!result.selected_features.is_empty());
    assert!(result.feature_scores.len() == features.ncols());
}

#[test]
#[allow(dead_code)]
fn test_forward_selection() {
    let (features, target) = create_test_data();
    let config = FeatureSelectionConfig {
        n_features: Some(3),
        ..Default::default()
    };

    let result = WrapperMethods::forward_selection(&features, &target, &config).unwrap();

    assert!(result.selected_features.len() <= 3);
    // Should prefer features 0 and 1
    assert!(result.selected_features.contains(&0) || result.selected_features.contains(&1));
}

#[test]
#[allow(dead_code)]
fn test_backward_elimination() {
    let (features, target) = create_test_data();
    let config = FeatureSelectionConfig {
        n_features: Some(5),
        ..Default::default()
    };

    let result = WrapperMethods::backward_elimination(&features, &target, &config).unwrap();

    assert!(result.selected_features.len() >= 5);
}

#[test]
#[allow(dead_code)]
fn test_recursive_feature_elimination() {
    let (features, target) = create_test_data();
    let config = FeatureSelectionConfig {
        n_features: Some(4),
        ..Default::default()
    };

    let result =
        WrapperMethods::recursive_feature_elimination(&features, &target, &config).unwrap();

    assert_eq!(result.selected_features.len(), 4);
}

#[test]
#[allow(dead_code)]
fn test_bidirectional_selection() {
    let (features, target) = create_test_data();
    let config = FeatureSelectionConfig {
        n_features: Some(5),
        max_iterations: 10,
        ..Default::default()
    };

    let result = WrapperMethods::bidirectional_selection(&features, &target, &config).unwrap();

    assert!(result.selected_features.len() <= 5);
}

#[test]
#[allow(dead_code)]
fn test_lasso_selection() {
    let (features, target) = create_test_data();

    let result = EmbeddedMethods::lasso_selection(&features, &target, 0.1, 100).unwrap();

    // LASSO should select some features and zero out others
    assert!(!result.selected_features.is_empty());
    assert!(result.selected_features.len() < features.ncols());
}

#[test]
#[allow(dead_code)]
fn test_ridge_selection() {
    let (features, target) = create_test_data();

    let result = EmbeddedMethods::ridge_selection(&features, &target, 1.0, Some(5)).unwrap();

    assert_eq!(result.selected_features.len(), 5);
}

#[test]
#[allow(dead_code)]
fn test_tree_based_selection() {
    let (features, target) = create_test_data();

    let result = EmbeddedMethods::tree_based_selection(&features, &target, Some(4)).unwrap();

    assert_eq!(result.selected_features.len(), 4);
}

#[test]
#[allow(dead_code)]
fn test_lag_based_selection() {
    let (features, target) = create_test_data();

    let result = TimeSeriesMethods::lag_based_selection(&features, &target, 3, Some(5)).unwrap();

    assert_eq!(result.selected_features.len(), 5);
}

#[test]
#[allow(dead_code)]
fn test_seasonal_importance_selection() {
    let (features, _) = create_test_data();

    let result = TimeSeriesMethods::seasonal_importance_selection(&features, 12, Some(4)).unwrap();

    assert_eq!(result.selected_features.len(), 4);
}

#[test]
#[allow(dead_code)]
fn test_cross_correlation_selection() {
    let (features, target) = create_test_data();

    let result =
        TimeSeriesMethods::cross_correlation_selection(&features, &target, 5, 0.1).unwrap();

    assert!(!result.selected_features.is_empty());
}

#[test]
#[allow(dead_code)]
fn test_granger_causality_selection() {
    let (features, target) = create_test_data();

    let result =
        TimeSeriesMethods::granger_causality_selection(&features, &target, 3, 0.05).unwrap();

    // May or may not select features depending on causality
    assert!(result.feature_scores.len() == features.ncols());
}

#[test]
#[allow(dead_code)]
fn test_auto_select() {
    let (features, target) = create_test_data();
    let config = FeatureSelectionConfig {
        n_features: Some(5),
        ..Default::default()
    };

    let result = FeatureSelector::auto_select(&features, Some(&target), &config).unwrap();

    assert!(result.selected_features.len() <= 5);
    assert_eq!(result.method, "AutoSelect");
}

#[test]
#[allow(dead_code)]
fn test_edge_cases() {
    // Test with insufficient data
    let small_features = Array2::zeros((2, 5));
    let small_target = Array1::zeros(2);

    let result = FilterMethods::correlation_selection(&small_features, &small_target, 0.1);
    assert!(result.is_err());

    // Test with dimension mismatch
    let features = Array2::zeros((10, 5));
    let mismatched_target = Array1::zeros(5);

    let result = FilterMethods::correlation_selection(&features, &mismatched_target, 0.1);
    assert!(result.is_err());
}

#[test]
#[allow(dead_code)]
fn test_feature_selection_result() {
    let result = FeatureSelectionResult {
        selected_features: vec![0, 2, 4],
        feature_scores: array![0.8, 0.2, 0.9, 0.1, 0.7],
        method: "Test".to_string(),
        metadata: HashMap::new(),
    };

    assert_eq!(result.selected_features.len(), 3);
    assert_eq!(result.method, "Test");
    assert_eq!(result.feature_scores.len(), 5);
}
