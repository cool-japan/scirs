use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};
use scirs2_metrics::fairness::{
    consistency_score, demographic_parity_difference, disparate_impact,
    equal_opportunity_difference, equalized_odds_difference,
};

#[test]
fn test_demographic_parity_difference() {
    // Test case 1: Perfect demographic parity
    // Both groups have 2/5 = 40% positive predictions
    let y_pred_1 = array![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let protected_1 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let dp_diff_1 = demographic_parity_difference(&y_pred_1, &protected_1).unwrap();
    assert_abs_diff_eq!(dp_diff_1, 0.0, epsilon = 1e-10);

    // Test case 2: Disparate predictions
    let y_pred_2 = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let protected_2 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Protected group: 80% positive, Unprotected group: 0% positive
    let dp_diff_2 = demographic_parity_difference(&y_pred_2, &protected_2).unwrap();
    assert_abs_diff_eq!(dp_diff_2, 0.8, epsilon = 1e-10);

    // Test case 3: No protected group members
    let y_pred_3 = array![1.0, 0.0, 1.0, 0.0];
    let protected_3 = array![0.0, 0.0, 0.0, 0.0];
    assert!(demographic_parity_difference(&y_pred_3, &protected_3).is_err());

    // Test case 4: No unprotected group members
    let y_pred_4 = array![1.0, 0.0, 1.0, 0.0];
    let protected_4 = array![1.0, 1.0, 1.0, 1.0];
    assert!(demographic_parity_difference(&y_pred_4, &protected_4).is_err());

    // Test case 5: Empty arrays
    let y_pred_5: ndarray::Array1<f64> = array![];
    let protected_5: ndarray::Array1<f64> = array![];
    assert!(demographic_parity_difference(&y_pred_5, &protected_5).is_err());

    // Test case 6: Non-binary predictions (treated as binary)
    let y_pred_6 = array![0.5, 0.7, 0.3, 0.8, 0.1, 0.9, 0.2, 0.4, 0.6, 0.3];
    let protected_6 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Values are thresholded at 0, so all non-zero values are considered positive
    let dp_diff_6 = demographic_parity_difference(&y_pred_6, &protected_6).unwrap();
    assert!((0.0..=1.0).contains(&dp_diff_6));
}

#[test]
fn test_disparate_impact() {
    // Test case 1: Perfect fairness
    // Both groups have 2/5 = 40% positive predictions
    let y_pred_1 = array![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let protected_1 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let di_1 = disparate_impact(&y_pred_1, &protected_1).unwrap();
    assert_abs_diff_eq!(di_1, 1.0, epsilon = 1e-10);

    // Test case 2: Disparate impact
    let y_pred_2 = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let protected_2 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Protected group: 80% positive, Unprotected group: 0% positive
    // Since unprotected rate is 0, and dividing by 0 is infinity, we handle this specially
    let di_2 = disparate_impact(&y_pred_2, &protected_2).unwrap();
    assert!(di_2 > 1.0); // Should be Infinity

    // Test case 3: Inverse disparate impact
    let y_pred_3 = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let protected_3 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Protected group: 0% positive, Unprotected group: 100% positive
    let di_3 = disparate_impact(&y_pred_3, &protected_3).unwrap();
    assert_abs_diff_eq!(di_3, 0.0, epsilon = 1e-10);

    // Test case 4: Zero positives in both groups
    let y_pred_4 = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let protected_4 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Special case: should be 1.0 (equal rates of 0%)
    let di_4 = disparate_impact(&y_pred_4, &protected_4).unwrap();
    assert_abs_diff_eq!(di_4, 1.0, epsilon = 1e-10);
}

#[test]
fn test_equalized_odds_difference() {
    // Test case 1: Perfect equalized odds
    let y_true_1 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_1 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected_1 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Both groups have perfect FPR and TPR
    let eod_1 = equalized_odds_difference(&y_true_1, &y_pred_1, &protected_1).unwrap();
    assert_abs_diff_eq!(eod_1, 0.0, epsilon = 1e-10);

    // Test case 2: Different error rates
    let y_true_2 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_2 = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected_2 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Protected: FPR=1/3, TPR=0/2
    // Unprotected: FPR=0/3, TPR=2/2
    // Max diff: max(|1/3 - 0/3|, |0/2 - 2/2|) = max(1/3, 1) = 1
    let eod_2 = equalized_odds_difference(&y_true_2, &y_pred_2, &protected_2).unwrap();
    assert_abs_diff_eq!(eod_2, 1.0, epsilon = 1e-10);

    // Test case 3: Only false positives differ
    let y_true_3 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_3 = array![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected_3 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Protected: FPR=1/3, TPR=2/2
    // Unprotected: FPR=0/3, TPR=2/2
    // Max diff: max(|1/3 - 0/3|, |2/2 - 2/2|) = max(1/3, 0) = 1/3
    let eod_3 = equalized_odds_difference(&y_true_3, &y_pred_3, &protected_3).unwrap();
    assert_abs_diff_eq!(eod_3, 1.0 / 3.0, epsilon = 1e-10);

    // Test case 4: Only true positives differ
    let y_true_4 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_4 = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected_4 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Protected: FPR=0/3, TPR=0/2
    // Unprotected: FPR=0/3, TPR=2/2
    // Max diff: max(|0/3 - 0/3|, |0/2 - 2/2|) = max(0, 1) = 1
    let eod_4 = equalized_odds_difference(&y_true_4, &y_pred_4, &protected_4).unwrap();
    assert_abs_diff_eq!(eod_4, 1.0, epsilon = 1e-10);
}

#[test]
fn test_equal_opportunity_difference() {
    // Test case 1: Perfect equal opportunity
    let y_true_1 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_1 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected_1 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Both groups have perfect TPR (2/2 and 2/2)
    let eo_1 = equal_opportunity_difference(&y_true_1, &y_pred_1, &protected_1).unwrap();
    assert_abs_diff_eq!(eo_1, 0.0, epsilon = 1e-10);

    // Test case 2: Different true positive rates
    let y_true_2 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_2 = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected_2 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Protected: TPR=0/2, Unprotected: TPR=2/2
    // Difference: |0 - 1| = 1
    let eo_2 = equal_opportunity_difference(&y_true_2, &y_pred_2, &protected_2).unwrap();
    assert_abs_diff_eq!(eo_2, 1.0, epsilon = 1e-10);

    // Test case 3: No positive examples in protected group
    let y_true_3 = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_3 = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let protected_3 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // No positive examples in protected group
    assert!(equal_opportunity_difference(&y_true_3, &y_pred_3, &protected_3).is_err());

    // Test case 4: No positive examples in unprotected group
    let y_true_4 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y_pred_4 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let protected_4 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // No positive examples in unprotected group
    assert!(equal_opportunity_difference(&y_true_4, &y_pred_4, &protected_4).is_err());

    // Test case 5: Partial true positive rate difference
    let y_true_5 = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y_pred_5 = array![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let protected_5 = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Protected: TPR=1/2=0.5, Unprotected: TPR=1/2=0.5
    // Difference: |0.5 - 0.5| = 0
    let eo_5 = equal_opportunity_difference(&y_true_5, &y_pred_5, &protected_5).unwrap();
    assert_abs_diff_eq!(eo_5, 0.0, epsilon = 1e-10);
}

#[test]
fn test_consistency_score() {
    // Test case 1: Perfectly consistent predictions
    // Create a feature matrix with 6 points in 2 clusters
    let features_1 = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.1, 0.2, // Cluster 1
            0.15, 0.22, 0.12, 0.18, 0.8, 0.9, // Cluster 2
            0.85, 0.91, 0.82, 0.88,
        ],
    )
    .unwrap();
    // Predictions match clusters perfectly
    let predictions_1 = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    // With k=2, each point should be perfectly consistent with its neighbors
    let consistency_1 = consistency_score(&features_1, &predictions_1, 2).unwrap();
    assert_abs_diff_eq!(consistency_1, 1.0, epsilon = 1e-10);

    // Test case 2: Inconsistent predictions
    // Same features but mixed predictions
    let features_2 = features_1.clone();
    let predictions_2 = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    // With k=2, neighbors often have different predictions
    let consistency_2 = consistency_score(&features_2, &predictions_2, 2).unwrap();
    assert!(consistency_2 < 1.0);

    // Test case 3: Wrong dimensions
    let features_3 =
        Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    let predictions_3 = array![0.0, 0.0, 0.0, 0.0, 0.0];
    assert!(consistency_score(&features_3, &predictions_3, 2).is_err());

    // Test case 4: k too large
    let features_4 =
        Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    let predictions_4 = array![0.0, 0.0, 0.0, 0.0];
    assert!(consistency_score(&features_4, &predictions_4, 4).is_err());

    // Test case 5: k=0
    let features_5 =
        Array2::from_shape_vec((4, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    let predictions_5 = array![0.0, 0.0, 0.0, 0.0];
    assert!(consistency_score(&features_5, &predictions_5, 0).is_err());

    // Test case 6: Empty arrays
    let features_6 = Array2::<f64>::from_shape_vec((0, 2), vec![]).unwrap();
    let predictions_6 = array![];
    assert!(consistency_score(&features_6, &predictions_6, 2).is_err());
}
