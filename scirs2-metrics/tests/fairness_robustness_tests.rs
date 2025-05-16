use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};
use scirs2_metrics::classification::accuracy_score;
use scirs2_metrics::fairness::robustness::{
    influence_function, performance_invariance, perturbation_sensitivity, PerturbationType,
};
use scirs2_metrics::fairness::{demographic_parity_difference, equalized_odds_difference};

#[test]
fn test_performance_invariance() {
    // Create a dataset with two groups
    let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    // Two protected attributes: gender and age
    let protected_groups = Array2::from_shape_vec(
        (8, 2),
        vec![
            // Gender (0=male, 1=female), Age (0=young, 1=old)
            0.0, 0.0, // male, young
            0.0, 1.0, // male, old
            0.0, 0.0, // male, young
            0.0, 1.0, // male, old
            1.0, 0.0, // female, young
            1.0, 1.0, // female, old
            1.0, 0.0, // female, young
            1.0, 1.0, // female, old
        ],
    )
    .unwrap();

    let group_names = vec!["gender".to_string(), "age".to_string()];

    // Calculate performance invariance using accuracy
    let result = performance_invariance(
        &y_true,
        &y_pred,
        &protected_groups,
        &group_names,
        |yt, yp| {
            let yt_array = ndarray::Array::from_vec(yt.to_vec());
            let yp_array = ndarray::Array::from_vec(yp.to_vec());
            accuracy_score(&yt_array, &yp_array).unwrap_or(0.0)
        },
    )
    .unwrap();

    // Check that we get metrics for each group
    assert!(result.group_metrics.contains_key("overall"));
    assert!(result.group_metrics.contains_key("gender=0"));
    assert!(result.group_metrics.contains_key("gender=1"));
    assert!(result.group_metrics.contains_key("age=0"));
    assert!(result.group_metrics.contains_key("age=1"));

    // Invariance score should be in a reasonable range
    assert!(result.invariance_score >= 0.0);
    assert!(result.invariance_score <= 1.0);

    // Test: error handling for wrong dimensions
    let wrong_y_true = array![0.0, 0.0, 1.0];
    assert!(performance_invariance(
        &wrong_y_true,
        &y_pred,
        &protected_groups,
        &group_names,
        |yt, yp| {
            let yt_array = ndarray::Array::from_vec(yt.to_vec());
            let yp_array = ndarray::Array::from_vec(yp.to_vec());
            accuracy_score(&yt_array, &yp_array).unwrap_or(0.0)
        }
    )
    .is_err());

    // Test: error handling for mismatched group names
    let wrong_group_names = vec!["gender".to_string()];
    assert!(performance_invariance(
        &y_true,
        &y_pred,
        &protected_groups,
        &wrong_group_names,
        |yt, yp| {
            let yt_array = ndarray::Array::from_vec(yt.to_vec());
            let yp_array = ndarray::Array::from_vec(yp.to_vec());
            accuracy_score(&yt_array, &yp_array).unwrap_or(0.0)
        }
    )
    .is_err());
}

#[test]
fn test_influence_function() {
    // Create a small dataset with a clear bias pattern
    let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let protected_group = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    // Calculate influence using demographic parity
    let influence_scores = influence_function(
        &y_true,
        &y_pred,
        &protected_group,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        None,
    )
    .unwrap();

    // We should get an influence score for each sample
    assert_eq!(influence_scores.len(), y_true.len());

    // Test with limited number of samples
    let limited_influence = influence_function(
        &y_true,
        &y_pred,
        &protected_group,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        Some(4),
    )
    .unwrap();

    // Should still have full length but only first 4 samples evaluated
    assert_eq!(limited_influence.len(), y_true.len());

    // Test: error handling for wrong dimensions
    let wrong_y_true = array![0.0, 0.0];
    assert!(influence_function(
        &wrong_y_true,
        &y_pred,
        &protected_group,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        None
    )
    .is_err());

    // Test: error handling for invalid n_samples
    assert!(influence_function(
        &y_true,
        &y_pred,
        &protected_group,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        Some(100) // More than available samples
    )
    .is_err());
}

#[test]
fn test_perturbation_sensitivity() {
    // Create test data
    let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let protected_group = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Test label flip perturbation
    let result_flip = perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::LabelFlip,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        0.2, // 20% of labels flipped
        5,   // 5 iterations
        Some(42),
    )
    .unwrap();

    // Check the result properties
    assert_eq!(result_flip.perturbed_values.len(), 5);
    assert!(result_flip.sensitivity_score >= 0.0);
    assert_eq!(result_flip.perturbation_type, "LabelFlip");
    assert_abs_diff_eq!(result_flip.perturbation_level, 0.2, epsilon = 1e-10);

    // Test subsample perturbation
    let result_subsample = perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::Subsample,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        0.8, // 80% subsample
        5,
        Some(42),
    )
    .unwrap();

    assert_eq!(result_subsample.perturbed_values.len(), 5);
    assert_eq!(result_subsample.perturbation_type, "Subsample");

    // Test noise perturbation
    let result_noise = perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::Noise,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        0.1, // Noise level
        5,
        Some(42),
    )
    .unwrap();

    assert_eq!(result_noise.perturbed_values.len(), 5);
    assert_eq!(result_noise.perturbation_type, "Noise");

    // Test: error handling for invalid perturbation level
    assert!(perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::LabelFlip,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        0.0, // Invalid - must be > 0
        5,
        None
    )
    .is_err());

    assert!(perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::LabelFlip,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        1.0, // Invalid - must be < 1
        5,
        None
    )
    .is_err());

    // Test: error handling for zero iterations
    assert!(perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::LabelFlip,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        0.2,
        0, // Invalid - must be > 0
        None
    )
    .is_err());
}

#[test]
fn test_perturbation_types() {
    // Create test data with more samples for better statistical properties
    let y_true = array![
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0
    ];
    let y_pred = array![
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        0.0, 1.0
    ];
    let protected_group = array![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ];

    // The original data is perfectly fair (demographic parity = 0)
    let original_dp = demographic_parity_difference(&y_pred, &protected_group).unwrap();
    assert_abs_diff_eq!(original_dp, 0.0, epsilon = 1e-10);

    // Test that different perturbation types produce different results
    let perturbation_level = 0.3;
    let n_iterations = 10;
    let seed = 42;

    // Label flip should create some unfairness
    let flip_result = perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::LabelFlip,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        perturbation_level,
        n_iterations,
        Some(seed),
    )
    .unwrap();

    // Noise should also create some unfairness
    let noise_result = perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::Noise,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        perturbation_level,
        n_iterations,
        Some(seed),
    )
    .unwrap();

    // Subsample might preserve fairness better
    let subsample_result = perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::Subsample,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        perturbation_level,
        n_iterations,
        Some(seed),
    )
    .unwrap();

    // The different perturbations should give different results
    // We don't assert specific values as they depend on random processes,
    // but we check that the sensitivity scores are positive
    assert!(flip_result.sensitivity_score > 0.0);
    assert!(noise_result.sensitivity_score > 0.0);
    assert!(subsample_result.sensitivity_score > 0.0);
}

#[test]
fn test_robustness_with_different_fairness_metrics() {
    // Create test data
    let y_true = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let y_pred = array![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let protected_group = array![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Test with demographic parity
    let dp_result = perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::LabelFlip,
        |yp, pg| demographic_parity_difference(yp, pg).unwrap_or(0.0),
        0.2,
        5,
        Some(42),
    )
    .unwrap();

    // Test with equalized odds
    let eo_result = perturbation_sensitivity(
        &y_true,
        &y_pred,
        &protected_group,
        PerturbationType::LabelFlip,
        |yp, pg| equalized_odds_difference(&y_true, yp, pg).unwrap_or(0.0),
        0.2,
        5,
        Some(42),
    )
    .unwrap();

    // Different fairness metrics should have different sensitivity profiles
    assert!(dp_result.original_fairness >= 0.0);
    assert!(eo_result.original_fairness >= 0.0);
}
