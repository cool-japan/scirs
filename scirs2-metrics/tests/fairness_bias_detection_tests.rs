use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};
use scirs2_metrics::classification::accuracy_score;
use scirs2_metrics::fairness::bias_detection::{
    intersectional_fairness, slice_analysis, subgroup_performance,
};

#[test]
fn test_slice_analysis() {
    // Create a sample dataset
    let features = Array2::from_shape_vec(
        (8, 3),
        vec![
            // age, gender(0=male, 1=female), region(0,1,2)
            25.0, 0.0, 0.0, 30.0, 0.0, 1.0, 22.0, 1.0, 0.0, 35.0, 1.0, 1.0, 40.0, 0.0, 2.0, 45.0,
            0.0, 0.0, 28.0, 1.0, 2.0, 50.0, 1.0, 0.0,
        ],
    )
    .unwrap();

    let y_true = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let y_pred = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];

    // Test: slice analysis with gender (column 1) and region (column 2)
    let results = slice_analysis(&features, &[1, 2], &y_true, &y_pred, |y_t, y_p| {
        // Convert Vec<f64> to Array1<f64> for accuracy_score
        let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
        let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
        accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
    })
    .unwrap();

    // Check that we get the overall accuracy
    assert!(results.contains_key("overall"));

    // The overall accuracy should be 75% (6/8 correct)
    assert_abs_diff_eq!(results["overall"], 0.75, epsilon = 1e-10);

    // Check that we get slices for gender
    assert!(results.contains_key("feature_1_0.0") || results.contains_key("feature_1_0")); // Male
    assert!(results.contains_key("feature_1_1.0") || results.contains_key("feature_1_1")); // Female

    // Check that we get slices for region
    assert!(results.contains_key("feature_2_0.0") || results.contains_key("feature_2_0")); // Region 0
    assert!(results.contains_key("feature_2_1.0") || results.contains_key("feature_2_1")); // Region 1
    assert!(results.contains_key("feature_2_2.0") || results.contains_key("feature_2_2")); // Region 2

    // Test: error handling for wrong dimensions
    let wrong_y_true = array![0.0, 0.0, 1.0];
    assert!(
        slice_analysis(&features, &[1, 2], &wrong_y_true, &y_pred, |y_t, y_p| {
            let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
            let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
            accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
        },)
        .is_err()
    );

    // Test: error handling for invalid feature index
    assert!(slice_analysis(
        &features,
        &[10], // Invalid column index
        &y_true,
        &y_pred,
        |y_t, y_p| {
            let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
            let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
            accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
        },
    )
    .is_err());
}

#[test]
fn test_subgroup_performance() {
    // Create sample dataset
    let y_true = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let y_pred = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];

    // Demographic groups: gender (0=male, 1=female) and age_group (0=young, 1=old)
    let groups = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, // male, young
            0.0, 1.0, // male, old
            1.0, 0.0, // female, young
            1.0, 0.0, // female, young
            0.0, 1.0, // male, old
            0.0, 1.0, // male, old
            1.0, 1.0, // female, old
            1.0, 1.0, // female, old
        ],
    )
    .unwrap();

    let group_names = vec!["gender".to_string(), "age_group".to_string()];

    // Analyze performance across subgroups
    let results = subgroup_performance(&y_true, &y_pred, &groups, &group_names, |y_t, y_p| {
        let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
        let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
        accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
    })
    .unwrap();

    // Check that we get the overall accuracy
    assert!(results.contains_key("overall"));

    // The overall accuracy should be 75% (6/8 correct)
    assert_abs_diff_eq!(results["overall"], 0.75, epsilon = 1e-10);

    // Check for individual subgroups
    assert!(results.contains_key("gender=0")); // Male
    assert!(results.contains_key("gender=1")); // Female
    assert!(results.contains_key("age_group=0")); // Young
    assert!(results.contains_key("age_group=1")); // Old

    // Check for intersectional subgroups
    assert!(results.contains_key("gender=0 & age_group=0")); // Male + Young
    assert!(results.contains_key("gender=0 & age_group=1")); // Male + Old
    assert!(results.contains_key("gender=1 & age_group=0")); // Female + Young
    assert!(results.contains_key("gender=1 & age_group=1")); // Female + Old

    // Test: error handling for wrong dimensions
    let wrong_y_true = array![0.0, 0.0, 1.0];
    assert!(
        subgroup_performance(&wrong_y_true, &y_pred, &groups, &group_names, |y_t, y_p| {
            let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
            let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
            accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
        },)
        .is_err()
    );

    // Test: error handling for mismatched group names
    let wrong_group_names = vec!["gender".to_string()];
    assert!(
        subgroup_performance(&y_true, &y_pred, &groups, &wrong_group_names, |y_t, y_p| {
            let y_t_array = ndarray::Array::from_vec(y_t.to_vec());
            let y_p_array = ndarray::Array::from_vec(y_p.to_vec());
            accuracy_score(&y_t_array, &y_p_array).unwrap_or(0.0)
        },)
        .is_err()
    );
}

#[test]
fn test_intersectional_fairness() {
    // Create sample dataset
    let y_true = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
    let y_pred = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];

    // Protected attributes: gender (0=male, 1=female) and race (0=group A, 1=group B)
    let protected_features = Array2::from_shape_vec(
        (8, 2),
        vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let feature_names = vec!["gender".to_string(), "race".to_string()];

    // Analyze intersectional fairness
    let results =
        intersectional_fairness(&y_true, &y_pred, &protected_features, &feature_names).unwrap();

    // Check that we get individual group results
    assert!(results.contains_key("gender=0")); // Male
    assert!(results.contains_key("gender=1")); // Female
    assert!(results.contains_key("race=0")); // Race group A
    assert!(results.contains_key("race=1")); // Race group B

    // Check for intersectional groups
    assert!(results.contains_key("gender=0 & race=0")); // Male + Race group A
    assert!(results.contains_key("gender=0 & race=1")); // Male + Race group B
    assert!(results.contains_key("gender=1 & race=0")); // Female + Race group A
    assert!(results.contains_key("gender=1 & race=1")); // Female + Race group B

    // Check that fairness metrics are calculated
    for (_, metrics) in results.iter() {
        assert!(metrics.demographic_parity >= 0.0 && metrics.demographic_parity <= 1.0);
        assert!(metrics.equalized_odds >= 0.0 && metrics.equalized_odds <= 1.0);
        assert!(metrics.equal_opportunity >= 0.0 && metrics.equal_opportunity <= 1.0);
    }

    // Test: error handling for wrong dimensions
    let wrong_y_true = array![0.0, 0.0, 1.0];
    assert!(
        intersectional_fairness(&wrong_y_true, &y_pred, &protected_features, &feature_names)
            .is_err()
    );

    // Test: error handling for mismatched feature names
    let wrong_feature_names = vec!["gender".to_string()];
    assert!(
        intersectional_fairness(&y_true, &y_pred, &protected_features, &wrong_feature_names)
            .is_err()
    );
}

#[test]
fn test_edge_cases() {
    // Edge case: all samples in one group (no valid protected groups)
    let y_true = array![0.0, 0.0, 1.0, 1.0];
    let y_pred = array![0.0, 0.0, 0.0, 1.0];

    let all_same_group = Array2::from_shape_vec(
        (4, 1),
        vec![1.0, 1.0, 1.0, 1.0], // All samples in the same group
    )
    .unwrap();

    let feature_names = vec!["group".to_string()];

    // This should work but might not have any intersectional results since all samples are in one group
    let results =
        intersectional_fairness(&y_true, &y_pred, &all_same_group, &feature_names).unwrap();
    assert_eq!(results.len(), 0); // Should have no results since all samples are in one group

    // Edge case: small groups
    let y_true_small = array![0.0, 1.0];
    let y_pred_small = array![0.0, 0.0];

    let small_groups = Array2::from_shape_vec(
        (2, 1),
        vec![0.0, 1.0], // One sample in each group
    )
    .unwrap();

    // Should work with small groups, but might have limited statistical significance
    let results_small =
        intersectional_fairness(&y_true_small, &y_pred_small, &small_groups, &feature_names)
            .unwrap();
    assert!(results_small.contains_key("group=0"));
    assert!(results_small.contains_key("group=1"));
}
