use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};
use scirs2_metrics::ranking::label::{
    coverage_error, coverage_error_multiple, label_ranking_average_precision_score,
    label_ranking_loss,
};

#[test]
fn test_coverage_error() {
    // Test case 1: Single sample, perfect ranking
    let y_true_1 = array![1.0, 1.0, 0.0, 0.0, 0.0];
    let y_score_1 = array![0.9, 0.8, 0.7, 0.6, 0.5];
    let coverage_1 = coverage_error(&y_true_1, &y_score_1).unwrap();
    // The last relevant label is at position 1 (0-indexed), so coverage is 2
    assert_abs_diff_eq!(coverage_1, 2.0, epsilon = 1e-10);

    // Test case 2: Single sample, imperfect ranking
    let y_true_2 = array![1.0, 1.0, 0.0, 0.0, 0.0];
    let y_score_2 = array![0.5, 0.9, 0.7, 0.6, 0.8];
    let coverage_2 = coverage_error(&y_true_2, &y_score_2).unwrap();
    // With the given scores, when sorting by score (descending):
    // [0.9, 0.8, 0.7, 0.6, 0.5] => [1.0, 0.0, 0.0, 0.0, 1.0]
    // The last relevant label (pos 4) requires covering 5 labels
    assert_abs_diff_eq!(coverage_2, 5.0, epsilon = 1e-10);

    // Test case 3: Single sample, worst possible ranking
    let y_true_3 = array![1.0, 1.0, 0.0, 0.0, 0.0];
    let y_score_3 = array![0.1, 0.2, 0.7, 0.6, 0.5];
    let coverage_3 = coverage_error(&y_true_3, &y_score_3).unwrap();
    // The last relevant label is at position 4 (0-indexed), so coverage is 5
    assert_abs_diff_eq!(coverage_3, 5.0, epsilon = 1e-10);
}

#[test]
fn test_coverage_error_multiple() {
    // Test case 1: Multiple samples, all perfect rankings
    let y_true_1 = Array2::from_shape_vec(
        (3, 5),
        vec![
            1.0, 1.0, 0.0, 0.0, 0.0, // Sample 1: labels 0 and 1 are relevant
            0.0, 0.0, 1.0, 1.0, 0.0, // Sample 2: labels 2 and 3 are relevant
            0.0, 0.0, 0.0, 0.0, 1.0, // Sample 3: only label 4 is relevant
        ],
    )
    .unwrap();

    let y_score_1 = Array2::from_shape_vec(
        (3, 5),
        vec![
            0.9, 0.8, 0.7, 0.6, 0.5, // Perfect ranking for sample 1
            0.5, 0.6, 0.9, 0.8, 0.7, // Perfect ranking for sample 2
            0.1, 0.2, 0.3, 0.4, 0.9, // Perfect ranking for sample 3
        ],
    )
    .unwrap();

    let coverage_1 = coverage_error_multiple(&y_true_1, &y_score_1).unwrap();
    // Average of [2, 2, 1] = 1.67
    assert_abs_diff_eq!(coverage_1, 1.67, epsilon = 0.01);

    // Test case 2: Multiple samples, mixed rankings
    let y_true_2 = Array2::from_shape_vec(
        (3, 5),
        vec![
            1.0, 1.0, 0.0, 0.0, 0.0, // Sample 1: labels 0 and 1 are relevant
            0.0, 0.0, 1.0, 1.0, 0.0, // Sample 2: labels 2 and 3 are relevant
            0.0, 0.0, 0.0, 0.0, 1.0, // Sample 3: only label 4 is relevant
        ],
    )
    .unwrap();

    let y_score_2 = Array2::from_shape_vec(
        (3, 5),
        vec![
            0.9, 0.8, 0.7, 0.6, 0.5, // Perfect ranking for sample 1
            0.5, 0.9, 0.6, 0.7,
            0.8, // Poor ranking for sample 2 (labels 2 and 3 ranked 3rd and 4th)
            0.9, 0.8, 0.7, 0.6, 0.5, // Worst ranking for sample 3 (label 4 ranked last)
        ],
    )
    .unwrap();

    let coverage_2 = coverage_error_multiple(&y_true_2, &y_score_2).unwrap();
    // Average of [2, 4, 5] = 3.67
    assert_abs_diff_eq!(coverage_2, 3.67, epsilon = 0.01);
}

#[test]
fn test_label_ranking_loss() {
    // Test case 1: Perfect rankings
    let y_true_1 = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 1.0, 0.0, 0.0, // Sample 1: labels 0 and 1 are relevant
            0.0, 0.0, 1.0, 1.0, // Sample 2: labels 2 and 3 are relevant
            1.0, 0.0, 1.0, 0.0, // Sample 3: labels 0 and 2 are relevant
        ],
    )
    .unwrap();

    let y_score_1 = Array2::from_shape_vec(
        (3, 4),
        vec![
            0.9, 0.8, 0.4, 0.3, // Perfect ranking for sample 1
            0.3, 0.4, 0.9, 0.8, // Perfect ranking for sample 2
            0.9, 0.3, 0.8, 0.4, // Perfect ranking for sample 3
        ],
    )
    .unwrap();

    let loss_1 = label_ranking_loss(&y_true_1, &y_score_1).unwrap();
    // All relevant labels are ranked higher than all irrelevant labels
    assert_abs_diff_eq!(loss_1, 0.0, epsilon = 1e-10);

    // Test case 2: Some errors in rankings
    let y_true_2 = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 1.0, 0.0, 0.0, // Sample 1: labels 0 and 1 are relevant
            0.0, 0.0, 1.0, 1.0, // Sample 2: labels 2 and 3 are relevant
            1.0, 0.0, 1.0, 0.0, // Sample 3: labels 0 and 2 are relevant
        ],
    )
    .unwrap();

    let y_score_2 = Array2::from_shape_vec(
        (3, 4),
        vec![
            0.9, 0.8, 0.85, 0.3, // One error: irrelevant label 2 > relevant label 1
            0.3, 0.95, 0.9, 0.8, // One error: irrelevant label 1 > relevant labels 2 and 3
            0.9, 0.95, 0.8, 0.4, // One error: irrelevant label 1 > relevant label 2
        ],
    )
    .unwrap();

    let loss_2 = label_ranking_loss(&y_true_2, &y_score_2).unwrap();
    // Sample 1: 1 error out of 4 possible pairs (1 relevant * 2 irrelevant)
    // Sample 2: 2 errors out of 4 possible pairs (2 relevant * 2 irrelevant)
    // Sample 3: 2 errors out of 4 possible pairs (2 relevant * 2 irrelevant)
    // Average: (1/4 + 2/4 + 2/4) / 3 = 5/12 = 0.417...
    assert_abs_diff_eq!(loss_2, 5.0 / 12.0, epsilon = 0.01);

    // Test case 3: Worst possible rankings
    let y_true_3 = Array2::from_shape_vec(
        (2, 4),
        vec![
            1.0, 1.0, 0.0, 0.0, // Sample 1: labels 0 and 1 are relevant
            0.0, 0.0, 1.0, 1.0, // Sample 2: labels 2 and 3 are relevant
        ],
    )
    .unwrap();

    let y_score_3 = Array2::from_shape_vec(
        (2, 4),
        vec![
            0.1, 0.2, 0.9, 0.8, // Worst ranking for sample 1
            0.9, 0.8, 0.1, 0.2, // Worst ranking for sample 2
        ],
    )
    .unwrap();

    let loss_3 = label_ranking_loss(&y_true_3, &y_score_3).unwrap();
    // All irrelevant labels are ranked higher than all relevant labels
    assert_abs_diff_eq!(loss_3, 1.0, epsilon = 1e-10);
}

#[test]
fn test_label_ranking_average_precision_score() {
    // Test case 1: Perfect rankings
    let y_true_1 = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 1.0, 0.0, 0.0, // Sample 1: labels 0 and 1 are relevant
            0.0, 0.0, 1.0, 1.0, // Sample 2: labels 2 and 3 are relevant
            1.0, 0.0, 1.0, 0.0, // Sample 3: labels 0 and 2 are relevant
        ],
    )
    .unwrap();

    let y_score_1 = Array2::from_shape_vec(
        (3, 4),
        vec![
            0.9, 0.8, 0.4, 0.3, // Perfect ranking for sample 1
            0.3, 0.4, 0.9, 0.8, // Perfect ranking for sample 2
            0.9, 0.3, 0.8, 0.4, // Perfect ranking for sample 3
        ],
    )
    .unwrap();

    let score_1 = label_ranking_average_precision_score(&y_true_1, &y_score_1).unwrap();
    // All relevant labels are ranked first, so the score should be 1.0
    assert_abs_diff_eq!(score_1, 1.0, epsilon = 1e-10);

    // Test case 2: Some errors in rankings
    let y_true_2 = Array2::from_shape_vec(
        (3, 4),
        vec![
            1.0, 1.0, 0.0, 0.0, // Sample 1: labels 0 and 1 are relevant
            0.0, 0.0, 1.0, 1.0, // Sample 2: labels 2 and 3 are relevant
            1.0, 0.0, 1.0, 0.0, // Sample 3: labels 0 and 2 are relevant
        ],
    )
    .unwrap();

    let y_score_2 = Array2::from_shape_vec(
        (3, 4),
        vec![
            0.9, 0.7, 0.8, 0.3, // Label 2 (irrelevant) ranked higher than label 1 (relevant)
            0.3, 0.9, 0.8,
            0.7, // Label 1 (irrelevant) ranked higher than labels 2 and 3 (relevant)
            0.8, 0.9, 0.7, 0.3, // Label 1 (irrelevant) ranked higher than label 2 (relevant)
        ],
    )
    .unwrap();

    let score_2 = label_ranking_average_precision_score(&y_true_2, &y_score_2).unwrap();
    // Score will be less than 1.0 due to ranking errors
    assert!(score_2 < 1.0);
    assert!(score_2 > 0.0);

    // Test case 3: Worst possible rankings
    let y_true_3 = Array2::from_shape_vec(
        (2, 4),
        vec![
            1.0, 1.0, 0.0, 0.0, // Sample 1: labels 0 and 1 are relevant
            0.0, 0.0, 1.0, 1.0, // Sample 2: labels 2 and 3 are relevant
        ],
    )
    .unwrap();

    let y_score_3 = Array2::from_shape_vec(
        (2, 4),
        vec![
            0.1, 0.2, 0.9, 0.8, // Worst ranking for sample 1 (relevant labels ranked last)
            0.9, 0.8, 0.1, 0.2, // Worst ranking for sample 2 (relevant labels ranked last)
        ],
    )
    .unwrap();

    let score_3 = label_ranking_average_precision_score(&y_true_3, &y_score_3).unwrap();
    // For the worst ranking, the score will be low but not necessarily 0
    assert!(score_3 < 0.5);
}

#[test]
fn test_edge_cases() {
    // Test case 1: Empty arrays
    let y_true_empty = Array2::<f64>::from_shape_vec((0, 4), vec![]).unwrap();
    let y_score_empty = Array2::<f64>::from_shape_vec((0, 4), vec![]).unwrap();

    assert!(coverage_error_multiple(&y_true_empty, &y_score_empty).is_err());
    assert!(label_ranking_loss(&y_true_empty, &y_score_empty).is_err());
    assert!(label_ranking_average_precision_score(&y_true_empty, &y_score_empty).is_err());

    // Test case 2: Different shapes
    let y_true_different =
        Array2::from_shape_vec((2, 4), vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]).unwrap();

    let y_score_different =
        Array2::from_shape_vec((2, 3), vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4]).unwrap();

    assert!(coverage_error_multiple(&y_true_different, &y_score_different).is_err());
    assert!(label_ranking_loss(&y_true_different, &y_score_different).is_err());
    assert!(label_ranking_average_precision_score(&y_true_different, &y_score_different).is_err());

    // Test case 3: No relevant labels
    let y_true_no_relevant =
        Array2::from_shape_vec((2, 4), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

    let y_score_no_relevant =
        Array2::from_shape_vec((2, 4), vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]).unwrap();

    assert!(
        label_ranking_average_precision_score(&y_true_no_relevant, &y_score_no_relevant).is_err()
    );
}
