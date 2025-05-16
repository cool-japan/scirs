use approx::assert_abs_diff_eq;
use ndarray::array;
use scirs2_metrics::classification::one_vs_one::{
    one_vs_one_accuracy, one_vs_one_f1_score, one_vs_one_precision_recall,
    weighted_one_vs_one_f1_score,
};

#[test]
fn test_one_vs_one_accuracy() {
    // Test case 1: Perfect prediction
    let y_true_1 = array![0, 1, 2, 0, 1, 2];
    let y_pred_1 = array![0, 1, 2, 0, 1, 2];
    let acc_1 = one_vs_one_accuracy(&y_true_1, &y_pred_1).unwrap();
    assert_abs_diff_eq!(acc_1, 1.0, epsilon = 1e-10);

    // Test case 2: Partial accuracy
    let y_true_2 = array![0, 1, 2, 0, 1, 2];
    let y_pred_2 = array![0, 2, 1, 0, 0, 2];
    let acc_2 = one_vs_one_accuracy(&y_true_2, &y_pred_2).unwrap();
    // In this case, classes 0 and 1 have 2/3 correct, 0 and 2 have 2/3 correct, and 1 and 2 have 0/2 correct
    // Average: (2/3 + 2/3 + 0/2) / 3 = 0.444...
    assert!(acc_2 > 0.0 && acc_2 < 1.0);

    // Test case 3: Single class
    let y_true_3 = array![0, 0, 0, 0];
    let y_pred_3 = array![0, 0, 0, 0];
    let acc_3 = one_vs_one_accuracy(&y_true_3, &y_pred_3).unwrap();
    assert_abs_diff_eq!(acc_3, 1.0, epsilon = 1e-10);

    // Test case 4: Different array lengths
    let y_true_4 = array![0, 1, 2];
    let y_pred_4 = array![0, 1, 2, 0];
    assert!(one_vs_one_accuracy(&y_true_4, &y_pred_4).is_err());

    // Test case 5: Empty arrays
    let y_true_5: ndarray::Array1<i32> = array![];
    let y_pred_5: ndarray::Array1<i32> = array![];
    assert!(one_vs_one_accuracy(&y_true_5, &y_pred_5).is_err());
}

#[test]
fn test_one_vs_one_precision_recall() {
    // Test case 1: Perfect prediction
    let y_true_1 = array![0, 1, 2, 0, 1, 2];
    let y_pred_1 = array![0, 1, 2, 0, 1, 2];
    let (precision_1, recall_1) = one_vs_one_precision_recall(&y_true_1, &y_pred_1).unwrap();

    // Each class should have precision and recall of 1.0
    assert_abs_diff_eq!(precision_1[&0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(precision_1[&1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(precision_1[&2], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(recall_1[&0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(recall_1[&1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(recall_1[&2], 1.0, epsilon = 1e-10);

    // Test case 2: Imperfect prediction
    let y_true_2 = array![0, 1, 2, 0, 1, 2];
    let y_pred_2 = array![0, 2, 1, 0, 0, 2];
    let (precision_2, recall_2) = one_vs_one_precision_recall(&y_true_2, &y_pred_2).unwrap();

    // Class 0: 2 TP, 1 FP, 0 FN => precision = 2/3, recall = 2/2 = 1.0
    assert_abs_diff_eq!(precision_2[&0], 2.0 / 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(recall_2[&0], 1.0, epsilon = 1e-10);

    // Class 1: 0 TP, 0 FP, 2 FN => precision = 0/0 = 0.0, recall = 0/2 = 0.0
    assert_abs_diff_eq!(precision_2[&1], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(recall_2[&1], 0.0, epsilon = 1e-10);

    // Class 2: 1 TP, 1 FP, 1 FN => precision = 1/2 = 0.5, recall = 1/2 = 0.5
    assert_abs_diff_eq!(precision_2[&2], 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(recall_2[&2], 0.5, epsilon = 1e-10);

    // Test case 3: Different array lengths
    let y_true_3 = array![0, 1, 2];
    let y_pred_3 = array![0, 1, 2, 0];
    assert!(one_vs_one_precision_recall(&y_true_3, &y_pred_3).is_err());
}

#[test]
fn test_one_vs_one_f1_score() {
    // Test case 1: Perfect prediction
    let y_true_1 = array![0, 1, 2, 0, 1, 2];
    let y_pred_1 = array![0, 1, 2, 0, 1, 2];
    let f1_scores_1 = one_vs_one_f1_score(&y_true_1, &y_pred_1).unwrap();

    // Each class should have F1 score of 1.0
    assert_abs_diff_eq!(f1_scores_1[&0], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(f1_scores_1[&1], 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(f1_scores_1[&2], 1.0, epsilon = 1e-10);

    // Test case 2: Imperfect prediction
    let y_true_2 = array![0, 1, 2, 0, 1, 2];
    let y_pred_2 = array![0, 2, 1, 0, 0, 2];
    let f1_scores_2 = one_vs_one_f1_score(&y_true_2, &y_pred_2).unwrap();

    // Class 0: precision = 2/3, recall = 1.0 => F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 2 * 2/3 / 5/3 = 4/5 = 0.8
    assert_abs_diff_eq!(f1_scores_2[&0], 0.8, epsilon = 1e-10);

    // Class 1: precision = 0.0, recall = 0.0 => F1 = 0.0
    assert_abs_diff_eq!(f1_scores_2[&1], 0.0, epsilon = 1e-10);

    // Class 2: precision = 0.5, recall = 0.5 => F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 2 * 0.25 / 1.0 = 0.5
    assert_abs_diff_eq!(f1_scores_2[&2], 0.5, epsilon = 1e-10);
}

#[test]
fn test_weighted_one_vs_one_f1_score() {
    // Test case 1: Perfect prediction
    let y_true_1 = array![0, 1, 2, 0, 1, 2];
    let y_pred_1 = array![0, 1, 2, 0, 1, 2];
    let weighted_f1_1 = weighted_one_vs_one_f1_score(&y_true_1, &y_pred_1).unwrap();
    assert_abs_diff_eq!(weighted_f1_1, 1.0, epsilon = 1e-10);

    // Test case 2: Imperfect prediction
    let y_true_2 = array![0, 1, 2, 0, 1, 2];
    let y_pred_2 = array![0, 2, 1, 0, 0, 2];
    let weighted_f1_2 = weighted_one_vs_one_f1_score(&y_true_2, &y_pred_2).unwrap();

    // Class 0: Count = 2, F1 = 0.8
    // Class 1: Count = 2, F1 = 0.0
    // Class 2: Count = 2, F1 = 0.5
    // Weighted F1 = (2*0.8 + 2*0.0 + 2*0.5) / 6 = (1.6 + 0 + 1.0) / 6 = 2.6 / 6 = 0.433...
    assert_abs_diff_eq!(weighted_f1_2, 0.433, epsilon = 1e-3);

    // Test case 3: Unbalanced classes
    let y_true_3 = array![0, 0, 0, 0, 1, 2];
    let y_pred_3 = array![0, 0, 0, 0, 2, 1];
    let weighted_f1_3 = weighted_one_vs_one_f1_score(&y_true_3, &y_pred_3).unwrap();

    // Class 0: Count = 4, F1 = 1.0
    // Class 1: Count = 1, F1 = 0.0
    // Class 2: Count = 1, F1 = 0.0
    // Weighted F1 = (4*1.0 + 1*0.0 + 1*0.0) / 6 = 4.0 / 6 = 0.667
    assert_abs_diff_eq!(weighted_f1_3, 0.667, epsilon = 1e-3);
}

#[test]
fn test_edge_cases() {
    // Different array lengths
    let y_true_1 = array![0, 1, 2];
    let y_pred_1 = array![0, 1, 2, 0];

    assert!(one_vs_one_accuracy(&y_true_1, &y_pred_1).is_err());
    assert!(one_vs_one_precision_recall(&y_true_1, &y_pred_1).is_err());
    assert!(one_vs_one_f1_score(&y_true_1, &y_pred_1).is_err());
    assert!(weighted_one_vs_one_f1_score(&y_true_1, &y_pred_1).is_err());

    // Empty arrays
    let y_true_2: ndarray::Array1<i32> = array![];
    let y_pred_2: ndarray::Array1<i32> = array![];

    assert!(one_vs_one_accuracy(&y_true_2, &y_pred_2).is_err());
    assert!(one_vs_one_precision_recall(&y_true_2, &y_pred_2).is_err());
    assert!(one_vs_one_f1_score(&y_true_2, &y_pred_2).is_err());
    assert!(weighted_one_vs_one_f1_score(&y_true_2, &y_pred_2).is_err());
}
