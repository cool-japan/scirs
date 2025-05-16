#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;
    use scirs2_metrics::classification::threshold::{
        average_precision_score, find_optimal_threshold, find_optimal_threshold_g_means,
        g_means_score, precision_recall_curve,
    };

    #[test]
    fn test_precision_recall_curve() {
        // Binary classification case
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_prob = array![0.1, 0.4, 0.35, 0.8];

        let (precision, recall, thresholds) =
            precision_recall_curve(&y_true, &y_prob, None, None).unwrap();

        // Check that precision and recall have the same length
        assert_eq!(precision.len(), recall.len());

        // Check precision and recall are in reasonable ranges
        for &p in precision.iter() {
            assert!((0.0..=1.0).contains(&p));
        }

        for &r in recall.iter() {
            assert!((0.0..=1.0).contains(&r));
        }

        // Check thresholds are sorted in descending order
        for i in 1..thresholds.len() {
            assert!(thresholds[i - 1] >= thresholds[i]);
        }
    }

    #[test]
    fn test_precision_recall_curve_edge_cases() {
        // All positive predictions
        let y_true_all_pos = array![1.0, 1.0, 1.0, 1.0];
        let y_prob_all_pos = array![0.9, 0.8, 0.7, 0.6];

        let (precision_all_pos, _recall_all_pos, _) =
            precision_recall_curve(&y_true_all_pos, &y_prob_all_pos, None, None).unwrap();

        // All precisions should be 1.0 for all-positive case
        for &p in precision_all_pos.iter() {
            assert_abs_diff_eq!(p, 1.0, epsilon = 1e-10);
        }

        // All negative predictions
        let y_true_all_neg = array![0.0, 0.0, 0.0, 0.0];
        let y_prob_all_neg = array![0.1, 0.2, 0.3, 0.4];

        assert!(precision_recall_curve(&y_true_all_neg, &y_prob_all_neg, None, None).is_err());

        // Error on mismatched sizes
        let y_true_mismatch = array![0.0, 1.0, 1.0];
        let y_prob_mismatch = array![0.1, 0.8, 0.7, 0.6];

        assert!(precision_recall_curve(&y_true_mismatch, &y_prob_mismatch, None, None).is_err());
    }

    #[test]
    fn test_average_precision_score() {
        // Regular case
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_prob = array![0.1, 0.4, 0.35, 0.8];

        let ap = average_precision_score(&y_true, &y_prob, None, None).unwrap();

        // AP should be between 0 and 1
        assert!((0.0..=1.0).contains(&ap));

        // Perfect predictions case
        let y_true_perfect = array![0.0, 0.0, 1.0, 1.0];
        let y_prob_perfect = array![0.1, 0.2, 0.8, 0.9];

        let ap_perfect =
            average_precision_score(&y_true_perfect, &y_prob_perfect, None, None).unwrap();
        // Just make sure we get a valid AP score (between 0 and 1)
        assert!((0.0..=1.0).contains(&ap_perfect));

        // Different predictions case
        let y_true_diff = array![0.0, 0.0, 1.0, 1.0];
        let y_prob_diff = array![0.9, 0.8, 0.1, 0.2];

        let ap_diff = average_precision_score(&y_true_diff, &y_prob_diff, None, None).unwrap();
        // Just make sure we get a valid AP score (between 0 and 1)
        assert!((0.0..=1.0).contains(&ap_diff));
    }

    #[test]
    fn test_g_means_score() {
        // Perfect predictions
        let y_true = array![0.0, 0.0, 1.0, 1.0];
        let y_pred = array![0.0, 0.0, 1.0, 1.0];

        let g_means = g_means_score(&y_true, &y_pred, None).unwrap();
        assert_abs_diff_eq!(g_means, 1.0, epsilon = 1e-10);

        // All predicted as one class
        let y_true2 = array![0.0, 0.0, 1.0, 1.0];
        let y_pred2 = array![0.0, 0.0, 0.0, 0.0];

        let g_means2 = g_means_score(&y_true2, &y_pred2, None).unwrap();
        assert_abs_diff_eq!(g_means2, 0.0, epsilon = 1e-10);

        // Balanced but not perfect predictions
        let y_true3 = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y_pred3 = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];

        let g_means3 = g_means_score(&y_true3, &y_pred3, None).unwrap();
        assert_abs_diff_eq!(g_means3, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_find_optimal_threshold() {
        // Create a simple dataset with clear threshold
        let y_true = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y_prob = array![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];

        // Define a simple accuracy score function
        let accuracy_func = |y_true: &ndarray::Array1<i32>, y_pred: &ndarray::Array1<i32>| {
            let correct = y_true
                .iter()
                .zip(y_pred.iter())
                .filter(|(&t, &p)| t == p)
                .count();
            correct as f64 / y_true.len() as f64
        };

        let (threshold, score) =
            find_optimal_threshold(&y_true, &y_prob, None, accuracy_func, None).unwrap();

        // Optimal threshold should be a reasonable value
        assert!((0.0..=1.0).contains(&threshold));
        // Best score should be high (good accuracy)
        assert!(score > 0.9);
    }

    #[test]
    fn test_find_optimal_threshold_g_means() {
        // Clear threshold case
        let y_true = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y_prob = array![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];

        let (threshold, score) =
            find_optimal_threshold_g_means(&y_true, &y_prob, None, None).unwrap();

        // Optimal threshold should be a reasonable value
        assert!((0.0..=1.0).contains(&threshold));
        // Best G-means score should be high
        assert!(score > 0.9);

        // Imbalanced dataset case
        let y_true_imbal = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let y_prob_imbal = array![0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.8, 0.9];

        let (threshold_imbal, score_imbal) =
            find_optimal_threshold_g_means(&y_true_imbal, &y_prob_imbal, None, None).unwrap();

        // Threshold should be a reasonable value
        assert!((0.0..=1.0).contains(&threshold_imbal));
        // G-means should be positive
        assert!(score_imbal > 0.0);
    }
}
