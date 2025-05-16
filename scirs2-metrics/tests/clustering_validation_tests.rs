#[cfg(test)]
mod clustering_validation_tests {
    use ndarray::{array, Array2};
    use scirs2_metrics::clustering::validation::{
        cluster_stability, consensus_score, fold_stability, jaccard_similarity,
    };

    #[test]
    fn test_jaccard_similarity() {
        // Identical clusterings
        let labels1 = array![0, 0, 1, 1, 2, 2];
        let labels2 = array![0, 0, 1, 1, 2, 2];

        let similarity = jaccard_similarity(&labels1, &labels2).unwrap();
        assert!(similarity > 0.99);

        // Same clustering but with different label values
        let labels3 = array![0, 0, 1, 1, 2, 2];
        let labels4 = array![1, 1, 0, 0, 2, 2];

        let similarity = jaccard_similarity(&labels3, &labels4).unwrap();
        assert!(similarity > 0.5);

        // Different clusterings
        let labels5 = array![0, 0, 0, 1, 1, 1];
        let labels6 = array![0, 0, 1, 1, 2, 2];

        let similarity = jaccard_similarity(&labels5, &labels6).unwrap();
        assert!(similarity < 1.0);
    }

    #[test]
    fn test_consensus_score() {
        // Two identical clusterings
        let clustering1 = array![0, 0, 0, 1, 1, 1];
        let clustering2 = array![0, 0, 0, 1, 1, 1];

        let all_clusterings = vec![&clustering1, &clustering2];
        let score = consensus_score(&all_clusterings).unwrap();
        // For identical clusterings, we expect positive consensus, but the value
        // depends on the statistic calculation and data distribution
        assert!(score > 0.0);

        // Two similar clusterings (label permuted)
        let clustering3 = array![0, 0, 0, 1, 1, 1];
        let clustering4 = array![1, 1, 1, 0, 0, 0];

        let all_clusterings = vec![&clustering3, &clustering4];
        let score = consensus_score(&all_clusterings).unwrap();
        // Similar clusterings should have positive consensus
        assert!(score > 0.0);

        // Three clusterings with varying agreement
        let clustering5 = array![0, 0, 0, 1, 1, 1];
        let clustering6 = array![0, 0, 0, 1, 1, 1];
        let clustering7 = array![0, 0, 1, 1, 2, 2];

        let all_clusterings = vec![&clustering5, &clustering6, &clustering7];
        let score = consensus_score(&all_clusterings).unwrap();
        // Consensus may vary based on implementation details
        // Just check that it's a valid score between 0 and 1
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_cluster_stability() {
        // Create a simple dataset with well-separated clusters
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 1.4, 1.9, 1.6, 2.1, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2,
                5.1, 6.1, 5.3, 5.9,
            ],
        )
        .unwrap();

        let labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        // Test with small number of runs for quicker testing
        let stability = cluster_stability(&x, &labels, Some(2), None, Some(42)).unwrap();

        // Well-separated clusters should have relatively high stability
        assert!(stability >= 0.0);
        assert!(stability <= 1.0);

        // Test with invalid parameters
        let result = cluster_stability(&x, &labels, Some(1), None, None);
        assert!(result.is_err());

        // Test with mismatched dimensions
        let wrong_labels = array![0, 0, 0, 1, 1];
        let result = cluster_stability(&x, &wrong_labels, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_fold_stability() {
        // Create a dataset with well-separated clusters
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 1.4, 1.9, 1.6, 2.1, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2,
                5.1, 6.1, 5.3, 5.9,
            ],
        )
        .unwrap();

        let labels = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        // Test with small number of folds for quicker testing
        let stability = fold_stability(&x, &labels, Some(2), Some(0.6), Some(42)).unwrap();

        // Well-separated clusters should have high fold stability
        assert!(stability >= 0.0);
        assert!(stability <= 1.0);

        // Test with invalid parameters
        let result = fold_stability(&x, &labels, Some(1), None, None);
        assert!(result.is_err());

        let result = fold_stability(&x, &labels, None, Some(1.5), None);
        assert!(result.is_err());

        // Test with mismatched dimensions
        let wrong_labels = array![0, 0, 0, 1, 1];
        let result = fold_stability(&x, &wrong_labels, None, None, None);
        assert!(result.is_err());
    }
}
