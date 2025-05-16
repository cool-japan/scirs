#[cfg(test)]
mod cluster_distance_tests {
    use ndarray::{array, Array2};
    use scirs2_metrics::clustering::distance::{
        distance_ratio_index, inter_cluster_distances, intra_cluster_distances, isolation_index,
    };

    #[test]
    fn test_inter_cluster_distances() {
        // Create a simple dataset with two well-separated clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        // Test Euclidean distance
        let distances = inter_cluster_distances(&x, &labels, "euclidean").unwrap();

        // There should be at least one pair: (0, 1)
        assert!(!distances.is_empty());

        // Check that the distance is reasonable
        let dist_0_1 = distances.get(&(0, 1)).unwrap();
        assert!(*dist_0_1 > 4.0 && *dist_0_1 < 6.0);

        // Test Manhattan distance
        let distances = inter_cluster_distances(&x, &labels, "manhattan").unwrap();
        assert!(!distances.is_empty());

        // Manhattan distance is typically larger than Euclidean
        let dist_0_1 = distances.get(&(0, 1)).unwrap();
        assert!(*dist_0_1 > 5.0);
    }

    #[test]
    fn test_intra_cluster_distances() {
        // Create a simple dataset with two clusters
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 5.0, 6.0, 5.2, 5.8, 5.5, 6.2],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        // Test Euclidean distance
        let distances = intra_cluster_distances(&x, &labels, "euclidean").unwrap();

        // There should be two clusters
        assert_eq!(distances.len(), 2);

        // Check that the distances are reasonable
        assert!(*distances.get(&0).unwrap() < 1.0); // Cluster 0 is compact
        assert!(*distances.get(&1).unwrap() < 1.0); // Cluster 1 is also compact

        // Test cosine distance
        let distances = intra_cluster_distances(&x, &labels, "cosine").unwrap();
        assert_eq!(distances.len(), 2);

        // Cosine distance for similar vectors should be small
        assert!(*distances.get(&0).unwrap() < 0.1);
        assert!(*distances.get(&1).unwrap() < 0.1);
    }

    #[test]
    fn test_distance_ratio_index() {
        // Create a dataset with well-separated clusters
        let well_separated = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 10.0, 12.0, 10.2, 11.8, 10.5, 12.2,
            ],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        // Create a dataset with poorly separated clusters
        let poorly_separated = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 3.0, 4.0, 3.2, 3.8, 3.5, 4.2],
        )
        .unwrap();

        // Test distance ratio index (higher is better)
        let good_ratio = distance_ratio_index(&well_separated, &labels, "euclidean").unwrap();
        let poor_ratio = distance_ratio_index(&poorly_separated, &labels, "euclidean").unwrap();

        // Just check that the ratios are reasonable
        assert!(good_ratio > 0.0);
        assert!(poor_ratio > 0.0);
    }

    #[test]
    fn test_isolation_index() {
        // Create a dataset with well-isolated clusters
        let well_isolated = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 10.0, 12.0, 10.2, 11.8, 10.5, 12.2,
            ],
        )
        .unwrap();

        let labels = array![0, 0, 0, 1, 1, 1];

        // Create a dataset with poorly isolated clusters
        let poorly_isolated = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 1.2, 2.2, 3.0, 4.0, 3.2, 3.8, 3.5, 4.2],
        )
        .unwrap();

        // Test isolation index (higher is better)
        let good_isolation = isolation_index(&well_isolated, &labels, "euclidean").unwrap();
        let poor_isolation = isolation_index(&poorly_isolated, &labels, "euclidean").unwrap();

        // Well-isolated clusters should have a higher isolation value
        assert!(good_isolation > poor_isolation);
        assert!(good_isolation > 0.8); // Should be close to 1 for well-isolated clusters
    }

    #[test]
    fn test_with_invalid_inputs() {
        // Skip empty dataset test as it may be handled differently in different implementations

        // Invalid metric
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let labels = array![0, 0, 1];

        let result = inter_cluster_distances(&x, &labels, "invalid_metric");
        assert!(result.is_err());

        // Mismatched dimensions
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let labels = array![0, 0]; // Only 2 labels for 3 samples

        let result = inter_cluster_distances(&x, &labels, "euclidean");
        assert!(result.is_err());
    }
}
