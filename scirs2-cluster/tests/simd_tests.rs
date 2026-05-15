//! SIMD-specific tests for clustering operations
//!
//! This module contains tests to verify the correctness of SIMD-accelerated
//! clustering implementations.

#[cfg(feature = "simd")]
mod simd_clustering_tests {
    use scirs2_cluster::density::{dbscan, DistanceMetric};
    use scirs2_cluster::gmm::{gaussian_mixture, GMMOptions};
    use scirs2_cluster::hierarchy::{linkage, LinkageMethod, Metric};
    use scirs2_core::ndarray::Array2;

    /// Test that DBSCAN with SIMD produces valid results
    #[test]
    fn test_dbscan_simd_euclidean() {
        let data = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 1.1, 2.1, 5.0, 7.0, 5.1, 6.8, 5.2, 7.1, 4.9, 7.2,
                0.0, 10.0, 10.0, 0.0,
            ],
        )
        .expect("Test: Failed to create data array");

        let labels = dbscan(data.view(), 0.8, 2, Some(DistanceMetric::Euclidean))
            .expect("Test: DBSCAN failed");

        assert_eq!(labels.len(), 10);

        // Check that we have some clusters (not all noise)
        let has_clusters = labels.iter().any(|&l| l >= 0);
        assert!(has_clusters, "DBSCAN should find at least one cluster");
    }

    /// Test DBSCAN with Manhattan distance metric and SIMD
    #[test]
    fn test_dbscan_simd_manhattan() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 1.2, 0.8, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9, 5.2, 4.8,
            ],
        )
        .expect("Test: Failed to create data array");

        let labels = dbscan(data.view(), 0.5, 2, Some(DistanceMetric::Manhattan))
            .expect("Test: DBSCAN failed");

        assert_eq!(labels.len(), 8);

        // Should find clusters with Manhattan distance
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert!(!unique_labels.is_empty());
    }

    /// Test DBSCAN with Chebyshev distance metric and SIMD
    #[test]
    fn test_dbscan_simd_chebyshev() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9],
        )
        .expect("Test: Failed to create data array");

        let labels = dbscan(data.view(), 0.3, 2, Some(DistanceMetric::Chebyshev))
            .expect("Test: DBSCAN failed");

        assert_eq!(labels.len(), 6);
    }

    /// Test GMM with SIMD acceleration
    #[test]
    fn test_gmm_simd_basic() {
        let data = Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 2.0, 1.2, 1.8, 0.8, 1.9, 1.1, 2.1, 1.3, 1.7, 0.9, 2.2, 5.0, 6.0, 5.2, 5.8,
                4.8, 5.9, 5.1, 6.1, 5.3, 5.7, 4.9, 6.2,
            ],
        )
        .expect("Test: Failed to create data array");

        let options = GMMOptions {
            n_components: 2,
            max_iter: 50,
            random_seed: Some(42),
            ..Default::default()
        };

        let labels = gaussian_mixture(data.view(), options).expect("Test: GMM failed");

        assert_eq!(labels.len(), 12);

        // Should find 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert!(unique_labels.len() <= 2);
        assert!(!unique_labels.is_empty());
    }

    /// Test hierarchical clustering with single linkage and SIMD
    #[test]
    fn test_hierarchy_simd_single() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9],
        )
        .expect("Test: Failed to create data array");

        let result = linkage(data.view(), LinkageMethod::Single, Metric::Euclidean);
        assert!(result.is_ok(), "Single linkage should succeed");

        let linkage_matrix = result.expect("Test: Linkage failed");
        assert_eq!(linkage_matrix.shape()[0], 5); // n-1 merges for n points
        assert_eq!(linkage_matrix.shape()[1], 4); // 4 columns
    }

    /// Test hierarchical clustering with complete linkage and SIMD
    #[test]
    fn test_hierarchy_simd_complete() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9],
        )
        .expect("Test: Failed to create data array");

        let result = linkage(data.view(), LinkageMethod::Complete, Metric::Euclidean);
        assert!(result.is_ok(), "Complete linkage should succeed");

        let linkage_matrix = result.expect("Test: Linkage failed");
        assert_eq!(linkage_matrix.shape()[0], 5);
        assert_eq!(linkage_matrix.shape()[1], 4);
    }

    /// Test hierarchical clustering with average linkage and SIMD
    #[test]
    fn test_hierarchy_simd_average() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9],
        )
        .expect("Test: Failed to create data array");

        let result = linkage(data.view(), LinkageMethod::Average, Metric::Euclidean);
        assert!(result.is_ok(), "Average linkage should succeed");

        let linkage_matrix = result.expect("Test: Linkage failed");
        assert_eq!(linkage_matrix.shape()[0], 5);
        assert_eq!(linkage_matrix.shape()[1], 4);
    }

    /// Test hierarchical clustering with Ward linkage and SIMD
    #[test]
    fn test_hierarchy_simd_ward() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9],
        )
        .expect("Test: Failed to create data array");

        let result = linkage(data.view(), LinkageMethod::Ward, Metric::Euclidean);
        assert!(result.is_ok(), "Ward linkage should succeed");

        let linkage_matrix = result.expect("Test: Linkage failed");
        assert_eq!(linkage_matrix.shape()[0], 5);
        assert_eq!(linkage_matrix.shape()[1], 4);
    }

    /// Test SIMD with various data sizes (including non-aligned)
    #[test]
    fn test_simd_various_sizes() {
        let sizes = vec![7, 13, 17, 31, 63]; // Intentionally non-power-of-2 sizes

        for size in sizes {
            // Generate test data
            let mut data_vec = Vec::with_capacity(size * 2);
            for i in 0..size {
                data_vec.push((i % 5) as f64);
                data_vec.push(((i + 2) % 5) as f64);
            }

            let data =
                Array2::from_shape_vec((size, 2), data_vec).expect("Test: Failed to create data");

            // Test DBSCAN
            let dbscan_result = dbscan(data.view(), 1.5, 2, Some(DistanceMetric::Euclidean));
            assert!(
                dbscan_result.is_ok(),
                "DBSCAN should work with size {}",
                size
            );

            // Test hierarchical clustering (only for smaller sizes due to O(n^3) complexity)
            if size <= 20 {
                let hier_result = linkage(data.view(), LinkageMethod::Single, Metric::Euclidean);
                assert!(
                    hier_result.is_ok(),
                    "Hierarchical clustering should work with size {}",
                    size
                );
            }
        }
    }

    /// Test edge cases with SIMD
    #[test]
    fn test_simd_edge_cases() {
        // Test with minimum size data
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0])
            .expect("Test: Failed to create data");

        let dbscan_result = dbscan(data.view(), 1.0, 2, Some(DistanceMetric::Euclidean));
        assert!(dbscan_result.is_ok(), "DBSCAN should work with 3 points");

        let hier_result = linkage(data.view(), LinkageMethod::Single, Metric::Euclidean);
        assert!(
            hier_result.is_ok(),
            "Hierarchical clustering should work with 3 points"
        );
    }

    /// Test with high-dimensional data
    #[test]
    fn test_simd_high_dimensional() {
        // 10 points in 10D space
        let n_samples = 10;
        let n_features = 10;
        let mut data_vec = Vec::with_capacity(n_samples * n_features);

        for i in 0..n_samples {
            for j in 0..n_features {
                data_vec.push((i as f64 + j as f64 * 0.1) % 10.0);
            }
        }

        let data = Array2::from_shape_vec((n_samples, n_features), data_vec)
            .expect("Test: Failed to create data");

        // Test DBSCAN
        let dbscan_result = dbscan(data.view(), 2.0, 2, Some(DistanceMetric::Euclidean));
        assert!(
            dbscan_result.is_ok(),
            "DBSCAN should work with high-dimensional data"
        );

        // Test GMM
        let gmm_options = GMMOptions {
            n_components: 2,
            max_iter: 20,
            random_seed: Some(42),
            ..Default::default()
        };

        let gmm_result = gaussian_mixture(data.view(), gmm_options);
        assert!(
            gmm_result.is_ok(),
            "GMM should work with high-dimensional data"
        );
    }

    /// Test consistency between f32 and f64 SIMD paths
    #[test]
    fn test_simd_f32_f64_consistency() {
        // Test with f64
        let data_f64 = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.1, 1.1, 0.9, 0.9, 1.2, 0.8, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9, 5.2, 4.8,
            ],
        )
        .expect("Test: Failed to create f64 data");

        let labels_f64 = dbscan(data_f64.view(), 0.5, 2, Some(DistanceMetric::Euclidean))
            .expect("Test: DBSCAN f64 failed");

        // Test with f32
        let data_f32 = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0f32, 1.0, 1.1, 1.1, 0.9, 0.9, 1.2, 0.8, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9, 5.2, 4.8,
            ],
        )
        .expect("Test: Failed to create f32 data");

        let labels_f32 = dbscan(data_f32.view(), 0.5f32, 2, Some(DistanceMetric::Euclidean))
            .expect("Test: DBSCAN f32 failed");

        // Both should produce valid results
        assert_eq!(labels_f64.len(), 8);
        assert_eq!(labels_f32.len(), 8);

        // The cluster structure should be similar (allowing for numerical differences)
        let unique_f64: std::collections::HashSet<_> = labels_f64.iter().cloned().collect();
        let unique_f32: std::collections::HashSet<_> = labels_f32.iter().cloned().collect();

        // Should have similar number of unique labels
        assert_eq!(
            unique_f64.len(),
            unique_f32.len(),
            "f32 and f64 should find similar cluster structures"
        );
    }
}

#[cfg(not(feature = "simd"))]
mod no_simd_tests {
    /// Placeholder test when SIMD is not enabled
    #[test]
    fn test_simd_not_enabled() {
        // This test just ensures the test suite runs even without SIMD
    }
}
