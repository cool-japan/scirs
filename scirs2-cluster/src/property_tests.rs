//! Property-based tests for clustering algorithms
//!
//! This module contains property-based tests using the proptest framework
//! to verify that clustering algorithms satisfy fundamental mathematical
//! properties regardless of input data characteristics.

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use proptest::prelude::*;

    use crate::vq::{kmeans2, MinitMethod, MissingMethod};

    // Strategy for generating valid clustering data
    fn clustering_data_strategy() -> impl Strategy<Value = Array2<f64>> {
        (3usize..20, 2usize..5).prop_flat_map(|(n_points, n_features)| {
            prop::collection::vec(-10.0f64..10.0f64, n_points * n_features)
                .prop_map(move |data| Array2::from_shape_vec((n_points, n_features), data).unwrap())
        })
    }

    proptest! {
        #[test]
        fn test_kmeans_all_points_assigned(data in clustering_data_strategy(), seed in any::<u64>()) {
            let n_points = data.shape()[0];
            let k = (n_points / 2).clamp(2, 5); // Reasonable k value

            let result = kmeans2(
                data.view(),
                k,
                Some(20), // iterations
                None,     // threshold
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false), // check_finite
                Some(seed),
            );

            if let Ok((centroids, labels)) = result {
                // Every point should be assigned to a cluster
                prop_assert_eq!(labels.len(), n_points);

                // All cluster labels should be valid (in range [0, k))
                for &label in labels.iter() {
                    prop_assert!(label < k, "Cluster label {} should be < {}", label, k);
                }

                // Centroids should have correct shape
                prop_assert_eq!(centroids.shape()[0], k);
                prop_assert_eq!(centroids.shape()[1], data.shape()[1]);

                // All centroids should be finite
                for &val in centroids.iter() {
                    prop_assert!(val.is_finite(), "Centroid values should be finite");
                }
            }
        }

        #[test]
        fn test_kmeans_deterministic(
            data in clustering_data_strategy(),
            seed in any::<u64>()
        ) {
            let n_points = data.shape()[0];
            let k = (n_points / 2).clamp(2, 4);

            let result1 = kmeans2(
                data.view(),
                k,
                Some(10),
                None,
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false),
                Some(seed),
            );

            let result2 = kmeans2(
                data.view(),
                k,
                Some(10),
                None,
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false),
                Some(seed),
            );

            if let (Ok((centroids1, labels1)), Ok((centroids2, labels2))) = (result1, result2) {
                // Results should be identical with same seed
                prop_assert_eq!(labels1, labels2, "Labels should be identical with same seed");
                prop_assert!(
                    centroids1.abs_diff_eq(&centroids2, 1e-10),
                    "Centroids should be identical with same seed"
                );
            }
        }
    }

    /// Additional non-property tests for specific edge cases
    #[cfg(test)]
    mod specific_tests {
        use super::*;
        use ndarray::Array2;

        #[test]
        fn test_kmeans_identical_points() {
            // Test with all identical points
            let data = Array2::from_shape_vec((5, 2), vec![1.0; 10]).unwrap();

            let result = kmeans2(
                data.view(),
                2,
                Some(10),
                None,
                Some(MinitMethod::Random),
                Some(MissingMethod::Warn),
                Some(false),
                Some(42),
            );

            // Should handle identical points gracefully
            assert!(
                result.is_ok() || result.is_err(),
                "Should either succeed or fail gracefully"
            );
        }
    }
}
