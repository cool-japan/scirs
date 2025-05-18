//! Unit tests for clustering algorithms

#[cfg(test)]
mod test {
    use crate::metrics::silhouette_score;
    use crate::vq::{kmeans2, whiten, MinitMethod, MissingMethod};
    use ndarray::{array, Array2};

    #[test]
    fn test_whiten() {
        let data: Array2<f64> =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.5, 2.5, 0.5, 1.5, 2.0, 3.0]).unwrap();

        let whitened = whiten(&data).unwrap();

        // Check that whitened data has unit variance
        let n_features = whitened.shape()[1];
        for j in 0..n_features {
            let column = whitened.column(j);
            let mean: f64 = column.mean().unwrap();
            let var: f64 = column.var(1.0);

            assert!((mean.abs()) < 1e-6, "Mean should be close to 0");
            assert!((var - 1.0).abs() < 1e-6, "Variance should be close to 1");
        }
    }

    #[test]
    fn test_kmeans2_all_init_methods() {
        let data =
            Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64 / 10.0).collect()).unwrap();

        let init_methods = vec![
            MinitMethod::Random,
            MinitMethod::Points,
            MinitMethod::PlusPlus,
        ];

        for method in init_methods {
            let (centroids, labels) = kmeans2(
                data.view(),
                3,
                Some(10),
                Some(1e-4),
                Some(method),
                Some(MissingMethod::Warn),
                Some(true),
                Some(42),
            )
            .unwrap();

            assert_eq!(centroids.shape()[0], 3);
            assert_eq!(centroids.shape()[1], 2);
            assert_eq!(labels.len(), 20);
        }
    }

    #[test]
    fn test_kmeans2_empty_cluster_handling() {
        // Create data that will likely result in empty clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 10.0, 10.0, 10.1, 10.1, 10.2, 10.2,
            ],
        )
        .unwrap();

        // Test with warning on empty clusters
        let result1 = kmeans2(
            data.view(),
            3,
            Some(5),
            Some(1e-4),
            Some(MinitMethod::Random),
            Some(MissingMethod::Warn),
            Some(true),
            Some(123),
        );

        // Should succeed with warning
        assert!(result1.is_ok());

        // Test with raising on empty clusters
        let result2 = kmeans2(
            data.view(),
            4, // More clusters than natural groups
            Some(5),
            Some(1e-4),
            Some(MinitMethod::Random),
            Some(MissingMethod::Raise),
            Some(true),
            Some(456),
        );

        // Might fail if empty cluster is created
        match result2 {
            Ok(_) => println!("Succeeded without empty clusters"),
            Err(e) => println!("Failed as expected: {}", e),
        }
    }

    #[test]
    fn test_silhouette_score() {
        // Create two well-separated clusters
        let data = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [1.2, 1.3],
            [10.0, 10.0],
            [10.5, 10.5],
            [10.2, 10.3],
        ];

        let labels = array![0, 0, 0, 1, 1, 1];

        let score = silhouette_score(data.view(), labels.view()).unwrap();

        // Well-separated clusters should have high silhouette score
        assert!(
            score > 0.8,
            "Silhouette score should be high for well-separated clusters"
        );
    }

    #[test]
    fn test_silhouette_with_noise() {
        // Create data with noise points (label -1)
        let data = array![
            [1.0, 1.0],
            [1.5, 1.5],
            [10.0, 10.0],
            [10.5, 10.5],
            [50.0, 50.0], // noise
        ];

        let labels = array![0, 0, 1, 1, -1];

        let score = silhouette_score(data.view(), labels.view()).unwrap();

        // Should handle noise points correctly
        assert!(score > 0.0);
    }
}
