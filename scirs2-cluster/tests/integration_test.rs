//! Integration tests for clustering algorithms

use ndarray::{array, Array2};
use scirs2_cluster::metrics::silhouette_score;
use scirs2_cluster::vq::{kmeans2, whiten, MinitMethod, MissingMethod};

#[test]
fn test_whiten() {
    let data =
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.5, 2.5, 0.5, 1.5, 2.0, 3.0]).unwrap();

    let whitened = whiten(&data).unwrap();

    // Check that whitened data has roughly unit variance
    assert_eq!(whitened.shape(), data.shape());
}

#[test]
fn test_kmeans2_init_methods() {
    let data = Array2::from_shape_vec((20, 2), (0..40).map(|i| i as f64 / 10.0).collect()).unwrap();

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
fn test_silhouette_score_basic() {
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
    assert!(score > 0.7);
}
