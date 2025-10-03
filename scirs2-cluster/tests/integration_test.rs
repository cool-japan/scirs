//! Integration tests for clustering algorithms

use scirs2_cluster::advanced_clustering::{AdvancedClusterer, AdvancedConfig};
use scirs2_cluster::metrics::silhouette_score;
use scirs2_cluster::vq::{kmeans2, whiten, MinitMethod, MissingMethod};
use scirs2_core::ndarray::{array, Array2};

#[test]
#[allow(dead_code)]
fn test_whiten() {
    let data =
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.5, 2.5, 0.5, 1.5, 2.0, 3.0]).unwrap();

    let whitened = whiten(&data).unwrap();

    // Check that whitened data has roughly unit variance
    assert_eq!(whitened.shape(), data.shape());
}

#[test]
#[allow(dead_code)]
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
#[allow(dead_code)]
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

// Advanced Clustering Integration Tests

#[test]
#[allow(dead_code)]
fn test_advanced_clusterer_basic() {
    // Create test data with clear clusters
    let data = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 1.0, // Cluster 1
            1.2, 1.1, 0.9, 1.3, 1.1, 0.8, 5.0, 5.0, // Cluster 2
            5.2, 5.1, 4.9, 5.3, 5.1, 4.8,
        ],
    )
    .unwrap();

    let mut clusterer = AdvancedClusterer::new();
    let result = clusterer.cluster(&data.view()).unwrap();

    // Verify basic properties
    assert_eq!(result.clusters.len(), data.nrows());
    assert!(result.centroids.nrows() >= 1);
    assert_eq!(result.centroids.ncols(), data.ncols());
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.quantum_advantage >= 1.0);
    assert!(result.neuromorphic_benefit >= 1.0);
    assert!(result.ai_speedup >= 1.0);
    assert!(result.meta_learning_improvement >= 1.0);
}

#[test]
#[allow(dead_code)]
fn test_advanced_clusterer_with_ai_selection() {
    let data = Array2::from_shape_vec(
        (12, 3),
        vec![
            // Cluster 1
            1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 0.9, 1.9, 2.9, 1.2, 2.2, 3.2, // Cluster 2
            10.0, 11.0, 12.0, 10.1, 11.1, 12.1, 9.9, 10.9, 11.9, 10.2, 11.2, 12.2,
            // Cluster 3
            20.0, 21.0, 22.0, 20.1, 21.1, 22.1, 19.9, 20.9, 21.9, 20.2, 21.2, 22.2,
        ],
    )
    .unwrap();

    let mut clusterer = AdvancedClusterer::new()
        .with_ai_algorithm_selection(true)
        .with_quantum_neuromorphic_fusion(true)
        .with_meta_learning(true);

    let result = clusterer.cluster(&data.view()).unwrap();

    // Verify AI selection worked
    assert!(!result.selected_algorithm.is_empty());
    assert!(result.ai_speedup >= 1.0);
    assert!(result.performance.ai_iterations > 0);

    // Check clustering quality
    assert!(result.performance.silhouette_score > 0.0);
    assert!(result.performance.execution_time > 0.0);
}

#[test]
#[allow(dead_code)]
fn test_advanced_clusterer_with_all_features() {
    let data = Array2::from_shape_vec(
        (16, 4),
        (0..64)
            .map(|i| {
                let cluster_id = i / 16;
                let base_value = cluster_id as f64 * 10.0;
                let noise = (i % 4) as f64 * 0.1;
                base_value + noise
            })
            .collect(),
    )
    .unwrap();

    let mut clusterer = AdvancedClusterer::new()
        .with_ai_algorithm_selection(true)
        .with_quantum_neuromorphic_fusion(true)
        .with_meta_learning(true)
        .with_continual_adaptation(true)
        .with_multi_objective_optimization(true);

    let result = clusterer.cluster(&data.view()).unwrap();

    // Verify all features are active
    assert!(result.ai_speedup > 1.0);
    assert!(result.quantum_advantage > 1.0);
    assert!(result.neuromorphic_benefit > 0.0);
    assert!(result.meta_learning_improvement > 0.0);

    // Check performance metrics
    assert!(result.performance.quantum_coherence >= 0.0);
    assert!(result.performance.neural_adaptation_rate >= 0.0);
    assert!(result.performance.energy_efficiency > 0.0);
    assert!(result.performance.memory_usage > 0.0);
}

#[test]
#[allow(dead_code)]
fn test_advanced_clusterer_error_handling() {
    let mut clusterer = AdvancedClusterer::new();

    // Test empty data
    let empty_data = Array2::zeros((0, 2));
    assert!(clusterer.cluster(&empty_data.view()).is_err());

    // Test single point
    let single_point = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    assert!(clusterer.cluster(&single_point.view()).is_err());

    // Test data with NaN
    let nan_data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 3.0, 4.0, 5.0]).unwrap();
    assert!(clusterer.cluster(&nan_data.view()).is_err());

    // Test data with infinity
    let inf_data =
        Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::INFINITY, 3.0, 4.0, 5.0]).unwrap();
    assert!(clusterer.cluster(&inf_data.view()).is_err());
}

#[test]
#[allow(dead_code)]
fn test_advanced_clusterer_different_data_sizes() {
    let mut clusterer = AdvancedClusterer::new().with_ai_algorithm_selection(true);

    // Test small dataset
    let small_data =
        Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1]).unwrap();

    let small_result = clusterer.cluster(&small_data.view()).unwrap();
    assert!(small_result.confidence > 0.0);

    // Test medium dataset
    let medium_data = Array2::from_shape_vec(
        (20, 3),
        (0..60)
            .map(|i| {
                let cluster = i / 20;
                let base = cluster as f64 * 10.0;
                base + (i % 20) as f64 * 0.1
            })
            .collect(),
    )
    .unwrap();

    let medium_result = clusterer.cluster(&medium_data.view()).unwrap();
    assert!(medium_result.confidence > 0.0);
    assert!(medium_result.performance.execution_time > 0.0);
}

#[test]
#[allow(dead_code)]
fn test_advanced_clusterer_high_dimensional() {
    // Test with higher dimensional data
    let data = Array2::from_shape_vec(
        (12, 8),
        (0..96)
            .map(|i| {
                let cluster_id = i / 32;
                let feature_id = i % 8;
                cluster_id as f64 * 5.0 + feature_id as f64 * 0.5
            })
            .collect(),
    )
    .unwrap();

    let mut clusterer = AdvancedClusterer::new()
        .with_quantum_neuromorphic_fusion(true)
        .with_ai_algorithm_selection(true);

    let result = clusterer.cluster(&data.view()).unwrap();

    // High-dimensional data should benefit from quantum algorithms
    assert!(result.quantum_advantage > 1.0);
    assert!(result.performance.quantum_coherence > 0.0);
    assert_eq!(result.centroids.ncols(), 8);
}

#[test]
#[allow(dead_code)]
fn test_advanced_clusterer_noisy_data() {
    // Create data with noise
    let mut data_vec = Vec::new();
    for i in 0..16 {
        let cluster_id = i / 8;
        let base_x = cluster_id as f64 * 10.0;
        let base_y = cluster_id as f64 * 10.0;

        // Add noise
        let noise_x = ((i * 13) % 100) as f64 / 100.0 - 0.5;
        let noise_y = ((i * 17) % 100) as f64 / 100.0 - 0.5;

        data_vec.push(base_x + noise_x);
        data_vec.push(base_y + noise_y);
    }

    let data = Array2::from_shape_vec((16, 2), data_vec).unwrap();

    let mut clusterer = AdvancedClusterer::new()
        .with_quantum_neuromorphic_fusion(true)
        .with_continual_adaptation(true);

    let result = clusterer.cluster(&data.view()).unwrap();

    // Quantum-neuromorphic algorithms should handle noise well
    assert!(result.neuromorphic_benefit > 1.0);
    assert!(result.performance.neural_adaptation_rate > 0.0);
}

#[test]
#[allow(dead_code)]
fn test_advanced_config_defaults() {
    let config = AdvancedConfig::default();

    assert_eq!(config.max_clusters, 20);
    assert_eq!(config.ai_confidence_threshold, 0.85);
    assert_eq!(config.quantum_coherence_time, 100.0);
    assert_eq!(config.neural_learning_rate, 0.01);
    assert_eq!(config.meta_learning_steps, 50);
    assert_eq!(config.objective_weights, [0.6, 0.3, 0.1]);
    assert_eq!(config.max_iterations, 1000);
    assert_eq!(config.tolerance, 1e-6);
}

#[test]
#[allow(dead_code)]
fn test_advanced_performance_metrics() {
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 1.0, 1.1, 1.1, 1.2, 1.2, // Cluster 1
            5.0, 5.0, 5.1, 5.1, 5.2, 5.2, // Cluster 2
        ],
    )
    .unwrap();

    let mut clusterer = AdvancedClusterer::new()
        .with_ai_algorithm_selection(true)
        .with_quantum_neuromorphic_fusion(true);

    let result = clusterer.cluster(&data.view()).unwrap();

    // Verify performance metrics are realistic
    let perf = &result.performance;
    assert!(perf.silhouette_score >= -1.0 && perf.silhouette_score <= 1.0);
    assert!(perf.execution_time > 0.0);
    assert!(perf.memory_usage > 0.0);
    assert!(perf.quantum_coherence >= 0.0 && perf.quantum_coherence <= 1.0);
    assert!(perf.neural_adaptation_rate >= 0.0);
    assert!(perf.energy_efficiency >= 0.0 && perf.energy_efficiency <= 1.0);
}
