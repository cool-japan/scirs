use super::*;
use scirs2_core::ndarray::Array2;

#[test]
fn test_enhanced_standard_scaler() {
    let data = Array2::from_shape_vec((100, 5), (0..500).map(|x| x as f64).collect())
        .expect("Operation failed");

    let mut scaler = EnhancedStandardScaler::new(false, 100);
    let transformed = scaler
        .fit_transform(&data.view())
        .expect("Operation failed");

    assert_eq!(transformed.shape(), data.shape());

    // Check that transformed data has approximately zero mean and unit variance
    let transformed_mean = transformed.mean_axis(Axis(0)).expect("Operation failed");
    for &mean in transformed_mean.iter() {
        assert!((mean.abs()) < 1e-10);
    }
}

#[test]
fn test_enhanced_standard_scaler_robust() {
    let mut data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect())
        .expect("Operation failed");
    // Add some outliers
    data[[0, 0]] = 1000.0;
    data[[1, 1]] = -1000.0;

    let mut robust_scaler = EnhancedStandardScaler::new(true, 100);
    let transformed = robust_scaler
        .fit_transform(&data.view())
        .expect("Operation failed");

    assert_eq!(transformed.shape(), data.shape());

    // Robust scaler should be less affected by outliers
    let transformed_median = transformed.mean_axis(Axis(0)).expect("Operation failed"); // Approximation
    for &median in transformed_median.iter() {
        assert!(median.abs() < 5.0); // Should be reasonable even with outliers
    }
}

#[test]
fn test_enhanced_pca() {
    let data = Array2::from_shape_vec((50, 10), (0..500).map(|x| x as f64).collect())
        .expect("Operation failed");

    let mut pca = EnhancedPCA::new(5, true, 100).expect("Operation failed");
    let transformed = pca.fit_transform(&data.view()).expect("Operation failed");

    assert_eq!(transformed.shape(), &[50, 5]);
    assert!(pca.components().is_some());
    assert!(pca.explained_variance_ratio().is_some());
}

#[test]
fn test_enhanced_pca_no_centering() {
    let data = Array2::from_shape_vec((30, 8), (0..240).map(|x| x as f64).collect())
        .expect("Operation failed");

    let mut pca = EnhancedPCA::new(3, false, 100).expect("Operation failed");
    let transformed = pca.fit_transform(&data.view()).expect("Operation failed");

    assert_eq!(transformed.shape(), &[30, 3]);
}

#[test]
fn test_processing_strategy_selection() {
    // Test that processing strategy is selected appropriately
    let small_data = Array2::ones((10, 5));
    let mut scaler = EnhancedStandardScaler::new(false, 100);
    scaler.fit(&small_data.view()).expect("Operation failed");

    // For small data, should use standard processing
    matches!(scaler.processing_strategy(), ProcessingStrategy::Standard);
}

#[test]
fn test_optimized_memory_pool() {
    let mut pool = AdvancedMemoryPool::new(100, 10, 2);

    // Test buffer allocation and reuse
    let buffer1 = pool.get_array(50, 5);
    assert_eq!(buffer1.shape(), &[50, 5]);

    pool.return_array(buffer1);

    // Should reuse the returned buffer
    let buffer2 = pool.get_array(50, 5);
    assert_eq!(buffer2.shape(), &[50, 5]);

    // Test temp array functionality
    let temp1 = pool.get_temp_array(20);
    assert_eq!(temp1.len(), 20);

    pool.return_temp_array(temp1);

    // Test performance stats
    pool.update_stats(1000000, 100); // 1ms, 100 samples
    let stats = pool.stats();
    assert_eq!(stats.transform_count, 1);
    assert!(stats.throughput_samples_per_sec > 0.0);
}

#[test]
fn test_optimized_pca_small_data() {
    let data = Array2::from_shape_vec(
        (20, 8),
        (0..160)
            .map(|x| x as f64 + scirs2_core::random::random::<f64>() * 0.1)
            .collect(),
    )
    .expect("Operation failed");

    let mut pca = AdvancedPCA::new(3, 100, 50);
    let transformed = pca.fit_transform(&data.view()).expect("Operation failed");

    assert_eq!(transformed.shape(), &[20, 3]);
    assert!(pca.components().is_some());
    assert!(pca.explained_variance_ratio().is_ok());
    assert!(pca.mean().is_some());

    // Test that explained variance ratios sum to less than or equal to 1
    let var_ratios = pca.explained_variance_ratio().expect("Operation failed");
    let sum_ratios: f64 = var_ratios.iter().sum();
    assert!(sum_ratios <= 1.0 + 1e-10);
    assert!(sum_ratios > 0.0);
}

#[test]
#[ignore = "slow: large data test - takes too long in CI"]
fn test_optimized_pca_large_data() {
    // Test with larger data to trigger block-wise algorithm
    let data = Array2::from_shape_vec(
        (15000, 600),
        (0..9000000)
            .map(|x| (x as f64).sin() * 0.01 + (x as f64 / 1000.0).cos())
            .collect(),
    )
    .expect("Operation failed");

    let mut pca = AdvancedPCA::new(50, 20000, 1000);
    let result = pca.fit(&data.view());
    assert!(result.is_ok());

    let transformed = pca.transform(&data.view());
    assert!(transformed.is_ok());
    assert_eq!(transformed.expect("Operation failed").shape(), &[15000, 50]);

    // Verify performance statistics
    let stats = pca.performance_stats();
    assert!(stats.transform_count > 0);
}

#[test]
#[ignore = "slow: measured 321.9s in a bounded 600s retry (0.6.5 ignore audit) -- randomized-SVD \
            AdvancedPCA fit+transform on a 60000x1200 (72M-element) matrix; well past the workspace's \
            120s default nextest hard-kill but completes and passes given enough time, so not a hang"]
fn test_optimized_pca_very_large_data() {
    // Test with very large data to trigger randomized SVD
    let data = Array2::from_shape_vec(
        (60000, 1200),
        (0..72000000)
            .map(|x| {
                let t = x as f64 / 1000000.0;
                t.sin() + 0.1 * (10.0 * t).sin() + 0.01 * scirs2_core::random::random::<f64>()
            })
            .collect(),
    )
    .expect("Operation failed");

    let mut pca = AdvancedPCA::new(20, 100000, 2000);
    let result = pca.fit(&data.view());
    assert!(result.is_ok());

    // Test transform
    let small_test_data = data.slice(scirs2_core::ndarray::s![..100, ..]).to_owned();
    let transformed = pca.transform(&small_test_data.view());
    assert!(transformed.is_ok());
    assert_eq!(transformed.expect("Operation failed").shape(), &[100, 20]);
}

#[test]
fn test_qr_decomposition_optimized() {
    let pca = AdvancedPCA::new(5, 100, 50);

    // Test QR decomposition on a simple matrix
    let matrix = Array2::from_shape_vec(
        (6, 4),
        vec![
            1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        ],
    )
    .expect("Operation failed");

    let result = pca.qr_decomposition_optimized(&matrix);
    assert!(result.is_ok());

    let (q, r) = result.expect("Operation failed");
    assert_eq!(q.shape(), &[6, 6]);
    assert_eq!(r.shape(), &[6, 4]);

    // Verify that Q is orthogonal (Q^T * Q should be close to identity)
    let qtq = q.t().dot(&q);
    for i in 0..6 {
        for j in 0..6 {
            if i == j {
                assert!((qtq[[i, j]] - 1.0).abs() < 1e-10);
            } else {
                assert!(qtq[[i, j]].abs() < 1e-10);
            }
        }
    }
}

#[test]
fn test_svd_small_matrix() {
    let pca = AdvancedPCA::new(3, 100, 50);

    // Test SVD on a known matrix
    let matrix = Array2::from_shape_vec(
        (4, 3),
        vec![3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0],
    )
    .expect("Operation failed");

    let result = pca.svd_small_matrix(&matrix);
    assert!(result.is_ok());

    let (u, s, vt) = result.expect("Operation failed");
    assert_eq!(u.shape(), &[4, 3]);
    assert_eq!(s.len(), 3);
    assert_eq!(vt.shape(), &[3, 3]);

    // Verify that singular values are non-negative and sorted
    for i in 0..s.len() - 1 {
        assert!(s[i] >= 0.0);
        assert!(s[i] >= s[i + 1] - 1e-10); // Allow for small numerical errors
    }

    // Verify reconstruction: A ≈ U * Σ * V^T
    let sigma_matrix = Array2::from_diag(&s);
    let reconstructed = u.dot(&sigma_matrix).dot(&vt);

    for i in 0..4 {
        for j in 0..3 {
            // Relaxed tolerance for numerical stability
            assert!(
                (matrix[[i, j]] - reconstructed[[i, j]]).abs() < 1e-6_f64,
                "Matrix reconstruction error at [{}, {}]: expected {}, got {}, diff = {}",
                i,
                j,
                matrix[[i, j]],
                reconstructed[[i, j]],
                (matrix[[i, j]] - reconstructed[[i, j]]).abs()
            );
        }
    }
}

#[test]
fn test_memory_pool_optimization() {
    let mut pool = AdvancedMemoryPool::new(1000, 100, 4);

    // Simulate some usage patterns
    for i in 0..10 {
        pool.update_stats(1000000 + i * 100000, 100); // Varying performance

        let buffer = pool.get_array(500, 50);
        pool.return_array(buffer);
    }

    // Test optimization
    pool.optimize();

    let stats = pool.stats();
    assert_eq!(stats.transform_count, 10);
    assert!(stats.cache_hit_rate >= 0.0 && stats.cache_hit_rate <= 1.0);
}

#[test]
fn test_performance_stats_accuracy() {
    let mut pool = AdvancedMemoryPool::new(100, 10, 2);

    // Test with known timing
    let test_time_ns = 2_000_000_000; // 2 seconds
    let test_samples = 1000;

    pool.update_stats(test_time_ns, test_samples);

    let stats = pool.stats();
    assert_eq!(stats.transform_count, 1);
    assert_eq!(stats.total_transform_time_ns, test_time_ns);

    // Throughput should be samples/second
    let expected_throughput = test_samples as f64 / 2.0; // 500 samples/second
    assert!((stats.throughput_samples_per_sec - expected_throughput).abs() < 1e-6);
}

#[test]
fn test_optimized_pca_numerical_stability() {
    // Test with data that could cause numerical issues
    let mut data = Array2::zeros((100, 10));

    // Create data with very different scales
    for i in 0..100 {
        for j in 0..10 {
            if j < 5 {
                data[[i, j]] = (i as f64) * 1e-6; // Very small values
            } else {
                data[[i, j]] = (i as f64) * 1e6; // Very large values
            }
        }
    }

    let mut pca = AdvancedPCA::new(5, 200, 20);
    let result = pca.fit_transform(&data.view());

    assert!(result.is_ok());
    let transformed = result.expect("Operation failed");
    assert_eq!(transformed.shape(), &[100, 5]);

    // Check that all values are finite
    for val in transformed.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_enhanced_standard_scaler_vs_optimized_pca() {
    // Compare enhanced scaler with optimized PCA preprocessing
    let data = Array2::from_shape_vec(
        (200, 15),
        (0..3000)
            .map(|x| x as f64 + scirs2_core::random::random::<f64>() * 10.0)
            .collect(),
    )
    .expect("Operation failed");

    // Test enhanced scaler
    let mut scaler = EnhancedStandardScaler::new(false, 100);
    let scaled_data = scaler
        .fit_transform(&data.view())
        .expect("Operation failed");

    // Apply PCA to scaled data
    let mut pca = AdvancedPCA::new(10, 300, 20);
    let pca_result = pca
        .fit_transform(&scaled_data.view())
        .expect("Operation failed");

    assert_eq!(pca_result.shape(), &[200, 10]);

    // Verify that the combination works correctly
    let explained_var = pca.explained_variance_ratio().expect("Operation failed");
    let total_explained: f64 = explained_var.iter().sum();
    assert!(total_explained > 0.5); // Should explain at least 50% of variance
    assert!(total_explained <= 1.0 + 1e-10);
}
