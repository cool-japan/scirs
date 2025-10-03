//! SIMD Optimization Demonstration
//!
//! This example demonstrates the SIMD optimization capabilities of the clustering algorithms,
//! comparing performance between standard and SIMD-accelerated implementations.

use scirs2_cluster::vq::{
    kmeans, kmeans_simd, vq, vq_simd, whiten, whiten_simd, KMeansInit, KMeansOptions,
    SimdOptimizationConfig,
};
use scirs2_core::ndarray::Array2;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 SIMD Optimization Demonstration for Clustering Algorithms");
    println!("============================================================");

    // Generate synthetic dataset for testing
    let (data, expected_labels) = generate_synthetic_data(1000, 10, 5);
    println!(
        "📊 Generated dataset: {} samples, {} features, {} clusters",
        data.nrows(),
        data.ncols(),
        5
    );

    // Configure SIMD optimizations
    let simd_config = SimdOptimizationConfig {
        simd_threshold: 32,
        enable_parallel: true,
        parallel_chunk_size: 256,
        cache_friendly: true,
        force_simd: false,
    };

    println!("\n⚙️  SIMD Configuration:");
    println!("   • SIMD Threshold: {}", simd_config.simd_threshold);
    println!("   • Parallel Processing: {}", simd_config.enable_parallel);
    println!(
        "   • Parallel Chunk Size: {}",
        simd_config.parallel_chunk_size
    );
    println!("   • Cache Friendly: {}", simd_config.cache_friendly);

    // Test 1: Data Whitening Performance
    println!("\n🧪 Test 1: Data Whitening Performance");
    println!("────────────────────────────────────");

    let start = Instant::now();
    let whitened_standard = whiten(&data)?;
    let standard_time = start.elapsed();

    let start = Instant::now();
    let whitened_simd = whiten_simd(&data, Some(&simd_config))?;
    let simd_time = start.elapsed();

    println!("   Standard whitening: {:?}", standard_time);
    println!("   SIMD whitening:     {:?}", simd_time);

    if simd_time < standard_time {
        let speedup = standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        println!("   🎯 SIMD speedup: {:.2}x faster", speedup);
    } else {
        println!("   📝 Standard implementation was faster (possibly due to overhead)");
    }

    // Verify results are equivalent
    let diff = (&whitened_standard - &whitened_simd)
        .mapv(|x| x.abs())
        .sum();
    println!(
        "   ✅ Results difference: {:.2e} (should be close to 0)",
        diff
    );

    // Test 2: Vector Quantization Performance
    println!("\n🧪 Test 2: Vector Quantization Performance");
    println!("─────────────────────────────────────────");

    // Generate centroids for testing
    let centroids = generate_centroids(5, 10);

    let start = Instant::now();
    let (labels_standard, distances_standard) = vq(whitened_standard.view(), centroids.view())?;
    let standard_time = start.elapsed();

    let start = Instant::now();
    let (labels_simd, distances_simd) =
        vq_simd(whitened_simd.view(), centroids.view(), Some(&simd_config))?;
    let simd_time = start.elapsed();

    println!("   Standard VQ: {:?}", standard_time);
    println!("   SIMD VQ:     {:?}", simd_time);

    if simd_time < standard_time {
        let speedup = standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        println!("   🎯 SIMD speedup: {:.2}x faster", speedup);
    } else {
        println!("   📝 Standard implementation was faster");
    }

    // Verify results are equivalent
    let label_diff = labels_standard
        .iter()
        .zip(labels_simd.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "   ✅ Label differences: {} (should be 0 or very small)",
        label_diff
    );

    // Test 3: Full K-means Performance
    println!("\n🧪 Test 3: Complete K-means Algorithm Performance");
    println!("──────────────────────────────────────────────────");

    let kmeans_options = KMeansOptions {
        max_iter: 100,
        tol: 1e-4,
        random_seed: Some(42),
        n_init: 1,
        init_method: KMeansInit::KMeansPlusPlus,
        // _cluster: removed invalid field
    };

    let start = Instant::now();
    let (centroids_standard, distortion_standard) =
        kmeans(data.view(), 5, Some(100), Some(1e-4), Some(true), Some(42))?;
    let standard_time = start.elapsed();

    let start = Instant::now();
    let (centroids_simd, labels_simd, inertia_simd) = kmeans_simd(
        data.view(),
        5,
        Some(kmeans_options),
        Some(simd_config.clone()),
    )?;
    let simd_time = start.elapsed();

    println!("   Standard K-means: {:?}", standard_time);
    println!("   SIMD K-means:     {:?}", simd_time);

    if simd_time < standard_time {
        let speedup = standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        println!("   🎯 SIMD speedup: {:.2}x faster", speedup);
    } else {
        println!("   📝 Standard implementation was faster");
    }

    println!("   📊 Standard distortion: {:.6}", distortion_standard);
    println!("   📊 SIMD inertia:        {:.6}", inertia_simd);

    // Test 4: Scalability Analysis
    println!("\n🧪 Test 4: Scalability Analysis");
    println!("────────────────────────────────");

    let sizes = [100, 500, 1000, 5000];

    for &size in &sizes {
        let (test_data, _labels) = generate_synthetic_data(size, 10, 3);

        let start = Instant::now();
        let _ = kmeans(
            test_data.view(),
            3,
            Some(50),
            Some(1e-4),
            Some(true),
            Some(42),
        )?;
        let standard_time = start.elapsed();

        let start = Instant::now();
        let _ = kmeans_simd(test_data.view(), 3, None, Some(simd_config.clone()))?;
        let simd_time = start.elapsed();

        let speedup = if simd_time.as_nanos() > 0 {
            standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64
        } else {
            0.0
        };

        println!(
            "   {} samples: Standard {:?}, SIMD {:?} (speedup: {:.2}x)",
            size, standard_time, simd_time, speedup
        );
    }

    // Test 5: Different SIMD Configurations
    println!("\n🧪 Test 5: SIMD Configuration Impact");
    println!("────────────────────────────────────");

    let configs = [
        (
            "Conservative",
            SimdOptimizationConfig {
                simd_threshold: 128,
                enable_parallel: false,
                parallel_chunk_size: 512,
                cache_friendly: true,
                force_simd: false,
            },
        ),
        (
            "Aggressive",
            SimdOptimizationConfig {
                simd_threshold: 16,
                enable_parallel: true,
                parallel_chunk_size: 128,
                cache_friendly: true,
                force_simd: true,
            },
        ),
        (
            "Parallel-focused",
            SimdOptimizationConfig {
                simd_threshold: 64,
                enable_parallel: true,
                parallel_chunk_size: 64,
                cache_friendly: false,
                force_simd: false,
            },
        ),
    ];

    for (name, config) in &configs {
        let start = Instant::now();
        let _ = kmeans_simd(data.view(), 5, None, Some(config.clone()))?;
        let time = start.elapsed();

        println!("   {}: {:?}", name, time);
    }

    // Test 6: Memory Usage Comparison
    println!("\n🧪 Test 6: Feature Capabilities Report");
    println!("──────────────────────────────────────");

    #[cfg(target_arch = "x86_64")]
    {
        println!("   🔧 Target Architecture: x86_64");
        println!("   🎯 SIMD Instructions: Available (likely SSE/AVX)");
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("   🔧 Target Architecture: AArch64");
        println!("   🎯 SIMD Instructions: Available (NEON)");
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("   🔧 Target Architecture: Other");
        println!("   ⚠️  SIMD Instructions: May not be available");
    }

    #[allow(unexpected_cfgs)]
    let parallel_status = if cfg!(feature = "parallel") {
        "Available"
    } else {
        "Core abstractions used"
    };

    println!("   📦 Parallel Processing: {}", parallel_status);

    println!("\n✅ SIMD Optimization Demonstration Complete!");
    println!("────────────────────────────────────────────");
    println!("The SIMD optimizations provide:");
    println!("• 🚀 Faster distance calculations using vector instructions");
    println!("• ⚡ Parallel processing for large datasets");
    println!("• 🎯 Automatic fallback to scalar implementations when needed");
    println!("• 🧠 Cache-friendly memory access patterns");
    println!("• 📊 Consistent results with improved performance");

    Ok(())
}

/// Generate synthetic clustered data for testing
#[allow(dead_code)]
fn generate_synthetic_data(
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
) -> (Array2<f64>, Vec<usize>) {
    use scirs2_core::random::Rng;

    let mut rng = scirs2_core::random::rng();
    let mut data = Array2::zeros((n_samples, n_features));
    let mut labels = Vec::with_capacity(n_samples);

    // Generate cluster centers
    let mut centers = Array2::zeros((n_clusters, n_features));
    for i in 0..n_clusters {
        for j in 0..n_features {
            centers[[i, j]] = rng.random_range(-10.0..10.0);
        }
    }

    // Generate points around centers
    let samples_per_cluster = n_samples / n_clusters;
    let mut sample_idx = 0;

    for cluster in 0..n_clusters {
        let end_idx = if cluster == n_clusters - 1 {
            n_samples
        } else {
            sample_idx + samples_per_cluster
        };

        while sample_idx < end_idx {
            for j in 0..n_features {
                let noise = rng.random_range(-1.0..1.0);
                data[[sample_idx, j]] = centers[[cluster, j]] + noise;
            }
            labels.push(cluster);
            sample_idx += 1;
        }
    }

    (data, labels)
}

/// Generate random centroids for testing
#[allow(dead_code)]
fn generate_centroids(n_clusters: usize, nfeatures: usize) -> Array2<f64> {
    use scirs2_core::random::Rng;

    let mut rng = scirs2_core::random::rng();
    let mut centroids = Array2::zeros((n_clusters, nfeatures));

    for i in 0..n_clusters {
        for j in 0..nfeatures {
            centroids[[i, j]] = rng.random_range(-5.0..5.0);
        }
    }

    centroids
}
