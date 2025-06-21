//! SIMD Performance Benchmark - Test SIMD vs Scalar Performance

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_spatial::distance::euclidean;
use scirs2_spatial::simd_distance::{
    parallel_pdist, simd_euclidean_distance, simd_manhattan_distance,
};
use std::time::Duration;

fn generate_test_data(n_points: usize, dimensions: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    Array2::from_shape_fn((n_points, dimensions), |_| rng.random_range(-10.0..10.0))
}

fn simd_vs_scalar_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");
    group.measurement_time(Duration::from_secs(3));

    // Test different vector sizes to show SIMD benefits
    for &size in &[4, 8, 16, 32, 64, 128] {
        let p1: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let p2: Vec<f64> = (0..size).map(|i| (i + 1) as f64).collect();

        // Scalar version
        group.bench_function(format!("scalar_euclidean_{}", size), |b| {
            b.iter(|| black_box(euclidean(&p1, &p2)))
        });

        // SIMD version
        group.bench_function(format!("simd_euclidean_{}", size), |b| {
            b.iter(|| black_box(simd_euclidean_distance(&p1, &p2).unwrap()))
        });

        // SIMD Manhattan
        group.bench_function(format!("simd_manhattan_{}", size), |b| {
            b.iter(|| black_box(simd_manhattan_distance(&p1, &p2).unwrap()))
        });
    }

    group.finish();
}

fn parallel_vs_sequential_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 200, 500];

    for &size in &sizes {
        let points = generate_test_data(size, 10);

        // Sequential (using our manual pdist implementation)
        group.bench_function(format!("sequential_pdist_{}", size), |b| {
            b.iter(|| {
                let mut distances = Vec::new();
                for i in 0..points.nrows() {
                    for j in (i + 1)..points.nrows() {
                        let p1 = points.row(i).to_vec();
                        let p2 = points.row(j).to_vec();
                        distances.push(euclidean(&p1, &p2));
                    }
                }
                black_box(distances.iter().sum::<f64>())
            })
        });

        // Parallel version
        group.bench_function(format!("parallel_pdist_{}", size), |b| {
            b.iter(|| {
                let distances = parallel_pdist(&points.view(), "euclidean").unwrap();
                black_box(distances.sum())
            })
        });
    }

    group.finish();
}

fn architecture_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("architecture");
    group.measurement_time(Duration::from_secs(1));

    // Report what SIMD features are available
    eprintln!("\n=== SIMD ARCHITECTURE DETECTION ===");

    #[cfg(target_arch = "x86_64")]
    {
        eprintln!("Architecture: x86_64");
        eprintln!("  SSE2: {}", is_x86_feature_detected!("sse2"));
        eprintln!("  AVX: {}", is_x86_feature_detected!("avx"));
        eprintln!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        eprintln!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        eprintln!("Architecture: aarch64");
        eprintln!(
            "  NEON: {}",
            std::arch::is_aarch64_feature_detected!("neon")
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        eprintln!("Architecture: Other (using scalar fallbacks)");
    }

    eprintln!("====================================");

    // Quick test
    let p1 = vec![1.0, 2.0, 3.0, 4.0];
    let p2 = vec![2.0, 3.0, 4.0, 5.0];

    group.bench_function("simd_detection_test", |b| {
        b.iter(|| black_box(simd_euclidean_distance(&p1, &p2).unwrap()))
    });

    group.finish();
}

criterion_group!(
    benches,
    architecture_detection,
    simd_vs_scalar_benchmark,
    parallel_vs_sequential_benchmark,
);

criterion_main!(benches);
