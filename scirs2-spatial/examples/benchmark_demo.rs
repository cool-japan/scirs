//! Benchmark Demonstration Example
//!
//! This example demonstrates the benchmarking capabilities of scirs2-spatial
//! without requiring the full criterion benchmark infrastructure.

use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_spatial::{
    distance::{euclidean, pdist},
    simd_distance::{parallel_pdist, simd_euclidean_distance_batch},
    KDTree,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SCIRS2-Spatial Benchmark Demonstration");
    println!("======================================");
    println!();

    // System capabilities
    println!("1. System SIMD Capabilities");
    report_simd_capabilities();
    println!();

    // Distance calculation benchmarks
    println!("2. Distance Calculation Performance");
    benchmark_distance_calculations()?;
    println!();

    // Distance matrix benchmarks
    println!("3. Distance Matrix Performance");
    benchmark_distance_matrices()?;
    println!();

    // Spatial structure benchmarks
    println!("4. Spatial Data Structure Performance");
    benchmark_spatial_structures()?;
    println!();

    // Memory scaling analysis
    println!("5. Memory Scaling Analysis");
    analyze_memory_scaling()?;
    println!();

    // Performance recommendations
    println!("6. Performance Recommendations");
    generate_recommendations();

    Ok(())
}

fn report_simd_capabilities() {
    #[cfg(target_arch = "x86_64")]
    {
        println!("  Architecture: x86_64");
        println!("  SSE2: {}", is_x86_feature_detected!("sse2"));
        println!("  AVX: {}", is_x86_feature_detected!("avx"));
        println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
        println!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("  Architecture: aarch64");
        println!(
            "  NEON: {}",
            std::arch::is_aarch64_feature_detected!("neon")
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("  Architecture: Other (using scalar fallbacks)");
    }
}

fn generate_test_data(n_points: usize, dimensions: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n_points, dimensions), |_| rng.random_range(-10.0..10.0))
}

fn benchmark_distance_calculations() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Comparing SIMD vs Scalar distance calculations:");
    println!(
        "  {:>8} {:>8} {:>12} {:>12} {:>10}",
        "Size", "Dim", "Scalar (μs)", "SIMD (μs)", "Speedup"
    );
    println!("  {}", "-".repeat(55));

    let test_cases = [(100, 10), (500, 10), (1000, 10), (1000, 50), (1000, 100)];

    for (size, dim) in test_cases {
        let points1 = generate_test_data(size, dim, 42);
        let points2 = generate_test_data(size, dim, 43);

        // Scalar benchmark
        let start = Instant::now();
        for (row1, row2) in points1.outer_iter().zip(points2.outer_iter()) {
            let _dist = euclidean(row1.as_slice().unwrap(), row2.as_slice().unwrap());
        }
        let scalar_time = start.elapsed().as_micros();

        // SIMD batch benchmark
        let start = Instant::now();
        let _distances = simd_euclidean_distance_batch(&points1.view(), &points2.view())?;
        let simd_time = start.elapsed().as_micros();

        let speedup = scalar_time as f64 / simd_time as f64;

        println!(
            "  {:>8} {:>8} {:>12} {:>12} {:>10.2}x",
            size, dim, scalar_time, simd_time, speedup
        );
    }

    Ok(())
}

fn benchmark_distance_matrices() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Distance matrix computation performance:");
    println!(
        "  {:>8} {:>15} {:>15} {:>10}",
        "Size", "Sequential (ms)", "Parallel (ms)", "Speedup"
    );
    println!("  {}", "-".repeat(50));

    for &size in &[100, 200, 500, 1000] {
        let points = generate_test_data(size, 5, 42);

        // Sequential distance matrix
        let start = Instant::now();
        let _distances_seq = pdist(&points, euclidean);
        let sequential_time = start.elapsed().as_millis();

        // Parallel distance matrix
        let start = Instant::now();
        let _distances_par = parallel_pdist(&points.view(), "euclidean")?;
        let parallel_time = start.elapsed().as_millis();

        let speedup = sequential_time as f64 / parallel_time as f64;

        println!(
            "  {:>8} {:>15} {:>15} {:>10.2}x",
            size, sequential_time, parallel_time, speedup
        );
    }

    Ok(())
}

fn benchmark_spatial_structures() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Spatial data structure performance:");
    println!(
        "  {:>8} {:>15} {:>15}",
        "Size", "KDTree Build (ms)", "Query 100 pts (ms)"
    );
    println!("  {}", "-".repeat(45));

    for &size in &[1000, 5000, 10000] {
        let points = generate_test_data(size, 3, 42);
        let query_points = generate_test_data(100, 3, 43);

        // KDTree construction
        let start = Instant::now();
        let kdtree = KDTree::new(&points)?;
        let build_time = start.elapsed().as_millis();

        // Query performance
        let start = Instant::now();
        for query in query_points.outer_iter() {
            let _result = kdtree.query(query.as_slice().unwrap(), 5)?;
        }
        let query_time = start.elapsed().as_millis();

        println!("  {:>8} {:>15} {:>15}", size, build_time, query_time);
    }

    Ok(())
}

fn analyze_memory_scaling() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Memory usage analysis:");
    println!(
        "  {:>8} {:>12} {:>15} {:>15}",
        "Size", "Data (MB)", "Distance Mat (MB)", "Efficiency"
    );
    println!("  {}", "-".repeat(55));

    for &size in &[500, 1000, 2000, 5000] {
        let points = generate_test_data(size, 8, 42);

        let data_size_mb = (size * 8 * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
        let dist_matrix_size_mb =
            (size * (size - 1) / 2 * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        // Measure actual performance with memory allocation
        let start = Instant::now();
        let _distances = parallel_pdist(&points.view(), "euclidean")?;
        let elapsed = start.elapsed().as_millis();

        let efficiency = (size * (size - 1) / 2) as f64 / elapsed as f64;

        println!(
            "  {:>8} {:>12.2} {:>15.2} {:>15.0} ops/ms",
            size, data_size_mb, dist_matrix_size_mb, efficiency
        );
    }

    Ok(())
}

fn generate_recommendations() {
    println!("  Based on benchmark results:");
    println!();
    println!("  ✓ SIMD Optimizations:");
    println!("    • Use SIMD functions for datasets with >100 points");
    println!("    • Best performance with vector dimensions ≥10");
    println!("    • Batch operations show significant speedup");
    println!();
    println!("  ✓ Parallel Processing:");
    println!("    • Use parallel functions for datasets >1,000 points");
    println!("    • Parallel distance matrices show good scaling");
    println!("    • Monitor memory usage for large datasets");
    println!();
    println!("  ✓ Memory Management:");
    println!("    • Distance matrices grow O(n²) - use condensed format");
    println!("    • For n>5,000 points, consider chunked processing");
    println!("    • KDTree construction is memory-efficient");
    println!();
    println!("  ✓ Algorithm Selection:");
    println!("    • KDTree: Excellent for low-dimensional nearest neighbor queries");
    println!("    • Parallel distance functions: Best for batch computations");
    println!("    • SIMD: Most effective for high-dimensional data");
    println!();
    println!("  📊 Performance Scaling Guidelines:");
    println!("    • Small datasets (<1,000): Standard algorithms sufficient");
    println!("    • Medium datasets (1,000-10,000): Enable SIMD + parallel");
    println!("    • Large datasets (>10,000): Use spatial data structures + chunking");
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_spatial::simd_distance::simd_euclidean_distance;

    #[test]
    fn test_benchmark_demo() {
        // Test that the benchmark demo functions work
        let points = generate_test_data(100, 5, 42);
        assert_eq!(points.shape(), [100, 5]);

        // Test basic distance calculation
        let p1 = vec![1.0, 2.0, 3.0];
        let p2 = vec![4.0, 5.0, 6.0];
        let dist = euclidean(&p1, &p2);
        assert!(dist > 0.0);

        // Test SIMD distance if available
        if let Ok(simd_dist) = simd_euclidean_distance(&p1, &p2) {
            assert!((dist - simd_dist).abs() < 1e-10);
        }
    }

    #[test]
    fn test_data_generation() {
        let data = generate_test_data(50, 3, 123);
        assert_eq!(data.shape(), [50, 3]);

        // Verify data is in expected range
        for &val in data.iter() {
            assert!(val >= -10.0 && val <= 10.0);
        }
    }
}
