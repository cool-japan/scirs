//! Detailed timing analysis with higher precision measurements

use ndarray::Array2;
use rand::{rngs::StdRng, Rng, SeedableRng};
use scirs2_spatial::{
    distance::euclidean,
    simd_distance::{parallel_pdist, simd_euclidean_distance},
};
use std::time::Instant;

fn generate_points(n_points: usize, dimensions: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n_points, dimensions), |_| rng.random_range(-10.0..10.0))
}

fn main() {
    println!("=== High-Precision Performance Analysis ===\n");

    // Single distance calculation timing
    println!("Single Distance Calculation Performance:");
    println!(
        "{:>8} {:>15} {:>15} {:>12}",
        "Dim", "Scalar (ns)", "SIMD (ns)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &dim in &[4, 8, 16, 32] {
        let p1: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let p2: Vec<f64> = (0..dim).map(|i| (i + 1) as f64).collect();

        // Scalar timing (many iterations for precision)
        let iterations = 100_000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = euclidean(&p1, &p2);
        }
        let scalar_total = start.elapsed();
        let scalar_per_op = scalar_total.as_nanos() / iterations;

        // SIMD timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = simd_euclidean_distance(&p1, &p2).unwrap();
        }
        let simd_total = start.elapsed();
        let simd_per_op = simd_total.as_nanos() / iterations;

        let speedup = scalar_per_op as f64 / simd_per_op as f64;

        println!(
            "{:>8} {:>15} {:>15} {:>12.2}x",
            dim, scalar_per_op, simd_per_op, speedup
        );
    }

    println!("\n=== Matrix Operations Detailed Timing ===");
    println!(
        "{:>8} {:>12} {:>15} {:>15}",
        "Size", "Operations", "Time (μs)", "Ops/sec"
    );
    println!("{}", "-".repeat(55));

    for &size in &[64, 128, 256] {
        let points = generate_points(size, 5, 12345);
        let expected_ops = size * (size - 1) / 2;

        let start = Instant::now();
        let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
        let elapsed = start.elapsed();

        let micros = elapsed.as_micros();
        let ops_per_sec = expected_ops as f64 / elapsed.as_secs_f64();

        println!(
            "{:>8} {:>12} {:>15} {:>15.0}",
            size, expected_ops, micros, ops_per_sec
        );
    }

    println!("\n=== Performance Characteristics Summary ===");

    // Test different problem sizes for scaling analysis
    let test_sizes = vec![32, 64, 128, 256, 512];
    println!("Scaling Analysis:");
    println!(
        "{:>8} {:>15} {:>15} {:>15}",
        "Size", "Time (μs)", "Ops/Million", "MB/sec"
    );
    println!("{}", "-".repeat(60));

    for &size in &test_sizes {
        if size > 400 {
            continue;
        } // Skip very large sizes to avoid long computation

        let points = generate_points(size, 10, 12345);
        let expected_ops = size * (size - 1) / 2;
        let data_size_mb = (size * 10 * 8) as f64 / (1024.0 * 1024.0); // Size in MB

        let start = Instant::now();
        let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
        let elapsed = start.elapsed();

        let micros = elapsed.as_micros();
        let ops_per_million = expected_ops as f64 / 1_000_000.0;
        let mb_per_sec = data_size_mb / elapsed.as_secs_f64();

        println!(
            "{:>8} {:>15} {:>15.2} {:>15.1}",
            size, micros, ops_per_million, mb_per_sec
        );
    }

    println!("\n=== Concrete Performance Numbers ===");

    // Single operation performance
    let p1: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let p2: Vec<f64> = (0..100).map(|i| (i + 1) as f64).collect();

    let iterations = 1_000_000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = simd_euclidean_distance(&p1, &p2).unwrap();
    }
    let total_time = start.elapsed();
    let per_op_nanos = total_time.as_nanos() / iterations;
    let ops_per_sec = 1_000_000_000.0 / per_op_nanos as f64;

    println!("Single Euclidean distance (100D): {} ns", per_op_nanos);
    println!(
        "Throughput: {:.1} million operations/second",
        ops_per_sec / 1_000_000.0
    );

    // Matrix performance
    let matrix_size = 100;
    let points = generate_points(matrix_size, 10, 12345);
    let expected_ops = matrix_size * (matrix_size - 1) / 2;

    let start = Instant::now();
    let _distances = parallel_pdist(&points.view(), "euclidean").unwrap();
    let elapsed = start.elapsed();

    println!(
        "\nMatrix 100×10 ({} distances): {} μs",
        expected_ops,
        elapsed.as_micros()
    );
    println!(
        "Matrix throughput: {:.1} million distance calculations/second",
        expected_ops as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );

    println!("\n✅ All performance measurements completed successfully!");
}
