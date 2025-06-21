//! SIMD Performance Optimization Example
//!
//! This example demonstrates the performance benefits of SIMD-accelerated
//! distance calculations and parallel spatial operations. It compares
//! scalar vs SIMD implementations across different data sizes and metrics.

use ndarray::Array2;
use rand::Rng;
use scirs2_spatial::simd_distance::{
    bench, parallel_pdist, simd_euclidean_distance, simd_euclidean_distance_batch, simd_knn_search,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SIMD Performance Optimization Example ===\n");

    // Example 1: System SIMD capabilities
    println!("1. System SIMD Capabilities");
    system_capabilities_example();
    println!();

    // Example 2: Single distance computation performance
    println!("2. Single Distance Computation Performance");
    single_distance_performance_example()?;
    println!();

    // Example 3: Batch distance computation
    println!("3. Batch Distance Computation Performance");
    batch_distance_performance_example()?;
    println!();

    // Example 4: Distance matrix computation
    println!("4. Distance Matrix Computation Performance");
    distance_matrix_performance_example()?;
    println!();

    // Example 5: K-nearest neighbors search
    println!("5. K-Nearest Neighbors Search Performance");
    knn_performance_example()?;
    println!();

    // Example 6: Scalability analysis
    println!("6. Scalability Analysis");
    scalability_analysis_example()?;
    println!();

    // Example 7: Memory efficiency comparison
    println!("7. Memory Efficiency and Cache Performance");
    memory_efficiency_example()?;

    Ok(())
}

fn system_capabilities_example() {
    println!("Detecting available SIMD instruction sets:");
    bench::report_simd_features();

    #[cfg(target_arch = "x86_64")]
    println!("\nOptimal SIMD width: {} elements (256-bit AVX)", 4);

    #[cfg(target_arch = "aarch64")]
    println!("\nOptimal SIMD width: {} elements (128-bit NEON)", 2);

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    println!("\nUsing scalar fallback implementations");
}

fn single_distance_performance_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing SIMD vs scalar for single distance computations:");

    // Test different vector sizes
    let dimensions = [3, 10, 50, 100, 500, 1000];
    let iterations = 100_000;

    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Dim", "Scalar (ms)", "SIMD (ms)", "Speedup"
    );
    println!("{}", "-".repeat(48));

    for &dim in &dimensions {
        let mut rng = rand::rng();

        // Generate random vectors
        let a: Vec<f64> = (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect();
        let b: Vec<f64> = (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect();

        // Scalar timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _dist = scirs2_spatial::distance::euclidean(&a, &b);
        }
        let scalar_time = start.elapsed().as_millis();

        // SIMD timing
        let start = Instant::now();
        for _ in 0..iterations {
            let _dist = simd_euclidean_distance(&a, &b)?;
        }
        let simd_time = start.elapsed().as_millis();

        let speedup = scalar_time as f64 / simd_time as f64;

        println!(
            "{:>8} {:>12} {:>12} {:>12.2}x",
            dim, scalar_time, simd_time, speedup
        );
    }

    Ok(())
}

fn batch_distance_performance_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Batch distance computation performance:");

    let n_points = 10_000;
    let dimensions = [2, 3, 5, 10, 20];

    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Dim", "Scalar (ms)", "SIMD (ms)", "Speedup"
    );
    println!("{}", "-".repeat(48));

    for &dim in &dimensions {
        // Generate random point sets
        let points1 = generate_random_points(n_points, dim);
        let points2 = generate_random_points(n_points, dim);

        // Measure performance
        let (scalar_time, simd_time) = bench::benchmark_distance_computation(
            &points1.view(),
            &points2.view(),
            1, // Single iteration since we have many points
        );

        let speedup = scalar_time / simd_time;

        println!(
            "{:>8} {:>12.1} {:>12.1} {:>12.2}x",
            dim,
            scalar_time * 1000.0,
            simd_time * 1000.0,
            speedup
        );
    }

    Ok(())
}

fn distance_matrix_performance_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Distance matrix computation performance comparison:");

    let point_counts = [100, 200, 500, 1000];
    let dim = 3;

    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Points", "Scalar (ms)", "Parallel (ms)", "Speedup"
    );
    println!("{}", "-".repeat(48));

    for &n_points in &point_counts {
        let points = generate_random_points(n_points, dim);

        // Scalar pdist
        let start = Instant::now();
        let _scalar_dists =
            scirs2_spatial::distance::pdist(&points, scirs2_spatial::distance::euclidean);
        let scalar_time = start.elapsed().as_millis();

        // Parallel SIMD pdist
        let start = Instant::now();
        let _parallel_dists = parallel_pdist(&points.view(), "euclidean")?;
        let parallel_time = start.elapsed().as_millis();

        let speedup = scalar_time as f64 / parallel_time as f64;

        println!(
            "{:>8} {:>12} {:>12} {:>12.2}x",
            n_points, scalar_time, parallel_time, speedup
        );
    }

    Ok(())
}

fn knn_performance_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("K-nearest neighbors search performance:");

    let n_data = 5000;
    let n_queries = 1000;
    let k_values = [1, 5, 10, 20];
    let dim = 5;

    let data_points = generate_random_points(n_data, dim);
    let query_points = generate_random_points(n_queries, dim);

    println!(
        "{:>6} {:>15} {:>12}",
        "k", "Time (ms)", "Throughput (queries/s)"
    );
    println!("{}", "-".repeat(35));

    for &k in &k_values {
        let start = Instant::now();
        let (_indices, _distances) =
            simd_knn_search(&query_points.view(), &data_points.view(), k, "euclidean")?;
        let elapsed = start.elapsed().as_millis();

        let throughput = (n_queries * 1000) / elapsed as usize;

        println!("{:>6} {:>15} {:>12}", k, elapsed, throughput);
    }

    // Compare different metrics
    println!("\nPerformance by distance metric (k=5):");
    let metrics = ["euclidean", "manhattan", "sqeuclidean", "chebyshev"];

    println!("{:>12} {:>12} {:>12}", "Metric", "Time (ms)", "Rel. Speed");
    println!("{}", "-".repeat(38));

    let mut base_time = 0;
    for (i, &metric) in metrics.iter().enumerate() {
        let start = Instant::now();
        let (_indices, _distances) =
            simd_knn_search(&query_points.view(), &data_points.view(), 5, metric)?;
        let elapsed = start.elapsed().as_millis();

        if i == 0 {
            base_time = elapsed;
        }

        let relative_speed = base_time as f64 / elapsed as f64;

        println!("{:>12} {:>12} {:>12.2}x", metric, elapsed, relative_speed);
    }

    Ok(())
}

fn scalability_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scalability analysis with increasing problem sizes:");

    let base_points = [1000, 2000, 5000, 10000];
    let dim = 4;

    println!(
        "{:>8} {:>12} {:>12} {:>15}",
        "Points", "Time (ms)", "Time/Point (μs)", "Memory (MB)"
    );
    println!("{}", "-".repeat(50));

    for &n_points in &base_points {
        let points = generate_random_points(n_points, dim);

        let start = Instant::now();
        let distances = parallel_pdist(&points.view(), "euclidean")?;
        let elapsed = start.elapsed().as_millis();

        let time_per_point = (elapsed as f64 * 1000.0) / (n_points as f64);
        let memory_mb = (distances.len() * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);

        println!(
            "{:>8} {:>12} {:>12.1} {:>15.2}",
            n_points, elapsed, time_per_point, memory_mb
        );
    }

    // Parallel efficiency analysis
    println!("\nParallel efficiency analysis:");
    test_parallel_efficiency()?;

    Ok(())
}

fn test_parallel_efficiency() -> Result<(), Box<dyn std::error::Error>> {
    let n_points = 2000;
    let dim = 5;
    let points = generate_random_points(n_points, dim);

    // Simulate different thread counts by controlling Rayon's thread pool
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Threads", "Time (ms)", "Speedup", "Efficiency"
    );
    println!("{}", "-".repeat(48));

    // Note: This is a simplified demonstration
    // In practice, you'd need to control Rayon's global thread pool
    let thread_counts = [1, 2, 4, 8];
    let mut base_time = 0;

    for (i, &threads) in thread_counts.iter().enumerate() {
        // For demonstration, we'll use the same parallel implementation
        // In a real scenario, you'd configure the thread pool
        let start = Instant::now();
        let _distances = parallel_pdist(&points.view(), "euclidean")?;
        let elapsed = start.elapsed().as_millis();

        if i == 0 {
            base_time = elapsed;
        }

        let speedup = base_time as f64 / elapsed as f64;
        let efficiency = speedup / threads as f64;

        println!(
            "{:>8} {:>12} {:>12.2}x {:>12.1}%",
            threads,
            elapsed,
            speedup,
            efficiency * 100.0
        );
    }

    Ok(())
}

fn memory_efficiency_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("Memory efficiency and cache performance analysis:");

    // Test different access patterns
    let n_points = 1000;
    let dimensions = [2, 5, 10, 20, 50];

    println!(
        "{:>8} {:>15} {:>15} {:>15}",
        "Dim", "Cache-friendly", "Random Access", "Improvement"
    );
    println!("{}", "-".repeat(60));

    for &dim in &dimensions {
        let points1 = generate_random_points(n_points, dim);
        let points2 = generate_random_points(n_points, dim);

        // Sequential access (cache-friendly)
        let start = Instant::now();
        let _distances = simd_euclidean_distance_batch(&points1.view(), &points2.view())?;
        let sequential_time = start.elapsed().as_millis();

        // Simulate random access pattern
        let start = Instant::now();
        let mut _sum = 0.0;
        for i in 0..n_points {
            let p1 = points1.row(i).to_vec();
            let p2 = points2.row(i).to_vec();
            _sum += simd_euclidean_distance(&p1, &p2)?;
        }
        let random_time = start.elapsed().as_millis();

        let improvement = random_time as f64 / sequential_time as f64;

        println!(
            "{:>8} {:>15} {:>15} {:>15.2}x",
            dim, sequential_time, random_time, improvement
        );
    }

    // Memory allocation analysis
    println!("\nMemory allocation patterns:");
    memory_allocation_analysis()?;

    Ok(())
}

fn memory_allocation_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let n_points = 2000;
    let dim = 10;

    println!("Operation                     Memory (MB)  Allocations");
    println!("{}", "-".repeat(50));

    // Point storage
    let _points = generate_random_points(n_points, dim);
    let points_memory = (n_points * dim * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    println!("{:<30} {:>10.2} {:>10}", "Point storage", points_memory, 1);

    // Distance matrix (condensed)
    let n_distances = n_points * (n_points - 1) / 2;
    let distance_memory = (n_distances * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    println!(
        "{:<30} {:>10.2} {:>10}",
        "Distance matrix", distance_memory, 1
    );

    // KNN results
    let k = 10;
    let knn_memory = (n_points * k * 2 * std::mem::size_of::<f64>()) as f64 / (1024.0 * 1024.0);
    println!("{:<30} {:>10.2} {:>10}", "KNN results", knn_memory, 2);

    // Total peak memory
    let total_memory = points_memory + distance_memory.max(knn_memory);
    println!("{:<30} {:>10.2} {:>10}", "Peak usage", total_memory, 3);

    Ok(())
}

/// Generate random points for testing
fn generate_random_points(n_points: usize, dim: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    Array2::from_shape_fn((n_points, dim), |_| rng.random_range(-10.0..10.0))
}

/// Demonstrate different optimization strategies
#[allow(dead_code)]
fn optimization_strategies_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nOptimization strategies comparison:");

    let n_points = 1000;
    let dim = 8;
    let points = generate_random_points(n_points, dim);

    // Strategy 1: Naive double loop
    let start = Instant::now();
    let mut naive_dists = Vec::new();
    for i in 0..n_points {
        for j in (i + 1)..n_points {
            let p1 = points.row(i).to_vec();
            let p2 = points.row(j).to_vec();
            naive_dists.push(scirs2_spatial::distance::euclidean(&p1, &p2));
        }
    }
    let naive_time = start.elapsed().as_millis();

    // Strategy 2: SIMD + sequential
    let start = Instant::now();
    let mut simd_dists = Vec::new();
    for i in 0..n_points {
        for j in (i + 1)..n_points {
            let p1 = points.row(i).to_vec();
            let p2 = points.row(j).to_vec();
            simd_dists.push(simd_euclidean_distance(&p1, &p2)?);
        }
    }
    let simd_time = start.elapsed().as_millis();

    // Strategy 3: Parallel + SIMD
    let start = Instant::now();
    let _parallel_dists = parallel_pdist(&points.view(), "euclidean")?;
    let parallel_time = start.elapsed().as_millis();

    println!("Strategy                      Time (ms)    Speedup");
    println!("{}", "-".repeat(45));
    println!("{:<30} {:>8} {:>8.1}x", "Naive scalar", naive_time, 1.0);
    println!(
        "{:<30} {:>8} {:>8.1}x",
        "SIMD sequential",
        simd_time,
        naive_time as f64 / simd_time as f64
    );
    println!(
        "{:<30} {:>8} {:>8.1}x",
        "SIMD parallel",
        parallel_time,
        naive_time as f64 / parallel_time as f64
    );

    Ok(())
}

/// Benchmark I/O intensive operations
#[allow(dead_code)]
fn benchmark_io_intensive() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nI/O intensive operations benchmark:");

    let sizes = [100, 500, 1000, 2000];
    let dim = 6;

    println!(
        "{:>8} {:>12} {:>15} {:>15}",
        "Size", "Read (ms)", "Compute (ms)", "Write (ms)"
    );
    println!("{}", "-".repeat(55));

    for &size in &sizes {
        // Simulate data reading
        let start = Instant::now();
        let points = generate_random_points(size, dim);
        let read_time = start.elapsed().as_millis();

        // Distance computation
        let start = Instant::now();
        let distances = parallel_pdist(&points.view(), "euclidean")?;
        let compute_time = start.elapsed().as_millis();

        // Simulate data writing
        let start = Instant::now();
        let _sum: f64 = distances.sum(); // Simulate processing
        let write_time = start.elapsed().as_millis();

        println!(
            "{:>8} {:>12} {:>15} {:>15}",
            size, read_time, compute_time, write_time
        );
    }

    Ok(())
}

/// Memory bandwidth utilization analysis
#[allow(dead_code)]
fn memory_bandwidth_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nMemory bandwidth utilization:");

    let n_points = 5000;
    let dim = 10;
    let points = generate_random_points(n_points, dim);

    // Calculate theoretical vs actual throughput
    let start = Instant::now();
    let _distances = parallel_pdist(&points.view(), "euclidean")?;
    let elapsed = start.elapsed().as_secs_f64();

    let data_size = n_points * dim * std::mem::size_of::<f64>();
    let bandwidth_gb_s = (data_size as f64) / (elapsed * 1e9);

    println!(
        "Data processed: {:.2} MB",
        data_size as f64 / (1024.0 * 1024.0)
    );
    println!("Time elapsed: {:.3} seconds", elapsed);
    println!("Effective bandwidth: {:.2} GB/s", bandwidth_gb_s);

    // Compare with theoretical peak
    #[cfg(target_arch = "x86_64")]
    println!("Theoretical peak (DDR4-3200): ~25.6 GB/s");

    Ok(())
}
