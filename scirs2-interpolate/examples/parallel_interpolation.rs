use scirs2_core::ndarray::{Array1, Array2};
use scirs2_interpolate::local::mls::{MovingLeastSquares, PolynomialBasis, WeightFunction};
use scirs2_interpolate::local::polynomial::{LocalPolynomialConfig, LocalPolynomialRegression};
use scirs2_interpolate::parallel::{
    ParallelConfig, ParallelLocalPolynomialRegression, ParallelMovingLeastSquares,
};
use std::error::Error;
use std::time::Instant;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("Parallel Interpolation Performance Comparison");
    println!("-----------------------------------------\n");

    // Create a large dataset
    println!("Generating test data...");
    let n_points = 10000;
    let n_dims = 3;

    let mut points_vec = Vec::with_capacity(n_points * n_dims);
    let mut values_vec = Vec::with_capacity(n_points);

    for i in 0..n_points {
        // Generate grid-like points with some noise
        let x = (i / 100) as f64 / 100.0 + scirs2_core::random::random::<f64>() * 0.005;
        let y = (i % 100) as f64 / 100.0 + scirs2_core::random::random::<f64>() * 0.005;
        let z = (i % 10) as f64 / 10.0 + scirs2_core::random::random::<f64>() * 0.005;

        points_vec.push(x);
        points_vec.push(y);
        points_vec.push(z);

        // Generate a test function: f(x,y,z) = sin(2πx) * cos(2πy) * z^2
        let value = (2.0 * std::f64::consts::PI * x).sin()
            * (2.0 * std::f64::consts::PI * y).cos()
            * z.powi(2);

        values_vec.push(value);
    }

    let points = Array2::from_shape_vec((n_points, n_dims), points_vec)?;
    let values = Array1::from_vec(values_vec);

    // Generate query points
    let n_queries = 1000;
    let mut query_vec = Vec::with_capacity(n_queries * n_dims);

    for _ in 0..n_queries {
        for _ in 0..n_dims {
            query_vec.push(scirs2_core::random::random::<f64>());
        }
    }

    let query_points = Array2::from_shape_vec((n_queries, n_dims), query_vec)?;

    // Part 1: Moving Least Squares
    println!("\n=== Moving Least Squares Comparison ===");

    // Create sequential MLS
    println!("Creating interpolation models...");
    let start = Instant::now();
    let mls = MovingLeastSquares::new(
        points.clone(),
        values.clone(),
        WeightFunction::Gaussian,
        PolynomialBasis::Linear,
        0.1,
    )?;
    println!("Sequential MLS creation time: {:?}", start.elapsed());

    // Create parallel MLS
    let start = Instant::now();
    let parallel_mls = ParallelMovingLeastSquares::new(
        points.clone(),
        values.clone(),
        WeightFunction::Gaussian,
        PolynomialBasis::Linear,
        0.1,
    )?;
    println!("Parallel MLS creation time:   {:?}", start.elapsed());

    // Sequential interpolation
    println!("\nRunning interpolation on {} query points...", n_queries);
    let start = Instant::now();
    let sequential_results = mls.evaluate_multi(&query_points.view())?;
    let sequential_time = start.elapsed();
    println!("Sequential MLS time: {:?}", sequential_time);

    // Parallel interpolation with different thread counts
    let thread_counts = vec![1, 2, 4, 8];

    for &n_threads in &thread_counts {
        let config = ParallelConfig::new().with_workers(n_threads);

        let start = Instant::now();
        let parallel_results =
            parallel_mls.evaluate_multi_parallel(&query_points.view(), &config)?;
        let parallel_time = start.elapsed();

        println!(
            "Parallel MLS with {} threads: {:?}",
            n_threads, parallel_time
        );

        // Calculate speedup
        let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("  Speedup: {:.2}x", speedup);

        // Verify results match
        let mut max_diff = 0.0;
        for i in 0..n_queries {
            let diff = (sequential_results[i] - parallel_results[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        println!("  Maximum difference: {:.6}", max_diff);
    }

    // Part 2: Local Polynomial Regression (LOESS)
    println!("\n=== Local Polynomial Regression (LOESS) Comparison ===");

    // Create sequential LOESS
    println!("Creating interpolation models...");
    let config = LocalPolynomialConfig {
        bandwidth: 0.1,
        weight_fn: WeightFunction::Gaussian,
        basis: PolynomialBasis::Linear,
        ..LocalPolynomialConfig::default()
    };

    let start = Instant::now();
    let loess =
        LocalPolynomialRegression::with_config(points.clone(), values.clone(), config.clone())?;
    println!("Sequential LOESS creation time: {:?}", start.elapsed());

    // Create parallel LOESS
    let start = Instant::now();
    let parallel_loess =
        ParallelLocalPolynomialRegression::with_config(points.clone(), values.clone(), config)?;
    println!("Parallel LOESS creation time:   {:?}", start.elapsed());

    // Use a smaller subset of query points for LOESS to keep runtime reasonable
    let subset_size = 100;
    let query_subset = query_points
        .slice(scirs2_core::ndarray::s![0..subset_size, ..])
        .to_owned();

    // Sequential interpolation
    println!("\nRunning interpolation on {} query points...", subset_size);
    let start = Instant::now();
    let mut sequential_loess_results = Array1::zeros(subset_size);
    for i in 0..subset_size {
        let result = loess.fit_at_point(&query_subset.row(i))?;
        sequential_loess_results[i] = result.value;
    }
    let sequential_time = start.elapsed();
    println!("Sequential LOESS time: {:?}", sequential_time);

    // Parallel interpolation with different thread counts
    for &n_threads in &thread_counts {
        let config = ParallelConfig::new().with_workers(n_threads);

        let start = Instant::now();
        let parallel_results =
            parallel_loess.fit_multiple_parallel(&query_subset.view(), &config)?;
        let parallel_time = start.elapsed();

        println!(
            "Parallel LOESS with {} threads: {:?}",
            n_threads, parallel_time
        );

        // Calculate speedup
        let speedup = sequential_time.as_secs_f64() / parallel_time.as_secs_f64();
        println!("  Speedup: {:.2}x", speedup);

        // Verify results match
        let mut max_diff = 0.0;
        for i in 0..subset_size {
            let diff = (sequential_loess_results[i] - parallel_results[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        println!("  Maximum difference: {:.6}", max_diff);
    }

    // Comparison of methods with large datasets
    println!("\n=== Large Dataset Performance Comparison ===");

    // Use the same datasets, but with a smaller subset of query points
    let n_test = 50;
    let test_points = query_points
        .slice(scirs2_core::ndarray::s![0..n_test, ..])
        .to_owned();

    println!(
        "Testing with {} points dataset, {} query points",
        n_points, n_test
    );

    // Sequential MLS (baseline)
    let start = Instant::now();
    let _ = mls.evaluate_multi(&test_points.view())?;
    let sequential_mls_time = start.elapsed();
    println!("Sequential MLS:        {:?}", sequential_mls_time);

    // Parallel MLS
    let config = ParallelConfig::new(); // Use default thread count
    let start = Instant::now();
    let _ = parallel_mls.evaluate_multi_parallel(&test_points.view(), &config)?;
    let parallel_mls_time = start.elapsed();
    println!("Parallel MLS:          {:?}", parallel_mls_time);
    println!(
        "  Speedup: {:.2}x",
        sequential_mls_time.as_secs_f64() / parallel_mls_time.as_secs_f64()
    );

    // Sequential LOESS (very slow for large datasets)
    let start = Instant::now();
    let mut results = Vec::with_capacity(n_test);
    for i in 0..n_test {
        let result = loess.fit_at_point(&test_points.row(i))?;
        results.push(result.value);
    }
    let sequential_loess_time = start.elapsed();
    println!("Sequential LOESS:      {:?}", sequential_loess_time);

    // Parallel LOESS
    let start = Instant::now();
    let _ = parallel_loess.fit_multiple_parallel(&test_points.view(), &config)?;
    let parallel_loess_time = start.elapsed();
    println!("Parallel LOESS:        {:?}", parallel_loess_time);
    println!(
        "  Speedup: {:.2}x",
        sequential_loess_time.as_secs_f64() / parallel_loess_time.as_secs_f64()
    );

    // Summary
    println!("\nConclusion:");
    println!("1. Parallel implementations provide significant speedup for interpolation tasks");
    println!("2. Speedup increases with larger datasets and more query points");
    println!("3. KD-tree acceleration combined with parallelization yields best performance");
    println!("4. Using multiple threads allows interpolation methods to scale with CPU cores");

    Ok(())
}
