//! Example: Parallel Monte Carlo Integration Performance
//!
//! This example demonstrates the performance benefits of parallel Monte Carlo
//! integration for computationally expensive integrand functions.
//! It compares sequential vs parallel Monte Carlo methods across different
//! problem complexities and system configurations.

use ndarray::ArrayView1;
use scirs2_integrate::monte_carlo::{monte_carlo, MonteCarloOptions};
use std::time::Instant;

#[cfg(feature = "parallel")]
use scirs2_integrate::monte_carlo_parallel::{
    adaptive_parallel_monte_carlo, parallel_monte_carlo, ParallelMonteCarloOptions,
};

use std::marker::PhantomData;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Parallel Monte Carlo Integration Performance Demo ===");

    #[cfg(not(feature = "parallel"))]
    {
        println!(
            "Parallel feature is not enabled. To see parallel performance benefits, compile with:"
        );
        println!("cargo run --features parallel --example parallel_monte_carlo_example");
        println!("\\nRunning sequential Monte Carlo methods for comparison...");
    }

    #[cfg(feature = "parallel")]
    {
        println!("Parallel feature is enabled! Comparing sequential vs parallel methods...");
    }

    // Test different integrand complexities
    test_simple_function()?;
    test_expensive_function()?;
    test_multidimensional_integration()?;
    test_adaptive_parallel_integration()?;

    println!("\\n=== Summary and Recommendations ===");
    print_performance_recommendations();

    Ok(())
}

/// Test integration of a simple polynomial function
fn test_simple_function() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n--- Test 1: Simple Polynomial Function ---");
    println!("Integrating f(x,y) = x² + y² over [0,1]×[0,1] (exact result: 2/3)");

    let f = |x: ArrayView1<f64>| x[0] * x[0] + x[1] * x[1];
    let ranges = [(0.0, 1.0), (0.0, 1.0)];
    let n_samples = 1_000_000;

    // Sequential Monte Carlo
    let seq_options = MonteCarloOptions {
        n_samples,
        seed: Some(42),
        _phantom: PhantomData,
        ..Default::default()
    };

    let start = Instant::now();
    let seq_result = monte_carlo(f, &ranges, Some(seq_options))?;
    let seq_time = start.elapsed();

    println!(
        "Sequential MC: value = {:.6}, error = {:.2e}, time = {:.2} ms",
        seq_result.value,
        seq_result.std_error,
        seq_time.as_millis()
    );

    // Parallel Monte Carlo (if available)
    #[cfg(feature = "parallel")]
    {
        let par_options = ParallelMonteCarloOptions {
            n_samples,
            seed: Some(42),
            n_threads: Some(4),
            batch_size: 50_000,
            use_chunking: true,
            _phantom: PhantomData,
            ..Default::default()
        };

        let start = Instant::now();
        let par_result = parallel_monte_carlo(f, &ranges, Some(par_options))?;
        let par_time = start.elapsed();

        println!(
            "Parallel MC:   value = {:.6}, error = {:.2e}, time = {:.2} ms",
            par_result.value,
            par_result.std_error,
            par_time.as_millis()
        );

        let speedup = seq_time.as_nanos() as f64 / par_time.as_nanos() as f64;
        println!("Speedup: {:.2}x", speedup);

        // Verify results are consistent
        let value_diff = (seq_result.value - par_result.value).abs();
        println!(
            "Value difference: {:.2e} (should be small due to randomness)",
            value_diff
        );
    }

    println!("Expected value: {:.6}", 2.0 / 3.0);

    Ok(())
}

/// Test integration of an expensive trigonometric function
fn test_expensive_function() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n--- Test 2: Expensive Trigonometric Function ---");
    println!("Integrating f(x,y) = sin(10πx) * cos(10πy) * exp(-x²-y²) over [-2,2]×[-2,2]");

    // Computationally expensive function
    let expensive_f = |x: ArrayView1<f64>| {
        let x_val = x[0];
        let y_val = x[1];
        let trig_part = (10.0 * std::f64::consts::PI * x_val).sin()
            * (10.0 * std::f64::consts::PI * y_val).cos();
        let exp_part = (-x_val * x_val - y_val * y_val).exp();
        trig_part * exp_part
    };

    let ranges = [(-2.0, 2.0), (-2.0, 2.0)];
    let n_samples = 500_000;

    // Sequential Monte Carlo
    let seq_options = MonteCarloOptions {
        n_samples,
        seed: Some(123),
        _phantom: PhantomData,
        ..Default::default()
    };

    let start = Instant::now();
    let seq_result = monte_carlo(expensive_f, &ranges, Some(seq_options))?;
    let seq_time = start.elapsed();

    println!(
        "Sequential MC: value = {:.6}, error = {:.2e}, time = {:.2} ms",
        seq_result.value,
        seq_result.std_error,
        seq_time.as_millis()
    );

    // Parallel Monte Carlo (if available)
    #[cfg(feature = "parallel")]
    {
        let par_options = ParallelMonteCarloOptions {
            n_samples,
            seed: Some(123),
            n_threads: Some(4),
            batch_size: 25_000,
            use_chunking: true,
            _phantom: PhantomData,
            ..Default::default()
        };

        let start = Instant::now();
        let par_result = parallel_monte_carlo(expensive_f, &ranges, Some(par_options))?;
        let par_time = start.elapsed();

        println!(
            "Parallel MC:   value = {:.6}, error = {:.2e}, time = {:.2} ms",
            par_result.value,
            par_result.std_error,
            par_time.as_millis()
        );

        let speedup = seq_time.as_nanos() as f64 / par_time.as_nanos() as f64;
        println!(
            "Speedup: {:.2}x (should be higher for expensive functions)",
            speedup
        );
    }

    Ok(())
}

/// Test high-dimensional integration
fn test_multidimensional_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n--- Test 3: High-Dimensional Integration ---");
    println!("Integrating f(x₁,...,x₆) = exp(-∑xᵢ²) over [-1,1]⁶");

    // 6-dimensional Gaussian-like function
    let multi_f = |x: ArrayView1<f64>| {
        let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
        (-sum_sq).exp()
    };

    let ranges: Vec<(f64, f64)> = vec![(-1.0, 1.0); 6]; // 6 dimensions
    let n_samples = 2_000_000;

    // Sequential Monte Carlo
    let seq_options = MonteCarloOptions {
        n_samples,
        seed: Some(456),
        _phantom: PhantomData,
        ..Default::default()
    };

    let start = Instant::now();
    let seq_result = monte_carlo(multi_f, &ranges, Some(seq_options))?;
    let seq_time = start.elapsed();

    println!(
        "Sequential MC: value = {:.6}, error = {:.2e}, time = {:.2} ms",
        seq_result.value,
        seq_result.std_error,
        seq_time.as_millis()
    );

    // Parallel Monte Carlo (if available)
    #[cfg(feature = "parallel")]
    {
        let par_options = ParallelMonteCarloOptions {
            n_samples,
            seed: Some(456),
            n_threads: Some(8), // Use more threads for high-dimensional problems
            batch_size: 100_000,
            use_chunking: true,
            _phantom: PhantomData,
            ..Default::default()
        };

        let start = Instant::now();
        let par_result = parallel_monte_carlo(multi_f, &ranges, Some(par_options))?;
        let par_time = start.elapsed();

        println!(
            "Parallel MC:   value = {:.6}, error = {:.2e}, time = {:.2} ms",
            par_result.value,
            par_result.std_error,
            par_time.as_millis()
        );

        let speedup = seq_time.as_nanos() as f64 / par_time.as_nanos() as f64;
        println!("Speedup: {:.2}x", speedup);

        // The exact value for this integral is π³ ≈ 31.006
        let exact_value = std::f64::consts::PI.powi(3);
        println!("Exact value: {:.6}", exact_value);
        println!(
            "Sequential error: {:.2e}",
            (seq_result.value - exact_value).abs()
        );
        println!(
            "Parallel error:   {:.2e}",
            (par_result.value - exact_value).abs()
        );
    }

    Ok(())
}

/// Test adaptive parallel Monte Carlo integration
fn test_adaptive_parallel_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\\n--- Test 4: Adaptive Parallel Monte Carlo ---");
    println!("Using adaptive sampling to reach target variance automatically");

    #[cfg(feature = "parallel")]
    {
        // Oscillatory function that's challenging to integrate
        let oscillatory_f = |x: ArrayView1<f64>| {
            let freq = 20.0;
            (freq * x[0]).sin() * (freq * x[1]).cos() * (-x[0] * x[0] - x[1] * x[1]).exp()
        };

        let ranges = [(-2.0, 2.0), (-2.0, 2.0)];
        let target_variance = 1e-4;
        let max_samples = 5_000_000;

        let options = ParallelMonteCarloOptions {
            n_samples: 100_000, // Initial samples
            seed: Some(789),
            n_threads: Some(4),
            batch_size: 50_000,
            _phantom: PhantomData,
            ..Default::default()
        };

        let start = Instant::now();
        let result = adaptive_parallel_monte_carlo(
            oscillatory_f,
            &ranges,
            target_variance,
            max_samples,
            Some(options),
        )?;
        let time = start.elapsed();

        println!("Adaptive MC result:");
        println!("  Value: {:.6}", result.value);
        println!("  Standard error: {:.2e}", result.std_error);
        println!("  Samples used: {}", result.n_evals);
        println!("  Time: {:.2} ms", time.as_millis());
        println!("  Target variance: {:.2e}", target_variance);

        if result.std_error <= target_variance {
            println!("  ✓ Target variance achieved!");
        } else {
            println!("  ⚠ Maximum samples reached before target variance");
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("Adaptive parallel Monte Carlo requires the 'parallel' feature to be enabled.");
    }

    Ok(())
}

/// Print performance recommendations
fn print_performance_recommendations() {
    println!("\\nParallel Monte Carlo is most beneficial when:");
    println!("• Integrand function is computationally expensive");
    println!("• High number of samples needed (>100,000)");
    println!("• Multiple CPU cores are available");
    println!("• High-dimensional integration problems");
    println!("• Statistical precision requirements are high");

    println!("\\nOptimization tips:");
    println!("• Use appropriate batch sizes (10,000-100,000 samples per batch)");
    println!("• Enable chunking for better load balancing");
    println!("• Consider antithetic sampling for variance reduction");
    println!("• Use adaptive methods for unknown convergence behavior");

    #[cfg(feature = "parallel")]
    {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        println!("\\n✓ Parallel feature is ENABLED");
        println!("  Available CPU cores: {}", num_threads);
        println!(
            "  Recommended thread count: {}",
            (num_threads as f64 * 0.75) as usize
        );
    }

    #[cfg(not(feature = "parallel"))]
    {
        println!("\\n⚠ Parallel feature is DISABLED");
        println!(
            "  To enable: cargo run --features parallel --example parallel_monte_carlo_example"
        );
    }
}
