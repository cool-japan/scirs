//! SIMD Performance Demonstration
//!
//! This example demonstrates the performance improvements possible with SIMD optimization
//! for special functions on large arrays.

use scirs2_core::ndarray::Array1;
use std::time::Instant;

#[cfg(feature = "simd")]
use scirs2_special::{
    benchmark_simd_performance, exp_f32_simd, gamma_f32_simd, gamma_f64_simd, j0_f32_simd,
    vectorized_special_ops,
};

use scirs2_special::{gamma, j0};

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SciRS2 Special Functions - SIMD Performance Demo");
    println!("{}", "=".repeat(60));

    // Check SIMD availability
    #[cfg(feature = "simd")]
    {
        println!("✓ SIMD features enabled");
        vectorized_special_ops()?;
        println!();
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("⚠ SIMD features not enabled. Run with --features simd for optimization demo.");
        println!();
    }

    // Test with different array sizes
    let sizes = vec![100, 1000, 10000, 100000];

    for &size in &sizes {
        println!("Performance comparison with {} elements:", size);
        println!("{}", "-".repeat(40));

        benchmark_gamma_performance(size)?;
        println!();

        benchmark_j0_performance(size)?;
        println!();

        #[cfg(feature = "simd")]
        {
            benchmark_simd_performance(size)?;
            println!();
        }
    }

    // Demonstrate precision comparison
    demonstrate_precision_comparison()?;

    Ok(())
}

#[allow(dead_code)]
fn benchmark_gamma_performance(size: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Create test data
    let data_f32: Array1<f32> =
        Array1::from_vec((0..size).map(|i| (i as f32) * 0.01 + 1.0).collect());
    let data_f64: Array1<f64> =
        Array1::from_vec((0..size).map(|i| (i as f64) * 0.01 + 1.0).collect());

    // Benchmark scalar implementation
    let start = Instant::now();
    let _scalar_f64: Array1<f64> = data_f64.mapv(gamma);
    let scalar_time = start.elapsed();

    let start = Instant::now();
    let _scalar_f32: Array1<f32> = data_f32.mapv(|x| gamma(x as f64) as f32);
    let scalar_f32_time = start.elapsed();

    println!("Gamma function performance:");
    println!("  Scalar f64: {:?}", scalar_time);
    println!("  Scalar f32: {:?}", scalar_f32_time);

    // Benchmark SIMD implementation if available
    #[cfg(feature = "simd")]
    {
        let start = Instant::now();
        let _simd_f64 = gamma_f64_simd(&data_f64.view())?;
        let simd_f64_time = start.elapsed();

        let start = Instant::now();
        let _simd_f32 = gamma_f32_simd(&data_f32.view())?;
        let simd_f32_time = start.elapsed();

        println!("  SIMD f64:   {:?}", simd_f64_time);
        println!("  SIMD f32:   {:?}", simd_f32_time);

        let speedup_f64 = scalar_time.as_nanos() as f64 / simd_f64_time.as_nanos() as f64;
        let speedup_f32 = scalar_f32_time.as_nanos() as f64 / simd_f32_time.as_nanos() as f64;

        println!("  Speedup f64: {:.2}x", speedup_f64);
        println!("  Speedup f32: {:.2}x", speedup_f32);
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("  SIMD: Not available (enable with --features simd)");
    }

    Ok(())
}

#[allow(dead_code)]
fn benchmark_j0_performance(size: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Create test data
    let data_f32: Array1<f32> = Array1::from_vec((0..size).map(|i| (i as f32) * 0.1).collect());

    // Benchmark scalar implementation
    let start = Instant::now();
    let _scalar_f32: Array1<f32> = data_f32.mapv(|x| j0(x as f64) as f32);
    let scalar_time = start.elapsed();

    println!("Bessel J0 function performance:");
    println!("  Scalar f32: {:?}", scalar_time);

    // Benchmark SIMD implementation if available
    #[cfg(feature = "simd")]
    {
        let start = Instant::now();
        let _simd_f32 = j0_f32_simd(&data_f32.view())?;
        let simd_time = start.elapsed();

        println!("  SIMD f32:   {:?}", simd_time);

        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
        println!("  Speedup:    {:.2}x", speedup);
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("  SIMD: Not available (enable with --features simd)");
    }

    Ok(())
}

#[allow(dead_code)]
fn demonstrate_precision_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("Precision Comparison:");
    println!("{}", "-".repeat(40));

    let test_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5];
    let _data_f32 = Array1::from_vec(test_values.iter().map(|&x| x as f32).collect());
    let data_f64 = Array1::from_vec(test_values.clone());

    println!("Gamma function precision (first 8 values):");
    println!("Value    Scalar f64      SIMD f64       Error");
    println!("{}", "-".repeat(50));

    #[cfg(feature = "simd")]
    {
        let scalar_results: Vec<f64> = data_f64.iter().map(|&x| gamma(x)).collect();
        let simd_results = gamma_f64_simd(&data_f64.view())?;

        for i in 0..test_values.len() {
            let error = (scalar_results[i] - simd_results[i]).abs();
            println!(
                "{:5.1}    {:12.6}    {:12.6}   {:10.2e}",
                test_values[i], scalar_results[i], simd_results[i], error
            );
        }
    }

    #[cfg(not(feature = "simd"))]
    {
        println!("SIMD precision comparison not available (enable with --features simd)");
    }

    Ok(())
}
