//! Example demonstrating performance optimization features
//!
//! This example shows how to use the performance optimization
//! module for efficient matrix operations on large matrices.

use scirs2_core::ndarray::{Array2, ShapeBuilder};
use scirs2_core::random::{thread_rng, Distribution, Uniform};
use scirs2_linalg::prelude::*;
use std::time::Instant;

// Helper function to create random arrays (POLICY-compliant)
fn random_array<Sh: ShapeBuilder<Dim = scirs2_core::ndarray::Ix2>>(
    shape: Sh,
    low: f64,
    high: f64,
) -> Array2<f64> {
    let dist = Uniform::new(low, high).unwrap();
    let mut rng = thread_rng();
    Array2::from_shape_fn(shape, |_| dist.sample(&mut rng))
}

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Performance Optimization Example");
    println!("================================\n");

    // Demonstrate blocked matrix multiplication
    demo_blocked_matmul()?;

    // Demonstrate in-place operations
    demo_inplace_operations()?;

    // Demonstrate optimized transpose
    demo_optimized_transpose()?;

    // Demonstrate adaptive algorithm selection
    demo_adaptive_algorithm()?;

    // Demonstrate parallel processing control
    demo_parallel_control()?;

    Ok(())
}

#[allow(dead_code)]
fn demo_blocked_matmul() -> Result<(), Box<dyn std::error::Error>> {
    println!("Blocked Matrix Multiplication");
    println!("----------------------------");

    let size = 1024;
    let a = random_array((size, size).f(), -1.0, 1.0);
    let b = random_array((size, size).f(), -1.0, 1.0);

    // Standard matrix multiplication
    let start = Instant::now();
    let c_standard = a.dot(&b);
    let time_standard = start.elapsed();

    // Blocked matrix multiplication
    let config = OptConfig::default()
        .with_blocksize(64)
        .with_parallel_threshold(256);

    let start = Instant::now();
    let c_blocked = blocked_matmul(&a.view(), &b.view(), &config)?;
    let time_blocked = start.elapsed();

    // Verify correctness (check a few elements)
    let tolerance = 1e-10;
    let mut max_diff: f64 = 0.0;
    for i in 0..10 {
        for j in 0..10 {
            let diff = (c_standard[[i, j]] - c_blocked[[i, j]]).abs();
            max_diff = max_diff.max(diff);
        }
    }

    println!("Matrix size: {}x{}", size, size);
    println!("Standard time: {:?}", time_standard);
    println!("Blocked time: {:?}", time_blocked);
    println!(
        "Speedup: {:.2}x",
        time_standard.as_secs_f64() / time_blocked.as_secs_f64()
    );
    println!("Max difference: {:.2e}", max_diff);
    println!("Results match: {}\n", max_diff < tolerance);

    Ok(())
}

#[allow(dead_code)]
fn demo_inplace_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("In-place Operations");
    println!("------------------");

    let size = 2048;
    let mut a = random_array((size, size).f(), -1.0, 1.0);
    let b = random_array((size, size).f(), -1.0, 1.0);

    // Standard addition (creates new array)
    let a_copy = a.clone();
    let start = Instant::now();
    let _c_standard = &a_copy + &b;
    let time_standard = start.elapsed();
    let memory_standard = size * size * std::mem::size_of::<f64>();

    // In-place addition
    let start = Instant::now();
    inplace_add(&mut a.view_mut(), &b.view())?;
    let time_inplace = start.elapsed();

    println!("Matrix size: {}x{}", size, size);
    println!("Standard addition time: {:?}", time_standard);
    println!("In-place addition time: {:?}", time_inplace);
    println!(
        "Speedup: {:.2}x",
        time_standard.as_secs_f64() / time_inplace.as_secs_f64()
    );
    println!(
        "Memory saved: {:.2} MB\n",
        memory_standard as f64 / (1024.0 * 1024.0)
    );

    // Demonstrate in-place scaling
    let mut a = random_array((size, size).f(), -1.0, 1.0);
    let scale = 2.5;

    // Standard scaling
    let a_copy = a.clone();
    let start = Instant::now();
    let _c_standard = &a_copy * scale;
    let time_standard = start.elapsed();

    // In-place scaling
    let start = Instant::now();
    let _ = inplace_scale(&mut a.view_mut(), scale);
    let time_inplace = start.elapsed();

    println!("Scaling operation:");
    println!("Standard time: {:?}", time_standard);
    println!("In-place time: {:?}", time_inplace);
    println!(
        "Speedup: {:.2}x\n",
        time_standard.as_secs_f64() / time_inplace.as_secs_f64()
    );

    Ok(())
}

#[allow(dead_code)]
fn demo_optimized_transpose() -> Result<(), Box<dyn std::error::Error>> {
    println!("Optimized Transpose");
    println!("------------------");

    let size = 2048;
    let a = random_array((size, size).f(), -1.0, 1.0);

    // Standard transpose
    let start = Instant::now();
    let _b_standard = a.t().to_owned();
    let time_standard = start.elapsed();

    // Optimized transpose
    let start = Instant::now();
    let _b_optimized = optimized_transpose(&a.view())?;
    let time_optimized = start.elapsed();

    println!("Matrix size: {}x{}", size, size);
    println!("Standard transpose time: {:?}", time_standard);
    println!("Optimized transpose time: {:?}", time_optimized);
    println!(
        "Speedup: {:.2}x\n",
        time_standard.as_secs_f64() / time_optimized.as_secs_f64()
    );

    Ok(())
}

#[allow(dead_code)]
fn demo_adaptive_algorithm() -> Result<(), Box<dyn std::error::Error>> {
    println!("Adaptive Algorithm Selection");
    println!("---------------------------");

    let sizes = [64, 256, 1024];

    for size in &sizes {
        let a = random_array((*size, *size).f(), -1.0, 1.0);
        let b = random_array((*size, *size).f(), -1.0, 1.0);

        // Use adaptive algorithm
        let config = OptConfig::default().with_algorithm(OptAlgorithm::Adaptive);

        let start = Instant::now();
        let _c = blocked_matmul(&a.view(), &b.view(), &config)?;
        let time = start.elapsed();

        // The adaptive algorithm will choose:
        // - Standard for small matrices (<= 128)
        // - Blocked for larger matrices
        // - Parallel blocked for very large matrices

        println!("Size: {}x{}, Time: {:?}", size, size, time);
    }
    println!();

    Ok(())
}

#[allow(dead_code)]
fn demo_parallel_control() -> Result<(), Box<dyn std::error::Error>> {
    println!("Parallel Processing Control");
    println!("--------------------------");

    let size = 1024;
    let a = random_array((size, size).f(), -1.0, 1.0);
    let b = random_array((size, size).f(), -1.0, 1.0);

    // Force serial execution
    let config_serial = OptConfig::default()
        .with_blocksize(64)
        .with_parallel_threshold(size * 2); // Threshold higher than matrix size

    let start = Instant::now();
    let _c_serial = blocked_matmul(&a.view(), &b.view(), &config_serial)?;
    let time_serial = start.elapsed();

    // Force parallel execution
    let config_parallel = OptConfig::default()
        .with_blocksize(64)
        .with_parallel_threshold(0); // Always use parallel

    let start = Instant::now();
    let _c_parallel = blocked_matmul(&a.view(), &b.view(), &config_parallel)?;
    let time_parallel = start.elapsed();

    println!("Matrix size: {}x{}", size, size);
    println!("Serial execution time: {:?}", time_serial);
    println!("Parallel execution time: {:?}", time_parallel);
    println!(
        "Parallel speedup: {:.2}x",
        time_serial.as_secs_f64() / time_parallel.as_secs_f64()
    );

    // Demonstrate benchmarking utility
    println!("\nBuilt-in benchmarking:");
    let config = OptConfig::default();
    let bench_results = matmul_benchmark(&a.view(), &b.view(), &config)?;
    println!("{}", bench_results);

    Ok(())
}
