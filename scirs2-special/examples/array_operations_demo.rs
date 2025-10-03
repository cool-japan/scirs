//! Array Operations Demonstration
//!
//! This example showcases the comprehensive array operation capabilities
//! of scirs2-special, including vectorized special functions, broadcasting,
//! memory-efficient processing, and complex number arrays.

use scirs2_core::ndarray::{arr1, arr2, Array1};
use scirs2_core::Complex64;
#[cfg(feature = "gpu")]
use scirs2_special::array_ops::{
    broadcasting, complex, convenience, memory_efficient, ArrayConfig,
};
use scirs2_special::gamma;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "gpu"))]
    {
        println!("This example requires the 'gpu' feature to be enabled.");
        println!("Run with: cargo run --example array_operations_demo --features gpu");
        return Ok(());
    }

    #[cfg(feature = "gpu")]
    {
        println!("=== SCIRS2-SPECIAL Array Operations Demo ===\n");

        // 1. Basic Vectorized Operations
        demo_vectorized_operations().await?;

        // 2. Multidimensional Arrays
        demo_multidimensional_arrays().await?;

        // 3. Broadcasting Operations
        demo_broadcasting()?;

        // 4. Complex Number Arrays
        demo_complex_arrays()?;

        // 5. Memory-Efficient Processing
        demo_memory_efficiency()?;

        // 6. Performance Comparison
        demo_performance_comparison().await?;

        println!("=== Array operations demo completed successfully! ===");

        Ok(())
    }
}

#[cfg(feature = "gpu")]
async fn demo_vectorized_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Vectorized Special Function Operations");
    println!("========================================");

    // Gamma function on arrays
    let gammainput = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let gamma_result = convenience::gamma_1d(&gammainput).await?;
    println!("Gamma function:");
    println!("  Input:  {:?}", gammainput);
    println!("  Output: {:?}", gamma_result);
    println!("  Expected: [1, 1, 2, 6, 24]");

    // Bessel J0 function on arrays
    let besselinput = arr1(&[0.0, 1.0, 2.4048, 5.0]);
    let bessel_result = convenience::j0_1d(&besselinput)?;
    println!("\nBessel J₀ function:");
    println!("  Input:  {:?}", besselinput);
    println!("  Output: {:?}", bessel_result);

    // Error function on arrays
    let erfinput = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let erf_result = convenience::erf_1d(&erfinput)?;
    println!("\nError function:");
    println!("  Input:  {:?}", erfinput);
    println!("  Output: {:?}", erf_result);

    // Factorial on arrays
    let factorialinput = arr1(&[0, 1, 2, 3, 4, 5]);
    let factorial_result = convenience::factorial_1d(&factorialinput)?;
    println!("\nFactorial function:");
    println!("  Input:  {:?}", factorialinput);
    println!("  Output: {:?}", factorial_result);

    // Softmax on arrays
    let softmax_input = arr1(&[1.0, 2.0, 3.0, 4.0]);
    let softmax_result = convenience::softmax_1d(&softmax_input)?;
    println!("\nSoftmax function:");
    println!("  Input:  {:?}", softmax_input);
    println!("  Output: {:?}", softmax_result);
    println!("  Sum:    {:.10}", softmax_result.sum());

    println!();
    Ok(())
}

#[cfg(feature = "gpu")]
async fn demo_multidimensional_arrays() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Multidimensional Array Operations");
    println!("====================================");

    // 2D gamma function
    let gamma_2d = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let gamma_2d_result = convenience::gamma_2d(&gamma_2d).await?;
    println!("2D Gamma function:");
    println!("Input:\n{:?}", gamma_2d);
    println!("Output:\n{:?}", gamma_2d_result);

    // Large array processing
    let largeinput: Array1<f64> = Array1::linspace(0.5, 10.0, 1000);
    println!("\nProcessing large array (1000 elements):");

    let start = Instant::now();
    let large_gamma_result = convenience::gamma_1d(&largeinput).await?;
    let duration = start.elapsed();

    println!(
        "  Input range: [{:.2}, {:.2}]",
        largeinput[0], largeinput[999]
    );
    println!(
        "  Output range: [{:.2e}, {:.2e}]",
        large_gamma_result[0], large_gamma_result[999]
    );
    println!("  Processing time: {:?}", duration);

    println!();
    Ok(())
}

#[allow(dead_code)]
#[cfg(feature = "gpu")]
fn demo_broadcasting() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Broadcasting Operations");
    println!("==========================");

    // Test broadcasting compatibility
    let shape1 = [3, 1];
    let shape2 = [1, 4];
    let can_broadcast = broadcasting::can_broadcast(&shape1, &shape2);
    println!(
        "Can broadcast {:?} and {:?}: {}",
        shape1, shape2, can_broadcast
    );

    if can_broadcast {
        let broadcastshape = broadcasting::broadcastshape(&shape1, &shape2)?;
        println!("Broadcast shape: {:?}", broadcastshape);
    }

    // Test various broadcasting scenarios
    let test_cases = [
        (vec![2, 3, 4], vec![3, 4]),
        (vec![5, 1], vec![1, 7]),
        (vec![3, 2], vec![4, 5]), // This should fail
        (vec![1, 10], vec![10, 1]),
    ];

    println!("\nBroadcasting compatibility tests:");
    for (shape1, shape2) in test_cases {
        let compatible = broadcasting::can_broadcast(&shape1, &shape2);
        print!("  {:?} × {:?}: {}", shape1, shape2, compatible);
        if compatible {
            if let Ok(resultshape) = broadcasting::broadcastshape(&shape1, &shape2) {
                print!(" → {:?}", resultshape);
            }
        }
        println!();
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
#[cfg(feature = "gpu")]
fn demo_complex_arrays() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Complex Number Array Operations");
    println!("==================================");

    // Lambert W function on complex arrays
    let complex_input = Array1::from(vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(2.0, 2.0),
    ]);

    println!("Lambert W function on complex arrays:");
    println!("Input:");
    for (i, z) in complex_input.iter().enumerate() {
        println!("  z[{}] = {:.3} + {:.3}i", i, z.re, z.im);
    }

    let config = ArrayConfig::default();
    let lambert_result = complex::lambert_w_array(&complex_input, 0, 1e-12, &config)?;

    println!("Output (W₀(z)):");
    for (i, w) in lambert_result.iter().enumerate() {
        println!("  W[{}] = {:.6} + {:.6}i", i, w.re, w.im);
    }

    // Verify the Lambert W property: W(z) * exp(W(z)) = z
    println!("\nVerification (W(z) * exp(W(z)) should equal z):");
    for (i, (&z, &w)) in complex_input.iter().zip(lambert_result.iter()).enumerate() {
        if w.is_finite() {
            let verification = w * w.exp();
            let error = (verification - z).norm();
            println!("  z[{}]: error = {:.2e}", i, error);
        }
    }

    println!();
    Ok(())
}

#[allow(dead_code)]
#[cfg(feature = "gpu")]
fn demo_memory_efficiency() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Memory-Efficient Processing");
    println!("==============================");

    // Memory usage estimation
    let shapes_to_test = [[100, 100], [1000, 1000], [5000, 5000]];

    println!("Memory usage estimation for f64 arrays:");
    for shape in shapes_to_test {
        let memory_single = memory_efficient::estimate_memory_usage::<f64>(&shape, 1);
        let memory_pair = memory_efficient::estimate_memory_usage::<f64>(&shape, 2);

        println!(
            "  {:?}: {:.1} MB (single), {:.1} MB (pair)",
            shape,
            memory_single as f64 / 1024.0 / 1024.0,
            memory_pair as f64 / 1024.0 / 1024.0
        );

        let config = ArrayConfig::default();
        let fits_in_limit = memory_efficient::check_memory_limit::<f64>(&shape, 2, &config);
        println!(
            "    Fits in memory limit ({}GB): {}",
            config.memory_limit / 1024 / 1024 / 1024,
            fits_in_limit
        );
    }

    // Configuration showcase
    println!("\nArray processing configurations:");

    let configs = [
        convenience::ConfigBuilder::new()
            .chunksize(512)
            .parallel(false)
            .memory_limit(512 * 1024 * 1024)
            .build(),
        convenience::ConfigBuilder::new()
            .chunksize(2048)
            .parallel(false)
            .memory_limit(2 * 1024 * 1024 * 1024)
            .build(),
        ArrayConfig::default(),
    ];

    for (i, config) in configs.iter().enumerate() {
        println!(
            "  Config {}: chunksize={}, parallel={}, memory_limit={}MB",
            i + 1,
            config.chunksize,
            config.parallel,
            config.memory_limit / 1024 / 1024
        );
    }

    println!();
    Ok(())
}

#[cfg(feature = "gpu")]
async fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Performance Comparison");
    println!("=========================");

    // Compare element-wise vs vectorized operations
    let sizes = [100, 1000, 10000];

    for size in sizes {
        println!("Array size: {} elements", size);

        let input: Array1<f64> = Array1::linspace(0.5, 10.0, size);

        // Element-wise operation
        let start = Instant::now();
        let _element_wise: Array1<f64> = input.mapv(gamma);
        let element_wise_time = start.elapsed();

        // Vectorized operation
        let start = Instant::now();
        let _vectorized = convenience::gamma_1d(&input).await?;
        let vectorized_time = start.elapsed();

        println!("  Element-wise: {:?}", element_wise_time);
        println!("  Vectorized:   {:?}", vectorized_time);

        if vectorized_time.as_nanos() > 0 {
            let speedup = element_wise_time.as_nanos() as f64 / vectorized_time.as_nanos() as f64;
            println!("  Speedup:      {:.2}x", speedup);
        }

        println!();
    }

    // Memory bandwidth test
    println!("Memory bandwidth test (large arrays):");
    let largesize = 100_000;
    let largeinput: Array1<f64> = Array1::linspace(0.1, 5.0, largesize);

    let start = Instant::now();
    let _large_result = convenience::gamma_1d(&largeinput).await?;
    let duration = start.elapsed();

    let throughput = largesize as f64 / duration.as_secs_f64();
    println!("  Processed {} elements in {:?}", largesize, duration);
    println!("  Throughput: {:.0} elements/second", throughput);

    let memory_bandwidth = (largesize * 16) as f64 / duration.as_secs_f64() / 1024.0 / 1024.0; // 16 bytes per f64 (input + output)
    println!("  Memory bandwidth: {:.1} MB/s", memory_bandwidth);

    println!();
    Ok(())
}
