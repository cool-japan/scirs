//! Advanced Array Operations Demo
//!
//! This example demonstrates the enhanced array operations in scirs2-special,
//! including lazy evaluation, GPU acceleration, and multidimensional support.

use ndarray::{Array, Array1};
use scirs2_special::array_ops::{
    convenience::{self, ConfigBuilder},
    ArrayConfig, Backend,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SCIRS2-SPECIAL Advanced Array Operations Demo ===\n");

    // 1. Basic array operations
    demo_basic_operations().await?;

    // 2. Configuration and backend selection
    demo_configuration_options().await?;

    // 3. Lazy evaluation
    #[cfg(feature = "lazy")]
    demo_lazy_evaluation().await?;

    // 4. GPU acceleration
    #[cfg(feature = "gpu")]
    demo_gpu_acceleration().await?;

    // 5. Large array processing
    demo_large_array_processing().await?;

    // 6. Batch processing
    demo_batch_processing().await?;

    // 7. Memory-efficient operations
    demo_memory_efficient_operations().await?;

    println!("=== Advanced array operations demo completed successfully! ===");
    Ok(())
}

async fn demo_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Array Operations");
    println!("========================");

    // 1D gamma function
    let input_1d = Array1::linspace(1.0, 5.0, 5);
    println!("Input 1D array: {:?}", input_1d);

    let result_1d = convenience::gamma_1d(&input_1d).await?;
    println!("Gamma(input): {:?}", result_1d);

    // 2D gamma function
    let input_2d = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    println!("\nInput 2D array:\n{:?}", input_2d);

    let result_2d = convenience::gamma_2d(&input_2d).await?;
    println!("Gamma(input) 2D:\n{:?}", result_2d);

    // Bessel J0 function
    let bessel_input = Array1::linspace(0.0, 5.0, 6);
    let bessel_result = convenience::j0_1d(&bessel_input)?;
    println!("\nBessel J0({:?}) = {:?}", bessel_input, bessel_result);

    // Error function
    let erf_input = Array1::linspace(-2.0, 2.0, 5);
    let erf_result = convenience::erf_1d(&erf_input)?;
    println!("erf({:?}) = {:?}", erf_input, erf_result);

    println!();
    Ok(())
}

async fn demo_configuration_options() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Configuration Options and Backend Selection");
    println!("=============================================");

    // Using configuration builder
    let custom_config = ConfigBuilder::new()
        .chunk_size(512)
        .parallel(true)
        .memory_limit(512 * 1024 * 1024) // 512MB
        .lazy_threshold(1000)
        .build();

    println!(
        "Custom config: chunk_size={}, parallel={}, memory_limit={}MB",
        custom_config.chunk_size,
        custom_config.parallel,
        custom_config.memory_limit / (1024 * 1024)
    );

    // Test with custom config
    let input = Array1::linspace(1.0, 4.0, 4);
    let result = convenience::gamma_1d_with_config(&input, &custom_config).await?;
    println!("Gamma with custom config: {:?}", result);

    // Predefined configurations
    let large_config = convenience::large_array_config();
    println!(
        "\nLarge array config: chunk_size={}, lazy_threshold={}",
        large_config.chunk_size, large_config.lazy_threshold
    );

    let small_config = convenience::small_array_config();
    println!(
        "Small array config: chunk_size={}, lazy_threshold={}",
        small_config.chunk_size, small_config.lazy_threshold
    );

    // Test different backends
    let backends = vec![Backend::Cpu];

    #[cfg(feature = "lazy")]
    let backends = {
        let mut b = backends;
        b.push(Backend::Lazy);
        b
    };

    #[cfg(feature = "gpu")]
    let backends = {
        let mut b = backends;
        b.push(Backend::Gpu);
        b
    };

    for backend in backends {
        let config = ArrayConfig {
            backend: backend.clone(),
            ..Default::default()
        };
        println!("Testing backend: {:?}", backend);

        let test_input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = convenience::gamma_1d_with_config(&test_input, &config).await?;
        println!("  Result: {:?}", result);
    }

    println!();
    Ok(())
}

#[cfg(feature = "lazy")]
async fn demo_lazy_evaluation() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Lazy Evaluation");
    println!("==================");

    // Create a large array that would benefit from lazy evaluation
    let large_input = Array::linspace(1.0, 100.0, 10000);
    println!("Created large array with {} elements", large_input.len());

    // Create lazy computation
    let lazy_gamma = convenience::gamma_lazy(&large_input, None)?;
    println!("Lazy computation created: {}", lazy_gamma.description());
    println!("Cost estimate: {} units", lazy_gamma.cost_estimate());
    println!("Is computed: {}", lazy_gamma.is_computed());

    // Force computation
    println!("Computing lazily...");
    let start_time = std::time::Instant::now();
    let result = lazy_gamma.compute()?;
    let duration = start_time.elapsed();

    println!("Computation completed in {:?}", duration);
    println!("Result shape: {:?}", result.shape());
    println!("First 5 values: {:?}", &result.as_slice().unwrap()[0..5]);

    // Demonstrate caching
    println!("\nTesting computation caching...");
    let lazy_gamma2 = convenience::gamma_lazy(&large_input, None)?;
    let start_time2 = std::time::Instant::now();
    let _result2 = lazy_gamma2.compute()?;
    let _result2_cached = lazy_gamma2.compute()?; // Should be cached
    let duration2 = start_time2.elapsed();
    println!("Second computation (with caching) took: {:?}", duration2);

    println!();
    Ok(())
}

#[cfg(feature = "gpu")]
async fn demo_gpu_acceleration() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. GPU Acceleration");
    println!("==================");

    let gpu_input = Array1::linspace(1.0, 10.0, 1000);
    println!("Processing array of {} elements on GPU", gpu_input.len());

    match convenience::gamma_gpu(&gpu_input).await {
        Ok(gpu_result) => {
            println!("GPU computation successful!");
            println!("Result shape: {:?}", gpu_result.shape());
            println!(
                "First 5 values: {:?}",
                &gpu_result.as_slice().unwrap()[0..5]
            );

            // Compare with CPU result
            let cpu_result = convenience::gamma_1d(&gpu_input).await?;
            let max_diff = gpu_result
                .iter()
                .zip(cpu_result.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            println!("Maximum difference between GPU and CPU: {:.2e}", max_diff);
        }
        Err(e) => {
            println!("GPU computation failed (falling back to CPU): {}", e);
            let cpu_result = convenience::gamma_1d(&gpu_input).await?;
            println!("CPU fallback result computed successfully");
            println!("Result shape: {:?}", cpu_result.shape());
        }
    }

    println!();
    Ok(())
}

async fn demo_large_array_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Large Array Processing");
    println!("=========================");

    // Create a large array
    let large_array = Array::linspace(0.1, 10.0, 50000);
    println!("Processing large array with {} elements", large_array.len());

    // Use configuration optimized for large arrays
    let config = convenience::large_array_config();

    let start_time = std::time::Instant::now();
    let result = convenience::gamma_1d_with_config(&large_array, &config).await?;
    let duration = start_time.elapsed();

    println!("Large array processing completed in {:?}", duration);
    println!("Result statistics:");
    println!(
        "  Min: {:.6}",
        result.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!(
        "  Max: {:.6}",
        result.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!("  Mean: {:.6}", result.mean().unwrap());

    // Memory usage estimation
    use scirs2_special::array_ops::memory_efficient;
    let memory_usage = memory_efficient::estimate_memory_usage::<f64>(result.shape(), 2);
    println!(
        "Estimated memory usage: {:.2} MB",
        memory_usage as f64 / (1024.0 * 1024.0)
    );

    println!();
    Ok(())
}

async fn demo_batch_processing() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Batch Processing");
    println!("==================");

    // Create multiple arrays to process
    let arrays = vec![
        Array1::linspace(1.0, 5.0, 100),
        Array1::linspace(2.0, 6.0, 100),
        Array1::linspace(3.0, 7.0, 100),
        Array1::linspace(0.5, 4.5, 100),
    ];

    println!("Processing batch of {} arrays", arrays.len());

    let config = ArrayConfig::default();
    let start_time = std::time::Instant::now();
    let results = convenience::batch_gamma(&arrays, &config).await?;
    let duration = start_time.elapsed();

    println!("Batch processing completed in {:?}", duration);
    println!("Results summary:");
    for (i, result) in results.iter().enumerate() {
        let mean = result.mean().unwrap();
        println!("  Array {}: mean = {:.6}", i + 1, mean);
    }

    println!();
    Ok(())
}

async fn demo_memory_efficient_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Memory-Efficient Operations");
    println!("==============================");

    // Create a moderately large 2D array
    let shape = (1000, 100);
    let array_2d = Array::ones(shape);
    println!("Created 2D array with shape {:?}", shape);

    // Check memory limits
    use scirs2_special::array_ops::memory_efficient;
    let memory_needed = memory_efficient::estimate_memory_usage::<f64>(
        array_2d.shape(),
        3, // input, output, temporary
    );
    println!(
        "Estimated memory needed: {:.2} MB",
        memory_needed as f64 / (1024.0 * 1024.0)
    );

    let config = ArrayConfig::default();
    let within_limit = memory_efficient::check_memory_limit::<f64>(array_2d.shape(), 3, &config);
    println!("Within memory limit: {}", within_limit);

    // Process with chunking for memory efficiency
    use scirs2_special::array_ops::vectorized;
    let chunked_result = vectorized::process_chunks(&array_2d, &config, |x: f64| x * 2.0 + 1.0)?;

    println!("Chunked processing completed");
    println!("Result shape: {:?}", chunked_result.shape());
    println!(
        "Sample values: {:?}",
        &chunked_result.as_slice().unwrap()[0..5]
    );

    // Parallel processing comparison
    #[cfg(feature = "parallel")]
    {
        let parallel_input = Array1::linspace(-1.0, 1.0, 10000);

        println!("\nParallel vs Sequential processing comparison:");

        // Sequential
        let start_time = std::time::Instant::now();
        let _seq_result = convenience::erf_1d(&parallel_input)?;
        let seq_duration = start_time.elapsed();

        // Parallel
        let start_time = std::time::Instant::now();
        let _par_result = convenience::erf_parallel(&parallel_input)?;
        let par_duration = start_time.elapsed();

        println!("  Sequential: {:?}", seq_duration);
        println!("  Parallel: {:?}", par_duration);
        println!(
            "  Speedup: {:.2}x",
            seq_duration.as_secs_f64() / par_duration.as_secs_f64()
        );
    }

    println!();
    Ok(())
}
