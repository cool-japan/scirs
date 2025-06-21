use std::thread;
/// This example demonstrates how the various new features can be integrated together
/// to create powerful computational pipelines.
use std::time::Duration;

// We'll conditionally import features based on what's enabled
#[cfg(feature = "logging")]
use scirs2_core::logging::{LogLevel, Logger, ProgressTracker};

#[cfg(feature = "profiling")]
use scirs2_core::profiling::{Profiler, Timer};

#[cfg(feature = "random")]
use scirs2_core::random::{DistributionExt, Random};

#[cfg(feature = "memory_management")]
use scirs2_core::memory::{BufferPool, ZeroCopyView};

#[cfg(feature = "types")]
use scirs2_core::types::NumericConversion;

fn main() {
    println!("Integrated Features Example");

    #[cfg(all(
        feature = "logging",
        feature = "profiling",
        feature = "random",
        feature = "memory_management",
        feature = "types"
    ))]
    run_integrated_example();

    #[cfg(not(all(
        feature = "logging",
        feature = "profiling",
        feature = "random",
        feature = "memory_management",
        feature = "types"
    )))]
    println!("Not all required features are enabled. Please run with --features=\"logging profiling random memory_management types\"");
}

#[cfg(all(
    feature = "logging",
    feature = "profiling",
    feature = "random",
    feature = "memory_management",
    feature = "types"
))]
fn run_integrated_example() {
    use ndarray::IxDyn;
    use rand_distr::Normal;

    // Initialize logging
    scirs2_core::logging::set_min_log_level(LogLevel::Debug);
    let logger = Logger::new("integrated_example");
    logger.info("Starting integrated example");

    // Initialize profiling
    Profiler::global().lock().unwrap().start();
    logger.debug("Profiling started");

    // Main processing pipeline with progress tracking
    let mut progress = ProgressTracker::new("Processing Pipeline", 4);

    // Step 1: Generate random data
    let timer_step1 = Timer::start("step1_generate_data");
    logger.debug("Generating random data");

    let mut rng = Random::default();
    let normal_distribution = Normal::new(0.0, 1.0).unwrap();

    // Use memory management for efficient buffer usage
    let mut buffer_pool = BufferPool::<f64>::new();
    let mut large_buffer = buffer_pool.acquire_vec(10000);

    // Fill the buffer with random data
    for elem in large_buffer.iter_mut() {
        *elem = rng.sample(normal_distribution);
    }

    // Create a 2D array from the buffer
    let array_data = normal_distribution.random_array(&mut rng, IxDyn(&[100, 100]));

    progress.update(1);
    timer_step1.stop();
    logger.info("Random data generation completed");

    // Step 2: Process the data
    let timer_step2 = Timer::start("step2_process_data");
    logger.debug("Processing data");

    // Create a zero-copy view for efficient transformation
    let array_view = ZeroCopyView::new(&array_data);

    // Process the data without copying
    let _squared_data = Timer::time_function("transform_data", || array_view.transform(|&x| x * x));

    progress.update(2);
    timer_step2.stop();
    logger.info("Data processing completed");

    // Step 3: Convert data types
    let timer_step3 = Timer::start("step3_convert_types");
    logger.debug("Converting data types");

    // Apply type conversions
    let conversion_result = Timer::time_function("type_conversion", || {
        let mut success_count = 0;
        let mut clamped_count = 0;
        let mut error_count = 0;

        // Try to convert values from the large buffer
        for &value in large_buffer.iter().take(100) {
            // Try normal conversion
            match value.to_numeric::<i16>() {
                Ok(_) => success_count += 1,
                Err(_) => {
                    // If normal conversion fails, use clamping
                    let _ = value.to_numeric_clamped::<i16>();
                    clamped_count += 1;
                    error_count += 1;
                }
            }
        }

        (success_count, clamped_count, error_count)
    });

    progress.update(3);
    timer_step3.stop();
    logger.info(&format!(
        "Type conversion completed: {} successful, {} clamped, {} errors",
        conversion_result.0, conversion_result.1, conversion_result.2
    ));

    // Step 4: Clean up resources
    let timer_step4 = Timer::start("step4_cleanup");
    logger.debug("Cleaning up resources");

    // Return the buffer to the pool
    buffer_pool.release_vec(large_buffer);

    // Simulate some cleanup work
    thread::sleep(Duration::from_millis(200));

    progress.update(4);
    timer_step4.stop();
    logger.info("Resource cleanup completed");

    // Complete the progress tracking
    progress.complete();

    // Print the profiling report
    println!("\n--- Performance Profile ---");
    Profiler::global().lock().unwrap().print_report();

    // Get specific timing information for the most intensive operation
    if let Some((calls, total, avg, max)) = Profiler::global()
        .lock()
        .unwrap()
        .get_timing_stats("transform_data")
    {
        logger.info(&format!(
            "Data transformation details: {} calls, {:.2}ms total, {:.2}ms average, {:.2}ms max",
            calls,
            total.as_secs_f64() * 1000.0,
            avg.as_secs_f64() * 1000.0,
            max.as_secs_f64() * 1000.0
        ));
    }

    // Final information
    logger.info("Integrated example completed successfully");

    // Stop the profiler
    Profiler::global().lock().unwrap().stop();
}
