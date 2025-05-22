use num_complex::Complex64;
use scirs2_fft::planning_adaptive::{AdaptiveExecutor, AdaptivePlanningConfig};
use std::time::{Duration, Instant};

fn main() {
    println!("Adaptive FFT Planning Example");
    println!("============================\n");

    // Configure the adaptive planner to switch strategies quickly for demonstration
    let config = AdaptivePlanningConfig {
        enabled: true,
        min_samples: 3, // Need at least 3 samples before switching
        evaluation_interval: Duration::from_millis(100), // Evaluate frequently
        max_strategy_switches: 10, // Allow multiple switches for demo
        enable_backend_switching: false, // Keep it simple
        improvement_threshold: 1.05, // Just 5% improvement needed for switching
    };

    // Create an adaptive executor for a specific size
    let size = 1024;
    let adaptive_executor = AdaptiveExecutor::new(&[size], true, Some(config));

    println!("Running FFT operations and adapting to performance...");

    // Create test data
    let mut input = vec![Complex64::default(); size];
    for i in 0..size {
        let phase = 2.0 * std::f64::consts::PI * (i as f64) / 100.0;
        input[i] = Complex64::new(phase.cos(), phase.sin());
    }
    let mut output = vec![Complex64::default(); size];

    // Run multiple iterations with stats tracking
    let iterations = 20;
    let mut total_time = Duration::from_nanos(0);

    for i in 1..=iterations {
        let start = Instant::now();

        // Execute FFT with adaptive planning
        adaptive_executor.execute(&input, &mut output).unwrap();

        let elapsed = start.elapsed();
        total_time += elapsed;

        // Print current stats every few iterations
        if i % 5 == 0 || i == 1 {
            println!("\nIteration {}/{}", i, iterations);
            println!(
                "  Current strategy: {:?}",
                adaptive_executor.current_strategy()
            );
            println!("  Execution time: {:?}", elapsed);
            println!("  Average time: {:?}", total_time / i as u32);

            // Print detailed statistics
            println!("\n  Strategy performance:");
            for (strategy, (avg_time, count)) in adaptive_executor.get_statistics() {
                if count > 0 {
                    println!("    {:?}: {:?} avg ({} samples)", strategy, avg_time, count);
                }
            }
        }

        // Artificially manipulate execution conditions on iteration 10
        // to simulate changing system load
        if i == 10 {
            println!("\nSimulating increased system load...");
            // Perform some busy work to slow things down
            for _ in 0..10_000_000 {
                std::hint::black_box(1 + 1);
            }
        }
    }

    println!(
        "\nFinal Strategy: {:?}",
        adaptive_executor.current_strategy()
    );
    println!(
        "Overall average execution time: {:?}",
        total_time / iterations as u32
    );

    // Show final stats for all strategies
    println!("\nFinal Strategy Statistics:");
    for (strategy, (avg_time, count)) in adaptive_executor.get_statistics() {
        if count > 0 {
            println!("  {:?}: {:?} avg ({} samples)", strategy, avg_time, count);
        }
    }
}
