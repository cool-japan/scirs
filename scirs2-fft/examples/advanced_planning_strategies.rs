use ndarray::{Array2, ShapeBuilder};
use num_complex::Complex64;
use scirs2_fft::{
    auto_tuning::{AutoTuneConfig, FftVariant, SizeRange, SizeStep},
    fft2, get_global_planner, init_global_planner, FftPlanExecutor, PlanBuilder, PlanningConfig,
    PlanningStrategy,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

// Helper function to create a test array with an impulse
fn create_test_array(size: usize) -> Array2<Complex64> {
    let mut array = Array2::zeros((size, size).f());
    array[[size / 4, size / 4]] = Complex64::new(1.0, 0.0);
    array
}

// Helper function to benchmark a strategy with multiple sizes
fn benchmark_strategy(
    strategy: PlanningStrategy,
    sizes: &[usize],
    iterations: usize,
) -> Vec<(usize, Duration, Duration)> {
    let mut results = Vec::new();

    for &size in sizes {
        let test_array = create_test_array(size);
        let input_flat: Vec<Complex64> = test_array.iter().cloned().collect();

        // Create the plan
        let builder = PlanBuilder::new()
            .shape(&[size, size])
            .forward(true)
            .strategy(strategy);

        // Measure plan creation time (first time)
        let start = Instant::now();
        let plan = match builder.build() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let plan_time = start.elapsed();

        // Create the executor
        let executor = FftPlanExecutor::new(plan);
        let mut result_data = vec![Complex64::default(); size * size];

        // Measure execution time (average over iterations)
        let mut total_exec_time = Duration::from_secs(0);

        for _ in 0..iterations {
            let start = Instant::now();
            if let Err(_) = executor.execute(&input_flat, &mut result_data) {
                continue;
            }
            total_exec_time += start.elapsed();
        }

        let avg_exec_time = total_exec_time / iterations as u32;

        results.push((size, plan_time, avg_exec_time));
    }

    results
}

fn main() {
    println!("Advanced FFT Planning Strategies Benchmark");
    println!("==========================================\n");

    // Configure auto-tuning
    let auto_tune_config = AutoTuneConfig {
        sizes: SizeRange {
            min: 64,
            max: 2048,
            step: SizeStep::PowersOfTwo,
        },
        repetitions: 10,
        warmup: 3,
        variants: vec![FftVariant::Standard, FftVariant::Cached],
        database_path: PathBuf::from(".fft_tuning_db.json"),
    };

    // Initialize global planner with auto-tuning
    let config = PlanningConfig {
        strategy: PlanningStrategy::AutoTuned,
        measure_performance: true,
        serialized_db_path: Some("./auto_tuned_plans.json".to_string()),
        auto_tune_config: Some(auto_tune_config),
        ..Default::default()
    };

    let _ = init_global_planner(config);

    // Sizes to benchmark
    let sizes = [128, 256, 512, 1024];
    let iterations = 5; // Number of iterations for averaging execution time

    // Benchmark different strategies
    println!("Benchmarking AlwaysNew strategy...");
    let always_new_results = benchmark_strategy(PlanningStrategy::AlwaysNew, &sizes, iterations);

    println!("Benchmarking CacheFirst strategy...");
    let cache_first_results = benchmark_strategy(PlanningStrategy::CacheFirst, &sizes, iterations);

    println!("Benchmarking SerializedFirst strategy...");
    let serialized_results =
        benchmark_strategy(PlanningStrategy::SerializedFirst, &sizes, iterations);

    println!("Benchmarking AutoTuned strategy...");
    let auto_tuned_results = benchmark_strategy(PlanningStrategy::AutoTuned, &sizes, iterations);

    // Also benchmark the standard FFT function for comparison
    println!("Benchmarking standard FFT function...");
    let mut standard_results = Vec::new();

    for &size in &sizes {
        let test_array = create_test_array(size);

        let mut total_time = Duration::from_secs(0);

        for _ in 0..iterations {
            let start = Instant::now();
            let _ = fft2(&test_array, None, None, None).unwrap();
            total_time += start.elapsed();
        }

        let avg_time = total_time / iterations as u32;
        standard_results.push((size, Duration::from_secs(0), avg_time)); // No separate planning time
    }

    // Print results
    println!("\nResults:");
    println!("========\n");
    println!(
        "{:<10} {:<15} {:<20} {:<20} {:<20}",
        "Size", "Strategy", "Plan Time", "Exec Time", "Total Time"
    );
    println!("{:-<85}", "");

    for (i, &size) in sizes.iter().enumerate() {
        let strategies = [
            ("Standard", &standard_results[i]),
            ("AlwaysNew", &always_new_results[i]),
            ("CacheFirst", &cache_first_results[i]),
            ("Serialized", &serialized_results[i]),
            ("AutoTuned", &auto_tuned_results[i]),
        ];

        for (name, result) in strategies {
            let (_, plan_time, exec_time) = *result;
            let total_time = plan_time + exec_time;

            println!(
                "{:<10} {:<15} {:<20?} {:<20?} {:<20?}",
                size, name, plan_time, exec_time, total_time
            );
        }

        println!("{:-<85}", "");
    }

    // Performance comparison across sizes
    println!("\nRelative Performance (lower is better, baseline is Standard FFT):");
    println!("{:<10} {:<15} {:<20}", "Size", "Strategy", "Relative Time");
    println!("{:-<45}", "");

    for (i, &size) in sizes.iter().enumerate() {
        let baseline = standard_results[i].2; // Standard FFT execution time

        let strategies = [
            ("Standard", standard_results[i].2),
            (
                "AlwaysNew",
                always_new_results[i].1 + always_new_results[i].2,
            ),
            (
                "CacheFirst",
                cache_first_results[i].1 + cache_first_results[i].2,
            ),
            (
                "Serialized",
                serialized_results[i].1 + serialized_results[i].2,
            ),
            (
                "AutoTuned",
                auto_tuned_results[i].1 + auto_tuned_results[i].2,
            ),
        ];

        for (name, time) in strategies {
            let relative = time.as_secs_f64() / baseline.as_secs_f64();
            println!("{:<10} {:<15} {:<20.2}", size, name, relative);
        }

        println!("{:-<45}", "");
    }

    // Get global planner stats
    let planner = get_global_planner();
    let planner_guard = planner.lock().unwrap();

    // Clear cache to make sure subsequent runs will benefit from serialized plans
    planner_guard.clear_cache();

    println!("\nDone! Plan database stored in auto_tuned_plans.json");
    println!("Run this benchmark again to see the benefits of serialized plans.");
}
