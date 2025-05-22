use ndarray::Array2;
use num_complex::Complex64;
use scirs2_fft::{
    fft2, get_global_planner, init_global_planner, plan_ahead_of_time, FftPlanExecutor,
    PlanBuilder, PlanningConfig, PlanningStrategy,
};
use std::time::{Duration, Instant};

fn main() {
    println!("FFT Plan Optimization Example");
    println!("============================\n");

    // Initialize a custom global planner configuration
    let config = PlanningConfig {
        strategy: PlanningStrategy::CacheFirst,
        measure_performance: true,
        serialized_db_path: Some("./fft_plans.json".to_string()),
        max_cached_plans: 256,
        max_plan_age: Duration::from_secs(7200), // 2 hours
        ..Default::default()
    };

    // Initialize the global planner with this configuration
    let _ = init_global_planner(config);

    // Plan ahead of time for common sizes
    println!("Planning ahead for common FFT sizes...");
    let common_sizes = [128, 256, 512, 1024, 2048];
    plan_ahead_of_time(&common_sizes, Some("./fft_plans.json")).unwrap();

    // Create test data
    println!("Creating test arrays...");
    let n = 512;
    let mut test_array = Array2::zeros((n, n));

    // Set a single impulse
    test_array[[n / 4, n / 4]] = Complex64::new(1.0, 0.0);

    // First, use the standard FFT function
    println!("Performing standard FFT (will create plans on demand)...");
    let start = Instant::now();
    let result1 = fft2(&test_array, None, None, None).unwrap();
    let std_time = start.elapsed();
    println!("Standard FFT completed in {:?}", std_time);

    // Now use the custom planner
    println!("\nTrying different planning strategies:");

    // 1. Always create new plans
    let builder = PlanBuilder::new()
        .shape(&[n, n])
        .forward(true)
        .strategy(PlanningStrategy::AlwaysNew);

    let start = Instant::now();
    let plan = builder.build().unwrap();
    let plan_time = start.elapsed();

    println!("1. PlanningStrategy::AlwaysNew");
    println!("   Plan creation took: {:?}", plan_time);

    // Execute the plan
    let executor = FftPlanExecutor::new(plan);
    let mut result2_data = vec![Complex64::default(); n * n];

    // Flatten the array for the executor
    let input_flat: Vec<Complex64> = test_array.iter().cloned().collect();

    let start = Instant::now();
    executor.execute(&input_flat, &mut result2_data).unwrap();
    let exec_time = start.elapsed();
    println!("   Plan execution took: {:?}", exec_time);
    println!("   Total time: {:?}", plan_time + exec_time);

    // 2. Use cached plans
    let builder = PlanBuilder::new()
        .shape(&[n, n])
        .forward(true)
        .strategy(PlanningStrategy::CacheFirst);

    let start = Instant::now();
    let plan = builder.build().unwrap();
    let plan_time = start.elapsed();

    println!("\n2. PlanningStrategy::CacheFirst");
    println!("   Plan lookup/creation took: {:?}", plan_time);

    // Execute the plan
    let executor = FftPlanExecutor::new(plan);
    let mut result3_data = vec![Complex64::default(); n * n];

    let start = Instant::now();
    executor.execute(&input_flat, &mut result3_data).unwrap();
    let exec_time = start.elapsed();
    println!("   Plan execution took: {:?}", exec_time);
    println!("   Total time: {:?}", plan_time + exec_time);

    // 3. Use the global planner for a second run (should be cached now)
    println!("\n3. Using the global planner (second run, should be cached)");
    let planner = get_global_planner();

    let start = Instant::now();
    let plan = {
        let mut planner_guard = planner.lock().unwrap();
        planner_guard
            .plan_fft(&[n, n], true, Default::default())
            .unwrap()
    };
    let plan_time = start.elapsed();
    println!("   Plan lookup took: {:?}", plan_time);

    // Execute the plan
    let executor = FftPlanExecutor::new(plan);
    let mut result4_data = vec![Complex64::default(); n * n];

    let start = Instant::now();
    executor.execute(&input_flat, &mut result4_data).unwrap();
    let exec_time = start.elapsed();
    println!("   Plan execution took: {:?}", exec_time);
    println!("   Total time: {:?}", plan_time + exec_time);

    // Verify results are the same
    println!("\nVerifying results...");
    let result2_flat = &result2_data[0..10]; // Just check the first few elements
    let result3_flat = &result3_data[0..10];
    let result4_flat = &result4_data[0..10];

    let result1_flat: Vec<Complex64> = result1.iter().take(10).cloned().collect();

    println!("First 10 elements from standard FFT:    {:?}", result1_flat);
    println!("First 10 elements from AlwaysNew:       {:?}", result2_flat);
    println!("First 10 elements from CacheFirst:      {:?}", result3_flat);
    println!("First 10 elements from global planner:  {:?}", result4_flat);

    // Compare performance
    println!("\nPerformance comparison:");
    println!("Standard FFT:           {:?}", std_time);
    println!("AlwaysNew + execution:  {:?}", plan_time + exec_time);
    println!("Cached + execution:     {:?}", plan_time + exec_time);

    println!("\nDone!");
}
