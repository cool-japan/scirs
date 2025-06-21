//! # Simple Adaptive Optimization Test
//!
//! This example tests the basic adaptive optimization functionality
//! without requiring complex dependencies.

#[cfg(feature = "profiling")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_core::profiling::adaptive::{
        AdaptiveOptimizer, OptimizationConfig, OptimizationGoal, Priority, WorkloadProfile,
        WorkloadType,
    };

    println!("🧠 Simple Adaptive Optimization Test");
    println!("====================================\n");

    // Create basic optimizer
    let config = OptimizationConfig::default()
        .with_goal(OptimizationGoal::Balanced)
        .with_learning_rate(0.01);

    let mut optimizer = AdaptiveOptimizer::new(config)?;
    println!("✅ Created adaptive optimizer");

    // Register a simple workload
    let workload = WorkloadProfile::builder()
        .with_name("test_workload")
        .with_workload_type(WorkloadType::Balanced)
        .with_priority(Priority::Medium)
        .build();

    optimizer.register_workload(workload)?;
    println!("📝 Registered workload");

    // Start optimization
    optimizer.start_optimization()?;
    println!("🚀 Started optimization");

    // Record some test metrics
    optimizer.record_metric("test_workload", "execution_time", 100.0)?;
    optimizer.record_metric("test_workload", "memory_usage", 1024.0)?;
    println!("📊 Recorded test metrics");

    // Get statistics
    let stats = optimizer.get_statistics();
    println!("📈 Optimizer Statistics:");
    println!("  - State: {:?}", stats.state);
    println!("  - Registered workloads: {}", stats.registered_workloads);

    // Stop optimization
    optimizer.stop_optimization()?;
    println!("🛑 Stopped optimization");

    println!("\n✨ Test completed successfully!");

    Ok(())
}

#[cfg(not(feature = "profiling"))]
fn main() {
    println!("⚠️  This example requires the 'profiling' feature to be enabled.");
    println!("Run with: cargo run --example adaptive_optimization_test --features profiling");
}
