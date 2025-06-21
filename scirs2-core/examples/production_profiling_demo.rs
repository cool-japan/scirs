//! # Production Profiling System Demo
//!
//! This example demonstrates the comprehensive production profiling system
//! for real-workload analysis and bottleneck identification in enterprise environments.

use scirs2_core::profiling::production::{ProductionProfiler, ProfileConfig, WorkloadType};
use scirs2_core::CoreResult;
use std::thread;
use std::time::Duration;

fn main() -> CoreResult<()> {
    println!("🚀 SciRS2 Core Production Profiling System Demo");
    println!("================================================\n");

    // Configuration for different environments
    demo_configuration_examples()?;
    println!();

    // Basic workload analysis
    demo_basic_workload_analysis()?;
    println!();

    // Advanced enterprise features
    demo_enterprise_features()?;
    println!();

    // Bottleneck identification
    demo_bottleneck_identification()?;
    println!();

    // Performance regression detection
    demo_regression_detection()?;
    println!();

    // Resource utilization monitoring
    demo_resource_monitoring()?;

    println!("\n✨ Production profiling demo completed successfully!");
    println!("\nThe production profiling system provides:");
    println!("  🔹 Real-workload analysis with minimal overhead");
    println!("  🔹 Automatic bottleneck identification with ML-powered insights");
    println!("  🔹 Performance regression detection against historical baselines");
    println!("  🔹 Comprehensive resource utilization tracking");
    println!("  🔹 Statistical analysis with confidence intervals");
    println!("  🔹 Enterprise-grade reporting and analytics");
    println!("  🔹 Integration with existing profiling infrastructure");
    println!("  🔹 Production-safe monitoring with configurable sampling");

    Ok(())
}

fn demo_configuration_examples() -> CoreResult<()> {
    println!("📋 Configuration Examples for Different Environments");
    println!("---------------------------------------------------");

    // Production environment - minimal overhead
    let production_config = ProfileConfig::production()
        .with_sampling_rate(0.001) // 0.1% sampling for minimal overhead
        .with_bottleneck_detection(true)
        .with_regression_detection(true);

    println!("🏭 Production Environment:");
    println!(
        "  - Sampling Rate: {:.3}%",
        production_config.sampling_rate * 100.0
    );
    println!(
        "  - Memory Limit: {} MB",
        production_config.max_memory_usage / (1024 * 1024)
    );
    println!(
        "  - Detailed Call Stacks: {}",
        production_config.detailed_call_stacks
    );
    println!(
        "  - Confidence Level: {:.1}%",
        production_config.confidence_level * 100.0
    );

    // Development environment - detailed tracking
    let dev_config = ProfileConfig::development()
        .with_sampling_rate(0.1) // 10% sampling for development
        .with_bottleneck_detection(true)
        .with_regression_detection(true);

    println!("\n🔧 Development Environment:");
    println!(
        "  - Sampling Rate: {:.1}%",
        dev_config.sampling_rate * 100.0
    );
    println!(
        "  - Memory Limit: {} MB",
        dev_config.max_memory_usage / (1024 * 1024)
    );
    println!(
        "  - Detailed Call Stacks: {}",
        dev_config.detailed_call_stacks
    );
    println!(
        "  - Bottleneck Threshold: {:.1}ms",
        dev_config.bottleneck_threshold_ms
    );

    // Staging environment - balanced approach
    let staging_config = ProfileConfig::default()
        .with_sampling_rate(0.05) // 5% sampling
        .with_bottleneck_detection(true)
        .with_regression_detection(true);

    println!("\n🎭 Staging Environment:");
    println!(
        "  - Sampling Rate: {:.1}%",
        staging_config.sampling_rate * 100.0
    );
    println!(
        "  - Regression Threshold: {:.1}%",
        staging_config.regression_threshold_percent
    );
    println!("  - Min Sample Size: {}", staging_config.min_sample_size);

    Ok(())
}

fn demo_basic_workload_analysis() -> CoreResult<()> {
    println!("⚡ Basic Workload Analysis");
    println!("-------------------------");

    let config = ProfileConfig::development(); // Use development config for demo
    let mut profiler = ProductionProfiler::new(config)?;

    // Analyze a compute-intensive workload
    println!("🔍 Starting workload analysis for 'matrix_operations'...");
    profiler.start_workload_analysis("matrix_operations", WorkloadType::ComputeIntensive)?;

    // Simulate matrix operations
    simulate_matrix_operations();

    let report = profiler
        .finish_workload_analysis_by_id("matrix_operations", WorkloadType::ComputeIntensive)?;

    println!("📊 Analysis Results:");
    println!("  - Workload ID: {}", report.workload_id);
    println!("  - Workload Type: {}", report.workload_type);
    println!("  - Duration: {:.2}s", report.duration.as_secs_f64());
    println!("  - Total Samples: {}", report.total_samples);
    println!("  - Analysis Quality: {}/100", report.analysis_quality);

    println!("\n📈 Performance Statistics:");
    println!(
        "  - Mean Time: {:.2}ms",
        report.statistics.mean_time.as_millis()
    );
    println!(
        "  - Median Time: {:.2}ms",
        report.statistics.median_time.as_millis()
    );
    println!(
        "  - 95th Percentile: {:.2}ms",
        report.statistics.p95_time.as_millis()
    );
    println!(
        "  - 99th Percentile: {:.2}ms",
        report.statistics.p99_time.as_millis()
    );
    println!(
        "  - Coefficient of Variation: {:.3}",
        report.statistics.coefficient_of_variation
    );

    println!("\n🔋 Resource Utilization:");
    println!(
        "  - CPU Usage: {:.1}%",
        report.resource_utilization.cpu_percent
    );
    println!(
        "  - Memory Usage: {:.1} MB",
        report.resource_utilization.memory_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  - Thread Count: {}",
        report.resource_utilization.thread_count
    );

    if report.has_bottlenecks() {
        println!("\n🔍 Identified Bottlenecks: {}", report.bottlenecks.len());
        for (i, bottleneck) in report.bottlenecks().iter().take(3).enumerate() {
            println!(
                "  {}. {} - {:.1}% impact ({:.2}ms avg)",
                i + 1,
                bottleneck.function,
                bottleneck.impact_percentage,
                bottleneck.average_time.as_millis()
            );
        }
    }

    Ok(())
}

fn demo_enterprise_features() -> CoreResult<()> {
    println!("🏢 Enterprise Features Demo");
    println!("---------------------------");

    let config = ProfileConfig::production()
        .with_sampling_rate(0.01) // 1% sampling for enterprise demo
        .with_bottleneck_detection(true)
        .with_regression_detection(true);

    let mut profiler = ProductionProfiler::new(config)?;

    // Demonstrate different workload types
    let workload_types = [
        ("data_processing", WorkloadType::MemoryIntensive),
        ("api_requests", WorkloadType::IOBound),
        ("network_sync", WorkloadType::NetworkBound),
        ("batch_computation", WorkloadType::ComputeIntensive),
        ("mixed_workload", WorkloadType::Mixed),
    ];

    for (workload_name, workload_type) in &workload_types {
        println!(
            "📊 Analyzing {} workload ({})...",
            workload_name, workload_type
        );

        profiler.start_workload_analysis(workload_name, workload_type.clone())?;

        // Simulate different types of work
        match workload_type {
            WorkloadType::ComputeIntensive => simulate_compute_work(),
            WorkloadType::MemoryIntensive => simulate_memory_work(),
            WorkloadType::IOBound => simulate_io_work(),
            WorkloadType::NetworkBound => simulate_network_work(),
            WorkloadType::Mixed => simulate_mixed_work(),
            WorkloadType::Custom(_) => simulate_compute_work(),
        }

        let report =
            profiler.finish_workload_analysis_by_id(workload_name, workload_type.clone())?;

        println!(
            "  ✅ Completed - Quality: {}/100, Samples: {}",
            report.analysis_quality, report.total_samples
        );

        if report.has_bottlenecks() {
            println!("    🔍 {} bottlenecks identified", report.bottlenecks.len());
        }

        if report.has_regressions() {
            println!("    ⚠️  {} regressions detected", report.regressions.len());
        }
    }

    println!("\n📄 Executive Summary Generation:");
    // Use the last report for executive summary demo
    profiler.start_workload_analysis("executive_demo", WorkloadType::Mixed)?;
    simulate_mixed_work();
    let final_report =
        profiler.finish_workload_analysis_by_id("executive_demo", WorkloadType::Mixed)?;

    println!("{}", final_report.executive_summary());

    Ok(())
}

fn demo_bottleneck_identification() -> CoreResult<()> {
    println!("🔍 Bottleneck Identification Demo");
    println!("---------------------------------");

    let config = ProfileConfig::development().with_bottleneck_detection(true);

    let mut profiler = ProductionProfiler::new(config)?;

    println!("🎯 Simulating workload with intentional bottlenecks...");
    profiler.start_workload_analysis("bottleneck_demo", WorkloadType::ComputeIntensive)?;

    // Simulate work with bottlenecks
    simulate_bottleneck_workload();

    let report = profiler
        .finish_workload_analysis_by_id("bottleneck_demo", WorkloadType::ComputeIntensive)?;

    if report.has_bottlenecks() {
        println!("\n🚨 Bottleneck Analysis Results:");
        println!("  - Total Bottlenecks Found: {}", report.bottlenecks.len());

        for (i, bottleneck) in report.bottlenecks().iter().enumerate() {
            println!("\n  🔍 Bottleneck #{}", i + 1);
            println!("    - Function: {}", bottleneck.function);
            println!(
                "    - Impact: {:.1}% of total execution time",
                bottleneck.impact_percentage
            );
            println!(
                "    - Average Time: {:.2}ms",
                bottleneck.average_time.as_millis()
            );
            println!("    - Sample Count: {}", bottleneck.sample_count);
            println!("    - Confidence: {:.1}%", bottleneck.confidence * 100.0);
            println!("    - Severity: {}/10", bottleneck.severity);

            if !bottleneck.optimizations.is_empty() {
                println!("    - Optimization Suggestions:");
                for suggestion in &bottleneck.optimizations {
                    println!("      • {}", suggestion);
                }
            }
        }
    } else {
        println!("✅ No significant bottlenecks detected in this workload.");
    }

    Ok(())
}

fn demo_regression_detection() -> CoreResult<()> {
    println!("📈 Performance Regression Detection Demo");
    println!("---------------------------------------");

    let config = ProfileConfig::development().with_regression_detection(true);

    let mut profiler = ProductionProfiler::new(config)?;

    // Record baseline performance
    println!("📊 Recording baseline performance...");
    profiler.record_performance("regression_test", Duration::from_millis(100))?;
    profiler.record_performance("regression_test", Duration::from_millis(95))?;
    profiler.record_performance("regression_test", Duration::from_millis(105))?;
    profiler.record_performance("regression_test", Duration::from_millis(98))?;
    profiler.record_performance("regression_test", Duration::from_millis(102))?;

    println!("⏱️  Baseline established: ~100ms average");

    // Simulate a performance regression
    println!("\n🔍 Analyzing current performance (simulating regression)...");
    profiler.start_workload_analysis("regression_test", WorkloadType::ComputeIntensive)?;

    // Simulate slower performance
    thread::sleep(Duration::from_millis(120)); // Simulate 20% performance regression

    let report = profiler
        .finish_workload_analysis_by_id("regression_test", WorkloadType::ComputeIntensive)?;

    if report.has_regressions() {
        println!("\n⚠️  Performance Regression Detected!");
        for regression in report.significant_regressions() {
            println!("  - Operation: {}", regression.operation);
            println!(
                "  - Baseline: {:.2}ms",
                regression.baseline_time.as_millis()
            );
            println!("  - Current: {:.2}ms", regression.current_time.as_millis());
            println!("  - Change: {:+.1}% slower", regression.change_percent);
            println!("  - Significance: {:.1}%", regression.significance * 100.0);
            println!("  - Detected At: {:?}", regression.detected_at);
        }
    } else {
        println!("✅ No performance regressions detected.");
    }

    Ok(())
}

fn demo_resource_monitoring() -> CoreResult<()> {
    println!("🖥️  Resource Utilization Monitoring Demo");
    println!("----------------------------------------");

    let config = ProfileConfig::development();
    let profiler = ProductionProfiler::new(config)?;

    println!("📊 Current Resource Utilization:");
    let resource_usage = profiler.get_resource_utilization()?;

    println!("  - CPU Usage: {:.1}%", resource_usage.cpu_percent);
    println!(
        "  - Memory Usage: {:.1} MB",
        resource_usage.memory_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("  - Active Threads: {}", resource_usage.thread_count);
    println!(
        "  - I/O Operations/sec: {:.1}",
        resource_usage.io_ops_per_sec
    );
    println!(
        "  - Network Throughput: {:.1} KB/s",
        resource_usage.network_bytes_per_sec / 1024.0
    );

    println!("\n📤 Data Export Capabilities:");
    let export_data = profiler.export_data("resource_demo")?;
    println!("  - Export Format: JSON");
    println!("  - Data Size: {} bytes", export_data.len());
    println!("  - Contains: Configuration, resource metrics, timestamps");

    Ok(())
}

// Simulation functions for different workload types

fn simulate_matrix_operations() {
    // Simulate CPU-intensive matrix operations
    for _ in 0..1000 {
        let _result: f64 = (0..100).map(|i| (i as f64).sin().cos()).sum();
    }
    thread::sleep(Duration::from_millis(10));
}

fn simulate_compute_work() {
    // Simulate compute-intensive work
    for _ in 0..500 {
        let _result: f64 = (0..50).map(|i| (i as f64).sqrt()).sum();
    }
    thread::sleep(Duration::from_millis(5));
}

fn simulate_memory_work() {
    // Simulate memory-intensive work
    let _large_vec: Vec<u64> = (0..10000).collect();
    thread::sleep(Duration::from_millis(8));
}

fn simulate_io_work() {
    // Simulate I/O-bound work
    thread::sleep(Duration::from_millis(15));
}

fn simulate_network_work() {
    // Simulate network-bound work
    thread::sleep(Duration::from_millis(12));
}

fn simulate_mixed_work() {
    // Simulate mixed workload
    simulate_compute_work();
    simulate_memory_work();
    thread::sleep(Duration::from_millis(3));
}

fn simulate_bottleneck_workload() {
    // Simulate a workload with clear bottlenecks

    // Fast operation
    for _ in 0..100 {
        let _result = 2 + 2;
    }

    // Bottleneck operation (simulated slow function)
    thread::sleep(Duration::from_millis(50));

    // Another fast operation
    for _ in 0..50 {
        let _result = 3 * 3;
    }

    // Medium bottleneck
    thread::sleep(Duration::from_millis(20));
}
