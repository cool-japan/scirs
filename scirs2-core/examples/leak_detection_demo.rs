//! # Memory Leak Detection System Demo
//!
//! This example demonstrates the comprehensive memory leak detection system
//! implemented in SciRS2 Core.

#[cfg(not(feature = "memory_management"))]
#[allow(dead_code)]
fn main() {
    println!("This example requires the 'memory_management' feature to be enabled.");
    println!("Run with: cargo run --example leak_detection_demo --features memory_management");
}

#[cfg(feature = "memory_management")]
use scirs2_core::memory::leak_detection::{LeakCheckGuard, LeakDetectionConfig, LeakDetector};
#[cfg(feature = "memory_management")]
use scirs2_core::CoreResult;

#[cfg(feature = "memory_management")]
#[allow(dead_code)]
fn main() -> CoreResult<()> {
    println!("🔍 SciRS2 Core Memory Leak Detection Demo");
    println!("==========================================\n");

    // Configure leak detection for development
    let config = LeakDetectionConfig::default()
        .development_mode()  // Enable detailed tracking
        .with_threshold_mb(10)  // Low threshold for demo
        .with_samplingrate(1.0); // Track all allocations

    println!("📋 Configuration:");
    println!(
        "  - Threshold: {} MB",
        config.growth_threshold_bytes / (1024 * 1024)
    );
    println!("  - Sampling rate: {:.1}%", config.samplingrate * 100.0);
    println!("  - Call stacks: {}", config.collect_call_stacks);
    println!("  - Production mode: {}\n", config.production_mode);

    // Create leak detector
    let detector = LeakDetector::new(config)?;
    println!("✅ Leak detector initialized");

    // Demonstrate checkpoint-based leak detection
    println!("\n🎯 Creating memory checkpoint...");
    let checkpoint = detector.create_checkpoint("demo_operation")?;
    println!(
        "  ✅ Checkpoint '{}' created at {}",
        checkpoint.name,
        checkpoint.timestamp.format("%H:%M:%S")
    );
    println!(
        "  📊 Initial memory usage: {} KB",
        checkpoint.memory_usage.rss_bytes / 1024
    );

    // Simulate some allocations
    println!("\n💾 Simulating memory allocations...");
    for i in 0..5 {
        let size = 1024 * (i + 1); // Increasing allocation sizes
        detector.track_allocation(size, 0x1000 + i)?;
        println!("  📦 Allocated {} bytes", size);
    }

    // Check for leaks
    println!("\n🔍 Checking for memory leaks...");
    let report = detector.check_leaks(&checkpoint)?;

    println!("📋 Leak Detection Report:");
    println!("  - Leaks detected: {}", report.has_leaks());
    println!("  - Memory growth: {} bytes", report.memory_growth);
    println!("  - Total leaks: {}", report.leaks.len());

    if report.has_leaks() {
        println!("  - Total leaked: {} bytes", report.total_leaked_bytes());
        println!("  - Max severity: {}", report.summary.max_severity);
        println!(
            "  - Avg confidence: {:.2}",
            report.summary.average_confidence
        );

        println!("\n💡 Recommendations:");
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
    } else {
        println!("  ✅ No leaks detected");
    }

    // Demonstrate RAII leak checking with guard
    println!("\n🛡️  Testing RAII leak guard...");
    {
        let guard = LeakCheckGuard::new(&detector, "raii_test")?;
        println!("  📦 Guard created - will check on drop");

        // Simulate more allocations within guard scope
        detector.track_allocation(2048, 0x2000)?;
        println!("  📦 Allocated 2048 bytes in guard scope");
    } // Guard drops here and automatically checks for leaks
    println!("  ✅ Guard scope ended");

    // Show leak types and their severity
    println!("\n📊 Leak Type Severity Levels:");
    use scirs2_core::memory::leak_detection::LeakType;
    for leak_type in [
        LeakType::Definite,
        LeakType::Indirect,
        LeakType::GrowthPattern,
        LeakType::Possible,
        LeakType::Reachable,
    ] {
        println!("  - {:?}: severity {}/10", leak_type, leak_type.severity());
    }

    // Demonstrate profiler tool detection
    println!("\n🔧 Available Profiler Tools:");
    use scirs2_core::memory::leak_detection::ProfilerTool;
    for tool in [
        ProfilerTool::Valgrind,
        ProfilerTool::AddressSanitizer,
        ProfilerTool::Heaptrack,
        ProfilerTool::Massif,
        ProfilerTool::Jemalloc,
    ] {
        println!("  - {}", tool.name());
    }

    // Show configuration options
    println!("\n⚙️  Configuration Examples:");

    let production_config = LeakDetectionConfig::default();
    println!("  📦 Production config:");
    println!(
        "    - Sampling: {:.1}%",
        production_config.samplingrate * 100.0
    );
    println!(
        "    - Call stacks: {}",
        production_config.collect_call_stacks
    );

    let development_config = LeakDetectionConfig::default().development_mode();
    println!("  🔧 Development config:");
    println!(
        "    - Sampling: {:.1}%",
        development_config.samplingrate * 100.0
    );
    println!(
        "    - Call stacks: {}",
        development_config.collect_call_stacks
    );

    println!("\n✨ Demo completed successfully!");
    println!("\nThe leak detection system provides:");
    println!("  🔹 Real-time allocation/deallocation tracking");
    println!("  🔹 Configurable thresholds and sampling rates");
    println!("  🔹 Multiple leak pattern detection algorithms");
    println!("  🔹 Integration with external profiler tools");
    println!("  🔹 RAII-based automatic leak checking");
    println!("  🔹 Production-safe monitoring with minimal overhead");
    println!("  🔹 Comprehensive reporting and recommendations");

    Ok(())
}
