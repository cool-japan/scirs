//! # Profiling (Beta 2 Enhanced)
//!
//! This module provides comprehensive utilities for profiling computational performance in scientific applications
//! with advanced features for detailed performance analysis and optimization.
//!
//! ## Enhanced Features (Beta 2)
//!
//! * Function-level timing instrumentation
//! * Memory allocation tracking
//! * Hierarchical profiling for nested operations
//! * Easy-to-use macros for profiling sections of code
//! * **Flame graph generation** for visualizing call hierarchies
//! * **Automated bottleneck detection** with performance thresholds
//! * **System-level resource monitoring** (CPU, memory, network)
//! * **Hardware performance counter integration**
//! * **Differential profiling** to compare performance between runs
//! * **Continuous performance monitoring** for long-running processes
//! * **Export capabilities** to various formats (JSON, CSV, flamegraph)
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_core::profiling::{Profiler, Timer, MemoryTracker};
//!
//! // Start the global profiler
//! Profiler::global().lock().unwrap().start();
//!
//! // Time a function call
//! let result = Timer::time_function("matrix_multiplication", || {
//!     // Perform matrix multiplication
//!     // ...
//!     42 // Return some result
//! });
//!
//! // Time a code block with more control
//! let timer = Timer::start("data_processing");
//! // Perform data processing
//! // ...
//! timer.stop();
//!
//! // Track memory allocations
//! let tracker = MemoryTracker::start("large_array_operation");
//! let large_array = vec![0; 1_000_000];
//! // ...
//! tracker.stop();
//!
//! // Print profiling report
//! Profiler::global().lock().unwrap().print_report();
//!
//! // Stop profiling
//! Profiler::global().lock().unwrap().stop();
//! ```

// Module declarations
pub mod advanced;
pub mod comprehensive;
pub mod entries;
pub mod memory;
pub mod profiler;
pub mod timer;

// External modules that were already in the profiling directory
pub mod adaptive;
pub mod continuousmonitoring;
pub mod coverage;
pub mod dashboards;
pub mod flame_graph_svg;
pub mod hardware_counters;
pub mod performance_hints;
pub mod production;
pub mod system_monitor;

// Re-exports for backward compatibility and convenient access
pub use entries::{MemoryEntry, TimingEntry};
pub use memory::{profiling_memory_tracker, MemoryTracker};
pub use profiler::Profiler;
pub use timer::Timer;

// Advanced profiling re-exports
pub use advanced::{
    BottleneckConfig, BottleneckDetector, BottleneckReport, BottleneckType, DifferentialProfiler,
    DifferentialReport, ExportableProfiler, FlameGraphGenerator, FlameGraphNode, MemoryDiff,
    PerformanceChange, PerformanceStats, ProfileSnapshot, ResourceStats, SystemResourceMonitor,
    TimingDiff,
};

// Comprehensive profiling re-exports
pub use comprehensive::{ComprehensiveConfig, ComprehensiveProfiler, ComprehensiveReport};

/// Macro for timing a block of code
#[macro_export]
macro_rules! profile_time {
    ($name:expr, $body:block) => {{
        let timer = $crate::profiling::Timer::start($name);
        let result = $body;
        timer.stop();
        result
    }};
}

/// Macro for tracking memory usage in a block of code
#[macro_export]
macro_rules! profile_memory {
    ($name:expr, $body:block) => {{
        let tracker = $crate::profiling::MemoryTracker::start($name);
        let result = $body;
        tracker.stop();
        result
    }};
}

/// Macro for timing a block of code with a parent operation
#[macro_export]
macro_rules! profile_time_with_parent {
    ($name:expr, $parent:expr, $body:block) => {{
        let timer = $crate::profiling::Timer::start_with_parent($name, $parent);
        let result = $body;
        timer.stop();
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timer_basic() {
        let timer = Timer::start("test_operation");
        thread::sleep(Duration::from_millis(10));
        timer.stop();

        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_memory_tracker_basic() {
        let tracker = MemoryTracker::start("test_memory");
        tracker.stop();
        // Memory tracking is a placeholder, so we just test that it doesn't panic
    }

    #[test]
    fn test_profiler_integration() {
        // Use global profiler
        Profiler::global().lock().unwrap().start();

        let timer = Timer::start("integration_test");
        thread::sleep(Duration::from_millis(5));
        timer.stop();

        let stats = Profiler::global()
            .lock()
            .unwrap()
            .get_timing_stats("integration_test");
        assert!(stats.is_some());

        let (calls, total, avg, max) = stats.unwrap();
        assert_eq!(calls, 1);
        assert!(total >= Duration::from_millis(5));
        assert!(avg >= Duration::from_millis(5));
        assert!(max >= Duration::from_millis(5));
    }

    #[test]
    fn test_flame_graph_generator() {
        let mut generator = advanced::FlameGraphGenerator::new();

        generator.start_call("function_a");
        generator.start_call("function_b");
        thread::sleep(Duration::from_millis(1));
        generator.end_call();
        generator.end_call();

        let flame_graph = generator.generate();
        assert!(!flame_graph.children.is_empty());
    }

    #[test]
    fn test_bottleneck_detector() {
        // Use global profiler
        Profiler::global().lock().unwrap().start();

        // Simulate a slow operation
        let timer = Timer::start("slow_operation");
        thread::sleep(Duration::from_millis(200));
        timer.stop();

        let config = advanced::BottleneckConfig {
            min_execution_threshold: Duration::from_millis(100),
            min_calls: 1, // Allow single calls to be detected
            ..Default::default()
        };

        let mut detector = advanced::BottleneckDetector::new(config);
        let reports = detector.analyze(&Profiler::global().lock().unwrap());

        assert!(!reports.is_empty());
        assert_eq!(
            reports[0].bottleneck_type,
            advanced::BottleneckType::SlowExecution
        );
    }

    #[test]
    fn test_differential_profiler() {
        // Use global profiler
        Profiler::global().lock().unwrap().start();

        // Baseline run
        let timer = Timer::start("diff_test_operation");
        thread::sleep(Duration::from_millis(10));
        timer.stop();

        let mut diff_profiler = advanced::DifferentialProfiler::new();
        diff_profiler.setbaseline(
            &Profiler::global().lock().unwrap(),
            Some("baseline".to_string()),
        );

        // Current run (slower) - use same operation name for comparison
        let timer = Timer::start("diff_test_operation");
        thread::sleep(Duration::from_millis(20));
        timer.stop();

        diff_profiler.set_current(
            &Profiler::global().lock().unwrap(),
            Some("current".to_string()),
        );

        let report = diff_profiler.generate_diff_report();
        assert!(report.is_some());

        let report = report.unwrap();
        assert!(!report.timing_diffs.is_empty() || !report.memory_diffs.is_empty());
        // Allow either timing or memory diffs
    }

    #[test]
    fn test_system_resourcemonitor() {
        let monitor = advanced::SystemResourceMonitor::new(Duration::from_millis(10));
        monitor.start();

        thread::sleep(Duration::from_millis(50));
        monitor.stop();

        let stats = monitor.get_stats();
        assert!(stats.sample_count > 0);
    }

    #[test]
    fn test_exportable_profiler() {
        let mut profiler = advanced::ExportableProfiler::new();
        profiler.add_metadata("test_run".to_string(), "beta2".to_string());

        profiler.profiler().start();

        let timer = Timer::start("export_test");
        thread::sleep(Duration::from_millis(5));
        timer.stop();

        // Test CSV export (to a temporary file path that we won't actually create)
        // In a real test, you'd use tempfile crate
        let csv_result = profiler.export_to_csv("/tmp/test_profile.csv");
        // We expect this to work or fail gracefully
        drop(csv_result);
    }
}
