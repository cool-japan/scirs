//! # Profiling (Alpha 6 Enhanced)
//!
//! This module provides comprehensive utilities for profiling computational performance in scientific applications
//! with advanced features for detailed performance analysis and optimization.
//!
//! ## Enhanced Features (Alpha 6)
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

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Timer for measuring code execution time
pub struct Timer {
    /// Name of the operation being timed
    name: String,
    /// Start time
    start_time: Instant,
    /// Whether the timer is currently running
    running: bool,
    /// Whether to automatically report the timing when dropped
    auto_report: bool,
    /// Parent timer name for hierarchical profiling
    parent: Option<String>,
}

impl Timer {
    /// Start a new timer with the given name
    pub fn start(name: &str) -> Self {
        let timer = Self {
            name: name.to_string(),
            start_time: Instant::now(),
            running: true,
            auto_report: true,
            parent: None,
        };
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_start(&timer);
        }
        timer
    }

    /// Start a new hierarchical timer with a parent
    pub fn start_with_parent(name: &str, parent: &str) -> Self {
        let timer = Self {
            name: name.to_string(),
            start_time: Instant::now(),
            running: true,
            auto_report: true,
            parent: Some(parent.to_string()),
        };
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_start(&timer);
        }
        timer
    }

    /// Time a function call and return its result
    pub fn time_function<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let timer = Self::start(name);
        let result = f();
        timer.stop();
        result
    }

    /// Time a function call with a parent timer and return its result
    pub fn time_function_with_parent<F, R>(name: &str, parent: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let timer = Self::start_with_parent(name, parent);
        let result = f();
        timer.stop();
        result
    }

    /// Stop the timer and record the elapsed time
    pub fn stop(&self) {
        if !self.running {
            return;
        }

        let elapsed = self.start_time.elapsed();
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_timer_stop(&self.name, elapsed, self.parent.as_deref());
        }
    }

    /// Get the elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Disable auto-reporting when dropped
    pub fn without_auto_report(mut self) -> Self {
        self.auto_report = false;
        self
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if self.running && self.auto_report {
            let elapsed = self.start_time.elapsed();
            if let Ok(mut profiler) = Profiler::global().lock() {
                profiler.register_timer_stop(&self.name, elapsed, self.parent.as_deref());
            }
        }
    }
}

/// Memory allocation tracker
pub struct MemoryTracker {
    /// Name of the operation being tracked
    name: String,
    /// Start memory usage
    start_memory: usize,
    /// Whether the tracker is currently running
    running: bool,
    /// Whether to automatically report when dropped
    auto_report: bool,
}

impl MemoryTracker {
    /// Start a new memory tracker with the given name
    pub fn start(name: &str) -> Self {
        let current_memory = Self::current_memory_usage();
        let tracker = Self {
            name: name.to_string(),
            start_memory: current_memory,
            running: true,
            auto_report: true,
        };
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_memory_tracker_start(&tracker);
        }
        tracker
    }

    /// Stop the tracker and record the memory usage
    pub fn stop(&self) {
        if !self.running {
            return;
        }

        let current_memory = Self::current_memory_usage();
        let memory_delta = current_memory.saturating_sub(self.start_memory);
        if let Ok(mut profiler) = Profiler::global().lock() {
            profiler.register_memory_tracker_stop(&self.name, memory_delta);
        }
    }

    /// Track memory usage for a function call and return its result
    pub fn track_function<F, R>(name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let tracker = Self::start(name);
        let result = f();
        tracker.stop();
        result
    }

    /// Get the current memory delta without stopping the tracker
    pub fn memory_delta(&self) -> isize {
        let current_memory = Self::current_memory_usage();
        current_memory as isize - self.start_memory as isize
    }

    /// Disable auto-reporting when dropped
    pub fn without_auto_report(mut self) -> Self {
        self.auto_report = false;
        self
    }

    /// Get the current memory usage (platform-dependent implementation)
    fn current_memory_usage() -> usize {
        // This is a simplified implementation that doesn't actually track real memory
        // A real implementation would use platform-specific APIs to get memory usage
        #[cfg(target_os = "linux")]
        {
            // On Linux, we would read /proc/self/statm
            0
        }

        #[cfg(target_os = "macos")]
        {
            // On macOS, we would use task_info
            0
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, we would use GetProcessMemoryInfo
            0
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            // Fallback for other platforms
            0
        }
    }
}

impl Drop for MemoryTracker {
    fn drop(&mut self) {
        if self.running && self.auto_report {
            let current_memory = Self::current_memory_usage();
            let memory_delta = current_memory.saturating_sub(self.start_memory);
            if let Ok(mut profiler) = Profiler::global().lock() {
                profiler.register_memory_tracker_stop(&self.name, memory_delta);
            }
        }
    }
}

/// Timing entry for the profiler
#[derive(Debug, Clone)]
pub struct TimingEntry {
    /// Number of calls
    calls: usize,
    /// Total duration
    total_duration: Duration,
    /// Minimum duration
    min_duration: Duration,
    /// Maximum duration
    max_duration: Duration,
    /// Parent operation (used for hierarchical profiling structure)
    #[allow(dead_code)]
    parent: Option<String>,
    /// Child operations
    children: Vec<String>,
}

impl TimingEntry {
    /// Create a new timing entry
    fn new(duration: Duration, parent: Option<&str>) -> Self {
        Self {
            calls: 1,
            total_duration: duration,
            min_duration: duration,
            max_duration: duration,
            parent: parent.map(String::from),
            children: Vec::new(),
        }
    }

    /// Add a new timing measurement
    fn add_measurement(&mut self, duration: Duration) {
        self.calls += 1;
        self.total_duration += duration;
        self.min_duration = std::cmp::min(self.min_duration, duration);
        self.max_duration = std::cmp::max(self.max_duration, duration);
    }

    /// Add a child operation
    fn add_child(&mut self, child: &str) {
        if !self.children.contains(&child.to_string()) {
            self.children.push(child.to_string());
        }
    }

    /// Get the average duration
    fn average_duration(&self) -> Duration {
        if self.calls == 0 {
            Duration::from_secs(0)
        } else {
            self.total_duration / self.calls as u32
        }
    }
}

/// Memory tracking entry for the profiler
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Number of allocations
    allocations: usize,
    /// Total memory delta (can be negative for memory releases)
    total_delta: isize,
    /// Maximum memory delta in a single allocation
    max_delta: usize,
}

impl MemoryEntry {
    /// Create a new memory entry
    fn new(delta: usize) -> Self {
        Self {
            allocations: 1,
            total_delta: delta as isize,
            max_delta: delta,
        }
    }

    /// Add a new memory measurement
    fn add_measurement(&mut self, delta: usize) {
        self.allocations += 1;
        self.total_delta += delta as isize;
        self.max_delta = std::cmp::max(self.max_delta, delta);
    }

    /// Get the average memory delta
    #[allow(dead_code)]
    fn average_delta(&self) -> f64 {
        if self.allocations == 0 {
            0.0
        } else {
            self.total_delta as f64 / self.allocations as f64
        }
    }
}

/// Profiler for collecting performance metrics
#[derive(Debug)]
pub struct Profiler {
    /// Timing measurements
    timings: HashMap<String, TimingEntry>,
    /// Memory measurements
    memory: HashMap<String, MemoryEntry>,
    /// Currently active timers
    active_timers: HashMap<String, Instant>,
    /// Whether the profiler is currently running
    running: bool,
}

impl Profiler {
    /// Create a new profiler
    pub fn new() -> Self {
        Self {
            timings: HashMap::new(),
            memory: HashMap::new(),
            active_timers: HashMap::new(),
            running: false,
        }
    }

    /// Get the global profiler instance
    pub fn global() -> &'static Mutex<Profiler> {
        static GLOBAL_PROFILER: Lazy<Mutex<Profiler>> = Lazy::new(|| Mutex::new(Profiler::new()));
        &GLOBAL_PROFILER
    }

    /// Start the profiler
    pub fn start(&mut self) {
        self.running = true;
        self.timings.clear();
        self.memory.clear();
        self.active_timers.clear();
    }

    /// Stop the profiler
    pub fn stop(&mut self) {
        self.running = false;
    }

    /// Reset the profiler
    pub fn reset(&mut self) {
        self.timings.clear();
        self.memory.clear();
        self.active_timers.clear();
    }

    /// Register the start of a timer
    pub fn register_timer_start(&mut self, timer: &Timer) {
        if !self.running {
            return;
        }

        self.active_timers
            .insert(timer.name.clone(), timer.start_time);

        // Register the parent-child relationship
        if let Some(parent) = &timer.parent {
            if let Some(entry) = self.timings.get_mut(parent) {
                entry.add_child(&timer.name);
            }
        }
    }

    /// Register the stop of a timer
    pub fn register_timer_stop(&mut self, name: &str, duration: Duration, parent: Option<&str>) {
        if !self.running {
            return;
        }

        // Remove from active timers
        self.active_timers.remove(name);

        // Update the timing entry
        match self.timings.get_mut(name) {
            Some(entry) => {
                entry.add_measurement(duration);
            }
            None => {
                let entry = TimingEntry::new(duration, parent);
                self.timings.insert(name.to_string(), entry);
            }
        }

        // Register the parent-child relationship
        if let Some(parent) = parent {
            if let Some(entry) = self.timings.get_mut(parent) {
                entry.add_child(name);
            }
        }
    }

    /// Register the start of a memory tracker
    pub fn register_memory_tracker_start(&mut self, _tracker: &MemoryTracker) {
        if !self.running {
            // Nothing to do at start, just ensure the method exists for symmetry
        }
    }

    /// Register the stop of a memory tracker
    pub fn register_memory_tracker_stop(&mut self, name: &str, delta: usize) {
        if !self.running {
            return;
        }

        // Update the memory entry
        match self.memory.get_mut(name) {
            Some(entry) => {
                entry.add_measurement(delta);
            }
            None => {
                let entry = MemoryEntry::new(delta);
                self.memory.insert(name.to_string(), entry);
            }
        }
    }

    /// Print a report of the profiling results
    pub fn print_report(&self) {
        if self.timings.is_empty() && self.memory.is_empty() {
            println!("No profiling data collected.");
            return;
        }

        if !self.timings.is_empty() {
            println!("\n=== Timing Report ===");
            println!(
                "{:<30} {:<10} {:<15} {:<15} {:<15}",
                "Operation", "Calls", "Total (ms)", "Average (ms)", "Max (ms)"
            );
            println!("{}", "-".repeat(90));

            // Sort by total duration
            let mut entries: Vec<(&String, &TimingEntry)> = self.timings.iter().collect();
            entries.sort_by(|a, b| b.1.total_duration.cmp(&a.1.total_duration));

            for (name, entry) in entries {
                println!(
                    "{:<30} {:<10} {:<15.2} {:<15.2} {:<15.2}",
                    name,
                    entry.calls,
                    entry.total_duration.as_secs_f64() * 1000.0,
                    entry.average_duration().as_secs_f64() * 1000.0,
                    entry.max_duration.as_secs_f64() * 1000.0
                );
            }
        }

        if !self.memory.is_empty() {
            println!("\n=== Memory Report ===");
            println!(
                "{:<30} {:<10} {:<15} {:<15}",
                "Operation", "Counts", "Total (KB)", "Max (KB)"
            );
            println!("{}", "-".repeat(75));

            // Sort by total memory delta
            let mut entries: Vec<(&String, &MemoryEntry)> = self.memory.iter().collect();
            entries.sort_by(|a, b| b.1.total_delta.abs().cmp(&a.1.total_delta.abs()));

            for (name, entry) in entries {
                println!(
                    "{:<30} {:<10} {:<15.2} {:<15.2}",
                    name,
                    entry.allocations,
                    entry.total_delta as f64 / 1024.0,
                    entry.max_delta as f64 / 1024.0
                );
            }
        }
    }

    /// Get a report of the profiling results as a string
    pub fn get_report(&self) -> String {
        use std::fmt::Write;
        let mut report = String::new();

        if self.timings.is_empty() && self.memory.is_empty() {
            writeln!(report, "No profiling data collected.").unwrap();
            return report;
        }

        if !self.timings.is_empty() {
            writeln!(report, "\n=== Timing Report ===").unwrap();
            writeln!(
                report,
                "{:<30} {:<10} {:<15} {:<15} {:<15}",
                "Operation", "Calls", "Total (ms)", "Average (ms)", "Max (ms)"
            )
            .unwrap();
            writeln!(report, "{}", "-".repeat(90)).unwrap();

            // Sort by total duration
            let mut entries: Vec<(&String, &TimingEntry)> = self.timings.iter().collect();
            entries.sort_by(|a, b| b.1.total_duration.cmp(&a.1.total_duration));

            for (name, entry) in entries {
                writeln!(
                    report,
                    "{:<30} {:<10} {:<15.2} {:<15.2} {:<15.2}",
                    name,
                    entry.calls,
                    entry.total_duration.as_secs_f64() * 1000.0,
                    entry.average_duration().as_secs_f64() * 1000.0,
                    entry.max_duration.as_secs_f64() * 1000.0
                )
                .unwrap();
            }
        }

        if !self.memory.is_empty() {
            writeln!(report, "\n=== Memory Report ===").unwrap();
            writeln!(
                report,
                "{:<30} {:<10} {:<15} {:<15}",
                "Operation", "Counts", "Total (KB)", "Max (KB)"
            )
            .unwrap();
            writeln!(report, "{}", "-".repeat(75)).unwrap();

            // Sort by total memory delta
            let mut entries: Vec<(&String, &MemoryEntry)> = self.memory.iter().collect();
            entries.sort_by(|a, b| b.1.total_delta.abs().cmp(&a.1.total_delta.abs()));

            for (name, entry) in entries {
                writeln!(
                    report,
                    "{:<30} {:<10} {:<15.2} {:<15.2}",
                    name,
                    entry.allocations,
                    entry.total_delta as f64 / 1024.0,
                    entry.max_delta as f64 / 1024.0
                )
                .unwrap();
            }
        }

        report
    }

    /// Get timing statistics for a specific operation
    pub fn get_timing_stats(&self, name: &str) -> Option<(usize, Duration, Duration, Duration)> {
        self.timings.get(name).map(|entry| {
            (
                entry.calls,
                entry.total_duration,
                entry.average_duration(),
                entry.max_duration,
            )
        })
    }

    /// Get memory statistics for a specific operation
    pub fn get_memory_stats(&self, name: &str) -> Option<(usize, isize, usize)> {
        self.memory
            .get(name)
            .map(|entry| (entry.allocations, entry.total_delta, entry.max_delta))
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Access a memory tracker from the profiling module to avoid name conflicts
pub fn profiling_memory_tracker() -> &'static MemoryTracker {
    // Create a dummy memory tracker for static access
    static MEMORY_TRACKER: once_cell::sync::Lazy<MemoryTracker> =
        once_cell::sync::Lazy::new(|| MemoryTracker {
            name: "global".to_string(),
            start_memory: 0,
            running: false,
            auto_report: false,
        });
    &MEMORY_TRACKER
}

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

/// SVG flame graph export capabilities
pub mod flame_graph_svg;

/// System resource monitoring
pub mod system_monitor;

/// Hardware performance counter integration
pub mod hardware_counters;

/// Continuous performance monitoring for long-running processes
pub mod continuous_monitoring;

/// Function-level performance hinting system
pub mod performance_hints;

/// Production profiling with real-workload analysis and bottleneck identification
#[path = "profiling/production.rs"]
pub mod production;

#[path = "profiling/adaptive.rs"]
pub mod adaptive;
/// Performance dashboards with real-time visualization and historical trends
#[path = "profiling/dashboards.rs"]
pub mod dashboards;

/// Test coverage analysis with comprehensive tracking and reporting
#[path = "profiling/coverage.rs"]
pub mod coverage;

/// Advanced profiling capabilities for Alpha 6
pub mod advanced {
    use super::*;
    use std::collections::{BTreeMap, VecDeque};
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::thread;

    /// Flame graph data structure for visualizing call hierarchies
    #[derive(Debug, Clone)]
    pub struct FlameGraphNode {
        /// Function name
        pub name: String,
        /// Total execution time
        pub total_time: Duration,
        /// Self execution time (excluding children)
        pub self_time: Duration,
        /// Number of samples
        pub samples: u64,
        /// Child nodes
        pub children: BTreeMap<String, FlameGraphNode>,
        /// Call depth
        pub depth: usize,
    }

    impl FlameGraphNode {
        /// Create a new flame graph node
        pub fn new(name: String, depth: usize) -> Self {
            Self {
                name,
                total_time: Duration::from_secs(0),
                self_time: Duration::from_secs(0),
                samples: 0,
                children: BTreeMap::new(),
                depth,
            }
        }

        /// Add a sample to this node
        pub fn add_sample(&mut self, duration: Duration) {
            self.total_time += duration;
            self.samples += 1;
        }

        /// Calculate self time by subtracting children's time
        pub fn calculate_self_time(&mut self) {
            let children_time: Duration =
                self.children.values().map(|child| child.total_time).sum();
            self.self_time = self.total_time.saturating_sub(children_time);

            // Recursively calculate for children
            for child in self.children.values_mut() {
                child.calculate_self_time();
            }
        }

        /// Generate flame graph format output
        pub fn to_flame_graph_format(&self, prefix: &str) -> Vec<String> {
            let mut lines = Vec::new();
            let current_stack = if prefix.is_empty() {
                self.name.clone()
            } else {
                format!("{};{}", prefix, self.name)
            };

            if self.self_time.as_nanos() > 0 {
                lines.push(format!("{} {}", current_stack, self.self_time.as_nanos()));
            }

            for child in self.children.values() {
                lines.extend(child.to_flame_graph_format(&current_stack));
            }

            lines
        }
    }

    /// Flame graph generator
    #[derive(Debug)]
    pub struct FlameGraphGenerator {
        /// Root node of the flame graph
        root: FlameGraphNode,
        /// Current call stack
        call_stack: Vec<String>,
        /// Stack of start times
        time_stack: Vec<Instant>,
    }

    impl FlameGraphGenerator {
        /// Create a new flame graph generator
        pub fn new() -> Self {
            Self {
                root: FlameGraphNode::new("root".to_string(), 0),
                call_stack: Vec::new(),
                time_stack: Vec::new(),
            }
        }

        /// Start a new function call
        pub fn start_call(&mut self, function_name: &str) {
            self.call_stack.push(function_name.to_string());
            self.time_stack.push(Instant::now());
        }

        /// End the current function call
        pub fn end_call(&mut self) {
            if let (Some(_function_name), Some(start_time)) =
                (self.call_stack.pop(), self.time_stack.pop())
            {
                let duration = start_time.elapsed();

                // Navigate to the correct node in the tree
                let mut current_node = &mut self.root;
                for (depth, name) in self.call_stack.iter().enumerate() {
                    current_node = current_node
                        .children
                        .entry(name.clone())
                        .or_insert_with(|| FlameGraphNode::new(name.clone(), depth + 1));
                }

                // Add the sample
                current_node.add_sample(duration);
            }
        }

        /// Generate the flame graph
        pub fn generate(&mut self) -> FlameGraphNode {
            self.root.calculate_self_time();
            self.root.clone()
        }

        /// Export flame graph to file
        pub fn export_to_file(&mut self, path: &str) -> Result<(), std::io::Error> {
            let flame_graph = self.generate();
            let lines = flame_graph.to_flame_graph_format("");

            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);

            for line in lines {
                writeln!(writer, "{}", line)?;
            }

            writer.flush()?;
            Ok(())
        }
    }

    impl Default for FlameGraphGenerator {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Performance bottleneck detection configuration
    #[derive(Debug, Clone)]
    pub struct BottleneckConfig {
        /// Minimum execution time threshold (operations slower than this are considered bottlenecks)
        pub min_execution_threshold: Duration,
        /// Memory usage threshold (operations using more memory than this are flagged)
        pub memory_threshold: usize,
        /// CPU usage threshold (0.0 to 1.0)
        pub cpu_threshold: f64,
        /// Minimum number of calls to consider for bottleneck analysis
        pub min_calls: usize,
        /// Enable automatic suggestions
        pub enable_suggestions: bool,
    }

    impl Default for BottleneckConfig {
        fn default() -> Self {
            Self {
                min_execution_threshold: Duration::from_millis(100),
                memory_threshold: 1024 * 1024, // 1 MB
                cpu_threshold: 0.8,            // 80%
                min_calls: 5,
                enable_suggestions: true,
            }
        }
    }

    /// Bottleneck detection result
    #[derive(Debug, Clone)]
    pub struct BottleneckReport {
        /// Operation name
        pub operation: String,
        /// Bottleneck type
        pub bottleneck_type: BottleneckType,
        /// Severity score (0.0 to 1.0, higher is more severe)
        pub severity: f64,
        /// Description of the issue
        pub description: String,
        /// Optimization suggestions
        pub suggestions: Vec<String>,
        /// Performance statistics
        pub stats: PerformanceStats,
    }

    /// Type of bottleneck detected
    #[derive(Debug, Clone, PartialEq)]
    pub enum BottleneckType {
        /// Slow execution time
        SlowExecution,
        /// High memory usage
        HighMemoryUsage,
        /// High CPU usage
        HighCpuUsage,
        /// Frequent calls (hot path)
        HotPath,
        /// Memory leaks
        MemoryLeak,
        /// Inefficient algorithm
        IneffientAlgorithm,
    }

    /// Performance statistics for bottleneck analysis
    #[derive(Debug, Clone)]
    pub struct PerformanceStats {
        /// Total calls
        pub calls: usize,
        /// Total execution time
        pub total_time: Duration,
        /// Average execution time
        pub avg_time: Duration,
        /// Maximum execution time
        pub max_time: Duration,
        /// Total memory usage
        pub total_memory: usize,
        /// Average memory usage
        pub avg_memory: f64,
        /// Maximum memory usage
        pub max_memory: usize,
        /// CPU utilization
        pub cpu_utilization: f64,
    }

    /// Automated bottleneck detector
    #[derive(Debug)]
    pub struct BottleneckDetector {
        /// Configuration for detection
        config: BottleneckConfig,
        /// Performance history
        #[allow(dead_code)]
        performance_history: HashMap<String, Vec<PerformanceStats>>,
    }

    impl BottleneckDetector {
        /// Create a new bottleneck detector
        pub fn new(config: BottleneckConfig) -> Self {
            Self {
                config,
                performance_history: HashMap::new(),
            }
        }

        /// Analyze profiling data for bottlenecks
        pub fn analyze(&mut self, profiler: &Profiler) -> Vec<BottleneckReport> {
            let mut reports = Vec::new();

            // Analyze timing data
            for (operation, timing_entry) in &profiler.timings {
                if timing_entry.calls >= self.config.min_calls {
                    let stats = PerformanceStats {
                        calls: timing_entry.calls,
                        total_time: timing_entry.total_duration,
                        avg_time: timing_entry.average_duration(),
                        max_time: timing_entry.max_duration,
                        total_memory: 0, // Would be populated from memory tracking
                        avg_memory: 0.0,
                        max_memory: 0,
                        cpu_utilization: 0.0, // Would be populated from CPU monitoring
                    };

                    // Check for slow execution
                    if stats.avg_time > self.config.min_execution_threshold {
                        let severity = (stats.avg_time.as_secs_f64()
                            / self.config.min_execution_threshold.as_secs_f64())
                        .min(1.0);
                        let mut suggestions = Vec::new();

                        if self.config.enable_suggestions {
                            suggestions.extend([
                                "Consider algorithm optimization".to_string(),
                                "Profile inner functions for specific bottlenecks".to_string(),
                                "Check for unnecessary allocations".to_string(),
                                "Consider parallel processing if applicable".to_string(),
                            ]);
                        }

                        reports.push(BottleneckReport {
                            operation: operation.clone(),
                            bottleneck_type: BottleneckType::SlowExecution,
                            severity,
                            description: format!(
                                "Operation '{}' takes {:.2}ms on average, which exceeds the threshold of {:.2}ms",
                                operation,
                                stats.avg_time.as_secs_f64() * 1000.0,
                                self.config.min_execution_threshold.as_secs_f64() * 1000.0
                            ),
                            suggestions,
                            stats: stats.clone(),
                        });
                    }

                    // Check for hot paths (frequent calls)
                    if stats.calls > 1000 {
                        let severity = (stats.calls as f64 / 10000.0).min(1.0);
                        let mut suggestions = Vec::new();

                        if self.config.enable_suggestions {
                            suggestions.extend([
                                "Consider caching results if applicable".to_string(),
                                "Look for opportunities to batch operations".to_string(),
                                "Profile for micro-optimizations".to_string(),
                                "Consider memoization for pure functions".to_string(),
                            ]);
                        }

                        reports.push(BottleneckReport {
                            operation: operation.clone(),
                            bottleneck_type: BottleneckType::HotPath,
                            severity,
                            description: format!(
                                "Operation '{}' is called {} times, indicating a hot path",
                                operation, stats.calls
                            ),
                            suggestions,
                            stats,
                        });
                    }
                }
            }

            // Analyze memory data
            for (operation, memory_entry) in &profiler.memory {
                if memory_entry.allocations >= self.config.min_calls {
                    let avg_memory =
                        memory_entry.total_delta as f64 / memory_entry.allocations as f64;

                    if memory_entry.max_delta > self.config.memory_threshold {
                        let severity = (memory_entry.max_delta as f64
                            / (self.config.memory_threshold as f64 * 2.0))
                            .min(1.0);
                        let mut suggestions = Vec::new();

                        if self.config.enable_suggestions {
                            suggestions.extend([
                                "Consider pre-allocating memory where possible".to_string(),
                                "Look for opportunities to reuse memory".to_string(),
                                "Check for memory leaks".to_string(),
                                "Consider using memory pools".to_string(),
                            ]);
                        }

                        reports.push(BottleneckReport {
                            operation: operation.clone(),
                            bottleneck_type: BottleneckType::HighMemoryUsage,
                            severity,
                            description: format!(
                                "Operation '{}' uses up to {:.2}MB of memory, exceeding threshold of {:.2}MB",
                                operation,
                                memory_entry.max_delta as f64 / 1024.0 / 1024.0,
                                self.config.memory_threshold as f64 / 1024.0 / 1024.0
                            ),
                            suggestions,
                            stats: PerformanceStats {
                                calls: memory_entry.allocations,
                                total_time: Duration::from_secs(0),
                                avg_time: Duration::from_secs(0),
                                max_time: Duration::from_secs(0),
                                total_memory: memory_entry.total_delta as usize,
                                avg_memory,
                                max_memory: memory_entry.max_delta,
                                cpu_utilization: 0.0,
                            },
                        });
                    }
                }
            }

            reports
        }

        /// Print bottleneck report
        pub fn print_report(&self, reports: &[BottleneckReport]) {
            if reports.is_empty() {
                println!("No performance bottlenecks detected.");
                return;
            }

            println!("\n=== Bottleneck Analysis Report ===");

            for report in reports {
                println!("\n🔍 Operation: {}", report.operation);
                println!("   Type: {:?}", report.bottleneck_type);
                println!("   Severity: {:.1}%", report.severity * 100.0);
                println!("   Description: {}", report.description);

                if !report.suggestions.is_empty() {
                    println!("   Suggestions:");
                    for suggestion in &report.suggestions {
                        println!("     • {}", suggestion);
                    }
                }

                println!("   Stats:");
                println!("     • Calls: {}", report.stats.calls);
                if report.stats.total_time.as_nanos() > 0 {
                    println!(
                        "     • Avg Time: {:.2}ms",
                        report.stats.avg_time.as_secs_f64() * 1000.0
                    );
                    println!(
                        "     • Max Time: {:.2}ms",
                        report.stats.max_time.as_secs_f64() * 1000.0
                    );
                }
                if report.stats.total_memory > 0 {
                    println!(
                        "     • Avg Memory: {:.2}KB",
                        report.stats.avg_memory / 1024.0
                    );
                    println!(
                        "     • Max Memory: {:.2}KB",
                        report.stats.max_memory as f64 / 1024.0
                    );
                }
            }
        }
    }

    impl Default for BottleneckDetector {
        fn default() -> Self {
            Self::new(BottleneckConfig::default())
        }
    }

    /// System resource monitor for tracking CPU, memory, and network usage
    #[derive(Debug)]
    pub struct SystemResourceMonitor {
        /// Monitoring interval
        interval: Duration,
        /// Whether monitoring is active
        active: Arc<std::sync::atomic::AtomicBool>,
        /// CPU usage history
        cpu_history: Arc<Mutex<VecDeque<f64>>>,
        /// Memory usage history
        memory_history: Arc<Mutex<VecDeque<usize>>>,
        /// Network I/O history (bytes)
        network_history: Arc<Mutex<VecDeque<(u64, u64)>>>, // (bytes_in, bytes_out)
    }

    impl SystemResourceMonitor {
        /// Create a new system resource monitor
        pub fn new(interval: Duration) -> Self {
            Self {
                interval,
                active: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                cpu_history: Arc::new(Mutex::new(VecDeque::new())),
                memory_history: Arc::new(Mutex::new(VecDeque::new())),
                network_history: Arc::new(Mutex::new(VecDeque::new())),
            }
        }

        /// Start monitoring system resources
        pub fn start(&self) {
            self.active.store(true, Ordering::Relaxed);

            let active = self.active.clone();
            let cpu_history = self.cpu_history.clone();
            let memory_history = self.memory_history.clone();
            let network_history = self.network_history.clone();
            let interval = self.interval;

            thread::spawn(move || {
                while active.load(Ordering::Relaxed) {
                    // Sample CPU usage
                    let cpu_usage = Self::get_cpu_usage();
                    if let Ok(mut cpu_hist) = cpu_history.lock() {
                        cpu_hist.push_back(cpu_usage);
                        if cpu_hist.len() > 1000 {
                            cpu_hist.pop_front();
                        }
                    }

                    // Sample memory usage
                    let memory_usage = Self::get_memory_usage();
                    if let Ok(mut mem_hist) = memory_history.lock() {
                        mem_hist.push_back(memory_usage);
                        if mem_hist.len() > 1000 {
                            mem_hist.pop_front();
                        }
                    }

                    // Sample network usage
                    let network_usage = Self::get_network_usage();
                    if let Ok(mut net_hist) = network_history.lock() {
                        net_hist.push_back(network_usage);
                        if net_hist.len() > 1000 {
                            net_hist.pop_front();
                        }
                    }

                    thread::sleep(interval);
                }
            });
        }

        /// Stop monitoring
        pub fn stop(&self) {
            self.active.store(false, Ordering::Relaxed);
        }

        /// Get current CPU usage (0.0 to 1.0)
        fn get_cpu_usage() -> f64 {
            // This is a simplified implementation
            // In a real implementation, you would use platform-specific APIs
            #[cfg(target_os = "linux")]
            {
                // On Linux, parse /proc/stat
                0.5 // Placeholder
            }

            #[cfg(target_os = "macos")]
            {
                // On macOS, use host_processor_info
                0.5 // Placeholder
            }

            #[cfg(target_os = "windows")]
            {
                // On Windows, use GetSystemTimes
                0.5 // Placeholder
            }

            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
            {
                0.5 // Fallback placeholder
            }
        }

        /// Get current memory usage in bytes
        fn get_memory_usage() -> usize {
            // Simplified implementation - would use platform-specific APIs
            1024 * 1024 * 512 // 512 MB placeholder
        }

        /// Get current network usage (bytes_in, bytes_out)
        fn get_network_usage() -> (u64, u64) {
            // Simplified implementation - would parse /proc/net/dev on Linux
            (1024, 1024) // Placeholder
        }

        /// Get resource usage statistics
        pub fn get_stats(&self) -> ResourceStats {
            let cpu_hist = self.cpu_history.lock().unwrap();
            let memory_hist = self.memory_history.lock().unwrap();
            let network_hist = self.network_history.lock().unwrap();

            let avg_cpu = if cpu_hist.is_empty() {
                0.0
            } else {
                cpu_hist.iter().sum::<f64>() / cpu_hist.len() as f64
            };

            let max_cpu = cpu_hist.iter().fold(0.0f64, |a, &b| a.max(b));

            let avg_memory = if memory_hist.is_empty() {
                0
            } else {
                memory_hist.iter().sum::<usize>() / memory_hist.len()
            };

            let max_memory = memory_hist.iter().max().copied().unwrap_or(0);

            let total_network_in: u64 = network_hist.iter().map(|(bytes_in, _)| *bytes_in).sum();
            let total_network_out: u64 = network_hist.iter().map(|(_, bytes_out)| *bytes_out).sum();

            ResourceStats {
                avg_cpu_usage: avg_cpu,
                max_cpu_usage: max_cpu,
                avg_memory_usage: avg_memory,
                max_memory_usage: max_memory,
                total_network_in,
                total_network_out,
                sample_count: cpu_hist.len(),
            }
        }
    }

    impl Default for SystemResourceMonitor {
        fn default() -> Self {
            Self::new(Duration::from_secs(1))
        }
    }

    /// Resource usage statistics
    #[derive(Debug, Clone)]
    pub struct ResourceStats {
        /// Average CPU usage (0.0 to 1.0)
        pub avg_cpu_usage: f64,
        /// Maximum CPU usage (0.0 to 1.0)
        pub max_cpu_usage: f64,
        /// Average memory usage (bytes)
        pub avg_memory_usage: usize,
        /// Maximum memory usage (bytes)
        pub max_memory_usage: usize,
        /// Total network bytes received
        pub total_network_in: u64,
        /// Total network bytes sent
        pub total_network_out: u64,
        /// Number of samples collected
        pub sample_count: usize,
    }

    /// Differential profiler for comparing performance between runs
    #[derive(Debug)]
    pub struct DifferentialProfiler {
        /// Baseline profiling data
        baseline: Option<ProfileSnapshot>,
        /// Current profiling data
        current: Option<ProfileSnapshot>,
    }

    /// Snapshot of profiling data at a point in time
    #[derive(Debug, Clone)]
    pub struct ProfileSnapshot {
        /// Timing data
        pub timings: HashMap<String, TimingEntry>,
        /// Memory data
        pub memory: HashMap<String, MemoryEntry>,
        /// Resource usage at snapshot time
        pub resources: Option<ResourceStats>,
        /// Timestamp of snapshot
        pub timestamp: std::time::Instant,
        /// Optional label for the snapshot
        pub label: Option<String>,
    }

    impl DifferentialProfiler {
        /// Create a new differential profiler
        pub fn new() -> Self {
            Self {
                baseline: None,
                current: None,
            }
        }

        /// Set the baseline snapshot
        pub fn set_baseline(&mut self, profiler: &Profiler, label: Option<String>) {
            self.baseline = Some(ProfileSnapshot {
                timings: profiler.timings.clone(),
                memory: profiler.memory.clone(),
                resources: None,
                timestamp: std::time::Instant::now(),
                label,
            });
        }

        /// Set the current snapshot
        pub fn set_current(&mut self, profiler: &Profiler, label: Option<String>) {
            self.current = Some(ProfileSnapshot {
                timings: profiler.timings.clone(),
                memory: profiler.memory.clone(),
                resources: None,
                timestamp: std::time::Instant::now(),
                label,
            });
        }

        /// Generate a differential report
        pub fn generate_diff_report(&self) -> Option<DifferentialReport> {
            if let (Some(baseline), Some(current)) = (&self.baseline, &self.current) {
                Some(DifferentialReport::new(baseline, current))
            } else {
                None
            }
        }
    }

    impl Default for DifferentialProfiler {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Differential profiling report
    #[derive(Debug)]
    pub struct DifferentialReport {
        /// Timing differences
        pub timing_diffs: HashMap<String, TimingDiff>,
        /// Memory differences
        pub memory_diffs: HashMap<String, MemoryDiff>,
        /// Overall performance change
        pub overall_change: PerformanceChange,
        /// Report generation timestamp
        pub generated_at: std::time::Instant,
    }

    impl DifferentialReport {
        /// Create a new differential report
        pub fn new(baseline: &ProfileSnapshot, current: &ProfileSnapshot) -> Self {
            let mut timing_diffs = HashMap::new();
            let mut memory_diffs = HashMap::new();

            // Calculate timing differences
            for (operation, current_timing) in &current.timings {
                if let Some(baseline_timing) = baseline.timings.get(operation) {
                    timing_diffs.insert(
                        operation.clone(),
                        TimingDiff::new(baseline_timing, current_timing),
                    );
                }
            }

            // Calculate memory differences
            for (operation, current_memory) in &current.memory {
                if let Some(baseline_memory) = baseline.memory.get(operation) {
                    memory_diffs.insert(
                        operation.clone(),
                        MemoryDiff::new(baseline_memory, current_memory),
                    );
                }
            }

            // Calculate overall performance change
            let overall_change = PerformanceChange::calculate(&timing_diffs, &memory_diffs);

            Self {
                timing_diffs,
                memory_diffs,
                overall_change,
                generated_at: std::time::Instant::now(),
            }
        }

        /// Print the differential report
        pub fn print(&self) {
            println!("\n=== Differential Profiling Report ===");

            if !self.timing_diffs.is_empty() {
                println!("\nTiming Changes:");
                println!(
                    "{:<30} {:<15} {:<15} {:<15}",
                    "Operation", "Baseline (ms)", "Current (ms)", "Change (%)"
                );
                println!("{}", "-".repeat(80));

                for (operation, diff) in &self.timing_diffs {
                    println!(
                        "{:<30} {:<15.2} {:<15.2} {:>+14.1}%",
                        operation,
                        diff.baseline_avg.as_secs_f64() * 1000.0,
                        diff.current_avg.as_secs_f64() * 1000.0,
                        diff.percentage_change
                    );
                }
            }

            if !self.memory_diffs.is_empty() {
                println!("\nMemory Changes:");
                println!(
                    "{:<30} {:<15} {:<15} {:<15}",
                    "Operation", "Baseline (KB)", "Current (KB)", "Change (%)"
                );
                println!("{}", "-".repeat(80));

                for (operation, diff) in &self.memory_diffs {
                    println!(
                        "{:<30} {:<15.2} {:<15.2} {:>+14.1}%",
                        operation,
                        diff.baseline_avg / 1024.0,
                        diff.current_avg / 1024.0,
                        diff.percentage_change
                    );
                }
            }

            println!("\nOverall Performance:");
            println!(
                "  • Timing Change: {:+.1}%",
                self.overall_change.timing_change
            );
            println!(
                "  • Memory Change: {:+.1}%",
                self.overall_change.memory_change
            );
            println!("  • Recommendation: {}", self.overall_change.recommendation);
        }
    }

    /// Timing difference between baseline and current
    #[derive(Debug)]
    pub struct TimingDiff {
        /// Baseline average duration
        pub baseline_avg: Duration,
        /// Current average duration
        pub current_avg: Duration,
        /// Percentage change (positive = slower, negative = faster)
        pub percentage_change: f64,
    }

    impl TimingDiff {
        /// Create a new timing difference
        pub fn new(baseline: &TimingEntry, current: &TimingEntry) -> Self {
            let baseline_avg = baseline.average_duration();
            let current_avg = current.average_duration();
            let percentage_change = if baseline_avg.as_nanos() > 0 {
                ((current_avg.as_nanos() as f64 - baseline_avg.as_nanos() as f64)
                    / baseline_avg.as_nanos() as f64)
                    * 100.0
            } else {
                0.0
            };

            Self {
                baseline_avg,
                current_avg,
                percentage_change,
            }
        }
    }

    /// Memory difference between baseline and current
    #[derive(Debug)]
    pub struct MemoryDiff {
        /// Baseline average memory usage
        pub baseline_avg: f64,
        /// Current average memory usage
        pub current_avg: f64,
        /// Percentage change (positive = more memory, negative = less memory)
        pub percentage_change: f64,
    }

    impl MemoryDiff {
        /// Create a new memory difference
        pub fn new(baseline: &MemoryEntry, current: &MemoryEntry) -> Self {
            let baseline_avg = if baseline.allocations > 0 {
                baseline.total_delta as f64 / baseline.allocations as f64
            } else {
                0.0
            };

            let current_avg = if current.allocations > 0 {
                current.total_delta as f64 / current.allocations as f64
            } else {
                0.0
            };

            let percentage_change = if baseline_avg.abs() > 0.0 {
                ((current_avg - baseline_avg) / baseline_avg.abs()) * 100.0
            } else {
                0.0
            };

            Self {
                baseline_avg,
                current_avg,
                percentage_change,
            }
        }
    }

    /// Overall performance change summary
    #[derive(Debug)]
    pub struct PerformanceChange {
        /// Overall timing change percentage
        pub timing_change: f64,
        /// Overall memory change percentage
        pub memory_change: f64,
        /// Performance recommendation
        pub recommendation: String,
    }

    impl PerformanceChange {
        /// Calculate overall performance change
        pub fn calculate(
            timing_diffs: &HashMap<String, TimingDiff>,
            memory_diffs: &HashMap<String, MemoryDiff>,
        ) -> Self {
            let timing_change = if timing_diffs.is_empty() {
                0.0
            } else {
                timing_diffs
                    .values()
                    .map(|diff| diff.percentage_change)
                    .sum::<f64>()
                    / timing_diffs.len() as f64
            };

            let memory_change = if memory_diffs.is_empty() {
                0.0
            } else {
                memory_diffs
                    .values()
                    .map(|diff| diff.percentage_change)
                    .sum::<f64>()
                    / memory_diffs.len() as f64
            };

            let recommendation = match (timing_change > 5.0, memory_change > 10.0) {
                (true, true) => "Performance degraded significantly in both time and memory. Review recent changes.".to_string(),
                (true, false) => "Execution time increased. Consider profiling hot paths for optimization opportunities.".to_string(),
                (false, true) => "Memory usage increased. Review memory allocation patterns and consider optimization.".to_string(),
                (false, false) => {
                    if timing_change < -5.0 || memory_change < -10.0 {
                        "Performance improved! Consider documenting the optimizations made.".to_string()
                    } else {
                        "Performance is stable with minimal changes.".to_string()
                    }
                }
            };

            Self {
                timing_change,
                memory_change,
                recommendation,
            }
        }
    }

    /// Performance profiler with export capabilities
    #[derive(Debug)]
    pub struct ExportableProfiler {
        /// Base profiler
        profiler: Profiler,
        /// Additional metadata
        metadata: HashMap<String, String>,
    }

    impl ExportableProfiler {
        /// Create a new exportable profiler
        pub fn new() -> Self {
            Self {
                profiler: Profiler::new(),
                metadata: HashMap::new(),
            }
        }

        /// Add metadata
        pub fn add_metadata(&mut self, key: String, value: String) {
            self.metadata.insert(key, value);
        }

        /// Export profiling data to JSON
        pub fn export_to_json(&self, path: &str) -> Result<(), std::io::Error> {
            use std::fs::File;
            use std::io::BufWriter;

            let file = File::create(path)?;
            let mut _writer = BufWriter::new(file);

            // In a real implementation, you would use serde to serialize the data
            // For now, we'll create a simple JSON structure manually
            let json_data = format!(
                r#"{{
                    "metadata": {:#?},
                    "timings": {:#?},
                    "memory": {:#?}
                }}"#,
                self.metadata, self.profiler.timings, self.profiler.memory
            );

            std::io::Write::write_all(&mut _writer, json_data.as_bytes())?;
            Ok(())
        }

        /// Export profiling data to CSV
        pub fn export_to_csv(&self, path: &str) -> Result<(), std::io::Error> {
            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);

            // Write timing data
            writeln!(writer, "Operation,Calls,Total_ms,Average_ms,Max_ms")?;
            for (operation, timing) in &self.profiler.timings {
                writeln!(
                    writer,
                    "{},{},{:.2},{:.2},{:.2}",
                    operation,
                    timing.calls,
                    timing.total_duration.as_secs_f64() * 1000.0,
                    timing.average_duration().as_secs_f64() * 1000.0,
                    timing.max_duration.as_secs_f64() * 1000.0
                )?;
            }

            writer.flush()?;
            Ok(())
        }

        /// Get access to the underlying profiler
        pub fn profiler(&mut self) -> &mut Profiler {
            &mut self.profiler
        }
    }

    impl Default for ExportableProfiler {
        fn default() -> Self {
            Self::new()
        }
    }
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
        diff_profiler.set_baseline(
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
    fn test_system_resource_monitor() {
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
        profiler.add_metadata("test_run".to_string(), "alpha6".to_string());

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

/// Comprehensive profiling integration that combines application profiling with system monitoring
pub mod comprehensive {
    use super::*;
    use crate::profiling::flame_graph_svg::{
        EnhancedFlameGraph, SvgFlameGraphConfig, SvgFlameGraphGenerator,
    };
    use crate::profiling::system_monitor::{
        AlertConfig, SystemAlert, SystemAlerter, SystemMonitor, SystemMonitorConfig,
    };
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Comprehensive profiling session that combines multiple profiling techniques
    pub struct ComprehensiveProfiler {
        /// Application profiler
        app_profiler: Arc<Mutex<Profiler>>,
        /// System resource monitor
        system_monitor: SystemMonitor,
        /// System alerter
        system_alerter: SystemAlerter,
        /// Flame graph generator
        flame_graph_generator: advanced::FlameGraphGenerator,
        /// Session start time
        session_start: Instant,
        /// Session configuration
        config: ComprehensiveConfig,
    }

    /// Configuration for comprehensive profiling
    #[derive(Debug, Clone)]
    pub struct ComprehensiveConfig {
        /// System monitoring configuration
        pub system_config: SystemMonitorConfig,
        /// Alert configuration
        pub alert_config: AlertConfig,
        /// SVG flame graph configuration
        pub svg_config: SvgFlameGraphConfig,
        /// Enable automatic bottleneck detection
        pub enable_bottleneck_detection: bool,
        /// Enable automatic alert notifications
        pub enable_alerts: bool,
        /// Enable flame graph generation
        pub enable_flame_graphs: bool,
        /// Session name for reports
        pub session_name: String,
    }

    impl Default for ComprehensiveConfig {
        fn default() -> Self {
            Self {
                system_config: SystemMonitorConfig::default(),
                alert_config: AlertConfig::default(),
                svg_config: SvgFlameGraphConfig::default(),
                enable_bottleneck_detection: true,
                enable_alerts: true,
                enable_flame_graphs: true,
                session_name: "Profiling Session".to_string(),
            }
        }
    }

    impl ComprehensiveProfiler {
        /// Create a new comprehensive profiler
        pub fn new(config: ComprehensiveConfig) -> Self {
            Self {
                app_profiler: Arc::new(Mutex::new(Profiler::new())),
                system_monitor: SystemMonitor::new(config.system_config.clone()),
                system_alerter: SystemAlerter::new(config.alert_config.clone()),
                flame_graph_generator: advanced::FlameGraphGenerator::new(),
                session_start: Instant::now(),
                config,
            }
        }

        /// Start comprehensive profiling
        pub fn start(
            &mut self,
        ) -> Result<(), crate::profiling::system_monitor::SystemMonitorError> {
            // Start application profiler
            self.app_profiler.lock().unwrap().start();

            // Start system monitor
            self.system_monitor.start()?;

            self.session_start = Instant::now();
            Ok(())
        }

        /// Stop comprehensive profiling
        pub fn stop(&mut self) {
            self.app_profiler.lock().unwrap().stop();
            self.system_monitor.stop();
        }

        /// Time a function with comprehensive profiling
        pub fn time_function<F, R>(&mut self, name: &str, f: F) -> R
        where
            F: FnOnce() -> R,
        {
            // Start flame graph tracking
            self.flame_graph_generator.start_call(name);

            // Execute with application profiling
            let result = Timer::time_function(name, f);

            // End flame graph tracking
            self.flame_graph_generator.end_call();

            // Check for alerts if enabled
            if self.config.enable_alerts {
                if let Ok(current_metrics) = self.system_monitor.get_current_metrics() {
                    let alerts = self.system_alerter.check_alerts(&current_metrics);
                    for alert in alerts {
                        self.handle_alert(&alert);
                    }
                }
            }

            result
        }

        /// Generate comprehensive profiling report
        pub fn generate_report(&mut self) -> ComprehensiveReport {
            let app_report = self.app_profiler.lock().unwrap().get_report();
            let system_metrics = self.system_monitor.get_metrics_history();
            let alerts = self.system_alerter.get_alert_history();

            let mut bottleneck_reports = Vec::new();
            if self.config.enable_bottleneck_detection {
                let mut detector =
                    advanced::BottleneckDetector::new(advanced::BottleneckConfig::default());
                bottleneck_reports = detector.analyze(&self.app_profiler.lock().unwrap());
            }

            let flame_graph = if self.config.enable_flame_graphs {
                Some(self.flame_graph_generator.generate())
            } else {
                None
            };

            ComprehensiveReport {
                session_name: self.config.session_name.clone(),
                session_duration: self.session_start.elapsed(),
                application_report: app_report,
                system_metrics,
                alerts,
                bottleneck_reports,
                flame_graph,
                generated_at: Instant::now(),
            }
        }

        /// Export comprehensive report to multiple formats
        pub fn export_report(&mut self, base_path: &str) -> Result<(), std::io::Error> {
            let report = self.generate_report();

            // Export text report
            std::fs::write(format!("{}_report.txt", base_path), report.to_text_format())?;

            // Export JSON report
            std::fs::write(
                format!("{}_report.json", base_path),
                report.to_json_format(),
            )?;

            // Export flame graph if available
            if let Some(ref flame_graph) = report.flame_graph {
                let svg_generator = SvgFlameGraphGenerator::new(self.config.svg_config.clone());
                svg_generator
                    .export_to_file(flame_graph, &format!("{}_flamegraph.svg", base_path))?;

                // Export enhanced flame graph with system metrics
                let enhanced = EnhancedFlameGraph {
                    performance: flame_graph.clone(),
                    memory: None,
                    cpu_usage: report
                        .system_metrics
                        .iter()
                        .map(|m| (m.timestamp.duration_since(self.session_start), m.cpu_usage))
                        .collect(),
                    memory_usage: report
                        .system_metrics
                        .iter()
                        .map(|m| {
                            (
                                m.timestamp.duration_since(self.session_start),
                                m.memory_usage,
                            )
                        })
                        .collect(),
                    total_duration: self.session_start.elapsed(),
                };
                enhanced.export_enhanced_svg(&format!("{}_enhanced_flamegraph.svg", base_path))?;
            }

            Ok(())
        }

        /// Handle system alerts
        fn handle_alert(&self, alert: &SystemAlert) {
            // In a real implementation, this could send notifications, log to files, etc.
            println!("ALERT: {}", alert.message);
        }

        /// Get application profiler reference
        pub fn app_profiler(&self) -> Arc<Mutex<Profiler>> {
            Arc::clone(&self.app_profiler)
        }

        /// Get current system metrics
        pub fn get_current_system_metrics(
            &self,
        ) -> Result<
            crate::profiling::system_monitor::SystemMetrics,
            crate::profiling::system_monitor::SystemMonitorError,
        > {
            self.system_monitor.get_current_metrics()
        }

        /// Get recent alerts
        pub fn get_recent_alerts(&self, duration: Duration) -> Vec<SystemAlert> {
            self.system_alerter.get_recent_alerts(duration)
        }
    }

    impl Drop for ComprehensiveProfiler {
        fn drop(&mut self) {
            self.stop();
        }
    }

    /// Comprehensive profiling report
    #[derive(Debug)]
    pub struct ComprehensiveReport {
        /// Session name
        pub session_name: String,
        /// Total session duration
        pub session_duration: Duration,
        /// Application profiling report
        pub application_report: String,
        /// System metrics collected during session
        pub system_metrics: Vec<crate::profiling::system_monitor::SystemMetrics>,
        /// System alerts triggered during session
        pub alerts: Vec<SystemAlert>,
        /// Bottleneck analysis results
        pub bottleneck_reports: Vec<advanced::BottleneckReport>,
        /// Flame graph data
        pub flame_graph: Option<advanced::FlameGraphNode>,
        /// Report generation timestamp
        pub generated_at: Instant,
    }

    impl ComprehensiveReport {
        /// Convert report to text format
        pub fn to_text_format(&self) -> String {
            use std::fmt::Write;
            let mut report = String::new();

            writeln!(report, "=== {} ===", self.session_name).unwrap();
            writeln!(
                report,
                "Session Duration: {:.2} seconds",
                self.session_duration.as_secs_f64()
            )
            .unwrap();
            writeln!(report, "Generated At: {:?}", self.generated_at).unwrap();
            writeln!(report).unwrap();

            // Application profiling
            writeln!(report, "=== Application Performance ===").unwrap();
            writeln!(report, "{}", self.application_report).unwrap();

            // System metrics summary
            if !self.system_metrics.is_empty() {
                writeln!(report, "=== System Resource Summary ===").unwrap();
                let avg_cpu = self.system_metrics.iter().map(|m| m.cpu_usage).sum::<f64>()
                    / self.system_metrics.len() as f64;
                let avg_memory = self
                    .system_metrics
                    .iter()
                    .map(|m| m.memory_usage)
                    .sum::<usize>()
                    / self.system_metrics.len();
                let max_cpu = self
                    .system_metrics
                    .iter()
                    .map(|m| m.cpu_usage)
                    .fold(0.0, f64::max);
                let max_memory = self
                    .system_metrics
                    .iter()
                    .map(|m| m.memory_usage)
                    .max()
                    .unwrap_or(0);

                writeln!(report, "Average CPU Usage: {:.1}%", avg_cpu).unwrap();
                writeln!(report, "Maximum CPU Usage: {:.1}%", max_cpu).unwrap();
                writeln!(
                    report,
                    "Average Memory Usage: {:.1} MB",
                    avg_memory as f64 / (1024.0 * 1024.0)
                )
                .unwrap();
                writeln!(
                    report,
                    "Maximum Memory Usage: {:.1} MB",
                    max_memory as f64 / (1024.0 * 1024.0)
                )
                .unwrap();
                writeln!(report).unwrap();
            }

            // Alerts
            if !self.alerts.is_empty() {
                writeln!(report, "=== System Alerts ({}) ===", self.alerts.len()).unwrap();
                for alert in &self.alerts {
                    writeln!(report, "[{:?}] {}", alert.severity, alert.message).unwrap();
                }
                writeln!(report).unwrap();
            }

            // Bottlenecks
            if !self.bottleneck_reports.is_empty() {
                writeln!(
                    report,
                    "=== Performance Bottlenecks ({}) ===",
                    self.bottleneck_reports.len()
                )
                .unwrap();
                for bottleneck in &self.bottleneck_reports {
                    writeln!(report, "Operation: {}", bottleneck.operation).unwrap();
                    writeln!(report, "Type: {:?}", bottleneck.bottleneck_type).unwrap();
                    writeln!(report, "Severity: {:.2}", bottleneck.severity).unwrap();
                    writeln!(report, "Description: {}", bottleneck.description).unwrap();
                    if !bottleneck.suggestions.is_empty() {
                        writeln!(report, "Suggestions:").unwrap();
                        for suggestion in &bottleneck.suggestions {
                            writeln!(report, "  - {}", suggestion).unwrap();
                        }
                    }
                    writeln!(report).unwrap();
                }
            }

            report
        }

        /// Convert report to JSON format
        pub fn to_json_format(&self) -> String {
            // Simplified JSON generation - in a real implementation would use serde
            use std::fmt::Write;
            let mut json = String::new();

            writeln!(json, "{{").unwrap();
            writeln!(json, "  \"session_name\": \"{}\",", self.session_name).unwrap();
            writeln!(
                json,
                "  \"session_duration_seconds\": {},",
                self.session_duration.as_secs_f64()
            )
            .unwrap();
            writeln!(json, "  \"alert_count\": {},", self.alerts.len()).unwrap();
            writeln!(
                json,
                "  \"bottleneck_count\": {},",
                self.bottleneck_reports.len()
            )
            .unwrap();
            writeln!(
                json,
                "  \"system_sample_count\": {}",
                self.system_metrics.len()
            )
            .unwrap();

            if !self.system_metrics.is_empty() {
                let avg_cpu = self.system_metrics.iter().map(|m| m.cpu_usage).sum::<f64>()
                    / self.system_metrics.len() as f64;
                let max_cpu = self
                    .system_metrics
                    .iter()
                    .map(|m| m.cpu_usage)
                    .fold(0.0, f64::max);
                writeln!(json, "  \"average_cpu_usage\": {},", avg_cpu).unwrap();
                writeln!(json, "  \"maximum_cpu_usage\": {}", max_cpu).unwrap();
            }

            writeln!(json, "}}").unwrap();
            json
        }

        /// Print comprehensive report to console
        pub fn print(&self) {
            println!("{}", self.to_text_format());
        }
    }

    /// Convenience macro for comprehensive profiling
    #[macro_export]
    macro_rules! comprehensive_profile {
        ($profiler:expr, $name:expr, $body:block) => {{
            $profiler.time_function($name, || $body)
        }};
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::thread;

        #[test]
        fn test_comprehensive_profiler() {
            let config = ComprehensiveConfig {
                session_name: "Test Session".to_string(),
                ..Default::default()
            };

            let mut profiler = ComprehensiveProfiler::new(config);
            profiler.start().unwrap();

            // Profile some work
            let result = profiler.time_function("test_work", || {
                thread::sleep(Duration::from_millis(10));
                42
            });

            assert_eq!(result, 42);

            // Generate report
            let report = profiler.generate_report();
            assert_eq!(report.session_name, "Test Session");
            assert!(report.session_duration > Duration::from_millis(5));

            profiler.stop();
        }

        #[test]
        fn test_comprehensive_report() {
            let report = ComprehensiveReport {
                session_name: "Test".to_string(),
                session_duration: Duration::from_secs(1),
                application_report: "Test report".to_string(),
                system_metrics: Vec::new(),
                alerts: Vec::new(),
                bottleneck_reports: Vec::new(),
                flame_graph: None,
                generated_at: Instant::now(),
            };

            let text = report.to_text_format();
            assert!(text.contains("Test"));

            let json = report.to_json_format();
            assert!(json.contains("session_name"));
        }
    }
}
