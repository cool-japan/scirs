//! Resource-aware prefetching system that adapts to system load.
//!
//! This module provides a prefetching system that monitors system resources (CPU, memory, IO)
//! and dynamically adjusts its prefetching strategy to avoid overloading the system.
//! This helps ensure that the prefetching system improves rather than hinders performance.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::prefetch::{PrefetchConfig, PrefetchStats};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Default sampling interval for resource monitoring
const DEFAULT_SAMPLING_INTERVAL: Duration = Duration::from_millis(500);

/// Default memory pressure threshold (percentage of available memory)
const DEFAULT_MEMORY_PRESSURE_THRESHOLD: f64 = 0.85;

/// Default CPU load threshold
const DEFAULT_CPU_LOAD_THRESHOLD: f64 = 0.85;

/// Default IO pressure threshold
const DEFAULT_IO_PRESSURE_THRESHOLD: f64 = 0.85;

/// Minimum interval between strategy adjustments
const MIN_ADJUSTMENT_INTERVAL: Duration = Duration::from_secs(1);

/// Maximum interval for resource snapshots
const MAX_SNAPSHOT_HISTORY: usize = 20;

/// Types of system resources that can be monitored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    /// CPU utilization
    CPU,

    /// Memory usage
    Memory,

    /// IO operations
    IO,

    /// Combined resource pressure
    Combined,
}

/// Snapshot of system resource usage at a point in time.
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: Instant,

    /// CPU usage (0.0 to 1.0)
    pub cpu_usage: f64,

    /// Memory usage (bytes)
    pub memory_usage: u64,

    /// Available memory (bytes)
    pub memory_available: u64,

    /// IO operations per second
    pub io_ops_per_sec: u64,

    /// IO bytes per second
    pub io_bytes_per_sec: u64,
}

impl ResourceSnapshot {
    /// Calculate memory pressure (0.0 to 1.0).
    pub fn memory_pressure(&self) -> f64 {
        if self.memory_available == 0 {
            0.0 // Avoid division by zero
        } else {
            self.memory_usage as f64 / (self.memory_usage + self.memory_available) as f64
        }
    }

    /// Calculate combined resource pressure (0.0 to 1.0).
    pub fn combined_pressure(&self) -> f64 {
        // Weight factors for different resources
        const CPU_WEIGHT: f64 = 0.4;
        const MEMORY_WEIGHT: f64 = 0.4;
        const IO_WEIGHT: f64 = 0.2;

        // Normalized IO pressure (estimate)
        let io_pressure = if self.io_bytes_per_sec > 100_000_000 {
            // Over 100MB/s is high IO
            0.9
        } else if self.io_bytes_per_sec > 50_000_000 {
            // 50-100MB/s is medium IO
            0.7
        } else if self.io_bytes_per_sec > 10_000_000 {
            // 10-50MB/s is moderate IO
            0.5
        } else {
            // Under 10MB/s is low IO
            0.2
        };

        // Combined pressure
        CPU_WEIGHT * self.cpu_usage
            + MEMORY_WEIGHT * self.memory_pressure()
            + IO_WEIGHT * io_pressure
    }
}

/// Configuration for resource-aware prefetching.
#[derive(Debug, Clone)]
pub struct ResourceAwareConfig {
    /// How often to sample system resources
    pub sampling_interval: Duration,

    /// Threshold for memory pressure (0.0 to 1.0)
    pub memory_pressure_threshold: f64,

    /// Threshold for CPU load (0.0 to 1.0)
    pub cpu_load_threshold: f64,

    /// Threshold for IO pressure (0.0 to 1.0)
    pub io_pressure_threshold: f64,

    /// Minimum time between strategy adjustments
    pub adjustment_interval: Duration,

    /// Whether to automatically adjust prefetching based on resources
    pub auto_adjust: bool,

    /// Whether to disable prefetching when system is under very high load
    pub disable_under_pressure: bool,

    /// Minimum prefetch count (even under load)
    pub min_prefetch_count: usize,

    /// Maximum prefetch count (when resources are abundant)
    pub max_prefetch_count: usize,
}

impl Default for ResourceAwareConfig {
    fn default() -> Self {
        Self {
            sampling_interval: DEFAULT_SAMPLING_INTERVAL,
            memory_pressure_threshold: DEFAULT_MEMORY_PRESSURE_THRESHOLD,
            cpu_load_threshold: DEFAULT_CPU_LOAD_THRESHOLD,
            io_pressure_threshold: DEFAULT_IO_PRESSURE_THRESHOLD,
            adjustment_interval: MIN_ADJUSTMENT_INTERVAL,
            auto_adjust: true,
            disable_under_pressure: true,
            min_prefetch_count: 1,
            max_prefetch_count: 8,
        }
    }
}

/// Builder for resource-aware configuration.
#[derive(Debug, Clone)]
pub struct ResourceAwareConfigBuilder {
    config: ResourceAwareConfig,
}

impl ResourceAwareConfigBuilder {
    /// Create a new resource-aware config builder with default settings.
    pub fn new() -> Self {
        Self {
            config: ResourceAwareConfig::default(),
        }
    }

    /// Set the sampling interval.
    pub fn with_sampling_interval(mut self, interval: Duration) -> Self {
        self.config.sampling_interval = interval;
        self
    }

    /// Set the memory pressure threshold.
    pub fn with_memory_pressure_threshold(mut self, threshold: f64) -> Self {
        self.config.memory_pressure_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the CPU load threshold.
    pub fn with_cpu_load_threshold(mut self, threshold: f64) -> Self {
        self.config.cpu_load_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the IO pressure threshold.
    pub fn with_io_pressure_threshold(mut self, threshold: f64) -> Self {
        self.config.io_pressure_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the adjustment interval.
    pub fn with_adjustment_interval(mut self, interval: Duration) -> Self {
        self.config.adjustment_interval = std::cmp::max(interval, MIN_ADJUSTMENT_INTERVAL);
        self
    }

    /// Enable or disable automatic adjustment.
    pub fn with_auto_adjust(mut self, auto_adjust: bool) -> Self {
        self.config.auto_adjust = auto_adjust;
        self
    }

    /// Enable or disable disabling prefetching under high load.
    pub fn with_disable_under_pressure(mut self, disable: bool) -> Self {
        self.config.disable_under_pressure = disable;
        self
    }

    /// Set the minimum prefetch count.
    pub fn with_min_prefetch_count(mut self, count: usize) -> Self {
        self.config.min_prefetch_count = count;
        self
    }

    /// Set the maximum prefetch count.
    pub fn with_max_prefetch_count(mut self, count: usize) -> Self {
        self.config.max_prefetch_count = count;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ResourceAwareConfig {
        self.config
    }
}

/// Resource monitor for tracking system load.
pub struct ResourceMonitor {
    /// Configuration for resource monitoring
    config: ResourceAwareConfig,

    /// History of resource snapshots
    snapshots: VecDeque<ResourceSnapshot>,

    /// Last time resources were sampled
    last_sample: Instant,

    /// Last time prefetching strategy was adjusted
    last_adjustment: Instant,

    /// Current pressure status (true if under pressure)
    under_pressure: bool,

    /// System info provider
    sys_info: Box<dyn SystemInfo + Send + Sync>,
}

impl ResourceMonitor {
    /// Create a new resource monitor with the given configuration.
    pub fn new(config: ResourceAwareConfig) -> Self {
        Self {
            config,
            snapshots: VecDeque::with_capacity(MAX_SNAPSHOT_HISTORY),
            last_sample: Instant::now(),
            last_adjustment: Instant::now(),
            under_pressure: false,
            sys_info: Box::new(DefaultSystemInfo),
        }
    }

    /// Take a snapshot of current system resources.
    pub fn take_snapshot(&mut self) -> ResourceSnapshot {
        // Get CPU usage
        let cpu_usage = self.sys_info.get_cpu_usage();

        // Get memory usage
        let (memory_usage, memory_available) = self.sys_info.get_memory_info();

        // Get IO stats
        let (io_ops_per_sec, io_bytes_per_sec) = self.sys_info.get_io_stats();

        // Create snapshot
        let snapshot = ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage,
            memory_usage,
            memory_available,
            io_ops_per_sec,
            io_bytes_per_sec,
        };

        // Update history
        self.snapshots.push_back(snapshot.clone());
        while self.snapshots.len() > MAX_SNAPSHOT_HISTORY {
            self.snapshots.pop_front();
        }

        // Update last sample time
        self.last_sample = Instant::now();

        snapshot
    }

    /// Check if it's time to take a new snapshot.
    pub fn should_take_snapshot(&self) -> bool {
        self.last_sample.elapsed() >= self.config.sampling_interval
    }

    /// Check if the system is under resource pressure.
    pub fn is_under_pressure(&mut self) -> bool {
        // Take a snapshot if needed
        if self.should_take_snapshot() {
            self.take_snapshot();
        }

        // Check if we have snapshots
        if self.snapshots.is_empty() {
            return false;
        }

        // Get the latest snapshot
        let latest = self.snapshots.back().unwrap();

        // Check each resource
        let cpu_pressure = latest.cpu_usage > self.config.cpu_load_threshold;
        let memory_pressure = latest.memory_pressure() > self.config.memory_pressure_threshold;

        // Calculate IO pressure
        let io_pressure = if latest.io_bytes_per_sec > 100_000_000 {
            // Over 100MB/s is considered high IO
            true
        } else {
            false
        };

        // Combined pressure
        self.under_pressure = cpu_pressure || memory_pressure || io_pressure;

        self.under_pressure
    }

    /// Get the optimal prefetch count based on current resources.
    pub fn get_optimal_prefetch_count(&mut self, base_prefetch_count: usize) -> usize {
        if !self.config.auto_adjust {
            return base_prefetch_count;
        }

        // Take a snapshot if needed
        if self.should_take_snapshot() {
            self.take_snapshot();
        }

        // Check if we have snapshots
        if self.snapshots.is_empty() {
            return base_prefetch_count;
        }

        // Get the latest snapshot
        let latest = self.snapshots.back().unwrap();

        // Calculate combined pressure
        let pressure = latest.combined_pressure();

        // Adjust prefetch count based on pressure
        if pressure > 0.90 && self.config.disable_under_pressure {
            // Very high pressure, drastically reduce or disable prefetching
            self.config.min_prefetch_count
        } else if pressure > 0.75 {
            // High pressure, reduce prefetching
            std::cmp::max(
                self.config.min_prefetch_count,
                (base_prefetch_count as f64 * 0.5).round() as usize,
            )
        } else if pressure > 0.6 {
            // Moderate pressure, slightly reduce prefetching
            std::cmp::max(
                self.config.min_prefetch_count,
                (base_prefetch_count as f64 * 0.75).round() as usize,
            )
        } else if pressure < 0.3 {
            // Low pressure, can increase prefetching
            std::cmp::min(
                self.config.max_prefetch_count,
                (base_prefetch_count as f64 * 1.5).round() as usize,
            )
        } else {
            // Normal pressure, use base count
            base_prefetch_count
        }
    }

    /// Adjust prefetching configuration based on resource pressure.
    pub fn adjust_prefetch_config(&mut self, config: &mut PrefetchConfig) -> bool {
        if !self.config.auto_adjust
            || self.last_adjustment.elapsed() < self.config.adjustment_interval
        {
            return false;
        }

        // Get optimal prefetch count
        let optimal_prefetch_count = self.get_optimal_prefetch_count(config.prefetch_count);

        // Check if we need to adjust
        if optimal_prefetch_count != config.prefetch_count {
            config.prefetch_count = optimal_prefetch_count;
            self.last_adjustment = Instant::now();
            return true;
        }

        false
    }

    /// Get the latest resource snapshot.
    pub fn get_latest_snapshot(&self) -> Option<ResourceSnapshot> {
        self.snapshots.back().cloned()
    }

    /// Get a summary of recent resource usage.
    pub fn get_resource_summary(&self) -> ResourceSummary {
        if self.snapshots.is_empty() {
            return ResourceSummary::default();
        }

        // Calculate averages
        let mut cpu_sum = 0.0;
        let mut memory_pressure_sum = 0.0;
        let mut io_bytes_sum = 0;

        for snapshot in &self.snapshots {
            cpu_sum += snapshot.cpu_usage;
            memory_pressure_sum += snapshot.memory_pressure();
            io_bytes_sum += snapshot.io_bytes_per_sec;
        }

        let count = self.snapshots.len();
        let avg_cpu = cpu_sum / count as f64;
        let avg_memory_pressure = memory_pressure_sum / count as f64;
        let avg_io_bytes = io_bytes_sum / count as u64;

        // Calculate trends (compare recent with older snapshots)
        let trend_duration = if count >= 2 {
            let oldest = &self.snapshots[0];
            let newest = self.snapshots.back().unwrap();

            newest.timestamp.duration_since(oldest.timestamp)
        } else {
            Duration::from_secs(0)
        };

        ResourceSummary {
            avg_cpu_usage: avg_cpu,
            avg_memory_pressure,
            avg_io_bytes_per_sec: avg_io_bytes,
            combined_pressure: self.snapshots.back().unwrap().combined_pressure(),
            snapshot_count: count,
            duration: trend_duration,
            under_pressure: self.under_pressure,
        }
    }
}

/// Summary of resource usage over time.
#[derive(Debug, Clone)]
pub struct ResourceSummary {
    /// Average CPU usage (0.0 to 1.0)
    pub avg_cpu_usage: f64,

    /// Average memory pressure (0.0 to 1.0)
    pub avg_memory_pressure: f64,

    /// Average IO bytes per second
    pub avg_io_bytes_per_sec: u64,

    /// Combined resource pressure (from latest snapshot)
    pub combined_pressure: f64,

    /// Number of snapshots used for the summary
    pub snapshot_count: usize,

    /// Duration covered by the snapshots
    pub duration: Duration,

    /// Whether the system is currently under pressure
    pub under_pressure: bool,
}

impl Default for ResourceSummary {
    fn default() -> Self {
        Self {
            avg_cpu_usage: 0.0,
            avg_memory_pressure: 0.0,
            avg_io_bytes_per_sec: 0,
            combined_pressure: 0.0,
            snapshot_count: 0,
            duration: Duration::from_secs(0),
            under_pressure: false,
        }
    }
}

/// Interface for getting system information.
pub trait SystemInfo {
    /// Get CPU usage (0.0 to 1.0).
    fn get_cpu_usage(&self) -> f64;

    /// Get memory information (usage, available).
    fn get_memory_info(&self) -> (u64, u64);

    /// Get IO statistics (ops/sec, bytes/sec).
    fn get_io_stats(&self) -> (u64, u64);
}

/// Default implementation of system info using sysinfo crate.
pub struct DefaultSystemInfo;

impl SystemInfo for DefaultSystemInfo {
    fn get_cpu_usage(&self) -> f64 {
        // Try to get CPU usage if sysinfo is available
        #[cfg(feature = "sysinfo")]
        {
            use sysinfo::{CpuExt, System, SystemExt};
            let mut system = System::new_all();
            system.refresh_cpu();

            // Calculate average CPU usage across all cores
            let cpu_usage: f64 = system
                .cpus()
                .iter()
                .map(|cpu| cpu.cpu_usage() as f64 / 100.0)
                .sum();
            cpu_usage / system.cpus().len() as f64
        }

        // Fallback to a reasonable estimate
        #[cfg(not(feature = "sysinfo"))]
        {
            // Without sysinfo, use getloadavg() if on Unix-like
            #[cfg(target_family = "unix")]
            {
                let mut loadavg = [0.0, 0.0, 0.0];
                if unsafe { libc::getloadavg(loadavg.as_mut_ptr(), 3) } == 3 {
                    // Normalize load average to 0.0-1.0 range
                    // (assuming a load of 1.0 per CPU core is "fully loaded")
                    let num_cpus = num_cpus::get() as f64;
                    return (loadavg[0] / num_cpus).min(1.0);
                }
            }

            // Fallback value if we can't get actual CPU usage
            0.5
        }
    }

    fn get_memory_info(&self) -> (u64, u64) {
        // Try to get memory info if sysinfo is available
        #[cfg(feature = "sysinfo")]
        {
            use sysinfo::{System, SystemExt};
            let mut system = System::new_all();
            system.refresh_memory();

            (
                system.used_memory() * 1024,
                system.available_memory() * 1024,
            )
        }

        // Fallback to reasonable defaults
        #[cfg(not(feature = "sysinfo"))]
        {
            // Check if we have sys-info available
            #[cfg(feature = "sys-info")]
            {
                if let Ok(mem) = sys_info::mem_info() {
                    let used = (mem.total - mem.free) * 1024;
                    let available = mem.free * 1024;
                    return (used, available);
                }
            }

            // Fallback values if we can't get actual memory info
            // Assume 50% of memory is being used
            (4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024) // 4GB used, 4GB available
        }
    }

    fn get_io_stats(&self) -> (u64, u64) {
        // Try to get IO stats if sysinfo is available
        #[cfg(feature = "sysinfo")]
        {
            use sysinfo::{DiskExt, System, SystemExt};
            let mut system = System::new_all();
            system.refresh_disks_list();

            // Sum IO activity across all disks
            let mut total_ops = 0;
            let mut total_bytes = 0;

            for disk in system.disks() {
                // Simple approximation of IO ops
                total_ops += 1; // Just a placeholder since sysinfo doesn't have this info
                total_bytes += disk.read_bytes() + disk.written_bytes();
            }

            (total_ops, total_bytes)
        }

        // Fallback to reasonable defaults
        #[cfg(not(feature = "sysinfo"))]
        {
            (10, 1024 * 1024) // 10 ops/sec, 1MB/sec
        }
    }
}

/// Resource-aware prefetching manager.
pub struct ResourceAwarePrefetcher {
    /// Resource monitor
    monitor: ResourceMonitor,

    /// Base prefetching configuration
    base_config: PrefetchConfig,

    /// Current prefetching configuration (adjusted for resources)
    current_config: PrefetchConfig,

    /// Whether prefetching is currently enabled
    enabled: bool,

    /// Performance statistics
    performance_stats: Arc<Mutex<PerformanceStats>>,

    /// Last time stats were updated
    last_stats_update: Instant,
}

/// Performance statistics for prefetching.
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    /// Prefetch hit rate
    pub hit_rate: f64,

    /// Average latency for prefetched blocks
    pub prefetch_latency_ns: f64,

    /// Average latency for non-prefetched blocks
    pub non_prefetch_latency_ns: f64,

    /// Number of blocks prefetched
    pub prefetch_count: usize,

    /// Number of blocks accessed
    pub access_count: usize,

    /// Resource summaries at different points in time
    pub resource_snapshots: Vec<(Instant, ResourceSummary)>,
}

impl ResourceAwarePrefetcher {
    /// Create a new resource-aware prefetcher.
    pub fn new(base_config: PrefetchConfig, resource_config: ResourceAwareConfig) -> Self {
        Self {
            monitor: ResourceMonitor::new(resource_config),
            base_config: base_config.clone(),
            current_config: base_config,
            enabled: true,
            performance_stats: Arc::new(Mutex::new(PerformanceStats::default())),
            last_stats_update: Instant::now(),
        }
    }

    /// Update prefetching configuration based on resource pressure.
    pub fn update_config(&mut self) -> bool {
        if !self.enabled {
            return false;
        }

        // Check if we should adjust prefetching based on resources
        let mut config = self.current_config.clone();
        let changed = self.monitor.adjust_prefetch_config(&mut config);

        if changed {
            self.current_config = config;

            // Take a resource snapshot and record it with stats
            if let Some(snapshot) = self.monitor.get_latest_snapshot() {
                let summary = self.monitor.get_resource_summary();
                if let Ok(mut stats) = self.performance_stats.lock() {
                    stats.resource_snapshots.push((Instant::now(), summary));

                    // Limit the number of snapshots
                    while stats.resource_snapshots.len() > 10 {
                        stats.resource_snapshots.remove(0);
                    }
                }
            }
        }

        changed
    }

    /// Record performance data from prefetching.
    pub fn record_performance(
        &mut self,
        prefetch_stats: &PrefetchStats,
        latency_ns: f64,
        is_prefetched: bool,
    ) {
        if let Ok(mut stats) = self.performance_stats.lock() {
            // Update overall stats
            stats.hit_rate = prefetch_stats.hit_rate;
            stats.prefetch_count = prefetch_stats.prefetch_count;
            stats.access_count = prefetch_stats.prefetch_hits + prefetch_stats.prefetch_misses;

            // Update latency based on whether this was a prefetched block
            if is_prefetched {
                // Moving average for prefetch latency
                if stats.prefetch_latency_ns == 0.0 {
                    stats.prefetch_latency_ns = latency_ns;
                } else {
                    stats.prefetch_latency_ns = stats.prefetch_latency_ns * 0.9 + latency_ns * 0.1;
                }
            } else {
                // Moving average for non-prefetch latency
                if stats.non_prefetch_latency_ns == 0.0 {
                    stats.non_prefetch_latency_ns = latency_ns;
                } else {
                    stats.non_prefetch_latency_ns =
                        stats.non_prefetch_latency_ns * 0.9 + latency_ns * 0.1;
                }
            }
        }

        // Take resource snapshots periodically
        if self.last_stats_update.elapsed() >= Duration::from_secs(5) {
            self.last_stats_update = Instant::now();

            // Take a snapshot
            let summary = self.monitor.get_resource_summary();
            if let Ok(mut stats) = self.performance_stats.lock() {
                stats.resource_snapshots.push((Instant::now(), summary));

                // Limit the number of snapshots
                while stats.resource_snapshots.len() > 10 {
                    stats.resource_snapshots.remove(0);
                }
            }
        }
    }

    /// Get the current prefetching configuration.
    pub fn get_current_config(&self) -> PrefetchConfig {
        self.current_config.clone()
    }

    /// Get the base prefetching configuration.
    pub fn get_base_config(&self) -> PrefetchConfig {
        self.base_config.clone()
    }

    /// Get a snapshot of the current resource usage.
    pub fn take_resource_snapshot(&mut self) -> ResourceSnapshot {
        self.monitor.take_snapshot()
    }

    /// Get a summary of resource usage.
    pub fn get_resource_summary(&self) -> ResourceSummary {
        self.monitor.get_resource_summary()
    }

    /// Get the performance statistics.
    pub fn get_performance_stats(&self) -> PerformanceStats {
        if let Ok(stats) = self.performance_stats.lock() {
            stats.clone()
        } else {
            PerformanceStats::default()
        }
    }

    /// Check if the system is under pressure.
    pub fn is_under_pressure(&mut self) -> bool {
        self.monitor.is_under_pressure()
    }

    /// Enable or disable prefetching.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if prefetching is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the optimal prefetch count based on current resources.
    pub fn get_optimal_prefetch_count(&mut self) -> usize {
        self.monitor
            .get_optimal_prefetch_count(self.base_config.prefetch_count)
    }

    /// Reset the prefetching configuration to the base configuration.
    pub fn reset_config(&mut self) {
        self.current_config = self.base_config.clone();
    }
}

/// Enhanced prefetching configuration with resource awareness.
#[derive(Debug, Clone)]
pub struct ResourceAwarePrefetchingConfig {
    /// Base prefetching configuration
    pub base_config: PrefetchConfig,

    /// Resource awareness configuration
    pub resource_config: ResourceAwareConfig,
}

impl ResourceAwarePrefetchingConfig {
    /// Create a new resource-aware prefetching configuration.
    pub fn new(base_config: PrefetchConfig, resource_config: ResourceAwareConfig) -> Self {
        Self {
            base_config,
            resource_config,
        }
    }

    /// Create a resource-aware prefetcher from this configuration.
    pub fn create_prefetcher(&self) -> ResourceAwarePrefetcher {
        ResourceAwarePrefetcher::new(self.base_config.clone(), self.resource_config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Mock implementation of SystemInfo for testing
    struct MockSystemInfo {
        cpu_usage: f64,
        memory_used: u64,
        memory_available: u64,
        io_ops: u64,
        io_bytes: u64,
    }

    impl SystemInfo for MockSystemInfo {
        fn get_cpu_usage(&self) -> f64 {
            self.cpu_usage
        }

        fn get_memory_info(&self) -> (u64, u64) {
            (self.memory_used, self.memory_available)
        }

        fn get_io_stats(&self) -> (u64, u64) {
            (self.io_ops, self.io_bytes)
        }
    }

    impl MockSystemInfo {
        fn new(cpu: f64, mem_used: u64, mem_avail: u64, io_ops: u64, io_bytes: u64) -> Self {
            Self {
                cpu_usage: cpu,
                memory_used: mem_used,
                memory_available: mem_avail,
                io_ops,
                io_bytes,
            }
        }
    }

    #[test]
    fn test_resource_snapshot() {
        let snapshot = ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage: 0.7,
            memory_usage: 8 * 1024 * 1024 * 1024,     // 8 GB
            memory_available: 8 * 1024 * 1024 * 1024, // 8 GB
            io_ops_per_sec: 100,
            io_bytes_per_sec: 10 * 1024 * 1024, // 10 MB/s
        };

        // Check memory pressure
        assert_eq!(snapshot.memory_pressure(), 0.5); // 8GB / (8GB + 8GB) = 0.5

        // Check combined pressure
        let combined = snapshot.combined_pressure();
        assert!(combined > 0.0 && combined < 1.0);
    }

    #[test]
    fn test_optimal_prefetch_count() {
        // Create a resource monitor with custom system info
        let config = ResourceAwareConfig {
            auto_adjust: true,
            min_prefetch_count: 1,
            max_prefetch_count: 10,
            ..Default::default()
        };

        let mut monitor = ResourceMonitor::new(config);

        // Replace the system info with our mock
        monitor.sys_info = Box::new(MockSystemInfo::new(
            0.2,                     // Low CPU usage
            2 * 1024 * 1024 * 1024,  // 2 GB used memory
            14 * 1024 * 1024 * 1024, // 14 GB available memory
            10,                      // 10 IO ops/sec
            1 * 1024 * 1024,         // 1 MB/s IO
        ));

        // Take a snapshot
        monitor.take_snapshot();

        // Check optimal prefetch count with low pressure
        let base_count = 4;
        let optimal = monitor.get_optimal_prefetch_count(base_count);
        assert!(optimal >= base_count); // Should be the same or higher under low pressure

        // Now test with high pressure
        monitor.sys_info = Box::new(MockSystemInfo::new(
            0.9,                     // High CPU usage
            14 * 1024 * 1024 * 1024, // 14 GB used memory
            2 * 1024 * 1024 * 1024,  // 2 GB available memory
            1000,                    // 1000 IO ops/sec
            100 * 1024 * 1024,       // 100 MB/s IO
        ));

        // Take a snapshot
        monitor.take_snapshot();

        // Check optimal prefetch count with high pressure
        let optimal = monitor.get_optimal_prefetch_count(base_count);
        assert!(optimal <= base_count); // Should be lower under high pressure
    }

    #[test]
    fn test_resource_aware_prefetcher() {
        // Create a resource-aware prefetcher
        let base_config = PrefetchConfig {
            prefetch_count: 5,
            ..Default::default()
        };

        let resource_config = ResourceAwareConfig {
            auto_adjust: true,
            min_prefetch_count: 1,
            max_prefetch_count: 10,
            ..Default::default()
        };

        let mut prefetcher = ResourceAwarePrefetcher::new(base_config, resource_config);

        // Record some performance data
        let stats = PrefetchStats {
            prefetch_count: 100,
            prefetch_hits: 80,
            prefetch_misses: 20,
            hit_rate: 0.8,
        };

        prefetcher.record_performance(&stats, 500_000.0, true); // 500µs latency, prefetched
        prefetcher.record_performance(&stats, 2_000_000.0, false); // 2ms latency, not prefetched

        // Get the performance stats
        let perf_stats = prefetcher.get_performance_stats();
        assert_eq!(perf_stats.hit_rate, 0.8);
        assert!(perf_stats.prefetch_latency_ns > 0.0);
        assert!(perf_stats.non_prefetch_latency_ns > 0.0);
        assert!(perf_stats.non_prefetch_latency_ns > perf_stats.prefetch_latency_ns);
        // Non-prefetched should be slower
    }
}
