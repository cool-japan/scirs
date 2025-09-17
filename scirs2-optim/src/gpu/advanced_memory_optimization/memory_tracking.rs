//! Memory usage tracking and profiling for GPU memory optimization
//!
//! This module provides comprehensive memory tracking capabilities including
//! usage monitoring, allocation event tracking, and memory pressure analysis.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::config::{AllocationType, PressureAction};

/// Memory usage tracking and profiling
#[derive(Debug, Clone, Default)]
pub struct MemoryUsageTracker {
    /// Current memory usage in bytes
    pub current_usage: usize,

    /// Peak memory usage since last reset
    pub peak_usage: usize,

    /// Memory usage history (limited to last N snapshots)
    pub usage_history: VecDeque<MemorySnapshot>,

    /// Memory allocation events (limited to last N events)
    pub allocation_events: VecDeque<AllocationEvent>,

    /// Memory pressure events (limited to last N events)
    pub pressure_events: VecDeque<PressureEvent>,

    /// Total GPU memory available
    pub total_gpu_memory: usize,

    /// Memory fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f32,

    /// Last memory synchronization timestamp
    pub last_sync: Instant,

    /// Maximum history size for each type of event
    max_history_size: usize,

    /// Minimum time between memory updates (to avoid overhead)
    min_update_interval: Duration,
}

impl MemoryUsageTracker {
    /// Create a new memory usage tracker
    pub fn new(total_gpu_memory: usize) -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            usage_history: VecDeque::new(),
            allocation_events: VecDeque::new(),
            pressure_events: VecDeque::new(),
            total_gpu_memory,
            fragmentation_ratio: 0.0,
            last_sync: Instant::now(),
            max_history_size: 1000, // Keep last 1000 events
            min_update_interval: Duration::from_millis(10), // Update at most every 10ms
        }
    }

    /// Update current memory usage
    pub fn update_usage(&mut self, new_usage: usize) {
        let now = Instant::now();

        // Rate limit updates to avoid overhead
        if now.duration_since(self.last_sync) < self.min_update_interval {
            return;
        }

        self.current_usage = new_usage;
        if new_usage > self.peak_usage {
            self.peak_usage = new_usage;
        }

        // Create snapshot
        let snapshot = MemorySnapshot {
            timestamp: now,
            usage_bytes: new_usage,
            pressure_level: self.calculate_pressure_level(),
            batch_size: 0, // This would be set externally
            active_checkpoints: 0, // This would be set externally
            offloaded_params: 0, // This would be set externally
        };

        self.add_snapshot(snapshot);
        self.last_sync = now;
    }

    /// Record an allocation event
    pub fn record_allocation(
        &mut self,
        size: usize,
        allocation_type: AllocationType,
        success: bool,
        latency: Duration,
    ) {
        let event = AllocationEvent {
            size,
            allocation_type,
            timestamp: Instant::now(),
            success,
            latency,
            pressure_level: self.calculate_pressure_level(),
        };

        self.add_allocation_event(event);

        // Update current usage if allocation was successful
        if success {
            self.update_usage(self.current_usage + size);
        }
    }

    /// Record a memory pressure event
    pub fn record_pressure_event(&mut self, action: PressureAction, memory_freed: usize) {
        let event = PressureEvent {
            timestamp: Instant::now(),
            pressure_level: self.calculate_pressure_level(),
            action,
            memory_freed,
        };

        self.add_pressure_event(event);

        // Update current usage if memory was freed
        if memory_freed > 0 {
            self.current_usage = self.current_usage.saturating_sub(memory_freed);
        }
    }

    /// Calculate current memory pressure level (0.0-1.0)
    pub fn calculate_pressure_level(&self) -> f32 {
        if self.total_gpu_memory == 0 {
            return 0.0;
        }
        (self.current_usage as f32 / self.total_gpu_memory as f32).min(1.0)
    }

    /// Get memory utilization percentage
    pub fn utilization_percentage(&self) -> f32 {
        self.calculate_pressure_level() * 100.0
    }

    /// Get memory statistics for the last N minutes
    pub fn get_statistics(&self, duration: Duration) -> MemoryStatistics {
        let cutoff_time = Instant::now() - duration;

        let recent_snapshots: Vec<&MemorySnapshot> = self
            .usage_history
            .iter()
            .filter(|s| s.timestamp >= cutoff_time)
            .collect();

        let recent_allocations: Vec<&AllocationEvent> = self
            .allocation_events
            .iter()
            .filter(|e| e.timestamp >= cutoff_time)
            .collect();

        if recent_snapshots.is_empty() {
            return MemoryStatistics::default();
        }

        let total_snapshots = recent_snapshots.len();
        let avg_usage = recent_snapshots.iter().map(|s| s.usage_bytes).sum::<usize>() / total_snapshots;
        let max_usage = recent_snapshots.iter().map(|s| s.usage_bytes).max().unwrap_or(0);
        let min_usage = recent_snapshots.iter().map(|s| s.usage_bytes).min().unwrap_or(0);

        let successful_allocations = recent_allocations.iter().filter(|e| e.success).count();
        let failed_allocations = recent_allocations.iter().filter(|e| !e.success).count();

        let avg_allocation_latency = if !recent_allocations.is_empty() {
            recent_allocations.iter().map(|e| e.latency).sum::<Duration>()
                / recent_allocations.len() as u32
        } else {
            Duration::ZERO
        };

        MemoryStatistics {
            avg_usage,
            max_usage,
            min_usage,
            current_usage: self.current_usage,
            peak_usage: self.peak_usage,
            pressure_level: self.calculate_pressure_level(),
            fragmentation_ratio: self.fragmentation_ratio,
            successful_allocations,
            failed_allocations,
            avg_allocation_latency,
            total_snapshots,
        }
    }

    /// Reset peak usage statistics
    pub fn reset_peak_usage(&mut self) {
        self.peak_usage = self.current_usage;
    }

    /// Clear all history (for memory conservation)
    pub fn clear_history(&mut self) {
        self.usage_history.clear();
        self.allocation_events.clear();
        self.pressure_events.clear();
    }

    /// Set configuration parameters
    pub fn configure(&mut self, max_history_size: usize, min_update_interval: Duration) {
        self.max_history_size = max_history_size;
        self.min_update_interval = min_update_interval;

        // Trim history if necessary
        self.trim_history();
    }

    // Private helper methods

    fn add_snapshot(&mut self, snapshot: MemorySnapshot) {
        self.usage_history.push_back(snapshot);
        if self.usage_history.len() > self.max_history_size {
            self.usage_history.pop_front();
        }
    }

    fn add_allocation_event(&mut self, event: AllocationEvent) {
        self.allocation_events.push_back(event);
        if self.allocation_events.len() > self.max_history_size {
            self.allocation_events.pop_front();
        }
    }

    fn add_pressure_event(&mut self, event: PressureEvent) {
        self.pressure_events.push_back(event);
        if self.pressure_events.len() > self.max_history_size {
            self.pressure_events.pop_front();
        }
    }

    fn trim_history(&mut self) {
        while self.usage_history.len() > self.max_history_size {
            self.usage_history.pop_front();
        }
        while self.allocation_events.len() > self.max_history_size {
            self.allocation_events.pop_front();
        }
        while self.pressure_events.len() > self.max_history_size {
            self.pressure_events.pop_front();
        }
    }
}

/// Memory usage snapshot at a point in time
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Timestamp when snapshot was taken
    pub timestamp: Instant,

    /// Memory usage in bytes
    pub usage_bytes: usize,

    /// Memory pressure level (0.0-1.0)
    pub pressure_level: f32,

    /// Active batch size at the time
    pub batch_size: usize,

    /// Number of active checkpoints
    pub active_checkpoints: usize,

    /// Number of offloaded parameters
    pub offloaded_params: usize,
}

/// Memory allocation event for profiling
#[derive(Debug, Clone)]
pub struct AllocationEvent {
    /// Size of allocation in bytes
    pub size: usize,

    /// Type of allocation
    pub allocation_type: AllocationType,

    /// Timestamp of allocation
    pub timestamp: Instant,

    /// Whether allocation was successful
    pub success: bool,

    /// Time taken for allocation
    pub latency: Duration,

    /// Memory pressure level at time of allocation
    pub pressure_level: f32,
}

/// Memory pressure event
#[derive(Debug, Clone)]
pub struct PressureEvent {
    /// Timestamp of event
    pub timestamp: Instant,

    /// Memory pressure level (0.0-1.0)
    pub pressure_level: f32,

    /// Action taken in response to pressure
    pub action: PressureAction,

    /// Amount of memory freed (bytes)
    pub memory_freed: usize,
}

/// Memory statistics over a time period
#[derive(Debug, Clone, Default)]
pub struct MemoryStatistics {
    /// Average memory usage (bytes)
    pub avg_usage: usize,

    /// Maximum memory usage (bytes)
    pub max_usage: usize,

    /// Minimum memory usage (bytes)
    pub min_usage: usize,

    /// Current memory usage (bytes)
    pub current_usage: usize,

    /// Peak memory usage since last reset (bytes)
    pub peak_usage: usize,

    /// Current pressure level (0.0-1.0)
    pub pressure_level: f32,

    /// Memory fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f32,

    /// Number of successful allocations
    pub successful_allocations: usize,

    /// Number of failed allocations
    pub failed_allocations: usize,

    /// Average allocation latency
    pub avg_allocation_latency: Duration,

    /// Total number of snapshots analyzed
    pub total_snapshots: usize,
}

impl MemoryStatistics {
    /// Get allocation success rate
    pub fn allocation_success_rate(&self) -> f32 {
        let total = self.successful_allocations + self.failed_allocations;
        if total == 0 {
            return 1.0;
        }
        self.successful_allocations as f32 / total as f32
    }

    /// Check if memory usage is stable (low variance)
    pub fn is_usage_stable(&self) -> bool {
        if self.max_usage == self.min_usage {
            return true;
        }
        let variance_ratio = (self.max_usage - self.min_usage) as f32 / self.avg_usage as f32;
        variance_ratio < 0.1 // Less than 10% variance
    }
}

/// Memory profiler for detailed analysis
#[derive(Debug, Clone, Default)]
pub struct MemoryProfiler {
    /// Allocation patterns by type
    allocation_patterns: std::collections::HashMap<AllocationType, AllocationPattern>,

    /// Memory hotspots (frequently allocated sizes)
    size_hotspots: std::collections::HashMap<usize, usize>,

    /// Temporal allocation patterns
    temporal_patterns: VecDeque<TemporalPattern>,

    /// Enable detailed profiling (may impact performance)
    detailed_profiling: bool,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable detailed profiling
    pub fn set_detailed_profiling(&mut self, enabled: bool) {
        self.detailed_profiling = enabled;
    }

    /// Record an allocation for profiling
    pub fn record_allocation(&mut self, event: &AllocationEvent) {
        if !self.detailed_profiling {
            return;
        }

        // Update allocation patterns
        let pattern = self.allocation_patterns
            .entry(event.allocation_type)
            .or_insert_with(AllocationPattern::default);
        pattern.record_allocation(event);

        // Update size hotspots
        *self.size_hotspots.entry(event.size).or_insert(0) += 1;

        // Record temporal pattern
        let temporal = TemporalPattern {
            timestamp: event.timestamp,
            allocation_type: event.allocation_type,
            size: event.size,
            success: event.success,
        };
        self.temporal_patterns.push_back(temporal);

        // Limit temporal pattern history
        if self.temporal_patterns.len() > 10000 {
            self.temporal_patterns.pop_front();
        }
    }

    /// Get profiling summary
    pub fn get_summary(&self) -> ProfilingSummary {
        ProfilingSummary {
            allocation_patterns: self.allocation_patterns.clone(),
            top_size_hotspots: self.get_top_size_hotspots(10),
            total_allocations: self.temporal_patterns.len(),
            detailed_profiling_enabled: self.detailed_profiling,
        }
    }

    fn get_top_size_hotspots(&self, count: usize) -> Vec<(usize, usize)> {
        let mut hotspots: Vec<_> = self.size_hotspots.iter().collect();
        hotspots.sort_by(|a, b| b.1.cmp(a.1));
        hotspots.into_iter().take(count).map(|(&size, &freq)| (size, freq)).collect()
    }
}

/// Allocation pattern analysis
#[derive(Debug, Clone, Default)]
pub struct AllocationPattern {
    /// Total number of allocations
    pub total_count: usize,

    /// Successful allocations
    pub success_count: usize,

    /// Total bytes allocated
    pub total_bytes: usize,

    /// Average allocation size
    pub avg_size: usize,

    /// Average allocation latency
    pub avg_latency: Duration,

    /// Peak allocation rate (allocations per second)
    pub peak_rate: f32,
}

impl AllocationPattern {
    fn record_allocation(&mut self, event: &AllocationEvent) {
        self.total_count += 1;
        if event.success {
            self.success_count += 1;
            self.total_bytes += event.size;
        }

        // Update average size
        if self.success_count > 0 {
            self.avg_size = self.total_bytes / self.success_count;
        }

        // Update average latency
        if self.total_count == 1 {
            self.avg_latency = event.latency;
        } else {
            let total_latency = self.avg_latency * (self.total_count - 1) as u32 + event.latency;
            self.avg_latency = total_latency / self.total_count as u32;
        }
    }
}

/// Temporal allocation pattern
#[derive(Debug, Clone)]
struct TemporalPattern {
    timestamp: Instant,
    allocation_type: AllocationType,
    size: usize,
    success: bool,
}

/// Profiling summary
#[derive(Debug, Clone)]
pub struct ProfilingSummary {
    /// Allocation patterns by type
    pub allocation_patterns: std::collections::HashMap<AllocationType, AllocationPattern>,

    /// Top size hotspots (size, frequency)
    pub top_size_hotspots: Vec<(usize, usize)>,

    /// Total number of tracked allocations
    pub total_allocations: usize,

    /// Whether detailed profiling is enabled
    pub detailed_profiling_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tracker_basic() {
        let mut tracker = MemoryUsageTracker::new(1024 * 1024 * 1024); // 1GB

        tracker.update_usage(512 * 1024 * 1024); // 512MB
        assert_eq!(tracker.current_usage, 512 * 1024 * 1024);
        assert_eq!(tracker.peak_usage, 512 * 1024 * 1024);
        assert!((tracker.calculate_pressure_level() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_allocation_recording() {
        let mut tracker = MemoryUsageTracker::new(1024 * 1024 * 1024);

        tracker.record_allocation(
            1024,
            AllocationType::Parameters,
            true,
            Duration::from_millis(1),
        );

        assert_eq!(tracker.allocation_events.len(), 1);
        assert_eq!(tracker.current_usage, 1024);
    }

    #[test]
    fn test_memory_statistics() {
        let mut tracker = MemoryUsageTracker::new(1024 * 1024 * 1024);

        // Record some usage patterns
        for i in 0..10 {
            tracker.update_usage(i * 1024);
            std::thread::sleep(Duration::from_millis(1));
        }

        let stats = tracker.get_statistics(Duration::from_secs(1));
        assert!(stats.total_snapshots > 0);
        assert!(stats.max_usage >= stats.min_usage);
    }

    #[test]
    fn test_profiler() {
        let mut profiler = MemoryProfiler::new();
        profiler.set_detailed_profiling(true);

        let event = AllocationEvent {
            size: 1024,
            allocation_type: AllocationType::Parameters,
            timestamp: Instant::now(),
            success: true,
            latency: Duration::from_millis(1),
            pressure_level: 0.5,
        };

        profiler.record_allocation(&event);

        let summary = profiler.get_summary();
        assert!(summary.allocation_patterns.contains_key(&AllocationType::Parameters));
        assert_eq!(summary.total_allocations, 1);
    }
}