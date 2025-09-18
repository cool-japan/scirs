//! Data structures for profiling entries (timing and memory)

use std::time::Duration;

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
    pub fn new(duration: Duration, parent: Option<&str>) -> Self {
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
    pub fn add_measurement(&mut self, duration: Duration) {
        self.calls += 1;
        self.total_duration += duration;
        self.min_duration = std::cmp::min(self.min_duration, duration);
        self.max_duration = std::cmp::max(self.max_duration, duration);
    }

    /// Add a child operation
    pub fn add_child(&mut self, child: &str) {
        if !self.children.contains(&child.to_string()) {
            self.children.push(child.to_string());
        }
    }

    /// Get the average duration
    pub fn average_duration(&self) -> Duration {
        if self.calls == 0 {
            Duration::from_secs(0)
        } else {
            self.total_duration / self.calls as u32
        }
    }

    /// Get the number of calls
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Get the total duration
    pub fn total_duration(&self) -> Duration {
        self.total_duration
    }

    /// Get the minimum duration
    pub fn min_duration(&self) -> Duration {
        self.min_duration
    }

    /// Get the maximum duration
    pub fn max_duration(&self) -> Duration {
        self.max_duration
    }

    /// Get the parent operation name
    pub fn parent(&self) -> Option<&str> {
        self.parent.as_deref()
    }

    /// Get the child operations
    pub fn children(&self) -> &[String] {
        &self.children
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
    pub fn new(delta: usize) -> Self {
        Self {
            allocations: 1,
            total_delta: delta as isize,
            max_delta: delta,
        }
    }

    /// Add a new memory measurement
    pub fn add_measurement(&mut self, delta: usize) {
        self.allocations += 1;
        self.total_delta += delta as isize;
        self.max_delta = std::cmp::max(self.max_delta, delta);
    }

    /// Get the average memory delta
    pub fn average_delta(&self) -> f64 {
        if self.allocations == 0 {
            0.0
        } else {
            self.total_delta as f64 / self.allocations as f64
        }
    }

    /// Get the number of allocations
    pub fn allocations(&self) -> usize {
        self.allocations
    }

    /// Get the total memory delta
    pub fn total_delta(&self) -> isize {
        self.total_delta
    }

    /// Get the maximum memory delta
    pub fn max_delta(&self) -> usize {
        self.max_delta
    }
}
