//! Memory metrics collection and analysis
//!
//! This module provides functionality for collecting and analyzing memory usage metrics.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::memory::metrics::event::{MemoryEvent, MemoryEventType};
use rand::prelude::*;
use rand::Rng;
#[cfg(feature = "memory_metrics")]
use serde::{Deserialize, Serialize};

// Define a simple Random struct for sampling when the random feature is not enabled
struct Random {
    rng: StdRng,
}

impl Default for Random {
    fn default() -> Self {
        Self {
            rng: StdRng::seed_from_u64(0), // Use a fixed seed for simplicity
        }
    }
}

impl Random {
    fn gen_range(&mut self, range: std::ops::Range<f64>) -> f64 {
        self.rng.gen_range(range)
    }
}

/// Memory metrics configuration
#[derive(Debug, Clone)]
pub struct MemoryMetricsConfig {
    /// Whether to collect events
    pub enabled: bool,
    /// Whether to capture call stacks (requires memory_call_stack feature)
    pub capture_call_stacks: bool,
    /// Maximum number of events to store
    pub max_events: usize,
    /// Whether to aggregate events in real-time
    pub real_time_aggregation: bool,
    /// Event sampling rate (1.0 = all events, 0.1 = 10% of events)
    pub samplingrate: f64,
}

impl Default for MemoryMetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            capture_call_stacks: cfg!(feature = "memory_call_stack"),
            max_events: 10000,
            real_time_aggregation: true,
            samplingrate: 1.0,
        }
    }
}

/// Allocation statistics for a component
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Number of allocations
    pub count: usize,
    /// Total bytes allocated
    pub total_bytes: usize,
    /// Average allocation size
    pub average_size: f64,
    /// Peak memory usage
    pub peak_usage: usize,
}

/// Component memory statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "memory_metrics", derive(Serialize, Deserialize))]
pub struct ComponentMemoryStats {
    /// Current memory usage
    pub current_usage: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of allocations
    pub allocation_count: usize,
    /// Total bytes allocated (including released memory)
    pub total_allocated: usize,
    /// Average allocation size
    pub avg_allocation_size: f64,
}

/// Memory usage report
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "memory_metrics",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct MemoryReport {
    /// Total current memory usage across all components
    pub total_current_usage: usize,
    /// Total peak memory usage across all components
    pub total_peak_usage: usize,
    /// Total number of allocations across all components
    pub total_allocation_count: usize,
    /// Total bytes allocated across all components
    pub total_allocated_bytes: usize,
    /// Component-specific statistics
    pub component_stats: HashMap<String, ComponentMemoryStats>,
    /// Duration since tracking started
    pub duration: Duration,
}

/// Memory metrics collector for tracking and analyzing memory usage
pub struct MemoryMetricsCollector {
    /// Configuration
    config: MemoryMetricsConfig,
    /// Collected events (if not aggregated)
    events: RwLock<VecDeque<MemoryEvent>>,
    /// Current memory usage by component
    current_usage: RwLock<HashMap<String, usize>>,
    /// Peak memory usage by component
    peak_usage: RwLock<HashMap<String, usize>>,
    /// Allocation count by component
    allocation_count: RwLock<HashMap<String, usize>>,
    /// Total allocated bytes by component
    total_allocated: RwLock<HashMap<String, usize>>,
    /// Average allocation size by component
    avg_allocation_size: RwLock<HashMap<String, f64>>,
    /// Start time
    start_time: Instant,
    /// Random number generator for sampling
    rng: Mutex<Random>,
}

impl MemoryMetricsCollector {
    /// Create a new memory metrics collector
    pub fn new(config: MemoryMetricsConfig) -> Self {
        Self {
            config,
            events: RwLock::new(VecDeque::with_capacity(1000)),
            current_usage: RwLock::new(HashMap::new()),
            peak_usage: RwLock::new(HashMap::new()),
            allocation_count: RwLock::new(HashMap::new()),
            total_allocated: RwLock::new(HashMap::new()),
            avg_allocation_size: RwLock::new(HashMap::new()),
            start_time: Instant::now(),
            rng: Mutex::new(Random::default()),
        }
    }

    /// Record a memory event
    pub fn record_event(&self, event: MemoryEvent) {
        if !self.config.enabled {
            return;
        }

        // Sample events if sampling rate < 1.0
        if self.config.samplingrate < 1.0 {
            let mut rng = self.rng.lock().unwrap();
            if rng.gen_range(0.0..1.0) > self.config.samplingrate {
                return;
            }
        }

        // Update aggregated metrics in real-time if enabled
        if self.config.real_time_aggregation {
            self.update_metrics(&event);
        }

        // Store the event if we're keeping raw events
        if self.config.max_events > 0 {
            let mut events = self.events.write().unwrap();
            events.push_back(event);

            // Limit the number of stored events
            while events.len() > self.config.max_events {
                events.pop_front();
            }
        }
    }

    /// Update aggregated metrics based on an event
    fn update_metrics(&self, event: &MemoryEvent) {
        match event.event_type {
            MemoryEventType::Allocation => {
                // Update current usage
                let mut current_usage = self.current_usage.write().unwrap();
                let component_usage = current_usage.entry(event.component.clone()).or_insert(0);
                *component_usage += event.size;

                // Update peak usage if current > peak
                let mut peak_usage = self.peak_usage.write().unwrap();
                let peak = peak_usage.entry(event.component.clone()).or_insert(0);
                *peak = (*peak).max(*component_usage);

                // Update allocation count
                let mut allocation_count = self.allocation_count.write().unwrap();
                let count = allocation_count.entry(event.component.clone()).or_insert(0);
                *count += 1;

                // Update total allocated
                let mut total_allocated = self.total_allocated.write().unwrap();
                let total = total_allocated.entry(event.component.clone()).or_insert(0);
                *total += event.size;

                // Update average allocation size
                let mut avg_allocation_size = self.avg_allocation_size.write().unwrap();
                let avg = avg_allocation_size
                    .entry(event.component.clone())
                    .or_insert(0.0);
                *avg = (*avg * (*count as f64 - 1.0) + event.size as f64) / *count as f64;
            }
            MemoryEventType::Deallocation => {
                // Update current usage
                let mut current_usage = self.current_usage.write().unwrap();
                let component_usage = current_usage.entry(event.component.clone()).or_insert(0);
                *component_usage = component_usage.saturating_sub(event.size);
            }
            MemoryEventType::Resize => {
                // Handle resize events (could be positive or negative change)
                if let Some(old_size) = event
                    .metadata
                    .get("old_size")
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    let size_diff = event.size as isize - old_size as isize;

                    let mut current_usage = self.current_usage.write().unwrap();
                    let component_usage = current_usage.entry(event.component.clone()).or_insert(0);

                    if size_diff > 0 {
                        *component_usage += size_diff as usize;
                    } else {
                        *component_usage = component_usage.saturating_sub((-size_diff) as usize);
                    }

                    // Update peak usage if needed
                    let mut peak_usage = self.peak_usage.write().unwrap();
                    let peak = peak_usage.entry(event.component.clone()).or_insert(0);
                    *peak = (*peak).max(*component_usage);
                }
            }
            MemoryEventType::Access | MemoryEventType::Transfer => {
                // These event types don't affect memory usage metrics
            }
        }
    }

    /// Get the current memory usage for a specific component
    pub fn get_current_usage(&self, component: &str) -> usize {
        let current_usage = self.current_usage.read().unwrap();
        *current_usage.get(component).unwrap_or(&0)
    }

    /// Get the peak memory usage for a specific component
    pub fn get_peak_usage(&self, component: &str) -> usize {
        let peak_usage = self.peak_usage.read().unwrap();
        *peak_usage.get(component).unwrap_or(&0)
    }

    /// Get total memory usage across all components
    pub fn get_total_current_usage(&self) -> usize {
        let current_usage = self.current_usage.read().unwrap();
        current_usage.values().sum()
    }

    /// Get peak memory usage across all components
    pub fn get_total_peak_usage(&self) -> usize {
        let peak_usage = self.peak_usage.read().unwrap();

        // Either sum of component peaks or peak of total current usage
        let component_sum: usize = peak_usage.values().sum();

        // In a real implementation, we'd track the total peak as well
        // For simplicity, just return the sum of component peaks
        component_sum
    }

    /// Get allocation statistics for a component
    pub fn get_allocation_stats(&self, component: &str) -> Option<AllocationStats> {
        let allocation_count = self.allocation_count.read().unwrap();
        let count = *allocation_count.get(component)?;

        let total_allocated = self.total_allocated.read().unwrap();
        let total = *total_allocated.get(component)?;

        let avg_allocation_size = self.avg_allocation_size.read().unwrap();
        let avg = *avg_allocation_size.get(component)?;

        let peak_usage = self.peak_usage.read().unwrap();
        let peak = *peak_usage.get(component)?;

        Some(AllocationStats {
            count,
            total_bytes: total,
            average_size: avg,
            peak_usage: peak,
        })
    }

    /// Generate a memory report
    pub fn generate_report(&self) -> MemoryReport {
        let current_usage = self.current_usage.read().unwrap();
        let peak_usage = self.peak_usage.read().unwrap();
        let allocation_count = self.allocation_count.read().unwrap();
        let total_allocated = self.total_allocated.read().unwrap();
        let avg_allocation_size = self.avg_allocation_size.read().unwrap();

        let mut component_stats = HashMap::new();

        // Collect all component names from all maps
        let mut components = std::collections::HashSet::new();
        components.extend(current_usage.keys().cloned());
        components.extend(peak_usage.keys().cloned());
        components.extend(allocation_count.keys().cloned());

        // Build component stats
        for component in components {
            let stats = ComponentMemoryStats {
                current_usage: *current_usage.get(&component).unwrap_or(&0),
                peak_usage: *peak_usage.get(&component).unwrap_or(&0),
                allocation_count: *allocation_count.get(&component).unwrap_or(&0),
                total_allocated: *total_allocated.get(&component).unwrap_or(&0),
                avg_allocation_size: *avg_allocation_size.get(&component).unwrap_or(&0.0),
            };

            component_stats.insert(component, stats);
        }

        MemoryReport {
            total_current_usage: current_usage.values().sum(),
            total_peak_usage: self.get_total_peak_usage(),
            total_allocation_count: allocation_count.values().sum(),
            total_allocated_bytes: total_allocated.values().sum(),
            component_stats,
            duration: self.start_time.elapsed(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        let mut events = self.events.write().unwrap();
        events.clear();

        let mut current_usage = self.current_usage.write().unwrap();
        current_usage.clear();

        let mut peak_usage = self.peak_usage.write().unwrap();
        peak_usage.clear();

        let mut allocation_count = self.allocation_count.write().unwrap();
        allocation_count.clear();

        let mut total_allocated = self.total_allocated.write().unwrap();
        total_allocated.clear();

        let mut avg_allocation_size = self.avg_allocation_size.write().unwrap();
        avg_allocation_size.clear();
    }

    /// Get all recorded events
    pub fn get_events(&self) -> Vec<MemoryEvent> {
        let events = self.events.read().unwrap();
        events.iter().cloned().collect()
    }

    /// Export the report as JSON (avoiding serialization of non-serializable fields)
    pub fn to_json(&self) -> serde_json::Value {
        // Generate a report first to get all the computed values
        let report = self.generate_report();

        let mut json_obj = serde_json::Map::new();

        json_obj.insert(
            "total_allocation_count".to_string(),
            serde_json::Value::Number(report.total_allocation_count.into()),
        );
        json_obj.insert(
            "total_peak_usage".to_string(),
            serde_json::Value::Number(report.total_peak_usage.into()),
        );
        json_obj.insert(
            "total_current_usage".to_string(),
            serde_json::Value::Number(report.total_current_usage.into()),
        );
        json_obj.insert(
            "total_allocated_bytes".to_string(),
            serde_json::Value::Number(report.total_allocated_bytes.into()),
        );

        // Serialize component stats manually
        let component_stats: serde_json::Value = report
            .component_stats
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    serde_json::json!({
                        "current_usage": v.current_usage,
                        "peak_usage": v.peak_usage,
                        "allocation_count": v.allocation_count,
                        "total_allocated": v.total_allocated,
                        "avg_allocation_size": v.avg_allocation_size
                    }),
                )
            })
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();

        json_obj.insert("component_stats".to_string(), component_stats);
        json_obj.insert(
            "duration_secs".to_string(),
            serde_json::Value::Number(report.duration.as_secs().into()),
        );

        serde_json::Value::Object(json_obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::metrics::event::MemoryEventType;

    #[test]
    fn test_memory_metrics_collector() {
        let config = MemoryMetricsConfig {
            enabled: true,
            capture_call_stacks: false,
            max_events: 100,
            real_time_aggregation: true,
            samplingrate: 1.0,
        };

        let collector = MemoryMetricsCollector::new(config);

        // Record allocation events
        collector.record_event(MemoryEvent::new(
            MemoryEventType::Allocation,
            "Component1",
            1024,
            0x1000,
        ));

        collector.record_event(MemoryEvent::new(
            MemoryEventType::Allocation,
            "Component1",
            2048,
            0x2000,
        ));

        collector.record_event(MemoryEvent::new(
            MemoryEventType::Allocation,
            "Component2",
            4096,
            0x3000,
        ));

        // Check current usage
        assert_eq!(collector.get_current_usage("Component1"), 3072);
        assert_eq!(collector.get_current_usage("Component2"), 4096);
        assert_eq!(collector.get_total_current_usage(), 7168);

        // Record deallocation event
        collector.record_event(MemoryEvent::new(
            MemoryEventType::Deallocation,
            "Component1",
            1024,
            0x1000,
        ));

        // Check updated usage
        assert_eq!(collector.get_current_usage("Component1"), 2048);
        assert_eq!(collector.get_total_current_usage(), 6144);

        // Check allocation stats
        let comp1_stats = collector.get_allocation_stats("Component1").unwrap();
        assert_eq!(comp1_stats.count, 2);
        assert_eq!(comp1_stats.total_bytes, 3072);
        assert_eq!(comp1_stats.peak_usage, 3072);

        // Generate report
        let report = collector.generate_report();
        assert_eq!(report.total_current_usage, 6144);
        assert_eq!(report.total_allocation_count, 3);

        // Check component stats in report
        let comp1_report = report.component_stats.get("Component1").unwrap();
        assert_eq!(comp1_report.current_usage, 2048);
        assert_eq!(comp1_report.allocation_count, 2);
    }
}
