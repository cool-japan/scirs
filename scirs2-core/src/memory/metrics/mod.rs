//! Memory metrics system for tracking and analyzing memory usage
//!
//! This module provides functionality for tracking, collecting, and reporting
//! memory usage metrics. It can be used to understand memory allocation patterns,
//! identify memory leaks, and optimize memory-intensive operations.
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_core::memory::metrics::{MemoryMetricsCollector, MemoryMetricsConfig, MemoryEvent, MemoryEventType};
//!
//! // Create a metrics collector
//! let config = MemoryMetricsConfig::default();
//! let collector = MemoryMetricsCollector::new(config);
//!
//! // Record memory events
//! collector.record_event(MemoryEvent::new(
//!     MemoryEventType::Allocation,
//!     "MyComponent",
//!     1024,
//!     0x1000,
//! ));
//!
//! // Generate a report
//! let report = collector.generate_report();
//! println!("{}", report.format());
//! ```

mod analytics;
mod collector;
mod event;
#[cfg(feature = "gpu")]
mod gpu;
mod profiler;
mod reporter;
mod snapshot;

#[cfg(test)]
mod test_utils;

pub use analytics::{
    AllocationPattern, LeakDetectionConfig, LeakDetectionResult, MemoryAnalytics,
    MemoryEfficiencyMetrics, MemoryIssue, MemoryPatternAnalysis, OptimizationRecommendation,
};
pub use collector::{
    AllocationStats, ComponentMemoryStats, MemoryMetricsCollector, MemoryMetricsConfig,
    MemoryReport,
};
pub use event::{MemoryEvent, MemoryEventType};
pub use profiler::{
    MemoryProfiler, MemoryProfilerConfig, PerformanceImpactAnalysis, ProfilingResult,
    ProfilingSession, ProfilingSummary, RiskAssessment,
};
pub use reporter::{format_bytes, format_duration};
pub use snapshot::{
    clear_snapshots, compare_snapshots, global_snapshot_manager, load_snapshots, save_snapshots,
    take_snapshot, ComponentStatsDiff, MemorySnapshot, SnapshotComponentStats, SnapshotDiff,
    SnapshotManager, SnapshotReport,
};

#[cfg(feature = "memory_visualization")]
pub use reporter::ChartFormat;

// Re-export snapshot visualization if feature is enabled

#[cfg(feature = "gpu")]
pub use gpu::{setup_gpu_memory_tracking, TrackedGpuBuffer, TrackedGpuContext};

use crate::memory::{BufferPool, ChunkProcessor, ChunkProcessor2D};
use ndarray::{ArrayBase, Data, Dimension, ViewRepr};
use once_cell::sync::Lazy;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;

/// Global memory metrics collector instance
static GLOBAL_METRICS_COLLECTOR: Lazy<Arc<MemoryMetricsCollector>> =
    Lazy::new(|| Arc::new(MemoryMetricsCollector::new(MemoryMetricsConfig::default())));

/// Get the global memory metrics collector
pub fn global_metrics_collector() -> Arc<MemoryMetricsCollector> {
    GLOBAL_METRICS_COLLECTOR.clone()
}

/// Track a memory allocation event in the global collector
pub fn track_allocation(component: impl Into<String>, size: usize, address: usize) {
    let event = MemoryEvent::new(MemoryEventType::Allocation, component, size, address);
    GLOBAL_METRICS_COLLECTOR.record_event(event);
}

/// Track a memory deallocation event in the global collector
pub fn track_deallocation(component: impl Into<String>, size: usize, address: usize) {
    let event = MemoryEvent::new(MemoryEventType::Deallocation, component, size, address);
    GLOBAL_METRICS_COLLECTOR.record_event(event);
}

/// Track a memory resize event in the global collector
pub fn track_resize(
    component: impl Into<String>,
    new_size: usize,
    old_size: usize,
    address: usize,
) {
    let event = MemoryEvent::new(MemoryEventType::Resize, component, new_size, address)
        .with_metadata("old_size", old_size.to_string());
    GLOBAL_METRICS_COLLECTOR.record_event(event);
}

/// Generate a memory report from the global collector
pub fn generate_memory_report() -> MemoryReport {
    GLOBAL_METRICS_COLLECTOR.generate_report()
}

/// Format the current memory report as a string
pub fn format_memory_report() -> String {
    GLOBAL_METRICS_COLLECTOR.generate_report().format()
}

/// Reset the global memory metrics collector
pub fn reset_memory_metrics() {
    GLOBAL_METRICS_COLLECTOR.reset();
}

/// A buffer pool that automatically tracks memory allocations and deallocations
pub struct TrackedBufferPool<T: Clone + Default> {
    inner: BufferPool<T>,
    component_name: String,
    _phantom: PhantomData<T>,
}

impl<T: Clone + Default> TrackedBufferPool<T> {
    /// Create a new tracked buffer pool with the given component name
    pub fn new(component_name: impl Into<String>) -> Self {
        Self {
            inner: BufferPool::new(),
            component_name: component_name.into(),
            _phantom: PhantomData,
        }
    }

    /// Acquire a vector from the pool, tracking the allocation
    pub fn acquire_vec(&mut self, capacity: usize) -> Vec<T> {
        let vec = self.inner.acquire_vec(capacity);
        let size = capacity * mem::size_of::<T>();

        // Track allocation
        track_allocation(&self.component_name, size, &vec as *const _ as usize);

        vec
    }

    /// Release a vector back to the pool, tracking the deallocation
    pub fn release_vec(&mut self, vec: Vec<T>) {
        let size = vec.capacity() * mem::size_of::<T>();

        // Track deallocation
        track_deallocation(&self.component_name, size, &vec as *const _ as usize);

        self.inner.release_vec(vec);
    }

    /// Acquire an ndarray from the pool, tracking the allocation
    pub fn acquire_array(&mut self, size: usize) -> ndarray::Array1<T> {
        let array = self.inner.acquire_array(size);
        let mem_size = size * mem::size_of::<T>();

        // Track allocation
        track_allocation(&self.component_name, mem_size, array.as_ptr() as usize);

        array
    }

    /// Release an ndarray back to the pool, tracking the deallocation
    pub fn release_array(&mut self, array: ndarray::Array1<T>) {
        let size = array.len() * mem::size_of::<T>();

        // Track deallocation
        track_deallocation(&self.component_name, size, array.as_ptr() as usize);

        self.inner.release_array(array);
    }

    /// Clear the pool, releasing all memory
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// A chunk processor that tracks memory usage during processing
pub struct TrackedChunkProcessor<'a, A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    inner: ChunkProcessor<'a, A, S, D>,
    component_name: String,
}

impl<'a, A, S, D> TrackedChunkProcessor<'a, A, S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Create a new tracked chunk processor
    pub fn new(
        array: &'a ArrayBase<S, D>,
        chunk_shape: D,
        component_name: impl Into<String>,
    ) -> Self {
        Self {
            inner: ChunkProcessor::new(array, chunk_shape),
            component_name: component_name.into(),
        }
    }

    /// Process the array in chunks, tracking memory usage for each chunk
    pub fn process_chunks<F>(&mut self, mut f: F)
    where
        F: FnMut(&ArrayBase<ViewRepr<&A>, D>, D),
    {
        // Create a wrapper function that tracks memory
        let component_name = self.component_name.clone();
        let tracked_f = move |chunk: &ArrayBase<ViewRepr<&A>, D>, coords: D| {
            // Calculate memory size (approximate)
            let size = chunk.len() * mem::size_of::<A>();

            // Track memory allocation for the chunk
            let address = chunk.as_ptr() as usize;
            track_allocation(&component_name, size, address);

            // Call the original function
            f(chunk, coords);

            // Track memory deallocation
            track_deallocation(&component_name, size, address);
        };

        // Process chunks with the tracked function
        self.inner.process_chunks(tracked_f);
    }

    /// Get the total number of chunks
    pub fn num_chunks(&self) -> usize {
        self.inner.num_chunks()
    }
}

/// A 2D chunk processor that tracks memory usage during processing
pub struct TrackedChunkProcessor2D<'a, A, S>
where
    S: Data<Elem = A>,
{
    inner: ChunkProcessor2D<'a, A, S>,
    component_name: String,
}

impl<'a, A, S> TrackedChunkProcessor2D<'a, A, S>
where
    S: Data<Elem = A>,
{
    /// Create a new tracked 2D chunk processor
    pub fn new(
        array: &'a ArrayBase<S, ndarray::Ix2>,
        chunk_shape: (usize, usize),
        component_name: impl Into<String>,
    ) -> Self {
        Self {
            inner: ChunkProcessor2D::new(array, chunk_shape),
            component_name: component_name.into(),
        }
    }

    /// Process the 2D array in chunks, tracking memory usage for each chunk
    pub fn process_chunks<F>(&mut self, mut f: F)
    where
        F: FnMut(&ArrayBase<ViewRepr<&A>, ndarray::Ix2>, (usize, usize)),
    {
        // Create a wrapper function that tracks memory
        let component_name = self.component_name.clone();
        let tracked_f = move |chunk: &ArrayBase<ViewRepr<&A>, ndarray::Ix2>,
                              coords: (usize, usize)| {
            // Calculate memory size (approximate)
            let size = chunk.len() * mem::size_of::<A>();

            // Track memory allocation for the chunk
            let address = chunk.as_ptr() as usize;
            track_allocation(&component_name, size, address);

            // Call the original function
            f(chunk, coords);

            // Track memory deallocation
            track_deallocation(&component_name, size, address);
        };

        // Process chunks with the tracked function
        self.inner.process_chunks(tracked_f);
    }
}

#[cfg(test)]
mod tests {
    use super::test_utils::MEMORY_METRICS_TEST_MUTEX;
    use super::*;

    #[test]
    fn test_global_memory_metrics() {
        // Lock the mutex to ensure test isolation
        let _lock = MEMORY_METRICS_TEST_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Reset metrics to start with clean state
        reset_memory_metrics();

        // Track some allocations
        track_allocation("TestComponent", 1024, 0x1000);
        track_allocation("TestComponent", 2048, 0x2000);
        track_allocation("OtherComponent", 4096, 0x3000);

        // Track a deallocation
        track_deallocation("TestComponent", 1024, 0x1000);

        // Generate a report
        let report = generate_memory_report();

        // Verify metrics
        assert_eq!(report.total_current_usage, 6144); // 2048 + 4096
        assert_eq!(report.total_allocation_count, 3);

        // Check component stats
        let test_comp = report.component_stats.get("TestComponent").unwrap();
        assert_eq!(test_comp.current_usage, 2048);
        assert_eq!(test_comp.allocation_count, 2);

        // Reset and verify
        reset_memory_metrics();
        let report = generate_memory_report();
        assert_eq!(report.total_current_usage, 0);
    }
}
