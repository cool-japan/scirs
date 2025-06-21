//! # Profiling Module
//!
//! Comprehensive profiling capabilities for `SciRS2` Core including production-level
//! performance analysis, bottleneck identification, and performance monitoring.

pub mod continuous_monitoring;
pub mod flame_graph_svg;
pub mod hardware_counters;
pub mod performance_hints;
pub mod production;
pub mod system_monitor;

// Re-export key types for convenience
pub use production::{
    PerformanceBottleneck, PerformanceRegression, ProductionProfiler, ProfileConfig,
    ResourceUsage, WorkloadAnalysisReport, WorkloadType,
};

// Production profiling types (using placeholders for types that don't exist in base module)
pub type ProfilerResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Basic profiling session for production profiler integration
#[derive(Debug)]
pub struct ProfilingSession {
    pub id: String,
    pub start_time: std::time::Instant,
}

impl ProfilingSession {
    pub fn new(id: &str) -> ProfilerResult<Self> {
        Ok(Self {
            id: id.to_string(),
            start_time: std::time::Instant::now(),
        })
    }
}

/// Profiling convenience functions
pub mod prelude {
    pub use super::production::{
        PerformanceBottleneck, PerformanceRegression, ProductionProfiler, ProfileConfig,
        ResourceUsage, WorkloadAnalysisReport, WorkloadType,
    };
}