//! Analytics and statistics types for JIT compilation

use crate::advanced_jit_compilation::cache::CacheStatistics;
use crate::advanced_jit_compilation::profiler::ProfilerAnalytics;
use std::time::Duration;

/// Compilation statistics
#[derive(Debug, Clone)]
pub struct CompilationStatistics {
    /// Total compilations
    pub total_compilations: u64,
    /// Successful compilations
    pub successful_compilations: u64,
    /// Failed compilations
    pub failed_compilations: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Average compilation time
    pub avg_compilation_time: Duration,
    /// Total compilation time
    pub total_compilation_time: Duration,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    /// Peak memory usage
    pub peak_memory_mb: f64,
    /// Average memory usage
    pub avg_memory_mb: f64,
    /// Memory allocations
    pub total_allocations: u64,
    /// Memory deallocations
    pub total_deallocations: u64,
}

/// JIT compilation analytics
#[derive(Debug)]
pub struct JitAnalytics {
    /// Compilation statistics
    pub compilation_stats: CompilationStatistics,
    /// Cache statistics
    pub cache_stats: CacheStatistics,
    /// Profiler statistics
    pub profiler_stats: ProfilerAnalytics,
    /// Overall performance score
    pub overall_performance: f64,
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
}

impl Default for CompilationStatistics {
    fn default() -> Self {
        Self {
            total_compilations: 0,
            successful_compilations: 0,
            failed_compilations: 0,
            cache_hits: 0,
            avg_compilation_time: Duration::default(),
            total_compilation_time: Duration::default(),
            memory_usage: MemoryUsageStats {
                peak_memory_mb: 0.0,
                avg_memory_mb: 0.0,
                total_allocations: 0,
                total_deallocations: 0,
            },
        }
    }
}
