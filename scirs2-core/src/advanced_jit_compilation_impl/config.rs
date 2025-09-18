//! Configuration types for the JIT compilation framework

use serde::{Deserialize, Serialize};

/// Configuration for JIT compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitCompilerConfig {
    /// Enable aggressive optimizations
    pub enable_aggressive_optimization: bool,
    /// Enable vectorization
    pub enable_vectorization: bool,
    /// Enable loop unrolling
    pub enable_loop_unrolling: bool,
    /// Enable function inlining
    pub enable_inlining: bool,
    /// Enable cross-module optimization
    pub enable_cross_module_optimization: bool,
    /// Target CPU architecture
    pub target_cpu: String,
    /// Target feature set
    pub target_features: Vec<String>,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Enable debugging information
    pub enable_debug_info: bool,
    /// Cache size limit (MB)
    pub cache_size_limit_mb: usize,
    /// Compilation timeout (seconds)
    pub compilation_timeout_seconds: u64,
    /// Enable profiling
    pub enable_profiling: bool,
    /// Enable adaptive compilation
    pub enable_adaptive_compilation: bool,
}

impl Default for JitCompilerConfig {
    fn default() -> Self {
        Self {
            enable_aggressive_optimization: true,
            enable_vectorization: true,
            enable_loop_unrolling: true,
            enable_inlining: true,
            enable_cross_module_optimization: true,
            target_cpu: "native".to_string(),
            target_features: vec!["avx2".to_string(), "fma".to_string(), "sse4.2".to_string()],
            optimization_level: 3,
            enable_debug_info: false,
            cache_size_limit_mb: 512,
            compilation_timeout_seconds: 30,
            enable_profiling: true,
            enable_adaptive_compilation: true,
        }
    }
}

/// Configuration for neuromorphic compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicConfig {
    /// Enable spike-based optimization
    pub enable_spike_optimization: bool,
    /// Enable plasticity learning
    pub enable_plasticity: bool,
    /// Enable temporal dynamics
    pub enable_temporal_dynamics: bool,
    /// Time step resolution (microseconds)
    pub time_step_resolution_us: f64,
    /// Maximum spike frequency (Hz)
    pub max_spike_frequency_hz: f64,
    /// Refractory period (milliseconds)
    pub refractory_period_ms: f64,
    /// Membrane time constant (milliseconds)
    pub membrane_time_constant_ms: f64,
    /// Synaptic delay range (milliseconds)
    pub synapticdelay_range_ms: (f64, f64),
    /// STDP learning window (milliseconds)
    pub stdp_window_ms: f64,
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            enable_spike_optimization: true,
            enable_plasticity: true,
            enable_temporal_dynamics: true,
            time_step_resolution_us: 100.0, // 0.1ms
            max_spike_frequency_hz: 1000.0,
            refractory_period_ms: 2.0,
            membrane_time_constant_ms: 10.0,
            synapticdelay_range_ms: (0.5, 5.0),
            stdp_window_ms: 20.0,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size
    pub maxsize_mb: usize,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable cache warming
    pub enable_cache_warming: bool,
    /// Cache persistence
    pub enable_persistence: bool,
    /// Persistence directory
    pub persistence_dir: Option<std::path::PathBuf>,
}

/// Cache eviction policies
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    Random,
    FIFO,
    Adaptive,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable execution profiling
    pub enable_execution_profiling: bool,
    /// Enable compilation profiling
    pub enable_compilation_profiling: bool,
    /// Sampling rate for profiling
    pub samplingrate: f64,
    /// Profile data retention time
    pub retention_hours: u32,
    /// Enable hotspot detection
    pub enable_hotspot_detection: bool,
    /// Hotspot threshold
    pub hotspot_threshold: f64,
}

/// Profiling session configuration
#[derive(Debug, Clone)]
pub struct ProfilingSessionConfig {
    /// Sampling interval
    pub sampling_interval: std::time::Duration,
    /// Include stack traces
    pub include_stack_traces: bool,
    /// Profile memory allocations
    pub profile_memory: bool,
    /// Profile system calls
    pub profile_syscalls: bool,
}

/// Pattern cache configuration
#[derive(Debug, Clone)]
pub struct PatternCacheConfig {
    /// Maximum patterns
    pub max_patterns: usize,
    /// TTL for patterns
    pub pattern_ttl: std::time::Duration,
    /// Enable LRU eviction
    pub enable_lru: bool,
}
