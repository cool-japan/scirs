//! Automatic kernel tuning and optimization for GPU operations
//!
//! This module provides capabilities for automatically tuning GPU kernel parameters
//! to achieve optimal performance on different hardware configurations and workloads.

use crate::gpu::{GpuBackend, GpuError, GpuKernelHandle};
use rand::Rng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error types for auto-tuning operations
#[derive(Error, Debug)]
pub enum AutoTuningError {
    /// No tuning configurations available
    #[error("No tuning configurations available for kernel: {0}")]
    NoConfigurations(String),

    /// Tuning process failed
    #[error("Auto-tuning failed: {0}")]
    TuningFailed(String),

    /// Invalid parameter configuration
    #[error("Invalid parameter configuration: {0}")]
    InvalidConfiguration(String),

    /// Benchmark execution failed
    #[error("Benchmark execution failed: {0}")]
    BenchmarkFailed(String),

    /// Underlying GPU error
    #[error("GPU error: {0}")]
    GpuError(#[from] GpuError),
}

/// Tunable kernel parameters
#[derive(Debug, Clone, PartialEq)]
pub struct KernelParameters {
    /// Work group size (local work size)
    pub work_group_size: [u32; 3],
    /// Global work size
    pub global_work_size: [u32; 3],
    /// Local memory usage per work group
    pub local_memory_size: usize,
    /// Register usage per thread
    pub register_usage: Option<usize>,
    /// Cache configuration hints
    pub cacheconfig: CacheConfig,
    /// Custom parameters for kernel-specific tuning
    pub custom_params: HashMap<String, ParameterValue>,
}

impl Default for KernelParameters {
    fn default() -> Self {
        Self {
            work_group_size: [16, 16, 1],
            global_work_size: [1024, 1024, 1],
            local_memory_size: 0,
            register_usage: None,
            cacheconfig: CacheConfig::Balanced,
            custom_params: HashMap::new(),
        }
    }
}

/// Parameter value types for kernel tuning
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    /// Integer parameter
    Int(i64),
    /// Floating point parameter
    Float(f64),
    /// String parameter
    String(String),
    /// Boolean parameter
    Bool(bool),
    /// Array of integers
    IntArray(Vec<i64>),
    /// Array of floats
    FloatArray(Vec<f64>),
}

impl ParameterValue {
    /// Convert to integer if possible
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParameterValue::Int(val) => Some(*val),
            ParameterValue::Float(val) => Some(*val as i64),
            _ => None,
        }
    }

    /// Convert to float if possible
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(val) => Some(*val),
            ParameterValue::Int(val) => Some(*val as f64),
            _ => None,
        }
    }

    /// Convert to string
    pub fn as_string(&self) -> String {
        match self {
            ParameterValue::String(val) => val.clone(),
            ParameterValue::Int(val) => val.to_string(),
            ParameterValue::Float(val) => val.to_string(),
            ParameterValue::Bool(val) => val.to_string(),
            _ => format!("{self:?}"),
        }
    }
}

/// Cache configuration strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheConfig {
    /// Prefer L1 cache for local memory
    PreferL1,
    /// Prefer shared memory over L1 cache
    PreferShared,
    /// Balanced cache usage
    Balanced,
    /// Optimize for read-only data
    ReadOnly,
    /// Optimize for write-through patterns
    WriteThrough,
}

/// Performance metrics for kernel execution
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Execution time
    pub execution_time: Duration,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Memory bandwidth utilization
    pub memorybandwidth_util: f64,
    /// Compute utilization
    pub compute_utilization: f64,
    /// Energy efficiency (operations per joule)
    pub energy_efficiency: Option<f64>,
    /// Cache hit rates
    pub cache_metrics: CacheMetrics,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time: Duration::from_millis(0),
            throughput: 0.0,
            memorybandwidth_util: 0.0,
            compute_utilization: 0.0,
            energy_efficiency: None,
            cache_metrics: CacheMetrics::default(),
        }
    }
}

/// Cache performance metrics
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    /// L1 cache hit rate
    pub l1_hit_rate: f64,
    /// L2 cache hit rate
    pub l2_hit_rate: f64,
    /// Shared memory bank conflicts
    pub shared_memory_conflicts: usize,
    /// Global memory coalescing efficiency
    pub coalescing_efficiency: f64,
    /// Memory throughput in GB/s
    pub memory_throughput: f64,
    /// Cache pressure indicator
    pub cache_pressure: f64,
}

/// Auto-tuning strategy configuration
#[derive(Debug, Clone)]
pub struct TuningStrategy {
    /// Search algorithm to use
    pub search_algorithm: SearchAlgorithm,
    /// Maximum number of configurations to test
    pub max_evaluations: usize,
    /// Time budget for tuning process
    pub time_budget: Duration,
    /// Number of benchmark runs per configuration
    pub benchmark_runs: usize,
    /// Convergence criteria
    pub convergence_threshold: f64,
    /// Whether to use historical data
    pub use_history: bool,
}

impl Default for TuningStrategy {
    fn default() -> Self {
        Self {
            search_algorithm: SearchAlgorithm::GridSearch,
            max_evaluations: 100,
            time_budget: Duration::from_secs(60),
            benchmark_runs: 3,
            convergence_threshold: 0.01, // 1% improvement required
            use_history: true,
        }
    }
}

/// Search algorithms for parameter optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchAlgorithm {
    /// Exhaustive grid search
    GridSearch,
    /// Random search
    RandomSearch,
    /// Bayesian optimization
    BayesianOptimization,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Differential evolution
    DifferentialEvolution,
    /// Particle swarm optimization
    ParticleSwarm,
}

/// Tuning configuration space
#[derive(Debug, Clone)]
pub struct TuningSpace {
    /// Work group size options
    pub work_group_sizes: Vec<[u32; 3]>,
    /// Local memory size options
    pub local_memory_sizes: Vec<usize>,
    /// Cache configuration options
    pub cache_configs: Vec<CacheConfig>,
    /// Custom parameter spaces
    pub custom_spaces: HashMap<String, Vec<ParameterValue>>,
}

impl Default for TuningSpace {
    fn default() -> Self {
        Self {
            work_group_sizes: vec![
                [8, 8, 1],
                [16, 16, 1],
                [32, 32, 1],
                [64, 8, 1],
                [8, 64, 1],
                [128, 1, 1],
                [256, 1, 1],
                [512, 1, 1],
            ],
            local_memory_sizes: vec![0, 1024, 2048, 4096, 8192, 16384],
            cache_configs: vec![
                CacheConfig::Balanced,
                CacheConfig::PreferL1,
                CacheConfig::PreferShared,
                CacheConfig::ReadOnly,
            ],
            custom_spaces: HashMap::new(),
        }
    }
}

/// Auto-tuning result
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Best parameters found
    pub best_params: KernelParameters,
    /// Best performance achieved
    pub best_performance: PerformanceMetrics,
    /// Number of configurations evaluated
    pub evaluations: usize,
    /// Total tuning time
    pub tuning_time: Duration,
    /// Convergence information
    pub converged: bool,
    /// Performance improvement over baseline
    pub improvement_factor: f64,
}

/// Automatic kernel tuner
#[derive(Debug)]
pub struct AutoTuner {
    backend: GpuBackend,
    strategy: TuningStrategy,
    tuning_cache: Arc<Mutex<HashMap<String, TuningResult>>>,
    device_info: DeviceInfo,
}

/// Device information for tuning
#[derive(Debug, Clone)]
struct DeviceInfo {
    compute_capability: String,
    #[allow(dead_code)]
    memory_size: usize,
    max_work_group_size: usize,
    max_local_memory_size: usize,
    #[allow(dead_code)]
    warp_size: usize,
}

impl AutoTuner {
    /// Create a new auto-tuner for the given backend
    pub fn new(backend: GpuBackend, strategy: TuningStrategy) -> Result<Self, AutoTuningError> {
        let device_info = Self::detect_device_info(backend)?;

        Ok(Self {
            backend,
            strategy,
            tuning_cache: Arc::new(Mutex::new(HashMap::new())),
            device_info,
        })
    }

    /// Auto-tune a kernel for optimal performance
    pub fn tune(
        &self,
        kernel: &GpuKernelHandle,
        kernel_name: &str,
        problemsize: &[usize],
        tuning_space: TuningSpace,
    ) -> Result<TuningResult, AutoTuningError> {
        let cache_key = self.generate_cache_key(kernel_name, problemsize);

        // Check cache first
        if self.strategy.use_history {
            if let Some(cached_result) = self.tuning_cache.lock().unwrap().get(&cache_key) {
                return Ok(cached_result.clone());
            }
        }

        let start_time = Instant::now();
        let mut best_params = KernelParameters::default();
        let mut best_performance: Option<PerformanceMetrics> = None;
        let mut evaluations = 0;

        // Generate parameter configurations to test
        let configurations = self.generate_configurations(&tuning_space)?;

        for (i, params) in configurations.iter().enumerate() {
            if start_time.elapsed() > self.strategy.time_budget {
                break;
            }

            if evaluations >= self.strategy.max_evaluations {
                break;
            }

            // Benchmark this configuration
            match self.benchmark_configuration(kernel, params, problemsize) {
                Ok(metrics) => {
                    evaluations += 1;

                    if best_performance.is_none()
                        || metrics.throughput > best_performance.as_ref().unwrap().throughput
                    {
                        best_params = params.clone();
                        best_performance = Some(metrics);
                    }

                    // Check convergence
                    if let Some(ref best) = best_performance {
                        if self.check_convergence(best, i) {
                            break;
                        }
                    }
                }
                Err(e) => {
                    // Log benchmark failure but continue
                    eprintln!("Benchmark failed for configuration {params:?}: {e}");
                }
            }
        }

        let best_performance = best_performance.ok_or_else(|| {
            AutoTuningError::TuningFailed("No successful configurations".to_string())
        })?;

        let tuning_time = start_time.elapsed();
        let improvement_factor = 1.0; // Would compare against baseline

        let result = TuningResult {
            best_params,
            best_performance,
            evaluations,
            tuning_time,
            converged: evaluations < self.strategy.max_evaluations,
            improvement_factor,
        };

        // Cache the result
        self.tuning_cache
            .lock()
            .unwrap()
            .insert(cache_key, result.clone());

        Ok(result)
    }

    /// Get cached tuning results
    pub fn get_cached_results(&self) -> HashMap<String, TuningResult> {
        self.tuning_cache.lock().unwrap().clone()
    }

    /// Clear tuning cache
    pub fn clear_cache(&self) {
        self.tuning_cache.lock().unwrap().clear();
    }

    /// Generate parameter configurations to test
    fn generate_configurations(
        &self,
        space: &TuningSpace,
    ) -> Result<Vec<KernelParameters>, AutoTuningError> {
        match self.strategy.search_algorithm {
            SearchAlgorithm::GridSearch => self.grid_search_configurations(space),
            SearchAlgorithm::RandomSearch => self.random_search_configurations(space),
            _ => {
                // For other algorithms, fall back to grid search for now
                self.grid_search_configurations(space)
            }
        }
    }

    /// Generate configurations using grid search
    fn grid_search_configurations(
        &self,
        space: &TuningSpace,
    ) -> Result<Vec<KernelParameters>, AutoTuningError> {
        let mut configurations = Vec::new();

        for &work_group_size in &space.work_group_sizes {
            for &local_memory_size in &space.local_memory_sizes {
                for &cache_config in &space.cache_configs {
                    // Validate configuration against device limits
                    if self.is_valid_configuration(work_group_size, local_memory_size) {
                        configurations.push(KernelParameters {
                            work_group_size,
                            global_work_size: [1024, 1024, 1], // Default
                            local_memory_size,
                            register_usage: None,
                            cacheconfig: cache_config,
                            custom_params: HashMap::new(),
                        });
                    }
                }
            }
        }

        Ok(configurations)
    }

    /// Generate configurations using random search
    fn random_search_configurations(
        &self,
        space: &TuningSpace,
    ) -> Result<Vec<KernelParameters>, AutoTuningError> {
        let mut configurations = Vec::new();
        let num_samples = self.strategy.max_evaluations.min(100);

        for _ in 0..num_samples {
            let work_group_size =
                space.work_group_sizes[rand::rng().random_range(0..space.work_group_sizes.len())];
            let local_memory_size = space.local_memory_sizes
                [rand::rng().random_range(0..space.local_memory_sizes.len())];
            let cache_config =
                space.cache_configs[rand::rng().random_range(0..space.cache_configs.len())];

            if self.is_valid_configuration(work_group_size, local_memory_size) {
                configurations.push(KernelParameters {
                    work_group_size,
                    global_work_size: [1024, 1024, 1],
                    local_memory_size,
                    register_usage: None,
                    cacheconfig: cache_config,
                    custom_params: HashMap::new(),
                });
            }
        }

        Ok(configurations)
    }

    /// Validate if a configuration is valid for the device
    fn is_valid_configuration(&self, work_group_size: [u32; 3], local_memorysize: usize) -> bool {
        let total_threads = work_group_size[0] * work_group_size[1] * work_group_size[2];

        total_threads <= self.device_info.max_work_group_size as u32
            && local_memorysize <= self.device_info.max_local_memory_size
    }

    /// Benchmark a specific configuration
    fn benchmark_configuration(
        &self,
        kernel: &GpuKernelHandle,
        params: &KernelParameters,
        problemsize: &[usize],
    ) -> Result<PerformanceMetrics, AutoTuningError> {
        let mut execution_times = Vec::new();

        // Run multiple iterations for stable timing
        for _ in 0..self.strategy.benchmark_runs {
            let start = Instant::now();

            // Execute kernel with these parameters
            kernel.dispatch(params.work_group_size);

            // In a real implementation, we would:
            // 1. Set up proper synchronization
            // 2. Configure kernel parameters
            // 3. Measure actual GPU execution time
            // 4. Collect performance counters

            let execution_time = start.elapsed();
            execution_times.push(execution_time);
        }

        // Calculate average execution time
        let avg_time = execution_times.iter().sum::<Duration>() / execution_times.len() as u32;

        // Calculate throughput (simplified)
        let total_ops = problemsize.iter().product::<usize>() as f64;
        let throughput = total_ops / avg_time.as_secs_f64();

        Ok(PerformanceMetrics {
            execution_time: avg_time,
            throughput,
            memorybandwidth_util: 0.8, // Mock value
            compute_utilization: 0.9,  // Mock value
            energy_efficiency: None,
            cache_metrics: CacheMetrics::default(),
        })
    }

    /// Check if tuning has converged
    fn check_convergence(&self, performance: &PerformanceMetrics, iteration: usize) -> bool {
        // Simple convergence check based on iteration count
        // In practice, would compare recent improvements
        iteration > 10 && iteration % 10 == 0
    }

    /// Generate cache key for tuning results
    fn generate_cache_key(&self, kernel_name: &str, problemsize: &[usize]) -> String {
        format!(
            "{}_{}_{}_{:?}",
            self.backend, self.device_info.compute_capability, kernel_name, problemsize
        )
    }

    /// Detect device information for tuning
    fn detect_device_info(backend: GpuBackend) -> Result<DeviceInfo, AutoTuningError> {
        // In a real implementation, this would query the actual device
        match backend {
            GpuBackend::Cuda => Ok(DeviceInfo {
                compute_capability: "8.0".to_string(),
                memory_size: 12 * 1024 * 1024 * 1024, // 12 GB
                max_work_group_size: 1024,
                max_local_memory_size: 48 * 1024, // 48 KB
                warp_size: 32,
            }),
            GpuBackend::Rocm => Ok(DeviceInfo {
                compute_capability: "RDNA2".to_string(),
                memory_size: 16 * 1024 * 1024 * 1024, // 16 GB
                max_work_group_size: 1024,
                max_local_memory_size: 64 * 1024, // 64 KB
                warp_size: 64,                    // Wavefront size
            }),
            _ => Ok(DeviceInfo {
                compute_capability: "Unknown".to_string(),
                memory_size: 8 * 1024 * 1024 * 1024, // 8 GB
                max_work_group_size: 256,
                max_local_memory_size: 16 * 1024, // 16 KB
                warp_size: 32,
            }),
        }
    }
}

/// Convenience functions for common auto-tuning scenarios
pub mod presets {
    use super::*;

    /// Get tuning space optimized for matrix multiplication
    pub fn matrix_multiply_space() -> TuningSpace {
        TuningSpace {
            work_group_sizes: vec![
                [16, 16, 1],
                [32, 32, 1],
                [8, 32, 1],
                [32, 8, 1],
                [64, 4, 1],
                [4, 64, 1],
                [128, 2, 1],
                [2, 128, 1],
            ],
            local_memory_sizes: vec![0, 2048, 4096, 8192, 16384],
            cache_configs: vec![CacheConfig::PreferShared, CacheConfig::Balanced],
            custom_spaces: HashMap::new(),
        }
    }

    /// Get tuning space optimized for convolution operations
    pub fn convolution_space() -> TuningSpace {
        TuningSpace {
            work_group_sizes: vec![
                [8, 8, 1],
                [16, 16, 1],
                [32, 8, 1],
                [8, 32, 1],
                [64, 1, 1],
                [32, 4, 1],
                [4, 32, 1],
            ],
            local_memory_sizes: vec![1024, 2048, 4096, 8192],
            cache_configs: vec![CacheConfig::PreferL1, CacheConfig::ReadOnly],
            custom_spaces: HashMap::new(),
        }
    }

    /// Get tuning space optimized for reduction operations
    pub fn reduction_space() -> TuningSpace {
        TuningSpace {
            work_group_sizes: vec![
                [64, 1, 1],
                [128, 1, 1],
                [256, 1, 1],
                [512, 1, 1],
                [1024, 1, 1],
                [32, 2, 1],
                [16, 4, 1],
            ],
            local_memory_sizes: vec![512, 1024, 2048, 4096],
            cache_configs: vec![CacheConfig::PreferShared],
            custom_spaces: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_value_conversion() {
        let int_val = ParameterValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));

        let float_val = ParameterValue::Float(3.5);
        assert_eq!(float_val.as_float(), Some(3.5));
        assert_eq!(float_val.as_int(), Some(3));
    }

    #[test]
    fn test_kernel_parameters_default() {
        let params = KernelParameters::default();
        assert_eq!(params.work_group_size, [16, 16, 1]);
        assert_eq!(params.local_memory_size, 0);
    }

    #[test]
    fn test_tuning_strategy_default() {
        let strategy = TuningStrategy::default();
        assert_eq!(strategy.search_algorithm, SearchAlgorithm::GridSearch);
        assert_eq!(strategy.max_evaluations, 100);
    }

    #[test]
    fn test_tuning_space_default() {
        let space = TuningSpace::default();
        assert!(!space.work_group_sizes.is_empty());
        assert!(!space.cache_configs.is_empty());
    }

    #[test]
    fn testmatrix_multiply_preset() {
        let space = presets::matrix_multiply_space();
        assert!(space.work_group_sizes.contains(&[16, 16, 1]));
        assert!(space.cache_configs.contains(&CacheConfig::PreferShared));
    }

    #[test]
    fn test_device_info_detection() {
        let device_info = AutoTuner::detect_device_info(GpuBackend::Cuda);
        assert!(device_info.is_ok());

        let info = device_info.unwrap();
        assert!(info.max_work_group_size > 0);
        assert!(info.max_local_memory_size > 0);
    }
}
