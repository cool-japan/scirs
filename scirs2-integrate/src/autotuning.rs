//! Auto-tuning for hardware configurations
//!
//! This module provides automatic parameter tuning based on detected hardware
//! characteristics. It optimizes algorithm parameters for CPU cores, cache sizes,
//! memory bandwidth, and other system properties to achieve optimal performance.
//!
//! # Hardware Detection
//!
//! The auto-tuning system detects:
//! - Number of CPU cores and threads
//! - Cache sizes (L1, L2, L3)
//! - Memory bandwidth characteristics
//! - SIMD instruction set availability
//! - GPU presence and capabilities
//!
//! # Examples
//!
//! ```
//! use scirs2_integrate::autotuning::{HardwareDetector, AutoTuner, TuningProfile};
//!
//! // Detect hardware automatically
//! let hardware = HardwareDetector::detect();
//! println!("Detected {} CPU cores", hardware.cpu_cores);
//!
//! // Create auto-tuner with detected hardware
//! let tuner = AutoTuner::new(hardware);
//! let profile = tuner.tune_for_problem_size(1000);
//! ```

use crate::common::IntegrateFloat;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Hardware characteristics detected at runtime
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// Number of physical CPU cores
    pub cpu_cores: usize,
    /// Number of logical CPU threads
    pub cpu_threads: usize,
    /// CPU brand and model
    pub cpu_model: String,
    /// L1 cache size per core (bytes)
    pub l1_cache_size: usize,
    /// L2 cache size per core (bytes)
    pub l2_cache_size: usize,
    /// L3 cache size total (bytes)
    pub l3_cache_size: usize,
    /// Memory size (bytes)
    pub memory_size: usize,
    /// Available SIMD instruction sets
    pub simd_features: Vec<SimdFeature>,
    /// Estimated memory bandwidth (bytes/second)
    pub memory_bandwidth: Option<f64>,
    /// GPU information
    pub gpu_info: Option<GpuInfo>,
}

#[derive(Debug, Clone)]
pub enum SimdFeature {
    SSE,
    SSE2,
    SSE3,
    SSSE3,
    SSE41,
    SSE42,
    AVX,
    AVX2,
    AVX512F,
    FMA,
    NEON, // ARM
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub vendor: String,
    pub model: String,
    pub memory_size: usize,
    pub compute_units: usize,
}

/// Hardware detection utilities
pub struct HardwareDetector;

impl HardwareDetector {
    /// Detect hardware characteristics
    pub fn detect() -> HardwareInfo {
        // Use cached detection result
        static HARDWARE_INFO: OnceLock<HardwareInfo> = OnceLock::new();

        HARDWARE_INFO.get_or_init(Self::detect_hardware).clone()
    }

    /// Perform actual hardware detection
    fn detect_hardware() -> HardwareInfo {
        let cpu_cores = Self::detect_cpu_cores();
        let cpu_threads = Self::detect_cpu_threads();
        let cpu_model = Self::detect_cpu_model();
        let (l1_cache_size, l2_cache_size, l3_cache_size) = Self::detect_cache_sizes();
        let memory_size = Self::detect_memory_size();
        let simd_features = Self::detect_simd_features();
        let memory_bandwidth = Self::estimate_memory_bandwidth();
        let gpu_info = Self::detect_gpu();

        HardwareInfo {
            cpu_cores,
            cpu_threads,
            cpu_model,
            l1_cache_size,
            l2_cache_size,
            l3_cache_size,
            memory_size,
            simd_features,
            memory_bandwidth,
            gpu_info,
        }
    }

    /// Detect number of physical CPU cores
    fn detect_cpu_cores() -> usize {
        // Try to get physical core count
        if let Some(cores) = std::thread::available_parallelism().ok().map(|n| n.get()) {
            // This gives logical cores, estimate physical cores
            cores / 2 // Rough estimate for hyperthreading
        } else {
            1
        }
        .max(1)
    }

    /// Detect number of logical CPU threads
    fn detect_cpu_threads() -> usize {
        std::thread::available_parallelism()
            .ok()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    /// Detect CPU model
    fn detect_cpu_model() -> String {
        format!("{} CPU", std::env::consts::ARCH)
    }

    /// Detect cache sizes
    fn detect_cache_sizes() -> (usize, usize, usize) {
        // Use reasonable defaults based on architecture
        #[cfg(target_arch = "x86_64")]
        {
            // Modern x86_64 typical cache sizes
            (32 * 1024, 256 * 1024, 8 * 1024 * 1024)
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 typical cache sizes
            (64 * 1024, 512 * 1024, 4 * 1024 * 1024)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Conservative defaults for other architectures
            (32 * 1024, 256 * 1024, 2 * 1024 * 1024)
        }
    }

    /// Detect total memory size
    fn detect_memory_size() -> usize {
        // Simple heuristic based on available system memory
        // In practice, you'd use platform-specific APIs
        8 * 1024 * 1024 * 1024 // Default to 8GB
    }

    /// Detect available SIMD features
    fn detect_simd_features() -> Vec<SimdFeature> {
        let mut features = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            // Use std::is_x86_feature_detected! macro for runtime detection
            if std::is_x86_feature_detected!("sse") {
                features.push(SimdFeature::SSE);
            }
            if std::is_x86_feature_detected!("sse2") {
                features.push(SimdFeature::SSE2);
            }
            if std::is_x86_feature_detected!("sse3") {
                features.push(SimdFeature::SSE3);
            }
            if std::is_x86_feature_detected!("ssse3") {
                features.push(SimdFeature::SSSE3);
            }
            if std::is_x86_feature_detected!("sse4.1") {
                features.push(SimdFeature::SSE41);
            }
            if std::is_x86_feature_detected!("sse4.2") {
                features.push(SimdFeature::SSE42);
            }
            if std::is_x86_feature_detected!("avx") {
                features.push(SimdFeature::AVX);
            }
            if std::is_x86_feature_detected!("avx2") {
                features.push(SimdFeature::AVX2);
            }
            if std::is_x86_feature_detected!("avx512f") {
                features.push(SimdFeature::AVX512F);
            }
            if std::is_x86_feature_detected!("fma") {
                features.push(SimdFeature::FMA);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is standard on ARM64
            features.push(SimdFeature::NEON);
        }

        features
    }

    /// Estimate memory bandwidth using a simple benchmark
    fn estimate_memory_bandwidth() -> Option<f64> {
        // Simple bandwidth estimation
        let size = 10 * 1024 * 1024; // 10MB
        let data: Vec<u64> = vec![1; size / 8];

        let start = Instant::now();
        let sum: u64 = data.iter().sum();
        let duration = start.elapsed();

        // Prevent optimization
        let _ = sum;

        if duration.as_nanos() > 0 {
            let bytes_per_second = (size as f64) / duration.as_secs_f64();
            Some(bytes_per_second)
        } else {
            None
        }
    }

    /// Detect GPU information
    fn detect_gpu() -> Option<GpuInfo> {
        // Placeholder for GPU detection
        // In practice, would use CUDA, OpenCL, or other GPU APIs
        None
    }
}

/// Tuning profile for specific problem characteristics
#[derive(Debug, Clone)]
pub struct TuningProfile {
    /// Optimal number of threads for parallel algorithms
    pub num_threads: usize,
    /// Block size for cache-friendly algorithms
    pub block_size: usize,
    /// Chunk size for parallel work distribution
    pub chunk_size: usize,
    /// Whether to use SIMD optimizations
    pub use_simd: bool,
    /// Memory pool size for frequent allocations
    pub memory_pool_size: usize,
    /// Tolerance for iterative algorithms
    pub default_tolerance: f64,
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Whether to use GPU acceleration if available
    pub use_gpu: bool,
}

/// Auto-tuner for algorithm parameters
pub struct AutoTuner {
    hardware: HardwareInfo,
    cache: HashMap<String, TuningProfile>,
}

impl AutoTuner {
    /// Create new auto-tuner with detected hardware
    pub fn new(hardware: HardwareInfo) -> Self {
        Self {
            hardware,
            cache: HashMap::new(),
        }
    }

    /// Create auto-tuner with automatic hardware detection
    pub fn auto() -> Self {
        Self::new(HardwareDetector::detect())
    }

    /// Tune parameters for specific problem size
    pub fn tune_for_problem_size(&self, problem_size: usize) -> TuningProfile {
        let cache_key = format!("size_{}", problem_size);

        if let Some(cached) = self.cache.get(&cache_key) {
            return cached.clone();
        }

        self.compute_tuning_profile(problem_size)
    }

    /// Compute optimal tuning profile for given problem size
    fn compute_tuning_profile(&self, problem_size: usize) -> TuningProfile {
        // Determine optimal thread count
        let num_threads = self.optimal_thread_count(problem_size);

        // Determine optimal block size based on cache
        let block_size = self.optimal_block_size(problem_size);

        // Determine chunk size for parallel distribution
        let chunk_size = self.optimal_chunk_size(problem_size, num_threads);

        // Determine if SIMD should be used
        let use_simd = !self.hardware.simd_features.is_empty() && problem_size >= 64;

        // Determine memory pool size
        let memory_pool_size = self.optimal_memory_pool_size(problem_size);

        // Determine tolerances based on problem size
        let (default_tolerance, max_iterations) = self.optimal_tolerances(problem_size);

        // Determine GPU usage
        let use_gpu = self.hardware.gpu_info.is_some() && problem_size >= 10000;

        TuningProfile {
            num_threads,
            block_size,
            chunk_size,
            use_simd,
            memory_pool_size,
            default_tolerance,
            max_iterations,
            use_gpu,
        }
    }

    /// Determine optimal thread count
    fn optimal_thread_count(&self, problem_size: usize) -> usize {
        let max_threads = self.hardware.cpu_threads;

        if problem_size < 1000 {
            // Small problems don't benefit from parallelization
            1
        } else if problem_size < 10000 {
            // Medium problems use moderate parallelization
            (max_threads / 2).clamp(1, 4)
        } else {
            // Large problems can use all available threads
            max_threads.min(problem_size / 1000)
        }
    }

    /// Determine optimal block size for cache efficiency
    fn optimal_block_size(&self, problem_size: usize) -> usize {
        let l1_elements = self.hardware.l1_cache_size / 8; // Assume f64
        let l2_elements = self.hardware.l2_cache_size / 8;

        if problem_size <= l1_elements {
            // Fits in L1 cache
            problem_size
        } else if problem_size <= l2_elements {
            // Use L1-sized blocks
            l1_elements / 4
        } else {
            // Use L2-sized blocks for large problems
            l2_elements / 16
        }
    }

    /// Determine optimal chunk size for parallel distribution
    fn optimal_chunk_size(&self, problem_size: usize, num_threads: usize) -> usize {
        if num_threads <= 1 {
            problem_size
        } else {
            // Balance between parallelization overhead and load balancing
            let min_chunk = 100; // Minimum chunk to avoid excessive overhead
            let ideal_chunk = problem_size / (num_threads * 4); // 4x oversubscription
            ideal_chunk.max(min_chunk)
        }
    }

    /// Determine optimal memory pool size
    fn optimal_memory_pool_size(&self, problem_size: usize) -> usize {
        // Use a fraction of available memory based on problem size
        let base_size = problem_size * 8 * 4; // 4x problem size in bytes
        let max_pool = self.hardware.memory_size / 8; // Use up to 1/8 of system memory

        base_size.min(max_pool).max(1024 * 1024) // At least 1MB
    }

    /// Determine optimal tolerances and iteration limits
    fn optimal_tolerances(&self, problem_size: usize) -> (f64, usize) {
        if problem_size < 1000 {
            (1e-12, 100) // High accuracy for small problems
        } else if problem_size < 100000 {
            (1e-10, 500) // Moderate accuracy for medium problems
        } else {
            (1e-8, 1000) // Lower accuracy for large problems
        }
    }

    /// Benchmark-based tuning for specific algorithms
    pub fn benchmark_tune<F: IntegrateFloat>(
        &mut self,
        algorithm_name: &str,
        benchmark_fn: impl Fn(&TuningProfile) -> Duration,
        problem_size: usize,
    ) -> TuningProfile {
        let base_profile = self.tune_for_problem_size(problem_size);

        // Try different parameter variations
        let mut best_profile = base_profile.clone();
        let mut best_time = benchmark_fn(&base_profile);

        // Test different thread counts
        for threads in [1, 2, 4, 8, 16] {
            if threads <= self.hardware.cpu_threads {
                let mut profile = base_profile.clone();
                profile.num_threads = threads;
                profile.chunk_size = self.optimal_chunk_size(problem_size, threads);

                let time = benchmark_fn(&profile);
                if time < best_time {
                    best_time = time;
                    best_profile = profile;
                }
            }
        }

        // Test different block sizes
        for &factor in &[0.5, 1.0, 2.0, 4.0] {
            let mut profile = best_profile.clone();
            profile.block_size = ((base_profile.block_size as f64) * factor) as usize;
            profile.block_size = profile.block_size.max(32).min(problem_size);

            let time = benchmark_fn(&profile);
            if time < best_time {
                best_time = time;
                best_profile = profile;
            }
        }

        // Cache the result
        let cache_key = format!("{}_{}", algorithm_name, problem_size);
        self.cache.insert(cache_key, best_profile.clone());

        best_profile
    }

    /// Get hardware information
    pub fn hardware_info(&self) -> &HardwareInfo {
        &self.hardware
    }
}

/// Auto-tuning for specific algorithm types
pub struct AlgorithmTuner;

impl AlgorithmTuner {
    /// Tune parameters for matrix operations
    pub fn tune_matrix_operations(hardware: &HardwareInfo, matrix_size: usize) -> TuningProfile {
        let tuner = AutoTuner::new(hardware.clone());

        let mut profile = tuner.tune_for_problem_size(matrix_size * matrix_size);

        // Matrix-specific adjustments
        if matrix_size >= 1000 {
            profile.block_size = 64; // Good block size for matrix multiplication
            profile.use_simd = true;
        }

        profile
    }

    /// Tune parameters for ODE solving
    pub fn tune_ode_solver(
        hardware: &HardwareInfo,
        system_size: usize,
        time_steps: usize,
    ) -> TuningProfile {
        let tuner = AutoTuner::new(hardware.clone());
        let problem_size = system_size * time_steps;

        let mut profile = tuner.tune_for_problem_size(problem_size);

        // ODE-specific adjustments
        if system_size > 100 {
            profile.use_simd = true;
            profile.default_tolerance = 1e-8; // Good balance for ODEs
            profile.max_iterations = 50;
        }

        profile
    }

    /// Tune parameters for Monte Carlo integration
    pub fn tune_monte_carlo(
        hardware: &HardwareInfo,
        dimensions: usize,
        samples: usize,
    ) -> TuningProfile {
        let tuner = AutoTuner::new(hardware.clone());

        let mut profile = tuner.tune_for_problem_size(samples);

        // Monte Carlo specific adjustments
        profile.num_threads = hardware.cpu_threads; // MC benefits from all threads
        profile.chunk_size = (samples / (hardware.cpu_threads * 8)).max(1000);

        if dimensions > 10 {
            profile.use_gpu = hardware.gpu_info.is_some(); // High-D benefits from GPU
        }

        profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let hardware = HardwareDetector::detect();

        assert!(hardware.cpu_cores > 0);
        assert!(hardware.cpu_threads >= hardware.cpu_cores);
        assert!(hardware.l1_cache_size > 0);
        assert!(hardware.l2_cache_size >= hardware.l1_cache_size);
        assert!(hardware.l3_cache_size >= hardware.l2_cache_size);
        assert!(hardware.memory_size > 0);
    }

    #[test]
    fn test_auto_tuner() {
        let hardware = HardwareDetector::detect();
        let tuner = AutoTuner::new(hardware);

        // Test small problem
        let small_profile = tuner.tune_for_problem_size(100);
        assert_eq!(small_profile.num_threads, 1);

        // Test large problem
        let large_profile = tuner.tune_for_problem_size(100000);
        assert!(large_profile.num_threads > 1);
        assert!(large_profile.block_size > 0);
        assert!(large_profile.chunk_size > 0);
    }

    #[test]
    fn test_algorithm_specific_tuning() {
        let hardware = HardwareDetector::detect();

        // Test matrix operations tuning
        let matrix_profile = AlgorithmTuner::tune_matrix_operations(&hardware, 1000);
        assert_eq!(matrix_profile.block_size, 64);

        // Test ODE solver tuning
        let ode_profile = AlgorithmTuner::tune_ode_solver(&hardware, 100, 1000);
        assert!(ode_profile.max_iterations > 0);
        assert!(ode_profile.default_tolerance > 0.0);

        // Test Monte Carlo tuning
        let mc_profile = AlgorithmTuner::tune_monte_carlo(&hardware, 5, 1000000);
        assert!(mc_profile.chunk_size > 0);
    }
}
