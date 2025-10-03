//! Advanced chunking strategies for optimal parallel performance
//!
//! This module provides intelligent chunking strategies that automatically
//! adapt to data size, CPU topology, and operation characteristics for
//! optimal parallel performance.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Chunking strategy for parallel operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// Fixed chunk size
    Fixed(usize),
    /// Adaptive chunk size based on data size and CPU count
    Adaptive,
    /// Chunk size optimized for CPU cache lines
    CacheOptimized,
    /// Chunk size optimized for memory bandwidth
    MemoryOptimized,
    /// Dynamic chunk size that adjusts during execution
    Dynamic,
    /// Work-stealing with balanced load distribution
    WorkStealingBalanced,
    /// NUMA-aware chunking for multi-socket systems
    NumaAware,
    /// Optimized for dense linear algebra operations
    LinearAlgebra,
    /// Optimized for sparse matrix operations
    SparseMatrix,
    /// Optimized for FFT and signal processing
    SignalProcessing,
    /// Optimized for image processing operations
    ImageProcessing,
    /// Optimized for Monte Carlo simulations
    MonteCarlo,
    /// Optimized for iterative solvers
    IterativeSolver,
    /// GPU-aware chunking for hybrid CPU/GPU workloads
    GpuAware,
    /// Custom chunking with user-defined function
    Custom(fn(usize, usize) -> usize),
}

/// Configuration for chunking operations
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Chunking strategy to use
    pub strategy: ChunkStrategy,
    /// Minimum chunk size (prevents too small chunks)
    pub min_chunk_size: usize,
    /// Maximum chunk size (prevents too large chunks)
    pub max_chunk_size: usize,
    /// Whether to prefer work-stealing over fixed scheduling
    pub prefer_work_stealing: bool,
    /// Memory access pattern hint
    pub memory_pattern: MemoryPattern,
    /// Computational intensity (ratio of compute to memory operations)
    pub compute_intensity: ComputeIntensity,
    /// Whether to enable performance monitoring
    pub enable_monitoring: bool,
    /// Target load balancing factor (0.0 = perfect balance, 1.0 = allow imbalance)
    pub load_balance_factor: f64,
    /// Cache-awareness level
    pub cache_awareness: CacheAwareness,
    /// NUMA topology consideration
    pub numa_strategy: NumaStrategy,
    /// GPU integration settings
    pub gpu_settings: Option<GpuChunkSettings>,
}

/// Memory access pattern hints
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPattern {
    /// Sequential memory access
    Sequential,
    /// Random memory access
    Random,
    /// Strided memory access with known stride
    Strided(usize),
    /// Block-wise access (e.g., matrix blocks)
    BlockWise,
    /// Sparse access patterns
    Sparse,
    /// Cache-friendly access pattern
    CacheFriendly,
}

/// Computational intensity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeIntensity {
    /// Memory-bound operations (low compute per memory access)
    MemoryBound,
    /// Balanced compute and memory operations
    Balanced,
    /// Compute-intensive operations (high compute per memory access)
    ComputeIntensive,
    /// CPU-bound operations with minimal memory access
    CpuBound,
}

/// Cache awareness levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAwareness {
    /// No cache considerations
    None,
    /// L1 cache awareness
    L1,
    /// L2 cache awareness
    L2,
    /// L3 cache awareness
    L3,
    /// Full cache hierarchy awareness
    Full,
}

/// NUMA topology strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumaStrategy {
    /// Ignore NUMA topology
    Ignore,
    /// Prefer local NUMA nodes
    LocalPreferred,
    /// Strictly enforce NUMA locality
    StrictLocal,
    /// Interleave across NUMA nodes
    Interleave,
    /// Custom NUMA strategy
    Custom,
}

/// GPU chunking settings
#[derive(Debug, Clone)]
pub struct GpuChunkSettings {
    /// Target GPU memory usage ratio (0.0 to 1.0)
    pub gpu_memory_ratio: f64,
    /// Minimum chunk size for GPU transfer efficiency
    pub gpu_min_chunk: usize,
    /// Whether to overlap CPU/GPU computation
    pub overlap_compute: bool,
    /// GPU memory bandwidth (bytes/second)
    pub gpu_bandwidth: Option<u64>,
    /// CPU-GPU transfer bandwidth (bytes/second)
    pub transfer_bandwidth: Option<u64>,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::Adaptive,
            min_chunk_size: 64,
            max_chunk_size: 8192,
            prefer_work_stealing: true,
            memory_pattern: MemoryPattern::Sequential,
            compute_intensity: ComputeIntensity::Balanced,
            enable_monitoring: false,
            load_balance_factor: 0.1,
            cache_awareness: CacheAwareness::L2,
            numa_strategy: NumaStrategy::LocalPreferred,
            gpu_settings: None,
        }
    }
}

impl Default for GpuChunkSettings {
    fn default() -> Self {
        Self {
            gpu_memory_ratio: 0.8,
            gpu_min_chunk: 4096,
            overlap_compute: true,
            gpu_bandwidth: None,
            transfer_bandwidth: None,
        }
    }
}

impl ChunkConfig {
    /// Create a configuration optimized for compute-intensive operations
    pub fn compute_intensive() -> Self {
        Self {
            strategy: ChunkStrategy::Adaptive,
            min_chunk_size: 32,
            max_chunk_size: 1024,
            prefer_work_stealing: true,
            memory_pattern: MemoryPattern::Sequential,
            compute_intensity: ComputeIntensity::ComputeIntensive,
            enable_monitoring: false,
            load_balance_factor: 0.05,
            cache_awareness: CacheAwareness::L1,
            numa_strategy: NumaStrategy::LocalPreferred,
            gpu_settings: None,
        }
    }

    /// Create a configuration optimized for memory-intensive operations
    pub fn memory_intensive() -> Self {
        Self {
            strategy: ChunkStrategy::MemoryOptimized,
            min_chunk_size: 256,
            max_chunk_size: 16384,
            prefer_work_stealing: false,
            memory_pattern: MemoryPattern::Sequential,
            compute_intensity: ComputeIntensity::MemoryBound,
            enable_monitoring: false,
            load_balance_factor: 0.2,
            cache_awareness: CacheAwareness::L3,
            numa_strategy: NumaStrategy::StrictLocal,
            gpu_settings: None,
        }
    }

    /// Create a configuration optimized for cache-friendly operations
    pub fn cache_friendly() -> Self {
        Self {
            strategy: ChunkStrategy::CacheOptimized,
            min_chunk_size: 64,
            max_chunk_size: 4096,
            prefer_work_stealing: true,
            memory_pattern: MemoryPattern::CacheFriendly,
            compute_intensity: ComputeIntensity::Balanced,
            enable_monitoring: false,
            load_balance_factor: 0.1,
            cache_awareness: CacheAwareness::Full,
            numa_strategy: NumaStrategy::LocalPreferred,
            gpu_settings: None,
        }
    }

    /// Create a configuration optimized for linear algebra operations
    pub fn linear_algebra() -> Self {
        Self {
            strategy: ChunkStrategy::LinearAlgebra,
            min_chunk_size: 256,
            max_chunk_size: 8192,
            prefer_work_stealing: false,
            memory_pattern: MemoryPattern::BlockWise,
            compute_intensity: ComputeIntensity::ComputeIntensive,
            enable_monitoring: false,
            load_balance_factor: 0.1,
            cache_awareness: CacheAwareness::L3,
            numa_strategy: NumaStrategy::LocalPreferred,
            gpu_settings: None,
        }
    }

    /// Create a configuration optimized for sparse matrix operations
    pub fn sparse_matrix() -> Self {
        Self {
            strategy: ChunkStrategy::SparseMatrix,
            min_chunk_size: 128,
            max_chunk_size: 4096,
            prefer_work_stealing: true,
            memory_pattern: MemoryPattern::Sparse,
            compute_intensity: ComputeIntensity::MemoryBound,
            enable_monitoring: true,
            load_balance_factor: 0.3,
            cache_awareness: CacheAwareness::L2,
            numa_strategy: NumaStrategy::LocalPreferred,
            gpu_settings: None,
        }
    }

    /// Create a configuration optimized for signal processing
    pub fn signal_processing() -> Self {
        Self {
            strategy: ChunkStrategy::SignalProcessing,
            min_chunk_size: 512,
            max_chunk_size: 16384,
            prefer_work_stealing: false,
            memory_pattern: MemoryPattern::Sequential,
            compute_intensity: ComputeIntensity::ComputeIntensive,
            enable_monitoring: false,
            load_balance_factor: 0.05,
            cache_awareness: CacheAwareness::L2,
            numa_strategy: NumaStrategy::LocalPreferred,
            gpu_settings: None,
        }
    }

    /// Create a configuration optimized for image processing
    pub fn image_processing() -> Self {
        Self {
            strategy: ChunkStrategy::ImageProcessing,
            min_chunk_size: 1024,
            max_chunk_size: 32768,
            prefer_work_stealing: true,
            memory_pattern: MemoryPattern::BlockWise,
            compute_intensity: ComputeIntensity::Balanced,
            enable_monitoring: false,
            load_balance_factor: 0.15,
            cache_awareness: CacheAwareness::L3,
            numa_strategy: NumaStrategy::LocalPreferred,
            gpu_settings: None,
        }
    }

    /// Create a configuration optimized for Monte Carlo simulations
    pub fn monte_carlo() -> Self {
        Self {
            strategy: ChunkStrategy::MonteCarlo,
            min_chunk_size: 1024,
            max_chunk_size: 65536,
            prefer_work_stealing: true,
            memory_pattern: MemoryPattern::Random,
            compute_intensity: ComputeIntensity::ComputeIntensive,
            enable_monitoring: true,
            load_balance_factor: 0.2,
            cache_awareness: CacheAwareness::L1,
            numa_strategy: NumaStrategy::Interleave,
            gpu_settings: None,
        }
    }

    /// Create a configuration optimized for iterative solvers
    pub fn iterative_solver() -> Self {
        Self {
            strategy: ChunkStrategy::IterativeSolver,
            min_chunk_size: 256,
            max_chunk_size: 8192,
            prefer_work_stealing: false,
            memory_pattern: MemoryPattern::Sequential,
            compute_intensity: ComputeIntensity::Balanced,
            enable_monitoring: true,
            load_balance_factor: 0.1,
            cache_awareness: CacheAwareness::L2,
            numa_strategy: NumaStrategy::StrictLocal,
            gpu_settings: None,
        }
    }

    /// Create a configuration for hybrid CPU/GPU workloads
    pub fn gpu_hybrid() -> Self {
        Self {
            strategy: ChunkStrategy::GpuAware,
            min_chunk_size: 4096,
            max_chunk_size: 131072,
            prefer_work_stealing: true,
            memory_pattern: MemoryPattern::Sequential,
            compute_intensity: ComputeIntensity::ComputeIntensive,
            enable_monitoring: true,
            load_balance_factor: 0.2,
            cache_awareness: CacheAwareness::L3,
            numa_strategy: NumaStrategy::LocalPreferred,
            gpu_settings: Some(GpuChunkSettings::default()),
        }
    }

    /// Enable performance monitoring for dynamic optimization
    pub fn with_monitoring(mut self) -> Self {
        self.enable_monitoring = true;
        self
    }

    /// Set NUMA strategy
    pub fn with_numa_strategy(mut self, strategy: NumaStrategy) -> Self {
        self.numa_strategy = strategy;
        self
    }

    /// Set memory pattern hint
    pub fn with_memory_pattern(mut self, pattern: MemoryPattern) -> Self {
        self.memory_pattern = pattern;
        self
    }

    /// Set compute intensity
    pub fn with_compute_intensity(mut self, intensity: ComputeIntensity) -> Self {
        self.compute_intensity = intensity;
        self
    }

    /// Add GPU settings
    pub fn with_gpu_settings(mut self, settings: GpuChunkSettings) -> Self {
        self.gpu_settings = Some(settings);
        self
    }
}

/// Enhanced chunking utilities
pub struct ChunkingUtils;

impl ChunkingUtils {
    /// Get the optimal chunk size based on data size and configuration
    pub fn optimal_chunk_size(data_size: usize, config: &ChunkConfig) -> usize {
        let thread_count = Self::thread_count();
        let cpu_info = Self::get_cpu_info();

        let chunk_size = match config.strategy {
            ChunkStrategy::Fixed(size) => size,
            ChunkStrategy::Adaptive => Self::adaptive_chunk_size(data_size, thread_count),
            ChunkStrategy::CacheOptimized => {
                Self::cache_optimized_chunk_size(data_size, thread_count)
            }
            ChunkStrategy::MemoryOptimized => {
                Self::memory_optimized_chunk_size(data_size, thread_count)
            }
            ChunkStrategy::Dynamic => Self::dynamic_chunk_size(data_size, thread_count),
            ChunkStrategy::WorkStealingBalanced => {
                Self::work_stealing_chunk_size(data_size, thread_count)
            }
            ChunkStrategy::NumaAware => {
                Self::numa_aware_chunk_size(data_size, thread_count, &cpu_info)
            }
            ChunkStrategy::LinearAlgebra => {
                Self::linear_algebra_chunk_size(data_size, thread_count, &cpu_info)
            }
            ChunkStrategy::SparseMatrix => Self::sparse_matrix_chunk_size(data_size, thread_count),
            ChunkStrategy::SignalProcessing => {
                Self::signal_processing_chunk_size(data_size, thread_count)
            }
            ChunkStrategy::ImageProcessing => {
                Self::image_processing_chunk_size(data_size, thread_count)
            }
            ChunkStrategy::MonteCarlo => Self::monte_carlo_chunk_size(data_size, thread_count),
            ChunkStrategy::IterativeSolver => {
                Self::iterative_solver_chunk_size(data_size, thread_count)
            }
            ChunkStrategy::GpuAware => Self::gpu_aware_chunk_size(data_size, thread_count, config),
            ChunkStrategy::Custom(func) => func(data_size, thread_count),
        };

        // Apply cache awareness adjustment
        let cache_adjusted =
            Self::apply_cache_awareness(chunk_size, config.cache_awareness, &cpu_info);

        // Apply memory pattern adjustment
        let pattern_adjusted = Self::apply_memory_pattern(cache_adjusted, config.memory_pattern);

        // Apply compute intensity adjustment
        let intensity_adjusted =
            Self::apply_compute_intensity(pattern_adjusted, config.compute_intensity);

        // Clamp to min/max bounds
        intensity_adjusted
            .max(config.min_chunk_size)
            .min(config.max_chunk_size)
    }

    /// Calculate adaptive chunk size based on data size and thread count
    fn adaptive_chunk_size(data_size: usize, thread_count: usize) -> usize {
        if data_size < 1000 {
            // Small data: use sequential processing
            data_size
        } else if data_size < 10000 {
            // Medium data: balance between overhead and parallelism
            (data_size / thread_count).max(64)
        } else {
            // Large data: optimize for work distribution
            let base_chunk = data_size / (thread_count * 4); // 4 chunks per thread
            base_chunk.max(256).min(8192)
        }
    }

    /// Calculate cache-optimized chunk size
    fn cache_optimized_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // Assume L2 cache line size of 64 bytes and L2 cache size of 256KB
        const CACHE_LINE_SIZE: usize = 64;
        const L2_CACHE_SIZE: usize = 256 * 1024;

        // Target chunk size that fits well in L2 cache
        let cache_friendly_size = L2_CACHE_SIZE / std::mem::size_of::<f32>() / 4; // 16K elements

        // Align to cache line boundaries
        let target_chunk = (data_size / thread_count).min(cache_friendly_size);
        let aligned_chunk = (target_chunk / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;

        aligned_chunk.max(CACHE_LINE_SIZE)
    }

    /// Calculate memory-optimized chunk size for bandwidth-bound operations
    fn memory_optimized_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // For memory-bound operations, use larger chunks to improve bandwidth utilization
        let target_chunk = data_size / thread_count;

        // Aim for chunks that allow good memory prefetching
        target_chunk.max(1024).min(32768)
    }

    /// Calculate dynamic chunk size (starts conservative, can be adjusted)
    fn dynamic_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // Start with a conservative chunk size that can be adjusted during execution
        let base_chunk = data_size / (thread_count * 8); // Start with more chunks
        base_chunk.max(128).min(2048)
    }

    /// Get the number of available threads
    fn thread_count() -> usize {
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads().max(1)
        }
        #[cfg(not(feature = "parallel"))]
        {
            1
        }
    }

    /// Get CPU information for optimization
    fn get_cpu_info() -> CpuInfo {
        CpuInfo {
            l1_cache_size: 32 * 1024,       // 32KB typical L1 cache
            l2_cache_size: 256 * 1024,      // 256KB typical L2 cache
            l3_cache_size: 8 * 1024 * 1024, // 8MB typical L3 cache
            cache_line_size: 64,            // 64 bytes typical cache line
            numa_nodes: 1,                  // Assume single NUMA node for simplicity
            cores_per_numa: Self::thread_count(),
        }
    }

    /// Work-stealing optimized chunk size
    fn work_stealing_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // Use smaller chunks for better work distribution
        let target_chunks = thread_count * 16; // 16 chunks per thread
        (data_size / target_chunks).max(32)
    }

    /// NUMA-aware chunk size calculation
    fn numa_aware_chunk_size(data_size: usize, thread_count: usize, cpu_info: &CpuInfo) -> usize {
        let chunks_per_numa = thread_count / cpu_info.numa_nodes;
        let chunk_size = data_size / (cpu_info.numa_nodes * chunks_per_numa);

        // Align to cache line boundaries
        let aligned = (chunk_size / cpu_info.cache_line_size) * cpu_info.cache_line_size;
        aligned.max(cpu_info.cache_line_size)
    }

    /// Linear algebra optimized chunk size
    fn linear_algebra_chunk_size(
        data_size: usize,
        thread_count: usize,
        cpu_info: &CpuInfo,
    ) -> usize {
        // Optimize for cache blocking in matrix operations
        let cache_size = cpu_info.l3_cache_size / 4; // Use 1/4 of L3 cache
        let elements_per_cache = cache_size / std::mem::size_of::<f64>();

        let sqrt_elements = (elements_per_cache as f64).sqrt() as usize;
        let block_size = sqrt_elements.max(64).min(512);

        // Ensure multiple of block size for efficient tiling
        ((data_size / thread_count) / block_size) * block_size
    }

    /// Sparse matrix optimized chunk size
    fn sparse_matrix_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // Use smaller chunks due to irregular memory access patterns
        let base_chunk = data_size / (thread_count * 8);
        base_chunk.max(128).min(2048)
    }

    /// Signal processing optimized chunk size
    fn signal_processing_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // Optimize for FFT-friendly sizes (powers of 2)
        let target_chunk = data_size / thread_count;
        let log2_chunk = (target_chunk as f64).log2().floor() as u32;
        let power_of_2_chunk = 2_usize.pow(log2_chunk);

        power_of_2_chunk.max(512).min(16384)
    }

    /// Image processing optimized chunk size
    fn image_processing_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // Optimize for 2D block processing
        let pixels_per_thread = data_size / thread_count;

        // Assume square blocks and find nearest square root
        let block_side = (pixels_per_thread as f64).sqrt() as usize;
        let block_size = block_side * block_side;

        block_size.max(1024).min(32768)
    }

    /// Monte Carlo optimized chunk size
    fn monte_carlo_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // Use large chunks for independent random sampling
        let chunk_size = data_size / thread_count;
        chunk_size.max(4096).min(131072)
    }

    /// Iterative solver optimized chunk size
    fn iterative_solver_chunk_size(data_size: usize, thread_count: usize) -> usize {
        // Balance between parallelism and convergence locality
        let chunk_size = data_size / (thread_count * 2);
        chunk_size.max(256).min(8192)
    }

    /// GPU-aware chunk size calculation
    fn gpu_aware_chunk_size(data_size: usize, thread_count: usize, config: &ChunkConfig) -> usize {
        if let Some(gpu_settings) = &config.gpu_settings {
            // Consider GPU memory capacity and transfer costs
            let cpu_chunk = data_size / thread_count;
            let gpu_preferred = gpu_settings.gpu_min_chunk;

            // Balance between CPU and GPU chunk preferences
            if data_size > gpu_preferred * 4 {
                gpu_preferred.max(cpu_chunk / 2)
            } else {
                cpu_chunk
            }
        } else {
            Self::adaptive_chunk_size(data_size, thread_count)
        }
    }

    /// Apply cache awareness to chunk size
    fn apply_cache_awareness(
        chunk_size: usize,
        awareness: CacheAwareness,
        cpu_info: &CpuInfo,
    ) -> usize {
        match awareness {
            CacheAwareness::None => chunk_size,
            CacheAwareness::L1 => {
                let l1_elements = cpu_info.l1_cache_size / std::mem::size_of::<f64>();
                chunk_size.min(l1_elements / 2)
            }
            CacheAwareness::L2 => {
                let l2_elements = cpu_info.l2_cache_size / std::mem::size_of::<f64>();
                chunk_size.min(l2_elements / 2)
            }
            CacheAwareness::L3 => {
                let l3_elements = cpu_info.l3_cache_size / std::mem::size_of::<f64>();
                chunk_size.min(l3_elements / 4)
            }
            CacheAwareness::Full => {
                // Use the most restrictive cache level
                let l1_elements = cpu_info.l1_cache_size / std::mem::size_of::<f64>();
                chunk_size.min(l1_elements / 4)
            }
        }
    }

    /// Apply memory pattern adjustment
    fn apply_memory_pattern(chunk_size: usize, pattern: MemoryPattern) -> usize {
        match pattern {
            MemoryPattern::Sequential => chunk_size,
            MemoryPattern::Random => chunk_size / 2, // Smaller chunks for random access
            MemoryPattern::Strided(stride) => {
                // Align chunk size to stride for efficient access
                ((chunk_size / stride) * stride).max(stride)
            }
            MemoryPattern::BlockWise => {
                // Optimize for block-wise access patterns
                let block_size = 64; // Typical block size
                ((chunk_size / block_size) * block_size).max(block_size)
            }
            MemoryPattern::Sparse => chunk_size / 4, // Much smaller chunks for sparse data
            MemoryPattern::CacheFriendly => chunk_size, // Already optimized
        }
    }

    /// Apply compute intensity adjustment
    fn apply_compute_intensity(chunk_size: usize, intensity: ComputeIntensity) -> usize {
        match intensity {
            ComputeIntensity::MemoryBound => chunk_size * 2, // Larger chunks for memory-bound
            ComputeIntensity::Balanced => chunk_size,
            ComputeIntensity::ComputeIntensive => chunk_size / 2, // Smaller chunks for better load balance
            ComputeIntensity::CpuBound => chunk_size / 4,         // Very small chunks for CPU-bound
        }
    }

    /// Process data with optimal chunking strategy
    pub fn chunked_map<T, R, F>(data: &[T], config: &ChunkConfig, map_fn: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync,
    {
        let chunk_size = Self::optimal_chunk_size(data.len(), config);

        #[cfg(feature = "parallel")]
        {
            if config.prefer_work_stealing {
                // Use work-stealing for better load balancing
                data.par_iter()
                    .with_min_len(chunk_size)
                    .map(|x| map_fn(x))
                    .collect()
            } else {
                // Use fixed chunking for predictable memory access patterns
                data.par_chunks(chunk_size)
                    .map(|chunk| chunk.iter().map(|x| map_fn(x)).collect::<Vec<_>>())
                    .flatten()
                    .collect()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            data.iter().map(map_fn).collect()
        }
    }

    /// Process pairs of data with optimal chunking
    pub fn chunked_zip_map<T, U, R, F>(
        data_a: &[T],
        data_b: &[U],
        config: &ChunkConfig,
        map_fn: F,
    ) -> Vec<R>
    where
        T: Sync,
        U: Sync,
        R: Send,
        F: Fn(&T, &U) -> R + Sync,
    {
        assert_eq!(
            data_a.len(),
            data_b.len(),
            "Arrays must have the same length"
        );

        let chunk_size = Self::optimal_chunk_size(data_a.len(), config);

        #[cfg(feature = "parallel")]
        {
            if config.prefer_work_stealing {
                data_a
                    .par_iter()
                    .with_min_len(chunk_size)
                    .zip(data_b.par_iter())
                    .map(|(a, b)| map_fn(a, b))
                    .collect()
            } else {
                data_a
                    .par_chunks(chunk_size)
                    .zip(data_b.par_chunks(chunk_size))
                    .map(|(chunk_a, chunk_b)| {
                        chunk_a
                            .iter()
                            .zip(chunk_b.iter())
                            .map(|(a, b)| map_fn(a, b))
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .collect()
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            data_a
                .iter()
                .zip(data_b.iter())
                .map(|(a, b)| map_fn(a, b))
                .collect()
        }
    }

    /// Reduce operation with optimal chunking
    pub fn chunked_reduce<T, F>(data: &[T], config: &ChunkConfig, identity: T, reduce_fn: F) -> T
    where
        T: Clone + Send + Sync,
        F: Fn(T, T) -> T + Sync,
    {
        let chunk_size = Self::optimal_chunk_size(data.len(), config);

        #[cfg(feature = "parallel")]
        {
            data.par_chunks(chunk_size)
                .map(|chunk| {
                    chunk
                        .iter()
                        .cloned()
                        .fold(identity.clone(), |acc, x| reduce_fn(acc, x))
                })
                .reduce(|| identity.clone(), |a, b| reduce_fn(a, b))
        }
        #[cfg(not(feature = "parallel"))]
        {
            data.iter().cloned().fold(identity, reduce_fn)
        }
    }
}

/// Extension trait for enhanced parallel operations on slices
pub trait ParallelSliceExt<T> {
    /// Map with automatic chunk optimization
    fn chunked_map<R, F>(&self, config: &ChunkConfig, map_fn: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync;

    /// Reduce with automatic chunk optimization
    fn chunked_reduce<F>(&self, config: &ChunkConfig, identity: T, reduce_fn: F) -> T
    where
        T: Clone + Send + Sync,
        F: Fn(T, T) -> T + Sync;
}

impl<T> ParallelSliceExt<T> for [T] {
    fn chunked_map<R, F>(&self, config: &ChunkConfig, map_fn: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync,
    {
        ChunkingUtils::chunked_map(self, config, map_fn)
    }

    fn chunked_reduce<F>(&self, config: &ChunkConfig, identity: T, reduce_fn: F) -> T
    where
        T: Clone + Send + Sync,
        F: Fn(T, T) -> T + Sync,
    {
        ChunkingUtils::chunked_reduce(self, config, identity, reduce_fn)
    }
}

/// CPU information for optimization
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub l1_cache_size: usize,
    pub l2_cache_size: usize,
    pub l3_cache_size: usize,
    pub cache_line_size: usize,
    pub numa_nodes: usize,
    pub cores_per_numa: usize,
}

/// Performance monitoring for dynamic chunk optimization
#[derive(Debug, Clone)]
pub struct ChunkPerformanceMonitor {
    measurements: Vec<ChunkMeasurement>,
    optimal_sizes: std::collections::HashMap<String, usize>,
}

/// Individual chunk performance measurement
#[derive(Debug, Clone)]
pub struct ChunkMeasurement {
    pub chunk_size: usize,
    pub data_size: usize,
    pub execution_time: std::time::Duration,
    pub throughput: f64,
    pub operation_type: String,
}

impl ChunkPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            optimal_sizes: std::collections::HashMap::new(),
        }
    }

    /// Record a performance measurement
    pub fn record_measurement(&mut self, measurement: ChunkMeasurement) {
        self.measurements.push(measurement);

        // Keep only recent measurements (last 1000)
        if self.measurements.len() > 1000 {
            self.measurements.remove(0);
        }

        self.update_optimal_sizes();
    }

    /// Update optimal chunk sizes based on measurements
    fn update_optimal_sizes(&mut self) {
        for measurement in &self.measurements {
            let key = format!(
                "{}_{}k",
                measurement.operation_type,
                measurement.data_size / 1000
            );

            let current_optimal = self
                .optimal_sizes
                .get(&key)
                .unwrap_or(&measurement.chunk_size);

            // Simple heuristic: prefer chunk sizes with higher throughput
            if measurement.throughput > self.get_throughput_for_size(*current_optimal, &key) {
                self.optimal_sizes.insert(key, measurement.chunk_size);
            }
        }
    }

    /// Get recorded throughput for a specific chunk size and operation
    fn get_throughput_for_size(&self, chunk_size: usize, operation_key: &str) -> f64 {
        self.measurements
            .iter()
            .filter(|m| {
                m.chunk_size == chunk_size
                    && format!("{}_{}k", m.operation_type, m.data_size / 1000) == operation_key
            })
            .map(|m| m.throughput)
            .fold(0.0, f64::max)
    }

    /// Get optimal chunk size for an operation type and data size
    pub fn get_optimal_size(&self, operation_type: &str, data_size: usize) -> Option<usize> {
        let key = format!("{}_{}k", operation_type, data_size / 1000);
        self.optimal_sizes.get(&key).copied()
    }

    /// Get performance statistics
    pub fn get_statistics(&self) -> ChunkStatistics {
        if self.measurements.is_empty() {
            return ChunkStatistics::default();
        }

        let throughputs: Vec<f64> = self.measurements.iter().map(|m| m.throughput).collect();
        let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let max_throughput = throughputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_throughput = throughputs.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        ChunkStatistics {
            total_measurements: self.measurements.len(),
            avg_throughput,
            max_throughput,
            min_throughput,
            optimal_operations: self.optimal_sizes.len(),
        }
    }
}

impl Default for ChunkPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from chunk performance monitoring
#[derive(Debug, Clone, Default)]
pub struct ChunkStatistics {
    pub total_measurements: usize,
    pub avg_throughput: f64,
    pub max_throughput: f64,
    pub min_throughput: f64,
    pub optimal_operations: usize,
}

/// Matrix-specific chunking utilities
pub struct MatrixChunking;

impl MatrixChunking {
    /// Calculate optimal chunk size for matrix multiplication
    pub fn matrix_multiply_chunks(
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) -> (usize, usize, usize) {
        // Cache-oblivious matrix multiplication blocking
        let cache_size = 256 * 1024; // L2 cache size
        let element_size = std::mem::size_of::<f64>();
        let cache_elements = cache_size / element_size;

        // Three matrices should fit in cache
        let block_size = ((cache_elements / 3) as f64).sqrt() as usize;
        let block_size = block_size.max(64).min(512);

        (
            block_size.min(rows_a),
            block_size.min(cols_a),
            block_size.min(cols_b),
        )
    }

    /// Calculate chunks for 2D array processing
    pub fn array_2d_chunks(rows: usize, cols: usize, thread_count: usize) -> (usize, usize) {
        // Try to maintain square-ish chunks
        let total_elements = rows * cols;
        let elements_per_thread = total_elements / thread_count;

        if rows >= cols {
            // More rows than columns - chunk by rows
            let chunk_rows = (elements_per_thread / cols).max(1);
            (chunk_rows, cols)
        } else {
            // More columns than rows - chunk by columns
            let chunk_cols = (elements_per_thread / rows).max(1);
            (rows, chunk_cols)
        }
    }

    /// Calculate chunks for 3D array processing
    pub fn array_3d_chunks(
        depth: usize,
        rows: usize,
        cols: usize,
        thread_count: usize,
    ) -> (usize, usize, usize) {
        let total_elements = depth * rows * cols;
        let elements_per_thread = total_elements / thread_count;

        // Prefer chunking along the largest dimension
        if depth >= rows && depth >= cols {
            let chunk_depth = (elements_per_thread / (rows * cols)).max(1);
            (chunk_depth, rows, cols)
        } else if rows >= cols {
            let chunk_rows = (elements_per_thread / (depth * cols)).max(1);
            (depth, chunk_rows, cols)
        } else {
            let chunk_cols = (elements_per_thread / (depth * rows)).max(1);
            (depth, rows, chunk_cols)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_size_calculation() {
        let config = ChunkConfig::default();

        // Small data should use larger relative chunk size
        let small_chunk = ChunkingUtils::optimal_chunk_size(100, &config);
        assert!(small_chunk >= config.min_chunk_size);

        // Large data should use reasonable chunk size
        let large_chunk = ChunkingUtils::optimal_chunk_size(100000, &config);
        assert!(large_chunk >= config.min_chunk_size);
        assert!(large_chunk <= config.max_chunk_size);
    }

    #[test]
    fn test_chunked_map() {
        let data: Vec<i32> = (0..1000).collect();
        let config = ChunkConfig::compute_intensive();

        let result = data.chunked_map(&config, |&x| x * x);
        let expected: Vec<i32> = (0..1000).map(|x| x * x).collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_chunked_zip_map() {
        let data_a: Vec<i32> = (0..1000).collect();
        let data_b: Vec<i32> = (1000..2000).collect();
        let config = ChunkConfig::memory_intensive();

        let result = ChunkingUtils::chunked_zip_map(&data_a, &data_b, &config, |&a, &b| a + b);
        let expected: Vec<i32> = (0..1000).map(|x| x + (x + 1000)).collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_chunked_reduce() {
        let data: Vec<i32> = (1..=100).collect();
        let config = ChunkConfig::cache_friendly();

        let result = data.chunked_reduce(&config, 0, |a, b| a + b);
        let expected = (1..=100).sum::<i32>();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_specialized_configs() {
        let linear_algebra = ChunkConfig::linear_algebra();
        assert_eq!(linear_algebra.strategy, ChunkStrategy::LinearAlgebra);
        assert_eq!(linear_algebra.memory_pattern, MemoryPattern::BlockWise);

        let sparse_matrix = ChunkConfig::sparse_matrix();
        assert_eq!(sparse_matrix.strategy, ChunkStrategy::SparseMatrix);
        assert_eq!(sparse_matrix.memory_pattern, MemoryPattern::Sparse);

        let monte_carlo = ChunkConfig::monte_carlo();
        assert_eq!(monte_carlo.strategy, ChunkStrategy::MonteCarlo);
        assert_eq!(monte_carlo.memory_pattern, MemoryPattern::Random);
    }

    #[test]
    fn test_matrix_chunking() {
        let (row_chunk, col_chunk_a, col_chunk_b) =
            MatrixChunking::matrix_multiply_chunks(1000, 800, 600);
        assert!(row_chunk > 0 && row_chunk <= 1000);
        assert!(col_chunk_a > 0 && col_chunk_a <= 800);
        assert!(col_chunk_b > 0 && col_chunk_b <= 600);

        let (chunk_rows, chunk_cols) = MatrixChunking::array_2d_chunks(100, 200, 4);
        assert!(chunk_rows > 0 && chunk_rows <= 100);
        assert!(chunk_cols > 0 && chunk_cols <= 200);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = ChunkPerformanceMonitor::new();

        let measurement = ChunkMeasurement {
            chunk_size: 1024,
            data_size: 10000,
            execution_time: std::time::Duration::from_millis(10),
            throughput: 1000.0,
            operation_type: "matrix_multiply".to_string(),
        };

        monitor.record_measurement(measurement);
        let stats = monitor.get_statistics();
        assert_eq!(stats.total_measurements, 1);
        assert_eq!(stats.max_throughput, 1000.0);
    }

    #[test]
    fn test_cache_awareness() {
        let cpu_info = CpuInfo {
            l1_cache_size: 32 * 1024,
            l2_cache_size: 256 * 1024,
            l3_cache_size: 8 * 1024 * 1024,
            cache_line_size: 64,
            numa_nodes: 1,
            cores_per_numa: 4,
        };

        let chunk_size = 10000;
        let l1_adjusted =
            ChunkingUtils::apply_cache_awareness(chunk_size, CacheAwareness::L1, &cpu_info);
        let l3_adjusted =
            ChunkingUtils::apply_cache_awareness(chunk_size, CacheAwareness::L3, &cpu_info);

        assert!(l1_adjusted <= chunk_size);
        assert!(l3_adjusted <= chunk_size);
        assert!(l1_adjusted <= l3_adjusted); // L1 should be more restrictive
    }

    #[test]
    fn test_memory_pattern_adjustment() {
        let base_size = 1000;

        let random_adjusted = ChunkingUtils::apply_memory_pattern(base_size, MemoryPattern::Random);
        let sparse_adjusted = ChunkingUtils::apply_memory_pattern(base_size, MemoryPattern::Sparse);
        let sequential_adjusted =
            ChunkingUtils::apply_memory_pattern(base_size, MemoryPattern::Sequential);

        assert!(random_adjusted <= base_size);
        assert!(sparse_adjusted <= base_size);
        assert_eq!(sequential_adjusted, base_size);
        assert!(sparse_adjusted < random_adjusted); // Sparse should be smallest
    }

    #[test]
    fn test_compute_intensity_adjustment() {
        let base_size = 1000;

        let memory_bound =
            ChunkingUtils::apply_compute_intensity(base_size, ComputeIntensity::MemoryBound);
        let compute_intensive =
            ChunkingUtils::apply_compute_intensity(base_size, ComputeIntensity::ComputeIntensive);
        let cpu_bound =
            ChunkingUtils::apply_compute_intensity(base_size, ComputeIntensity::CpuBound);

        assert!(memory_bound >= base_size); // Memory-bound should use larger chunks
        assert!(compute_intensive <= base_size); // Compute-intensive should use smaller chunks
        assert!(cpu_bound <= base_size); // CPU-bound should use smallest chunks
        assert!(cpu_bound <= compute_intensive);
    }

    #[test]
    fn test_gpu_chunk_settings() {
        let gpu_settings = GpuChunkSettings::default();
        assert_eq!(gpu_settings.gpu_memory_ratio, 0.8);
        assert_eq!(gpu_settings.gpu_min_chunk, 4096);
        assert!(gpu_settings.overlap_compute);

        let config = ChunkConfig::gpu_hybrid();
        assert!(config.gpu_settings.is_some());
        assert_eq!(config.strategy, ChunkStrategy::GpuAware);
    }
}
