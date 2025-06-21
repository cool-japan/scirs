//! # Optimization Parameters Generation
//!
//! This module generates optimal parameters for various operations
//! based on detected system resources.

use super::{
    cpu::CpuInfo, gpu::GpuInfo, memory::MemoryInfo, network::NetworkInfo, storage::StorageInfo,
};
use crate::error::CoreResult;

/// Optimization parameters for system operations
#[derive(Debug, Clone)]
pub struct OptimizationParams {
    /// Recommended thread count for parallel operations
    pub thread_count: usize,
    /// Recommended chunk size for memory operations (bytes)
    pub chunk_size: usize,
    /// Enable SIMD operations
    pub enable_simd: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Enable memory prefetching
    pub enable_prefetch: bool,
    /// NUMA-aware memory allocation
    pub numa_aware: bool,
    /// Cache-friendly parameters
    pub cache_params: CacheParams,
    /// I/O optimization parameters
    pub io_params: IoParams,
    /// GPU-specific parameters
    pub gpu_params: Option<GpuParams>,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            thread_count: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            chunk_size: 64 * 1024, // 64KB default
            enable_simd: false,
            enable_gpu: false,
            enable_prefetch: true,
            numa_aware: false,
            cache_params: CacheParams::default(),
            io_params: IoParams::default(),
            gpu_params: None,
        }
    }
}

impl OptimizationParams {
    /// Generate optimization parameters from system resources
    pub fn generate(
        cpu: &CpuInfo,
        memory: &MemoryInfo,
        gpu: Option<&GpuInfo>,
        network: &NetworkInfo,
        storage: &StorageInfo,
    ) -> CoreResult<Self> {
        let thread_count = Self::calculate_optimal_thread_count(cpu, memory);
        let chunk_size = Self::calculate_optimal_chunk_size(cpu, memory, storage);
        let enable_simd = Self::should_enable_simd(cpu);
        let enable_gpu = Self::should_enable_gpu(gpu);
        let enable_prefetch = Self::should_enable_prefetch(memory, storage);
        let numa_aware = memory.numa_nodes > 1;

        let cache_params = CacheParams::from_cpu(cpu);
        let io_params = IoParams::from_resources(network, storage);
        let gpu_params = gpu.map(GpuParams::from_gpu);

        Ok(Self {
            thread_count,
            chunk_size,
            enable_simd,
            enable_gpu,
            enable_prefetch,
            numa_aware,
            cache_params,
            io_params,
            gpu_params,
        })
    }

    /// Calculate optimal thread count
    fn calculate_optimal_thread_count(cpu: &CpuInfo, memory: &MemoryInfo) -> usize {
        let base_threads = cpu.physical_cores;

        // Add hyperthreading benefit for certain workloads
        let ht_benefit = if cpu.logical_cores > cpu.physical_cores {
            (cpu.logical_cores - cpu.physical_cores) / 2
        } else {
            0
        };

        // Consider memory pressure
        let memory_factor = if memory.is_under_pressure() {
            0.75 // Reduce threads under memory pressure
        } else {
            1.0
        };

        let optimal = ((base_threads + ht_benefit) as f64 * memory_factor) as usize;
        optimal.max(1).min(cpu.logical_cores)
    }

    /// Calculate optimal chunk size
    fn calculate_optimal_chunk_size(
        cpu: &CpuInfo,
        memory: &MemoryInfo,
        storage: &StorageInfo,
    ) -> usize {
        // Base on CPU cache size
        let cache_based = cpu.cache_l3_kb * 1024 / 4; // Use 1/4 of L3 cache

        // Base on memory bandwidth
        let memory_based = memory.optimal_chunk_size();

        // Base on storage characteristics
        let storage_based = storage.optimal_io_size;

        // Take the geometric mean to balance all factors
        let geometric_mean = ((cache_based as f64 * memory_based as f64 * storage_based as f64)
            .powf(1.0 / 3.0)) as usize;

        // Ensure it's a reasonable size (between 4KB and 64MB)
        geometric_mean.clamp(4 * 1024, 64 * 1024 * 1024)
    }

    /// Determine if SIMD should be enabled
    fn should_enable_simd(cpu: &CpuInfo) -> bool {
        cpu.simd_capabilities.sse4_2 || cpu.simd_capabilities.avx2 || cpu.simd_capabilities.neon
    }

    /// Determine if GPU should be enabled
    fn should_enable_gpu(gpu: Option<&GpuInfo>) -> bool {
        gpu.map(|g| g.is_compute_capable()).unwrap_or(false)
    }

    /// Determine if prefetching should be enabled
    fn should_enable_prefetch(memory: &MemoryInfo, storage: &StorageInfo) -> bool {
        // Enable prefetch if we have sufficient memory and storage supports it
        !memory.is_under_pressure() && storage.supports_async_io()
    }

    /// Get scaling factor for different problem sizes
    pub fn scaling_factor(&self, problem_size: usize) -> f64 {
        let base_size = 1024 * 1024; // 1MB base
        if problem_size <= base_size {
            1.0
        } else {
            let ratio = problem_size as f64 / base_size as f64;
            // Use square root scaling to avoid excessive resource usage
            ratio.sqrt()
        }
    }

    /// Adjust parameters for specific workload type
    pub fn adjust_for_workload(&mut self, workload: WorkloadType) {
        match workload {
            WorkloadType::CpuIntensive => {
                // Maximize CPU utilization
                self.thread_count = self.thread_count.max(
                    std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(4),
                );
                self.chunk_size = self.chunk_size.max(1024 * 1024); // Larger chunks
            }
            WorkloadType::MemoryIntensive => {
                // Optimize for memory bandwidth
                self.enable_prefetch = true;
                self.chunk_size = self.chunk_size.min(256 * 1024); // Smaller chunks
            }
            WorkloadType::IoIntensive => {
                // Optimize for I/O throughput
                self.thread_count = (self.thread_count * 2).min(16); // More threads for I/O
                self.chunk_size = self.io_params.optimal_buffer_size;
            }
            WorkloadType::GpuIntensive => {
                // Favor GPU over CPU
                if self.enable_gpu {
                    self.thread_count = self.thread_count.min(4); // Fewer CPU threads
                }
            }
        }
    }
}

/// Cache optimization parameters
#[derive(Debug, Clone)]
pub struct CacheParams {
    /// L1 cache line size
    pub cache_line_size: usize,
    /// Optimal data alignment
    pub alignment: usize,
    /// Prefetch distance
    pub prefetch_distance: usize,
    /// Cache-friendly loop tiling size
    pub tile_size: usize,
}

impl Default for CacheParams {
    fn default() -> Self {
        Self {
            cache_line_size: 64,
            alignment: 64,
            prefetch_distance: 64,
            tile_size: 64,
        }
    }
}

impl CacheParams {
    /// Generate cache parameters from CPU info
    pub fn from_cpu(cpu: &CpuInfo) -> Self {
        let cache_line_size = 64; // Most modern CPUs use 64-byte cache lines
        let alignment = cache_line_size;

        // Prefetch distance based on cache size
        let prefetch_distance = (cpu.cache_l1_kb * 1024 / 16).clamp(64, 1024);

        // Tile size based on L1 cache
        let tile_size = (cpu.cache_l1_kb * 1024 / 8).clamp(64, 4096);

        Self {
            cache_line_size,
            alignment,
            prefetch_distance,
            tile_size,
        }
    }
}

/// I/O optimization parameters
#[derive(Debug, Clone)]
pub struct IoParams {
    /// Optimal buffer size for I/O operations
    pub optimal_buffer_size: usize,
    /// Number of concurrent I/O operations
    pub concurrent_operations: usize,
    /// Enable asynchronous I/O
    pub enable_async_io: bool,
    /// Enable I/O caching
    pub enable_io_cache: bool,
}

impl Default for IoParams {
    fn default() -> Self {
        Self {
            optimal_buffer_size: 64 * 1024, // 64KB
            concurrent_operations: 4,
            enable_async_io: true,
            enable_io_cache: true,
        }
    }
}

impl IoParams {
    /// Generate I/O parameters from network and storage info
    pub fn from_resources(network: &NetworkInfo, storage: &StorageInfo) -> Self {
        let optimal_buffer_size = storage.optimal_io_size.max(network.mtu);
        let concurrent_operations = storage.queue_depth.min(16);
        let enable_async_io = storage.supports_async_io();
        let enable_io_cache = !storage.is_ssd() || storage.capacity > 512 * 1024 * 1024 * 1024; // Cache for HDD or large SSDs

        Self {
            optimal_buffer_size,
            concurrent_operations,
            enable_async_io,
            enable_io_cache,
        }
    }
}

/// GPU optimization parameters
#[derive(Debug, Clone)]
pub struct GpuParams {
    /// Optimal workgroup/block size
    pub workgroup_size: usize,
    /// Number of workgroups to launch
    pub workgroup_count: usize,
    /// Shared memory usage per workgroup
    pub shared_memory_size: usize,
    /// Enable unified memory
    pub use_unified_memory: bool,
    /// Optimal data transfer strategy
    pub transfer_strategy: GpuTransferStrategy,
}

impl GpuParams {
    /// Generate GPU parameters from GPU info
    pub fn from_gpu(gpu: &GpuInfo) -> Self {
        let workgroup_size = gpu.optimal_workgroup_size();
        let workgroup_count = (gpu.compute_units * 4).min(65535); // 4 workgroups per compute unit, capped
        let shared_memory_size = 16 * 1024; // 16KB default shared memory
        let use_unified_memory = gpu.features.unified_memory;

        let transfer_strategy = if gpu.memory_bandwidth_gbps > 500.0 {
            GpuTransferStrategy::HighBandwidth
        } else if use_unified_memory {
            GpuTransferStrategy::Unified
        } else {
            GpuTransferStrategy::Standard
        };

        Self {
            workgroup_size,
            workgroup_count,
            shared_memory_size,
            use_unified_memory,
            transfer_strategy,
        }
    }
}

/// GPU data transfer strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTransferStrategy {
    /// Standard host-device transfers
    Standard,
    /// High bandwidth optimized transfers
    HighBandwidth,
    /// Unified memory
    Unified,
    /// Zero-copy transfers
    ZeroCopy,
}

/// Workload type classifications
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
    /// CPU-intensive computations
    CpuIntensive,
    /// Memory-intensive operations
    MemoryIntensive,
    /// I/O-intensive operations
    IoIntensive,
    /// GPU-intensive computations
    GpuIntensive,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_params_generation() {
        let cpu = CpuInfo::default();
        let memory = MemoryInfo::default();
        let gpu = Some(GpuInfo::default());
        let network = NetworkInfo::default();
        let storage = StorageInfo::default();

        let params = OptimizationParams::generate(&cpu, &memory, gpu.as_ref(), &network, &storage);
        assert!(params.is_ok());

        let params = params.unwrap();
        assert!(params.thread_count > 0);
        assert!(params.chunk_size > 0);
    }

    #[test]
    fn test_thread_count_calculation() {
        let cpu = CpuInfo {
            physical_cores: 8,
            logical_cores: 16,
            ..Default::default()
        };
        let memory = MemoryInfo::default();

        let thread_count = OptimizationParams::calculate_optimal_thread_count(&cpu, &memory);
        assert!(thread_count >= 8);
        assert!(thread_count <= 16);
    }

    #[test]
    fn test_chunk_size_calculation() {
        let cpu = CpuInfo {
            cache_l3_kb: 8192, // 8MB L3 cache
            ..Default::default()
        };
        let memory = MemoryInfo::default();
        let storage = StorageInfo::default();

        let chunk_size = OptimizationParams::calculate_optimal_chunk_size(&cpu, &memory, &storage);
        assert!(chunk_size >= 4 * 1024); // At least 4KB
        assert!(chunk_size <= 64 * 1024 * 1024); // At most 64MB
    }

    #[test]
    fn test_workload_adjustment() {
        let mut params = OptimizationParams::default();
        let original_thread_count = params.thread_count;

        params.adjust_for_workload(WorkloadType::CpuIntensive);
        assert!(params.thread_count >= original_thread_count);

        params.adjust_for_workload(WorkloadType::MemoryIntensive);
        assert!(params.enable_prefetch);
    }

    #[test]
    fn test_cache_params() {
        let cpu = CpuInfo {
            cache_l1_kb: 32,
            ..Default::default()
        };

        let cache_params = CacheParams::from_cpu(&cpu);
        assert_eq!(cache_params.cache_line_size, 64);
        assert!(cache_params.tile_size > 0);
    }

    #[test]
    fn test_gpu_params() {
        let gpu = GpuInfo {
            vendor: super::super::gpu::GpuVendor::Nvidia,
            compute_units: 2048,
            features: super::super::gpu::GpuFeatures {
                unified_memory: true,
                ..Default::default()
            },
            ..Default::default()
        };

        let gpu_params = GpuParams::from_gpu(&gpu);
        assert_eq!(gpu_params.workgroup_size, 256); // NVIDIA typical
        assert!(gpu_params.use_unified_memory);
        assert_eq!(gpu_params.transfer_strategy, GpuTransferStrategy::Unified);
    }

    #[test]
    fn test_scaling_factor() {
        let params = OptimizationParams::default();

        assert_eq!(params.scaling_factor(1024), 1.0); // Small problem
        assert!(params.scaling_factor(1024 * 1024 * 4) > 1.0); // Larger problem
    }
}

// Import statements are already handled above
