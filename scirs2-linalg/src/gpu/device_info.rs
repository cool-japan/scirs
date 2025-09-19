//! GPU device information and capabilities detection

use super::{GpuDeviceInfo, GpuDeviceType};
use crate::error::LinalgResult;

/// Device capability flags for different GPU features
#[derive(Debug, Clone, Copy)]
pub struct DeviceCapabilities {
    /// Supports double precision (fp64) operations
    pub supports_fp64: bool,
    /// Supports half precision (fp16) operations
    pub supports_fp16: bool,
    /// Supports unified memory
    pub supports_unified_memory: bool,
    /// Supports peer-to-peer memory access
    pub supports_p2p: bool,
    /// Maximum threads per block/work group
    pub max_threads_per_block: usize,
    /// Maximum shared memory per block
    pub max_shared_memory: usize,
    /// Warp/wavefront size
    pub warpsize: usize,
}

/// Performance characteristics of a GPU device
#[derive(Debug, Clone)]
pub struct DevicePerformance {
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f64,
    /// Peak compute performance in GFLOPS (single precision)
    pub peak_gflops_fp32: f64,
    /// Peak compute performance in GFLOPS (double precision)
    pub peak_gflops_fp64: f64,
    /// Memory latency in nanoseconds
    pub memory_latency_ns: f64,
    /// Cache size in bytes
    pub cachesize: usize,
}

/// Extended device information with capabilities and performance data
#[derive(Debug, Clone)]
pub struct ExtendedDeviceInfo {
    /// Basic device information
    pub basic_info: GpuDeviceInfo,
    /// Device capabilities
    pub capabilities: DeviceCapabilities,
    /// Performance characteristics
    pub performance: DevicePerformance,
}

impl ExtendedDeviceInfo {
    /// Create extended device info from basic info
    pub fn from_basic(basicinfo: GpuDeviceInfo) -> Self {
        // Estimate capabilities and performance based on device type
        let (capabilities, performance) = match basicinfo.device_type {
            GpuDeviceType::Cuda => estimate_cuda_specs(&basicinfo),
            GpuDeviceType::OpenCl => estimate_opencl_specs(&basicinfo),
            GpuDeviceType::Rocm => estimate_rocm_specs(&basicinfo),
            GpuDeviceType::Vulkan => estimate_vulkan_specs(&basicinfo),
            GpuDeviceType::Metal => estimate_metal_specs(&basicinfo),
            GpuDeviceType::OneApi => estimate_oneapi_specs(&basicinfo),
            GpuDeviceType::WebGpu => estimate_webgpu_specs(&basicinfo),
        };

        Self {
            basic_info: basicinfo,
            capabilities,
            performance,
        }
    }

    /// Check if this device is suitable for a given workload size
    pub fn is_suitable_for_workload(&self, elements: usize, requiresfp64: bool) -> bool {
        // Check memory requirements (assume 8 bytes per element for safety)
        let memory_required = elements * 8;
        let memory_available = self.basic_info.total_memory;

        if memory_required > memory_available / 2 {
            return false; // Need at least 50% memory available
        }

        // Check precision requirements
        if requiresfp64 && !self.capabilities.supports_fp64 {
            return false;
        }

        true
    }

    /// Estimate performance for matrix multiplication
    pub fn estimate_matmul_performance(&self, m: usize, n: usize, k: usize) -> f64 {
        let ops = 2.0 * m as f64 * n as f64 * k as f64; // FMA operations
        let peak_ops_per_sec = self.performance.peak_gflops_fp32 * 1e9;

        // Simple estimate assuming 50% of peak performance
        ops / (peak_ops_per_sec * 0.5)
    }

    /// Get recommended block size for this device
    pub fn recommended_blocksize(&self) -> (usize, usize) {
        match self.basic_info.device_type {
            GpuDeviceType::Cuda => (32, 32),   // Common CUDA block size
            GpuDeviceType::OpenCl => (16, 16), // Conservative OpenCL size
            GpuDeviceType::Rocm => (32, 32),   // Similar to CUDA
            GpuDeviceType::Vulkan => (32, 32),
            GpuDeviceType::Metal => (16, 16),
            GpuDeviceType::OneApi => (16, 16), // Conservative size for Intel
            GpuDeviceType::WebGpu => (8, 8),   // Small size for web compatibility
        }
    }
}

#[allow(dead_code)]
fn estimate_cuda_specs(info: &GpuDeviceInfo) -> (DeviceCapabilities, DevicePerformance) {
    let capabilities = DeviceCapabilities {
        supports_fp64: info.supports_fp64,
        supports_fp16: info.supports_fp16,
        supports_unified_memory: true, // Most modern CUDA devices
        supports_p2p: true,
        max_threads_per_block: 1024,
        max_shared_memory: 48 * 1024, // 48KB typical
        warpsize: 32,
    };

    // Rough estimates based on compute units and clock frequency
    let estimated_cores = info.compute_units * 64; // Estimate cores per SM
    let peak_gflops = estimated_cores as f64 * info.clock_frequency as f64 * 2.0 / 1000.0;

    let performance = DevicePerformance {
        memory_bandwidth: (info.total_memory as f64 / 1e9) * 10.0, // Rough estimate
        peak_gflops_fp32: peak_gflops,
        peak_gflops_fp64: peak_gflops * 0.5, // Typical ratio
        memory_latency_ns: 400.0,
        cachesize: 256 * 1024, // L2 cache estimate
    };

    (capabilities, performance)
}

#[allow(dead_code)]
fn estimate_opencl_specs(info: &GpuDeviceInfo) -> (DeviceCapabilities, DevicePerformance) {
    let capabilities = DeviceCapabilities {
        supports_fp64: info.supports_fp64,
        supports_fp16: info.supports_fp16,
        supports_unified_memory: false, // Conservative assumption
        supports_p2p: false,
        max_threads_per_block: info.max_work_groupsize,
        max_shared_memory: 32 * 1024, // Conservative estimate
        warpsize: 64,                 // AMD wavefront size or conservative estimate
    };

    let estimated_cores = info.compute_units * 64;
    let peak_gflops = estimated_cores as f64 * info.clock_frequency as f64 * 2.0 / 1000.0;

    let performance = DevicePerformance {
        memory_bandwidth: (info.total_memory as f64 / 1e9) * 8.0,
        peak_gflops_fp32: peak_gflops,
        peak_gflops_fp64: peak_gflops * 0.25, // More conservative
        memory_latency_ns: 500.0,
        cachesize: 128 * 1024,
    };

    (capabilities, performance)
}

#[allow(dead_code)]
fn estimate_rocm_specs(info: &GpuDeviceInfo) -> (DeviceCapabilities, DevicePerformance) {
    let capabilities = DeviceCapabilities {
        supports_fp64: info.supports_fp64,
        supports_fp16: info.supports_fp16,
        supports_unified_memory: true, // Modern AMD GPUs
        supports_p2p: true,
        max_threads_per_block: 1024,
        max_shared_memory: 64 * 1024, // LDS on AMD
        warpsize: 64,                 // AMD wavefront size
    };

    let estimated_cores = info.compute_units * 64;
    let peak_gflops = estimated_cores as f64 * info.clock_frequency as f64 * 2.0 / 1000.0;

    let performance = DevicePerformance {
        memory_bandwidth: (info.total_memory as f64 / 1e9) * 12.0, // AMD typically good bandwidth
        peak_gflops_fp32: peak_gflops,
        peak_gflops_fp64: peak_gflops * 0.5,
        memory_latency_ns: 350.0,
        cachesize: 512 * 1024,
    };

    (capabilities, performance)
}

#[allow(dead_code)]
fn estimate_vulkan_specs(info: &GpuDeviceInfo) -> (DeviceCapabilities, DevicePerformance) {
    // Similar to OpenCL but more conservative
    let capabilities = DeviceCapabilities {
        supports_fp64: info.supports_fp64,
        supports_fp16: info.supports_fp16,
        supports_unified_memory: false,
        supports_p2p: false,
        max_threads_per_block: info.max_work_groupsize,
        max_shared_memory: 32 * 1024,
        warpsize: 32, // Conservative estimate
    };

    let estimated_cores = info.compute_units * 32;
    let peak_gflops = estimated_cores as f64 * info.clock_frequency as f64 * 2.0 / 1000.0;

    let performance = DevicePerformance {
        memory_bandwidth: (info.total_memory as f64 / 1e9) * 6.0,
        peak_gflops_fp32: peak_gflops,
        peak_gflops_fp64: peak_gflops * 0.25,
        memory_latency_ns: 600.0,
        cachesize: 64 * 1024,
    };

    (capabilities, performance)
}

#[allow(dead_code)]
fn estimate_metal_specs(info: &GpuDeviceInfo) -> (DeviceCapabilities, DevicePerformance) {
    let capabilities = DeviceCapabilities {
        supports_fp64: info.supports_fp64,
        supports_fp16: true,           // Apple GPUs generally support fp16
        supports_unified_memory: true, // Apple's unified memory architecture
        supports_p2p: false,
        max_threads_per_block: 1024,
        max_shared_memory: 32 * 1024,
        warpsize: 32, // SIMD group size on Apple GPUs
    };

    let estimated_cores = info.compute_units * 32; // Conservative estimate
    let peak_gflops = estimated_cores as f64 * info.clock_frequency as f64 * 2.0 / 1000.0;

    let performance = DevicePerformance {
        memory_bandwidth: (info.total_memory as f64 / 1e9) * 15.0, // Apple's unified memory is fast
        peak_gflops_fp32: peak_gflops,
        peak_gflops_fp64: peak_gflops * 0.5,
        memory_latency_ns: 200.0, // Unified memory has lower latency
        cachesize: 128 * 1024,
    };

    (capabilities, performance)
}

#[allow(dead_code)]
fn estimate_oneapi_specs(info: &GpuDeviceInfo) -> (DeviceCapabilities, DevicePerformance) {
    let capabilities = DeviceCapabilities {
        supports_fp64: info.supports_fp64,
        supports_fp16: true,           // Intel GPUs support fp16
        supports_unified_memory: true, // Intel unified memory
        supports_p2p: false,
        max_threads_per_block: 256,
        max_shared_memory: 64 * 1024,
        warpsize: 16, // Intel GPU execution units
    };

    let estimated_cores = info.compute_units * 16;
    let peak_gflops = estimated_cores as f64 * info.clock_frequency as f64 * 2.0 / 1000.0;

    let performance = DevicePerformance {
        memory_bandwidth: (info.total_memory as f64 / 1e9) * 8.0,
        peak_gflops_fp32: peak_gflops,
        peak_gflops_fp64: peak_gflops * 0.5,
        memory_latency_ns: 300.0,
        cachesize: 32 * 1024,
    };

    (capabilities, performance)
}

#[allow(dead_code)]
fn estimate_webgpu_specs(info: &GpuDeviceInfo) -> (DeviceCapabilities, DevicePerformance) {
    let capabilities = DeviceCapabilities {
        supports_fp64: false, // WebGPU typically doesn't support fp64
        supports_fp16: false, // Limited fp16 support in WebGPU
        supports_unified_memory: false,
        supports_p2p: false,
        max_threads_per_block: 256,
        max_shared_memory: 16 * 1024, // Limited by web constraints
        warpsize: 32,
    };

    let estimated_cores = info.compute_units * 8; // Conservative for web
    let peak_gflops = estimated_cores as f64 * info.clock_frequency as f64 * 1.0 / 1000.0;

    let performance = DevicePerformance {
        memory_bandwidth: (info.total_memory as f64 / 1e9) * 2.0, // Limited by web APIs
        peak_gflops_fp32: peak_gflops,
        peak_gflops_fp64: 0.0,     // No fp64 support
        memory_latency_ns: 1000.0, // Higher latency in web environment
        cachesize: 8 * 1024,
    };

    (capabilities, performance)
}

/// Benchmark a device to get actual performance characteristics
#[allow(dead_code)]
pub fn benchmark_device_performance(
    device_info: &GpuDeviceInfo,
) -> LinalgResult<DevicePerformance> {
    // This would run actual benchmarks on the device
    // For now, return estimates
    let extended_info = ExtendedDeviceInfo::from_basic(device_info.clone());
    Ok(extended_info.performance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extended_device_info_creation() {
        let basic_info = GpuDeviceInfo {
            device_type: GpuDeviceType::Cuda,
            name: "Test GPU".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB
            compute_units: 80,
            clock_frequency: 1500,
            supports_fp64: true,
            supports_fp16: true,
            max_work_groupsize: 1024,
            memory_bandwidth: 900.0,
            l2_cachesize: 6 * 1024 * 1024,      // 6MB
            shared_memory_per_block: 48 * 1024, // 48KB
            registers_per_block: 65536,
            warpsize: 32,
            max_threads_per_mp: 2048,
            multiprocessor_count: 80,
            supports_tensor_cores: true,
            supports_mixed_precision: true,
            vendor: "NVIDIA".to_string(),
        };

        let extended_info = ExtendedDeviceInfo::from_basic(basic_info);

        assert_eq!(extended_info.basic_info.device_type, GpuDeviceType::Cuda);
        assert_eq!(extended_info.capabilities.warpsize, 32);
        assert!(extended_info.performance.peak_gflops_fp32 > 0.0);
    }

    #[test]
    fn test_workload_suitability() {
        let basic_info = GpuDeviceInfo {
            device_type: GpuDeviceType::Cuda,
            name: "Test GPU".to_string(),
            total_memory: 1024 * 1024 * 1024, // 1GB
            compute_units: 10,
            clock_frequency: 1000,
            supports_fp64: false,
            supports_fp16: true,
            max_work_groupsize: 1024,
            memory_bandwidth: 400.0,
            l2_cachesize: 1024 * 1024,          // 1MB
            shared_memory_per_block: 32 * 1024, // 32KB
            registers_per_block: 32768,
            warpsize: 32,
            max_threads_per_mp: 1536,
            multiprocessor_count: 10,
            supports_tensor_cores: false,
            supports_mixed_precision: false,
            vendor: "Test".to_string(),
        };

        let extended_info = ExtendedDeviceInfo::from_basic(basic_info);

        // Small workload, no fp64 required - should be suitable
        assert!(extended_info.is_suitable_for_workload(1000, false));

        // Large workload - should not be suitable (memory)
        assert!(!extended_info.is_suitable_for_workload(100_000_000, false));

        // Requires fp64 but device doesn't support it
        assert!(!extended_info.is_suitable_for_workload(1000, true));
    }

    #[test]
    fn test_performance_estimation() {
        let basic_info = GpuDeviceInfo {
            device_type: GpuDeviceType::Cuda,
            name: "Test GPU".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,
            compute_units: 80,
            clock_frequency: 1500,
            supports_fp64: true,
            supports_fp16: true,
            max_work_groupsize: 1024,
            memory_bandwidth: 900.0,
            l2_cachesize: 6 * 1024 * 1024,      // 6MB
            shared_memory_per_block: 48 * 1024, // 48KB
            registers_per_block: 65536,
            warpsize: 32,
            max_threads_per_mp: 2048,
            multiprocessor_count: 80,
            supports_tensor_cores: true,
            supports_mixed_precision: true,
            vendor: "Test".to_string(),
        };

        let extended_info = ExtendedDeviceInfo::from_basic(basic_info);

        // Estimate time for 1000x1000x1000 matrix multiplication
        let time_estimate = extended_info.estimate_matmul_performance(1000, 1000, 1000);
        assert!(time_estimate > 0.0);
        assert!(time_estimate < 1.0); // Should be less than 1 second
    }
}
