//! # GPU Detection and Capabilities
//!
//! This module provides GPU detection and capability assessment for
//! accelerated computing workloads.

use crate::error::{CoreError, CoreResult};

/// GPU information and capabilities
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU name/model
    pub name: String,
    /// GPU vendor
    pub vendor: GpuVendor,
    /// Total GPU memory in bytes
    pub memory_total: usize,
    /// Available GPU memory in bytes
    pub memory_available: usize,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Number of compute units (CUDA cores, stream processors, etc.)
    pub compute_units: usize,
    /// Base clock frequency in MHz
    pub base_clock_mhz: usize,
    /// Memory clock frequency in MHz
    pub memory_clock_mhz: usize,
    /// Compute capability/architecture
    pub compute_capability: ComputeCapability,
    /// Supported features
    pub features: GpuFeatures,
    /// Performance characteristics
    pub performance: GpuPerformance,
}

impl GpuInfo {
    /// Detect GPU information
    pub fn detect() -> CoreResult<Self> {
        #[cfg(feature = "gpu")]
        {
            // Try different GPU detection methods
            if let Ok(gpu) = Self::detect_cuda() {
                return Ok(gpu);
            }

            if let Ok(gpu) = Self::detect_opencl() {
                return Ok(gpu);
            }

            if let Ok(gpu) = Self::detect_vulkan() {
                return Ok(gpu);
            }
        }

        // Try platform-specific detection
        #[cfg(target_os = "linux")]
        if let Ok(gpu) = Self::detect_linux() {
            return Ok(gpu);
        }

        #[cfg(target_os = "windows")]
        if let Ok(gpu) = Self::detect_windows() {
            return Ok(gpu);
        }

        #[cfg(target_os = "macos")]
        if let Ok(gpu) = Self::detect_macos() {
            return Ok(gpu);
        }

        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("No GPU detected"),
        ))
    }

    /// Detect CUDA-capable GPU
    #[cfg(feature = "gpu")]
    fn detect_cuda() -> CoreResult<Self> {
        // In a real implementation, this would use CUDA runtime API
        // For now, return a placeholder
        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("CUDA detection not implemented"),
        ))
    }

    /// Detect OpenCL-capable GPU
    #[cfg(feature = "gpu")]
    fn detect_opencl() -> CoreResult<Self> {
        // In a real implementation, this would use OpenCL API
        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("OpenCL detection not implemented"),
        ))
    }

    /// Detect Vulkan-capable GPU
    #[cfg(feature = "gpu")]
    fn detect_vulkan() -> CoreResult<Self> {
        // In a real implementation, this would use Vulkan API
        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("Vulkan detection not implemented"),
        ))
    }

    /// Detect GPU on Linux via sysfs
    #[cfg(target_os = "linux")]
    fn detect_linux() -> CoreResult<Self> {
        use std::fs;

        // Try to detect via /sys/class/drm
        if let Ok(entries) = fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy().starts_with("card") {
                        // Try to read device information
                        let device_path = path.join("device");
                        if let Ok(vendor) = fs::read_to_string(device_path.join("vendor")) {
                            if let Ok(device) = fs::read_to_string(device_path.join("device")) {
                                let vendor_id = vendor.trim();
                                let device_id = device.trim();

                                return Ok(Self::create_from_pci_ids(vendor_id, device_id));
                            }
                        }
                    }
                }
            }
        }

        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("No GPU detected on Linux"),
        ))
    }

    /// Detect GPU on Windows
    #[cfg(target_os = "windows")]
    fn detect_windows() -> CoreResult<Self> {
        // In a real implementation, this would use DXGI or WMI
        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("Windows GPU detection not implemented"),
        ))
    }

    /// Detect GPU on macOS
    #[cfg(target_os = "macos")]
    fn detect_macos() -> CoreResult<Self> {
        // For Apple Silicon, we know it has integrated GPU
        #[cfg(target_arch = "aarch64")]
        {
            return Ok(Self {
                name: "Apple GPU".to_string(),
                vendor: GpuVendor::Apple,
                memory_total: 8 * 1024 * 1024 * 1024, // Unified memory
                memory_available: 6 * 1024 * 1024 * 1024,
                memory_bandwidth_gbps: 200.0,
                compute_units: 8,
                base_clock_mhz: 1000,
                memory_clock_mhz: 2000,
                compute_capability: ComputeCapability::Metal,
                features: GpuFeatures {
                    unified_memory: true,
                    double_precision: true,
                    half_precision: true,
                    tensor_cores: false,
                    ray_tracing: false,
                },
                performance: GpuPerformance {
                    fp32_gflops: 2600.0,
                    fp16_gflops: 5200.0,
                    memory_bandwidth_gbps: 200.0,
                    efficiency_score: 0.9,
                },
            });
        }

        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("macOS GPU detection not implemented"),
        ))
    }

    /// Create GPU info from PCI vendor/device IDs
    fn create_from_pci_ids(vendor_id: &str, device_id: &str) -> Self {
        let vendor = match vendor_id {
            "0x10de" => GpuVendor::Nvidia,
            "0x1002" => GpuVendor::Amd,
            "0x8086" => GpuVendor::Intel,
            _ => GpuVendor::Unknown,
        };

        // This is a simplified mapping - real implementation would have
        // comprehensive device databases
        let (name, memory_gb, compute_units) = match (vendor_id, device_id) {
            ("0x10de", _) => ("NVIDIA GPU".to_string(), 8, 2048),
            ("0x1002", _) => ("AMD GPU".to_string(), 8, 64),
            ("0x8086", _) => ("Intel GPU".to_string(), 4, 96),
            _ => ("Unknown GPU".to_string(), 4, 32),
        };

        Self {
            name,
            vendor,
            memory_total: memory_gb * 1024 * 1024 * 1024,
            memory_available: (memory_gb * 1024 * 1024 * 1024 * 3) / 4, // 75% available
            memory_bandwidth_gbps: 500.0,
            compute_units,
            base_clock_mhz: 1500,
            memory_clock_mhz: 7000,
            compute_capability: ComputeCapability::Unknown,
            features: GpuFeatures::default(),
            performance: GpuPerformance::default(),
        }
    }

    /// Calculate performance score (0.0 to 1.0)
    pub fn performance_score(&self) -> f64 {
        let memory_score = (self.memory_total as f64 / (24.0 * 1024.0 * 1024.0 * 1024.0)).min(1.0); // Normalize to 24GB
        let compute_score = (self.compute_units as f64 / 4096.0).min(1.0); // Normalize to 4096 units
        let bandwidth_score = (self.memory_bandwidth_gbps / 1000.0).min(1.0); // Normalize to 1000 GB/s
        let efficiency_score = self.performance.efficiency_score;

        (memory_score + compute_score + bandwidth_score + efficiency_score) / 4.0
    }

    /// Get optimal workgroup/block size
    pub fn optimal_workgroup_size(&self) -> usize {
        match self.vendor {
            GpuVendor::Nvidia => 256, // Typical for NVIDIA
            GpuVendor::Amd => 64,     // Typical for AMD
            GpuVendor::Intel => 128,  // Typical for Intel
            GpuVendor::Apple => 32,   // Typical for Apple
            GpuVendor::Unknown => 64,
        }
    }

    /// Check if suitable for compute workloads
    pub fn is_compute_capable(&self) -> bool {
        self.memory_total >= 2 * 1024 * 1024 * 1024 && // At least 2GB
        self.compute_units >= 32 // At least 32 compute units
    }

    /// Check if suitable for machine learning
    pub fn is_ml_capable(&self) -> bool {
        self.is_compute_capable() && (self.features.tensor_cores || self.features.half_precision)
    }
}

/// GPU vendor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    /// NVIDIA
    Nvidia,
    /// AMD
    Amd,
    /// Intel
    Intel,
    /// Apple
    Apple,
    /// Unknown vendor
    Unknown,
}

/// GPU compute capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeCapability {
    /// CUDA compute capability
    Cuda(u32, u32), // major, minor
    /// OpenCL version
    OpenCL(u32, u32), // major, minor
    /// Vulkan version
    Vulkan(u32, u32), // major, minor
    /// Metal (Apple)
    Metal,
    /// DirectCompute (Microsoft)
    DirectCompute,
    /// Unknown capability
    Unknown,
}

/// GPU feature support
#[derive(Debug, Clone)]
pub struct GpuFeatures {
    /// Unified memory support
    pub unified_memory: bool,
    /// Double precision (FP64) support
    pub double_precision: bool,
    /// Half precision (FP16) support
    pub half_precision: bool,
    /// Tensor cores or equivalent
    pub tensor_cores: bool,
    /// Ray tracing support
    pub ray_tracing: bool,
}

impl Default for GpuFeatures {
    fn default() -> Self {
        Self {
            unified_memory: false,
            double_precision: true,
            half_precision: false,
            tensor_cores: false,
            ray_tracing: false,
        }
    }
}

/// GPU performance characteristics
#[derive(Debug, Clone)]
pub struct GpuPerformance {
    /// FP32 performance in GFLOPS
    pub fp32_gflops: f64,
    /// FP16 performance in GFLOPS
    pub fp16_gflops: f64,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Overall efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
}

impl Default for GpuPerformance {
    fn default() -> Self {
        Self {
            fp32_gflops: 1000.0,
            fp16_gflops: 2000.0,
            memory_bandwidth_gbps: 500.0,
            efficiency_score: 0.7,
        }
    }
}

/// Multi-GPU information
#[derive(Debug, Clone)]
pub struct MultiGpuInfo {
    /// List of detected GPUs
    pub gpus: Vec<GpuInfo>,
    /// Total combined memory
    pub total_memory: usize,
    /// Whether GPUs support peer-to-peer communication
    pub p2p_capable: bool,
    /// SLI/CrossFire configuration
    pub multi_gpu_config: MultiGpuConfig,
}

impl MultiGpuInfo {
    /// Detect all available GPUs
    pub fn detect() -> CoreResult<Self> {
        let mut gpus = Vec::new();

        // Try to detect multiple GPUs
        // This is simplified - real implementation would enumerate all devices
        if let Ok(gpu) = GpuInfo::detect() {
            gpus.push(gpu);
        }

        let total_memory = gpus.iter().map(|gpu| gpu.memory_total).sum();

        Ok(Self {
            gpus,
            total_memory,
            p2p_capable: false,
            multi_gpu_config: MultiGpuConfig::Single,
        })
    }

    /// Get the best GPU for compute workloads
    pub fn best_compute_gpu(&self) -> Option<&GpuInfo> {
        self.gpus
            .iter()
            .filter(|gpu| gpu.is_compute_capable())
            .max_by(|a, b| {
                a.performance_score()
                    .partial_cmp(&b.performance_score())
                    .unwrap()
            })
    }

    /// Get total compute capability
    pub fn total_compute_units(&self) -> usize {
        self.gpus.iter().map(|gpu| gpu.compute_units).sum()
    }
}

/// Multi-GPU configuration types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiGpuConfig {
    /// Single GPU
    Single,
    /// SLI (NVIDIA)
    Sli,
    /// CrossFire (AMD)
    CrossFire,
    /// NVLink (NVIDIA)
    NvLink,
    /// Independent GPUs
    Independent,
}

impl Default for GpuInfo {
    fn default() -> Self {
        Self {
            name: "Default GPU".to_string(),
            vendor: GpuVendor::Unknown,
            memory_total: 4 * 1024 * 1024 * 1024,     // 4GB
            memory_available: 3 * 1024 * 1024 * 1024, // 3GB
            memory_bandwidth_gbps: 200.0,
            compute_units: 512,
            base_clock_mhz: 1000,
            memory_clock_mhz: 4000,
            compute_capability: ComputeCapability::Unknown,
            features: GpuFeatures::default(),
            performance: GpuPerformance::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_vendor() {
        assert_eq!(GpuVendor::Nvidia, GpuVendor::Nvidia);
        assert_ne!(GpuVendor::Nvidia, GpuVendor::Amd);
    }

    #[test]
    fn test_compute_capability() {
        let cuda_cap = ComputeCapability::Cuda(7, 5);
        assert_eq!(cuda_cap, ComputeCapability::Cuda(7, 5));
        assert_ne!(cuda_cap, ComputeCapability::Metal);
    }

    #[test]
    fn test_gpu_features() {
        let features = GpuFeatures {
            unified_memory: true,
            tensor_cores: true,
            ..Default::default()
        };

        assert!(features.unified_memory);
        assert!(features.tensor_cores);
        assert!(!features.ray_tracing);
    }

    #[test]
    fn test_gpu_performance() {
        let perf = GpuPerformance::default();
        assert!(perf.fp32_gflops > 0.0);
        assert!(perf.efficiency_score >= 0.0 && perf.efficiency_score <= 1.0);
    }

    #[test]
    fn test_pci_id_parsing() {
        let gpu = GpuInfo::create_from_pci_ids("0x10de", "0x1234");
        assert_eq!(gpu.vendor, GpuVendor::Nvidia);
        assert!(gpu.name.contains("NVIDIA"));
    }

    #[test]
    fn test_multi_gpu_config() {
        assert_eq!(MultiGpuConfig::Single, MultiGpuConfig::Single);
        assert_ne!(MultiGpuConfig::Single, MultiGpuConfig::Sli);
    }

    #[test]
    fn test_optimal_workgroup_size() {
        let nvidia_gpu = GpuInfo {
            vendor: GpuVendor::Nvidia,
            ..Default::default()
        };
        assert_eq!(nvidia_gpu.optimal_workgroup_size(), 256);

        let amd_gpu = GpuInfo {
            vendor: GpuVendor::Amd,
            ..Default::default()
        };
        assert_eq!(amd_gpu.optimal_workgroup_size(), 64);
    }
}
