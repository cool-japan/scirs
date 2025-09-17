//! GPU device detection and management
//!
//! This module handles the detection and management of GPU devices across
//! different backends (CUDA, OpenCL, Metal, ROCm) with automatic fallback to CPU.

use std::fmt::Debug;

use super::config::{GpuBackend, GpuCapabilities, TensorCoresGeneration};
use crate::error::{Result, TimeSeriesError};

/// GPU device manager for detecting and managing GPU devices
#[derive(Debug)]
pub struct GpuDeviceManager {
    /// Available devices
    devices: Vec<GpuCapabilities>,
    /// Current device
    current_device: Option<usize>,
}

impl GpuDeviceManager {
    /// Create a new device manager
    pub fn new() -> Result<Self> {
        // Detect actual GPU devices when dependencies are available
        let mut devices = Vec::new();

        // Try to detect CUDA devices
        if let Some(cuda_devices) = Self::detect_cuda_devices() {
            devices.extend(cuda_devices);
        }

        // Try to detect OpenCL devices
        if let Some(opencl_devices) = Self::detect_opencl_devices() {
            devices.extend(opencl_devices);
        }

        // Try to detect Metal devices (Apple Silicon)
        if let Some(metal_devices) = Self::detect_metal_devices() {
            devices.extend(metal_devices);
        }

        // Try to detect ROCm devices (AMD)
        if let Some(rocm_devices) = Self::detect_rocm_devices() {
            devices.extend(rocm_devices);
        }

        // Always provide CPU fallback if no GPU devices found
        if devices.is_empty() {
            devices.push(GpuCapabilities {
                backend: GpuBackend::CpuFallback,
                compute_capability: None,
                memory: Self::get_system_memory(),
                multiprocessors: Self::get_cpu_cores(),
                supports_fp16: false,
                supports_tensor_cores: false,
                max_threads_per_block: 1,
                tensor_cores_generation: None,
                memory_bandwidth: 100.0, // GB/s - rough estimate for system memory
                tensor_performance: None,
            });
        }

        Ok(Self {
            devices,
            current_device: Some(0), // Default to first device
        })
    }

    /// Get available devices
    pub fn get_devices(&self) -> &[GpuCapabilities] {
        &self.devices
    }

    /// Set current device
    pub fn set_device(&mut self, deviceid: usize) -> Result<()> {
        if deviceid >= self.devices.len() {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Device {deviceid} not available"
            )));
        }
        self.current_device = Some(deviceid);
        Ok(())
    }

    /// Get current device capabilities
    pub fn current_device_capabilities(&self) -> Option<&GpuCapabilities> {
        self.current_device.map(|id| &self.devices[id])
    }

    /// Check if GPU acceleration is available
    pub fn is_gpu_available(&self) -> bool {
        self.devices
            .iter()
            .any(|dev| !matches!(dev.backend, GpuBackend::CpuFallback))
    }

    /// Detect CUDA devices
    fn detect_cuda_devices() -> Option<Vec<GpuCapabilities>> {
        // In a real implementation, this would use CUDA Runtime API
        // For now, simulate detection by checking for common NVIDIA indicators
        #[cfg(target_os = "linux")]
        {
            if std::path::Path::new("/dev/nvidia0").exists()
                || std::path::Path::new("/proc/driver/nvidia").exists()
            {
                return Some(vec![GpuCapabilities {
                    backend: GpuBackend::Cuda,
                    compute_capability: Some((8, 0)), // Simulated A100 capability
                    memory: 40 * 1024 * 1024 * 1024,  // 40GB simulated
                    multiprocessors: 108,
                    supports_fp16: true,
                    supports_tensor_cores: true,
                    max_threads_per_block: 1024,
                    tensor_cores_generation: Some(TensorCoresGeneration::V3), // A100 is gen 3
                    memory_bandwidth: 1555.0,                                 // GB/s for A100
                    tensor_performance: Some(312.0),                          // TOPS for A100 BF16
                }]);
            }
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, could check for nvidia-ml.dll or query WMI
            // For simulation, assume no CUDA devices
        }

        None
    }

    /// Detect OpenCL devices
    fn detect_opencl_devices() -> Option<Vec<GpuCapabilities>> {
        // In a real implementation, this would use OpenCL API
        // Check for common OpenCL indicators
        #[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
        {
            // Simulated OpenCL device detection
            // In real implementation, would enumerate platforms and devices
            if Self::has_opencl_drivers() {
                return Some(vec![GpuCapabilities {
                    backend: GpuBackend::OpenCL,
                    compute_capability: None,
                    memory: 8 * 1024 * 1024 * 1024, // 8GB simulated
                    multiprocessors: 64,
                    supports_fp16: true,
                    supports_tensor_cores: false,
                    max_threads_per_block: 256,
                    tensor_cores_generation: None,
                    memory_bandwidth: 500.0, // GB/s estimate
                    tensor_performance: None,
                }]);
            }
        }

        None
    }

    /// Detect Metal devices (Apple Silicon)
    fn detect_metal_devices() -> Option<Vec<GpuCapabilities>> {
        #[cfg(target_os = "macos")]
        {
            // Check for Apple Silicon or dedicated GPU
            if Self::is_apple_silicon() || Self::has_metal_gpu() {
                return Some(vec![GpuCapabilities {
                    backend: GpuBackend::Metal,
                    compute_capability: None,
                    memory: 16 * 1024 * 1024 * 1024, // 16GB unified memory
                    multiprocessors: 32,             // GPU cores
                    supports_fp16: true,
                    supports_tensor_cores: true, // Neural Engine
                    max_threads_per_block: 1024,
                    tensor_cores_generation: Some(TensorCoresGeneration::V3), // Apple Silicon Neural Engine
                    memory_bandwidth: 400.0,                                  // GB/s for M1 Pro/Max
                    tensor_performance: Some(15.8), // TOPS for M1 Neural Engine
                }]);
            }
        }

        None
    }

    /// Detect ROCm devices (AMD)
    fn detect_rocm_devices() -> Option<Vec<GpuCapabilities>> {
        #[cfg(target_os = "linux")]
        {
            // Check for AMD ROCm installation
            if std::path::Path::new("/opt/rocm").exists()
                || std::path::Path::new("/dev/kfd").exists()
            {
                return Some(vec![GpuCapabilities {
                    backend: GpuBackend::Rocm,
                    compute_capability: None,
                    memory: 32 * 1024 * 1024 * 1024, // 32GB simulated
                    multiprocessors: 120,
                    supports_fp16: true,
                    supports_tensor_cores: false, // AMD uses Matrix Cores, not Tensor Cores
                    max_threads_per_block: 1024,
                    tensor_cores_generation: None, // AMD has MFMA instructions instead
                    memory_bandwidth: 1600.0,      // GB/s for MI250X
                    tensor_performance: Some(383.0), // TOPS for MI250X BF16
                }]);
            }
        }

        None
    }

    /// Check for OpenCL drivers
    fn has_opencl_drivers() -> bool {
        #[cfg(target_os = "linux")]
        {
            std::path::Path::new("/usr/lib/x86_64-linux-gnu/libOpenCL.so").exists()
                || std::path::Path::new("/usr/lib64/libOpenCL.so").exists()
        }
        #[cfg(target_os = "windows")]
        {
            std::path::Path::new("C:/Windows/System32/OpenCL.dll").exists()
        }
        #[cfg(target_os = "macos")]
        {
            std::path::Path::new("/System/Library/Frameworks/OpenCL.framework").exists()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            false
        }
    }

    /// Check if running on Apple Silicon
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn is_apple_silicon() -> bool {
        std::env::consts::ARCH == "aarch64"
    }

    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)]
    fn is_apple_silicon() -> bool {
        false
    }

    /// Check for Metal GPU
    #[cfg(target_os = "macos")]
    #[allow(dead_code)]
    fn has_metal_gpu() -> bool {
        std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists()
    }

    #[cfg(not(target_os = "macos"))]
    #[allow(dead_code)]
    fn has_metal_gpu() -> bool {
        false
    }

    /// Get system memory size
    fn get_system_memory() -> usize {
        #[cfg(target_os = "linux")]
        {
            // Try to read from /proc/meminfo
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                for line in contents.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert KB to bytes
                            }
                        }
                    }
                }
            }
        }

        // Default to 8GB if detection fails
        8 * 1024 * 1024 * 1024
    }

    /// Get number of CPU cores
    fn get_cpu_cores() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4) // Default to 4 cores
    }
}

impl Default for GpuDeviceManager {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            devices: vec![],
            current_device: None,
        })
    }
}
